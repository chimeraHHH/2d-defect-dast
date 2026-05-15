from __future__ import annotations
from math import pi
import numbers

from ase.units import Bohr, Ha
from ase.utils.timing import timer
import numpy as np

from gpaw.band_descriptor import BandDescriptor
from gpaw.blacs import BlacsDescriptor, BlacsGrid, Redistributor
from gpaw.lfc import BasisFunctions
from gpaw.matrix_descriptor import MatrixDescriptor
from gpaw.pw.descriptor import PWDescriptor
from gpaw.pw.lfc import PWLFC
from gpaw.typing import Array2D
from gpaw.utilities import unpack_hermitian
from gpaw.utilities.blas import axpy
from gpaw.utilities.progressbar import ProgressBar
from gpaw.wavefunctions.arrays import PlaneWaveExpansionWaveFunctions
from gpaw.wavefunctions.fdpw import FDPWWaveFunctions
from gpaw.wavefunctions.mode import Mode
import gpaw
import gpaw.cgpaw as cgpaw
import gpaw.fftw as fftw


class PW(Mode):
    name = 'pw'

    def __init__(self,
                 ecut: float = 340,
                 *,
                 fftwflags: int = fftw.MEASURE,
                 cell: Array2D | None = None,
                 gammacentered=False,  # Deprecated
                 pulay_stress: float | None = None,
                 dedecut: float | str | None = None,
                 force_complex_dtype: bool = False,
                 interpolation: str | int = 'fft'):
        """Plane-wave basis mode.

        Only one of ``dedecut`` and ``pulay_stress`` can be used.

        Parameters
        ==========
        ecut: float
            Plane-wave cutoff in eV.
        gammacentered: bool
            Center the grid of chosen plane waves around the
            gamma point or q/k-vector
        dedecut:
            Estimate of derivative of total energy with respect to
            plane-wave cutoff.  Used to calculate the Pulay-stress.
        pulay_stress: float or None
            Pulay-stress correction.
        fftwflags: int
            Flags for making an FFTW plan.  There are 4 possibilities
            (default is MEASURE)::

                from gpaw.fftw import ESTIMATE, MEASURE, PATIENT, EXHAUSTIVE

        cell: np.ndarray
            Use this unit cell to chose the plane-waves.
        interpolation : str or int
            Interpolation scheme to construct the density on the fine grid.
            Default is ``'fft'`` and alternatively a stencil (integer) can
            be given to perform an explicit real-space interpolation.
        """

        assert not gammacentered
        self.ecut = ecut / Ha
        # Don't do expensive planning in dry-run mode:
        self.fftwflags = fftwflags if not gpaw.dry_run else fftw.MEASURE
        self.dedecut = dedecut
        self.interpolation = interpolation
        self.pulay_stress = (None
                             if pulay_stress is None
                             else pulay_stress * Bohr**3 / Ha)

        assert pulay_stress is None or dedecut is None

        if cell is None:
            self.cell_cv = None
        else:
            self.cell_cv = cell / Bohr

        Mode.__init__(self, force_complex_dtype)

    def __call__(self, parallel, initksl, gd, **kwargs):
        dedepsilon = 0.0

        if self.cell_cv is None:
            ecut = self.ecut
        else:
            volume0 = abs(np.linalg.det(self.cell_cv))
            ecut = self.ecut * (volume0 / gd.volume)**(2 / 3.0)

        if self.pulay_stress is not None:
            dedepsilon = self.pulay_stress * gd.volume
        elif self.dedecut is not None:
            if self.dedecut == 'estimate':
                dedepsilon = 'estimate'
            else:
                dedepsilon = self.dedecut * 2 / 3 * ecut

        wfs = PWWaveFunctions(ecut,
                              self.fftwflags, dedepsilon,
                              parallel, initksl, gd=gd,
                              **kwargs)

        return wfs

    def todict(self):
        dct = Mode.todict(self)
        dct['ecut'] = self.ecut * Ha

        if self.cell_cv is not None:
            dct['cell'] = self.cell_cv * Bohr
        if self.pulay_stress is not None:
            dct['pulay_stress'] = self.pulay_stress * Ha / Bohr**3
        if self.dedecut is not None:
            dct['dedecut'] = self.dedecut
        if self.interpolation != 'fft':
            dct['interpolation'] = self.interpolation
        return dct


class Preconditioner:
    """Preconditioner for KS equation.

    From:

      Teter, Payne and Allen, Phys. Rev. B 40, 12255 (1989)

    as modified by:

      Kresse and Furthmüller, Phys. Rev. B 54, 11169 (1996)
    """

    def __init__(self, G2_qG, pd, _scale=1.0):
        self.G2_qG = G2_qG
        self.pd = pd
        self._scale = _scale

    def calculate_kinetic_energy(self, psit_xG, kpt):
        if psit_xG.ndim == 1:
            return self.calculate_kinetic_energy(psit_xG[np.newaxis], kpt)[0]
        G2_G = self.G2_qG[kpt.q]
        return np.array([self.pd.integrate(0.5 * G2_G * psit_G, psit_G).real
                         for psit_G in psit_xG]) * self._scale

    def __call__(self, R_xG, kpt, ekin_x, out=None):
        if out is None:
            out = np.empty_like(R_xG)
        G2_G = self.G2_qG[kpt.q]
        if R_xG.ndim == 1:
            cgpaw.pw_precond(G2_G, R_xG, ekin_x, out)
        else:
            for PR_G, R_G, ekin in zip(out, R_xG, ekin_x):
                cgpaw.pw_precond(G2_G, R_G, ekin, PR_G)
        return out


class NonCollinearPreconditioner(Preconditioner):
    def calculate_kinetic_energy(self, psit_xsG, kpt):
        shape = psit_xsG.shape
        ekin_xs = Preconditioner.calculate_kinetic_energy(
            self, psit_xsG.reshape((-1, shape[-1])), kpt)
        return ekin_xs.reshape(shape[:-1]).sum(-1)

    def __call__(self, R_sG, kpt, ekin, out=None):
        return Preconditioner.__call__(self, R_sG, kpt, [ekin, ekin], out)


class PWWaveFunctions(FDPWWaveFunctions):
    mode = 'pw'

    def __init__(self, ecut, fftwflags, dedepsilon,
                 parallel, initksl,
                 reuse_wfs_method, collinear,
                 gd, nvalence, setups, bd, dtype,
                 world, kd, kptband_comm, timer):
        self.ecut = ecut
        self.fftwflags = fftwflags
        self.dedepsilon = dedepsilon  # Pulay correction for stress tensor

        self.ng_k = None  # number of G-vectors for all IBZ k-points

        FDPWWaveFunctions.__init__(self, parallel, initksl,
                                   reuse_wfs_method=reuse_wfs_method,
                                   collinear=collinear,
                                   gd=gd, nvalence=nvalence, setups=setups,
                                   bd=bd, dtype=dtype, world=world, kd=kd,
                                   kptband_comm=kptband_comm, timer=timer)
        self.read_from_file_init_wfs_dm = False

    def empty(self, n=(), global_array=False, realspace=False, q=None):
        if isinstance(n, numbers.Integral):
            n = (n,)
        if realspace:
            return self.gd.empty(n, self.dtype, global_array)
        elif global_array:
            return np.zeros(n + (self.pd.ngmax,), complex)
        elif q is None:
            return np.zeros(n + (self.pd.maxmyng,), complex)
        else:
            return self.pd.empty(n, self.dtype, q)

    def integrate(self, a_xg, b_yg=None, global_integral=True):
        return self.pd.integrate(a_xg, b_yg, global_integral)

    def bytes_per_wave_function(self):
        return 16 * self.pd.maxmyng

    def set_setups(self, setups):
        self.timer.start('PWDescriptor')
        self.pd = PWDescriptor(self.ecut, self.gd, self.dtype, self.kd,
                               self.fftwflags)
        self.timer.stop('PWDescriptor')

        # Build array of number of plane wave coefficiants for all k-points
        # in the IBZ:
        self.ng_k = np.zeros(self.kd.nibzkpts, dtype=int)
        for kpt in self.kpt_u:
            if kpt.s != 1:  # avoid double counting (only sum over s=0 or None)
                self.ng_k[kpt.k] = len(self.pd.Q_qG[kpt.q])
        self.kd.comm.sum(self.ng_k)

        self.pt = PWLFC([setup.pt_j for setup in setups], self.pd)

        FDPWWaveFunctions.set_setups(self, setups)

        if self.dedepsilon == 'estimate':
            dedecut = self.setups.estimate_dedecut(self.ecut)
            self.dedepsilon = dedecut * 2 / 3 * self.ecut

    def get_pseudo_partial_waves(self):
        return PWLFC([setup.get_partial_waves_for_atomic_orbitals()
                      for setup in self.setups], self.pd)

    def __str__(self):
        s = 'Wave functions: Plane wave expansion\n'
        s += '  Cutoff energy: %.3f eV\n' % (self.pd.ecut * Ha)

        if self.dtype == float:
            s += ('  Number of coefficients: %d (reduced to %d)\n' %
                  (self.pd.ngmax * 2 - 1, self.pd.ngmax))
        else:
            s += ('  Number of coefficients (min, max): %d, %d\n' %
                  (self.pd.ngmin, self.pd.ngmax))

        stress = self.dedepsilon / self.gd.volume * Ha / Bohr**3
        dedecut = 1.5 * self.dedepsilon / self.ecut
        s += ('  Pulay-stress correction: {:.6f} eV/Ang^3 '
              '(de/decut={:.6f})\n'.format(stress, dedecut))

        if fftw.have_fftw():
            s += '  Using FFTW library\n'
        else:
            s += "  Using Numpy's FFT\n"
        return s + FDPWWaveFunctions.__str__(self)

    def make_preconditioner(self, block=1):
        if self.collinear:
            return Preconditioner(self.pd.G2_qG, self.pd)
        return NonCollinearPreconditioner(self.pd.G2_qG, self.pd)

    @timer('Apply H')
    def apply_pseudo_hamiltonian(self, kpt, ham, psit_xG, Htpsit_xG):
        """Apply the pseudo Hamiltonian i.e. without PAW corrections."""
        if not self.collinear:
            self.apply_pseudo_hamiltonian_nc(kpt, ham, psit_xG, Htpsit_xG)
            return

        N = len(psit_xG)
        S = self.gd.comm.size

        vt_R = self.gd.collect(ham.vt_sG[kpt.s], broadcast=True)
        Q_G = self.pd.Q_qG[kpt.q]
        T_G = 0.5 * self.pd.G2_qG[kpt.q]

        for n1 in range(0, N, S):
            n2 = min(n1 + S, N)
            psit_G = self.pd.alltoall1(psit_xG[n1:n2], kpt.q)
            with self.timer('HMM T'):
                np.multiply(T_G, psit_xG[n1:n2], Htpsit_xG[n1:n2])
            if psit_G is not None:
                psit_R = self.pd.ifft(psit_G, kpt.q, local=True, safe=False)
                psit_R *= vt_R
                self.pd.fftplan.execute()
                vtpsit_G = self.pd.tmp_Q.ravel()[Q_G]
            else:
                vtpsit_G = self.pd.tmp_G
            self.pd.alltoall2(vtpsit_G, kpt.q, Htpsit_xG[n1:n2])

        ham.xc.apply_orbital_dependent_hamiltonian(
            kpt, psit_xG, Htpsit_xG, ham.dH_asp)

    def apply_pseudo_hamiltonian_nc(self, kpt, ham, psit_xG, Htpsit_xG):
        Htpsit_xG[:] = 0.5 * self.pd.G2_qG[kpt.q] * psit_xG
        v, x, y, z = ham.vt_xG
        iy = y * 1j
        for psit_sG, Htpsit_sG in zip(psit_xG, Htpsit_xG):
            a = self.pd.ifft(psit_sG[0], kpt.q)
            b = self.pd.ifft(psit_sG[1], kpt.q)
            Htpsit_sG[0] += self.pd.fft(a * (v + z) + b * (x - iy), kpt.q)
            Htpsit_sG[1] += self.pd.fft(a * (x + iy) + b * (v - z), kpt.q)

    def add_orbital_density(self, nt_G, kpt, n):
        axpy(1.0, abs(self.pd.ifft(kpt.psit_nG[n], kpt.q))**2, nt_G)

    def add_to_density_from_k_point_with_occupation(self, nt_xR, kpt, f_n):
        if not self.collinear:
            self.add_to_density_from_k_point_with_occupation_nc(
                nt_xR, kpt, f_n)
            return

        comm = self.gd.comm

        nt_R = self.gd.zeros(global_array=True)

        for n1 in range(0, self.bd.mynbands, comm.size):
            n2 = min(n1 + comm.size, self.bd.mynbands)
            psit_G = self.pd.alltoall1(kpt.psit.array[n1:n2], kpt.q)
            if psit_G is not None:
                f = f_n[n1 + comm.rank]
                psit_R = self.pd.ifft(psit_G, kpt.q, local=True, safe=False)
                # Same as nt_R += f * abs(psit_R)**2, but much faster:
                cgpaw.add_to_density(f, psit_R, nt_R)

        comm.sum(nt_R)
        nt_R = self.gd.distribute(nt_R)
        nt_xR[kpt.s] += nt_R

    def add_to_density_from_k_point_with_occupation_nc(self, nt_xR, kpt, f_n):
        for f, psit_sG in zip(f_n, kpt.psit.array):
            p1 = self.pd.ifft(psit_sG[0], kpt.q)
            p2 = self.pd.ifft(psit_sG[1], kpt.q)
            p11 = p1.real**2 + p1.imag**2
            p22 = p2.real**2 + p2.imag**2
            p12 = p1.conj() * p2
            nt_xR[0] += f * (p11 + p22)
            nt_xR[1] += 2 * f * p12.real
            nt_xR[2] += 2 * f * p12.imag
            nt_xR[3] += f * (p11 - p22)

    def add_to_kinetic_energy_density_kpt(self, kpt, psit_xG, taut_xR):
        N = self.bd.mynbands
        S = self.gd.comm.size
        Gpsit_xG = np.empty((S,) + psit_xG.shape[1:], dtype=psit_xG.dtype)
        taut_R = self.gd.zeros(global_array=True)
        G_Gv = self.pd.get_reciprocal_vectors(q=kpt.q)
        comm = self.gd.comm

        for v in range(3):
            for n1 in range(0, N, S):
                n2 = min(n1 + S, N)
                dn = n2 - n1
                Gpsit_xG[:dn] = 1j * G_Gv[:, v] * psit_xG[n1:n2]
                Gpsit_G = self.pd.alltoall1(Gpsit_xG[:dn], kpt.q)
                if Gpsit_G is not None:
                    f = kpt.f_n[n1 + comm.rank]
                    a_R = self.pd.ifft(Gpsit_G, kpt.q, local=True, safe=False)
                    cgpaw.add_to_density(0.5 * f, a_R, taut_R)

        comm.sum(taut_R)
        taut_R = self.gd.distribute(taut_R)
        taut_xR[kpt.s] += taut_R

    def add_to_ke_crossterms_kpt(self, kpt, psit_xG, taut_xR):
        N = self.bd.mynbands
        S = self.gd.comm.size
        Gpsit_xG = np.empty((S,) + psit_xG.shape[1:], dtype=psit_xG.dtype)
        taut_wR = self.gd.zeros((6,), global_array=True)
        G_Gv = self.pd.get_reciprocal_vectors(q=kpt.q)
        comm = self.gd.comm

        for n1 in range(0, N, S):
            n2 = min(n1 + S, N)
            dn = n2 - n1
            a_vR = {}
            for v in range(3):
                Gpsit_xG[:dn] = 1j * G_Gv[:, v] * psit_xG[n1:n2]
                Gpsit_G = self.pd.alltoall1(Gpsit_xG[:dn], kpt.q)
                if Gpsit_G is not None:
                    f = kpt.f_n[n1 + comm.rank]
                    a_vR[v] = self.pd.ifft(Gpsit_G, kpt.q,
                                           local=True, safe=True)
            if len(a_vR) == 3:
                f = kpt.f_n[n1 + comm.rank]
                w = 0
                for v1 in range(3):
                    for v2 in range(v1, 3):
                        # imaginary parts should cancel
                        taut_wR[w] += f * (a_vR[v1].conj()
                                           * a_vR[v2]).real
                        w += 1
            elif len(a_vR) == 0:
                pass
            else:
                raise RuntimeError('Parallelization issue')

        comm.sum(taut_wR)
        taut_wR = self.gd.distribute(taut_wR)
        taut_xR[kpt.s, :] += taut_wR

    def calculate_kinetic_energy_density(self):
        if self.kpt_u[0].f_n is None:
            return None

        taut_sR = self.gd.zeros(self.nspins)
        for kpt in self.kpt_u:
            self.add_to_kinetic_energy_density_kpt(kpt, kpt.psit_nG, taut_sR)

        self.kptband_comm.sum(taut_sR)
        for taut_R in taut_sR:
            self.kd.symmetry.symmetrize(taut_R, self.gd)
        return taut_sR

    def calculate_kinetic_energy_density_crossterms(self):
        if self.kpt_u[0].f_n is None:
            return None

        taut_svvR = self.gd.zeros((self.nspins, 6))
        for kpt in self.kpt_u:
            self.add_to_ke_crossterms_kpt(kpt, kpt.psit_nG, taut_svvR)

        self.kptband_comm.sum(taut_svvR)
        for taut_R in taut_svvR.reshape(-1, *taut_svvR.shape[-3:]):
            self.kd.symmetry.symmetrize(taut_R, self.gd)
        return taut_svvR

    def apply_mgga_orbital_dependent_hamiltonian(self, kpt, psit_xG,
                                                 Htpsit_xG, dH_asp,
                                                 dedtaut_R):
        N = len(psit_xG)
        S = self.gd.comm.size
        Q_G = self.pd.Q_qG[kpt.q]
        Gpsit_xG = np.empty((S,) + psit_xG.shape[1:], dtype=psit_xG.dtype)
        tmp_xG = np.empty((S,) + psit_xG.shape[1:], dtype=Htpsit_xG.dtype)
        G_Gv = self.pd.get_reciprocal_vectors(q=kpt.q)

        dedtaut_R = self.gd.collect(dedtaut_R, broadcast=True)

        for v in range(3):
            for n1 in range(0, N, S):
                n2 = min(n1 + S, N)
                dn = n2 - n1
                Gpsit_xG[:dn] = 1j * G_Gv[:, v] * psit_xG[n1:n2]
                tmp_xG[:] = 0
                Gpsit_G = self.pd.alltoall1(Gpsit_xG[:dn], kpt.q)
                if Gpsit_G is not None:
                    a_R = self.pd.ifft(Gpsit_G, kpt.q, local=True, safe=False)
                    a_R *= dedtaut_R
                    self.pd.fftplan.execute()
                    a_R = self.pd.tmp_Q.ravel()[Q_G]
                else:
                    a_R = self.pd.tmp_G
                self.pd.alltoall2(a_R, kpt.q, tmp_xG[:dn])
                axpy(-0.5, (1j * G_Gv[:, v] * tmp_xG[:dn]).ravel(),
                     Htpsit_xG[n1:n2].ravel())

    def _get_wave_function_array(self, u, n, realspace=True, periodic=False):
        kpt = self.kpt_u[u]
        psit_G = kpt.psit_nG[n]

        if realspace:
            if psit_G.ndim == 2:
                psit_R = np.array([self.pd.ifft(psits_G, kpt.q)
                                   for psits_G in psit_G])
            else:
                psit_R = self.pd.ifft(psit_G, kpt.q)
            if self.kd.gamma or periodic:
                return psit_R

            k_c = self.kd.ibzk_kc[kpt.k]
            eikr_R = self.gd.plane_wave(k_c)
            return psit_R * eikr_R

        return psit_G

    def get_wave_function_array(self, n, k, s, realspace=True,
                                cut=True, periodic=False):
        kpt_rank, q = self.kd.get_rank_and_index(k)
        u = q * self.nspins + s
        band_rank, myn = self.bd.who_has(n)

        rank = self.world.rank
        if (self.kd.comm.rank == kpt_rank and
            self.bd.comm.rank == band_rank):
            psit_G = self._get_wave_function_array(u, myn, realspace, periodic)

            if realspace:
                psit_G = self.gd.collect(psit_G)
            else:
                assert not cut
                tmp_G = self.pd.gather(psit_G, self.kpt_u[u].q)
                if tmp_G is not None:
                    ng = self.pd.ngmax
                    if self.collinear:
                        psit_G = np.zeros(ng, complex)
                    else:
                        psit_G = np.zeros((2, ng), complex)
                    psit_G[..., :tmp_G.shape[-1]] = tmp_G

            if rank == 0:
                return psit_G

            # Domain master send this to the global master
            if self.gd.comm.rank == 0:
                self.world.ssend(psit_G, 0, 1398)

        if rank == 0:
            # allocate full wave function and receive
            shape = () if self.collinear else (2,)
            psit_G = self.empty(shape, global_array=True,
                                realspace=realspace)
            # XXX this will fail when using non-standard nesting
            # of communicators.
            world_rank = (kpt_rank * self.gd.comm.size *
                          self.bd.comm.size +
                          band_rank * self.gd.comm.size)
            self.world.receive(psit_G, world_rank, 1398)
            return psit_G

        # We return a number instead of None on all the slaves.  Most of
        # the time the return value will be ignored on the slaves, but
        # in some cases it will be multiplied by some other number and
        # then ignored.  Allowing for this will simplify some code here
        # and there.
        return np.nan

    def write(self, writer, write_wave_functions=False):
        FDPWWaveFunctions.write(self, writer)

        if not write_wave_functions:
            return

        if self.collinear:
            shape = (self.nspins,
                     self.kd.nibzkpts, self.bd.nbands, self.pd.ngmax)
        else:
            shape = (self.kd.nibzkpts, self.bd.nbands, 2, self.pd.ngmax)

        writer.add_array('coefficients', shape, complex)

        c = Bohr**-1.5
        for s in range(self.nspins):
            for k in range(self.kd.nibzkpts):
                for n in range(self.bd.nbands):
                    psit_G = self.get_wave_function_array(n, k, s,
                                                          realspace=False,
                                                          cut=False)
                    writer.fill(psit_G * c)

        writer.add_array('indices', (self.kd.nibzkpts, self.pd.ngmax),
                         np.int32)

        if self.bd.comm.rank > 0:
            return

        Q_G = np.empty(self.pd.ngmax, np.int32)
        kk = 0
        for r in range(self.kd.comm.size):
            for q, k in enumerate(self.kd.get_indices(r)):
                ng = self.ng_k[k]
                if r == self.kd.comm.rank:
                    Q_G[:ng] = self.pd.Q_qG[q]
                    if r > 0:
                        self.kd.comm.send(Q_G, 0)
                if self.kd.comm.rank == 0:
                    if r > 0:
                        self.kd.comm.receive(Q_G, r)
                    Q_G[ng:] = -1
                    writer.fill(Q_G)
                    assert k == kk
                    kk += 1

    def read(self, reader):
        FDPWWaveFunctions.read(self, reader)

        if 'coefficients' not in reader.wave_functions:
            return

        Q_kG = reader.wave_functions.indices
        for kpt in self.kpt_u:
            if kpt.s == 0:
                Q_G = Q_kG[kpt.k]
                ng = self.ng_k[kpt.k]
                assert (Q_G[:ng] == self.pd.Q_qG[kpt.q]).all()
                assert (Q_G[ng:] == -1).all()

        c = reader.bohr**1.5
        if reader.version < 0:
            c = 1  # old gpw file
        elif reader.version >= 4:
            c *= self.gd.N_c.prod()
        for kpt in self.kpt_u:
            ng = self.ng_k[kpt.k]
            index = (kpt.s, kpt.k) if self.collinear else (kpt.k,)
            psit_nG = reader.wave_functions.proxy('coefficients', *index)
            psit_nG.scale = c
            psit_nG.length_of_last_dimension = ng

            kpt.psit = PlaneWaveExpansionWaveFunctions(
                self.bd.nbands, self.pd, self.dtype, psit_nG,
                kpt=kpt.q, dist=(self.bd.comm, self.bd.comm.size),
                spin=kpt.s, collinear=self.collinear)

        if self.world.size > 1:
            # Read to memory:
            for kpt in self.kpt_u:
                kpt.psit.read_from_file()
                self.read_from_file_init_wfs_dm = True

    def hs(self, ham, q=-1, s=0, md=None):
        npw = len(self.pd.Q_qG[q])
        N = self.pd.tmp_R.size

        if md is None:
            H_GG = np.zeros((npw, npw), complex)
            S_GG = np.zeros((npw, npw), complex)
            G1 = 0
            G2 = npw
        else:
            H_GG = md.zeros(dtype=complex)
            S_GG = md.zeros(dtype=complex)
            if S_GG.size == 0:
                return H_GG, S_GG
            G1, G2 = next(md.my_blocks(S_GG))[:2]

        H_GG.ravel()[G1::npw + 1] = (0.5 * self.pd.gd.dv / N *
                                     self.pd.G2_qG[q][G1:G2])
        for G in range(G1, G2):
            x_G = self.pd.zeros(q=q)
            x_G[G] = 1.0
            H_GG[G - G1] += (self.pd.gd.dv / N *
                             self.pd.fft(ham.vt_sG[s] *
                                         self.pd.ifft(x_G, q), q))

        if ham.xc.type == 'MGGA':
            G_Gv = self.pd.get_reciprocal_vectors(q=q)
            for G in range(G1, G2):
                x_G = self.pd.zeros(q=q)
                x_G[G] = 1.0
                for v in range(3):
                    a_R = self.pd.ifft(1j * G_Gv[:, v] * x_G, q)
                    H_GG[G - G1] += (self.pd.gd.dv / N *
                                     (-0.5) * 1j * G_Gv[:, v] *
                                     self.pd.fft(ham.xc.dedtaut_sG[s] *
                                                 a_R, q))

        S_GG.ravel()[G1::npw + 1] = self.pd.gd.dv / N

        f_GI = self.pt.expand(q)
        nI = f_GI.shape[1]
        dH_II = np.zeros((nI, nI))
        dS_II = np.zeros((nI, nI))
        I1 = 0
        for a in self.pt.my_atom_indices:
            dH_ii = unpack_hermitian(ham.dH_asp[a][s])
            dS_ii = self.setups[a].dO_ii
            I2 = I1 + len(dS_ii)
            dH_II[I1:I2, I1:I2] = dH_ii / N**2
            dS_II[I1:I2, I1:I2] = dS_ii / N**2
            I1 = I2

        H_GG += np.dot(f_GI[G1:G2].conj(), np.dot(dH_II, f_GI.T))
        S_GG += np.dot(f_GI[G1:G2].conj(), np.dot(dS_II, f_GI.T))

        return H_GG, S_GG

    @timer('Full diag')
    def diagonalize_full_hamiltonian(self, ham, atoms, log,
                                     nbands=None, ecut=None, scalapack=None,
                                     expert=False):

        if self.dtype != complex:
            raise ValueError(
                'Please use mode=PW(..., force_complex_dtype=True)')

        if self.gd.comm.size > 1:
            raise ValueError(
                "Please use parallel={'domain': 1}")

        S = self.bd.comm.size

        if nbands is None and ecut is None:
            nbands = self.pd.ngmin // S * S
        elif nbands is None:
            ecut /= Ha
            # XXX I have seen this nbands expression elsewhere,
            # extract to function!
            nbands = int(self.gd.volume * ecut**1.5 * 2**0.5 / 3 / pi**2)

        if nbands % S != 0:
            nbands += S - nbands % S

        assert nbands <= self.pd.ngmin

        if expert:
            iu = nbands
        else:
            iu = None

        self.bd = bd = BandDescriptor(nbands, self.bd.comm)
        self.occupations.bd = bd

        log(f'Diagonalizing full Hamiltonian ({nbands} lowest bands)')
        log('Matrix size (min, max): {}, {}'.format(self.pd.ngmin,
                                                    self.pd.ngmax))
        mem = 3 * self.pd.ngmax**2 * 16 / S / 1024**2
        log('Approximate memory used per core to store H_GG, S_GG: {:.3f} MB'
            .format(mem))
        log('Notice: Up to twice the amount of memory might be allocated\n'
            'during diagonalization algorithm.')
        log('The least memory is required when the parallelization is purely\n'
            'over states (bands) and not k-points, set '
            "GPAW(..., parallel={'kpt': 1}, ...).")

        if S > 1:
            if isinstance(scalapack, (list, tuple)):
                nprow, npcol, b = scalapack
                assert nprow * npcol == S, (nprow, npcol, S)
            else:
                nprow = int(round(S**0.5))
                while S % nprow != 0:
                    nprow -= 1
                npcol = S // nprow
                b = 64
            log(f'ScaLapack grid: {nprow}x{npcol},',
                'block-size:', b)
            bg = BlacsGrid(bd.comm, S, 1)
            bg2 = BlacsGrid(bd.comm, nprow, npcol)
            scalapack = True
        else:
            scalapack = False

        self.set_positions(atoms.get_scaled_positions())
        self.kpt_u[0].projections = None
        self.allocate_arrays_for_projections(self.pt.my_atom_indices)

        myslice = bd.get_slice()

        pb = ProgressBar(log.fd)
        nkpt = len(self.kpt_u)

        for u, kpt in enumerate(self.kpt_u):
            pb.update(u / nkpt)
            npw = len(self.pd.Q_qG[kpt.q])
            if scalapack:
                mynpw = -(-npw // S)
                md = BlacsDescriptor(bg, npw, npw, mynpw, npw)
                md2 = BlacsDescriptor(bg2, npw, npw, b, b)
            else:
                md = md2 = MatrixDescriptor(npw, npw)

            with self.timer('Build H and S'):
                H_GG, S_GG = self.hs(ham, kpt.q, kpt.s, md)

            if scalapack:
                r = Redistributor(bd.comm, md, md2)
                H_GG = r.redistribute(H_GG)
                S_GG = r.redistribute(S_GG)

            psit_nG = md2.empty(dtype=complex)
            eps_n = np.empty(npw)

            with self.timer('Diagonalize'):
                if not scalapack:
                    md2.general_diagonalize_dc(H_GG, S_GG, psit_nG, eps_n,
                                               iu=iu)
                else:
                    md2.general_diagonalize_dc(H_GG, S_GG, psit_nG, eps_n)
                    if eps_n[0] < -1000:
                        msg = f"""Lowest eigenvalue is {eps_n[0]} Hartree.
You might be suffering from MKL library bug MKLD-11440.
See issue #241 in GPAW. Creashing to prevent corrupted results."""
                        raise RuntimeError(msg)

            del H_GG, S_GG

            kpt.eps_n = eps_n[myslice].copy()

            if scalapack:
                md3 = BlacsDescriptor(bg, npw, npw, bd.maxmynbands, npw)
                r = Redistributor(bd.comm, md2, md3)
                psit_nG = r.redistribute(psit_nG)

            kpt.psit = PlaneWaveExpansionWaveFunctions(
                self.bd.nbands, self.pd, self.dtype,
                psit_nG[:bd.mynbands].copy(),
                kpt=kpt.q, dist=(self.bd.comm, self.bd.comm.size),
                spin=kpt.s, collinear=self.collinear)
            del psit_nG

            with self.timer('Projections'):
                self.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

            kpt.f_n = None

        pb.finish()

        self.calculate_occupation_numbers()

        return nbands

    def initialize_from_lcao_coefficients(self,
                                          basis_functions: BasisFunctions,
                                          block_size: int = 10) -> None:
        """Convert from LCAO to PW coefficients."""
        nlcao = len(self.kpt_qs[0][0].C_nM)

        # We go from LCAO to real-space and then to PW's.
        # It's too expensive to allocate one big real-space array:
        block_size = min(max(nlcao, 1), block_size)
        psit_nR = self.gd.empty(block_size, self.dtype)

        for kpt in self.kpt_u:
            if self.kd.gamma:
                emikr_R = 1.0
            else:
                k_c = self.kd.ibzk_kc[kpt.k]
                emikr_R = self.gd.plane_wave(-k_c)
            kpt.psit = PlaneWaveExpansionWaveFunctions(
                self.bd.nbands, self.pd, self.dtype, kpt=kpt.q,
                dist=(self.bd.comm, -1, 1),
                spin=kpt.s, collinear=self.collinear)
            psit_nG = kpt.psit.array
            if psit_nG.ndim == 3:  # non-collinear calculation
                N, S, G = psit_nG.shape
                psit_nG = psit_nG.reshape((N * S, G))
            for n1 in range(0, nlcao, block_size):
                n2 = min(n1 + block_size, nlcao)
                psit_nR[:] = 0.0
                basis_functions.lcao_to_grid(kpt.C_nM[n1:n2],
                                             psit_nR[:n2 - n1],
                                             kpt.q,
                                             block_size)
                for psit_R, psit_G in zip(psit_nR, psit_nG[n1:n2]):
                    psit_G[:] = self.pd.fft(psit_R * emikr_R, kpt.q)
            kpt.C_nM = None

    def random_wave_functions(self, mynao):
        rs = np.random.RandomState(self.world.rank)
        for kpt in self.kpt_u:
            if kpt.psit is None:
                kpt.psit = PlaneWaveExpansionWaveFunctions(
                    self.bd.nbands, self.pd, self.dtype, kpt=kpt.q,
                    dist=(self.bd.comm, -1, 1),
                    spin=kpt.s, collinear=self.collinear)

            array = kpt.psit.array[mynao:]
            weight_G = 1.0 / (1.0 + self.pd.G2_qG[kpt.q])
            array.real = rs.uniform(-1, 1, array.shape) * weight_G
            array.imag = rs.uniform(-1, 1, array.shape) * weight_G
            if self.gd.comm.rank == 0:
                array[:, 0].imag = 0.0

    def estimate_memory(self, mem):
        FDPWWaveFunctions.estimate_memory(self, mem)
        self.pd.estimate_memory(mem.subnode('PW-descriptor'))

    def get_kinetic_stress(self):
        sigma_vv = np.zeros((3, 3), dtype=complex)
        pd = self.pd
        dOmega = pd.gd.dv / pd.gd.N_c.prod()
        if pd.dtype == float:
            dOmega *= 2
        K_qv = self.pd.K_qv
        for kpt in self.kpt_u:
            G_Gv = pd.get_reciprocal_vectors(q=kpt.q, add_q=False)
            psit2_G = 0.0
            for n, f in enumerate(kpt.f_n):
                psit2_G += f * np.abs(kpt.psit_nG[n])**2
            for alpha in range(3):
                Ga_G = G_Gv[:, alpha] + K_qv[kpt.q, alpha]
                for beta in range(3):
                    Gb_G = G_Gv[:, beta] + K_qv[kpt.q, beta]
                    sigma_vv[alpha, beta] += (psit2_G * Ga_G * Gb_G).sum()

        sigma_vv *= -dOmega
        self.world.sum(sigma_vv)
        return sigma_vv
