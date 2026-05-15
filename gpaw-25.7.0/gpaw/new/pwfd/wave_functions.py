from __future__ import annotations

from functools import partial
from math import pi
from typing import Optional, Callable

import numpy as np
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core.atom_arrays import AtomArrays, AtomDistribution
from gpaw.core.atom_centered_functions import AtomCenteredFunctions
from gpaw.core.plane_waves import PWArray
from gpaw.core.uniform_grid import UGArray, UGDesc
from gpaw.fftw import get_efficient_fft_size
from gpaw.gpu import as_np, XP
from gpaw.mpi import receive, send
from gpaw.new import prod, trace, zips
from gpaw.new.potential import Potential
from gpaw.new.wave_functions import WaveFunctions
from gpaw.setup import Setups
from gpaw.typing import Array2D, Array3D, Vector
from gpaw.utilities import as_real_dtype


class PWFDWaveFunctions(WaveFunctions, XP):
    def __init__(self,
                 psit_nX: XArray,
                 *,
                 spin: int,
                 q: int,
                 k: int,
                 setups: Setups,
                 relpos_ac: Array2D,
                 atomdist: AtomDistribution,
                 weight: float = 1.0,
                 ncomponents: int = 1,
                 qspiral_v: Vector | None = None):
        # assert isinstance(atomdist, AtomDistribution)
        self.psit_nX = psit_nX
        nbands = psit_nX.dims[0]
        super().__init__(setups=setups,
                         nbands=nbands,
                         spin=spin,
                         q=q,
                         k=k,
                         kpt_c=psit_nX.desc.kpt_c,
                         relpos_ac=relpos_ac,
                         atomdist=atomdist,
                         weight=weight,
                         ncomponents=ncomponents,
                         qspiral_v=qspiral_v,
                         dtype=psit_nX.desc.dtype,
                         domain_comm=psit_nX.desc.comm,
                         band_comm=psit_nX.comm)
        self._pt_aiX: Optional[AtomCenteredFunctions] = None
        self.orthonormalized = False
        self.bytes_per_band = (prod(self.array_shape(global_shape=True)) *
                               psit_nX.desc.itemsize)
        XP.__init__(self, self.psit_nX.xp)

    @classmethod
    def from_wfs(cls,
                 wfs: PWFDWaveFunctions,
                 psit_nX: XArray,
                 relpos_ac=None,
                 atomdist=None) -> PWFDWaveFunctions:
        return cls(
            psit_nX,
            spin=wfs.spin,
            q=wfs.q,
            k=wfs.k,
            setups=wfs.setups,
            relpos_ac=wfs.relpos_ac if relpos_ac is None else relpos_ac,
            atomdist=atomdist or wfs.atomdist,
            weight=wfs.weight,
            ncomponents=wfs.ncomponents,
            qspiral_v=wfs.qspiral_v)

    def __del__(self):
        # We could be reading from a gpw-file
        data = self.psit_nX.data
        if hasattr(data, 'fd'):
            data.fd.close()

    def _short_string(self, global_shape: tuple[int]) -> str:
        return self.psit_nX.desc._short_string(global_shape)

    def array_shape(self, global_shape=False):
        if global_shape:
            shape = self.psit_nX.desc.global_shape()
        else:
            shape = self.psit_nX.desc.myshape
        if self.ncomponents == 4:
            shape = (2,) + shape
        return shape

    @property
    def pt_aiX(self) -> AtomCenteredFunctions:
        """PAW projector functions.

        :::

            ~a _
            p (r)
             i
        """
        if self._pt_aiX is None:
            self._pt_aiX = self.psit_nX.desc.atom_centered_functions(
                [setup.pt_j for setup in self.setups],
                self.relpos_ac,
                atomdist=self.atomdist,
                qspiral_v=self.qspiral_v,
                xp=self.psit_nX.xp)
        return self._pt_aiX

    @property
    def P_ani(self) -> AtomArrays:
        """PAW projections.

        :::

             ~a  ~
            <p | ψ >
              i   n
        """
        if self._P_ani is None:
            self._P_ani = self.pt_aiX.empty(self.psit_nX.dims,
                                            self.psit_nX.comm)
            if self.psit_nX.data is None:
                raise RuntimeError('There are no projections or wavefunctions')
            self.pt_aiX.integrate(self.psit_nX, self._P_ani)
        return self._P_ani

    def move(self,
             relpos_ac: Array2D,
             atomdist: AtomDistribution,
             move_wave_functions: Callable[..., None]) -> None:
        if self.psit_nX.data is not None:
            move_wave_functions(
                self.relpos_ac,
                relpos_ac,
                self.P_ani,
                self.psit_nX,
                self.setups)
        super().move(relpos_ac, atomdist, move_wave_functions)
        self.orthonormalized = False
        assert self.pt_aiX is not None
        self.pt_aiX.move(relpos_ac, atomdist)

    def add_to_density(self,
                       nt_sR: UGArray,
                       D_asii: AtomArrays) -> None:
        occ_n = self.weight * self.spin_degeneracy * self.myocc_n

        self.add_to_atomic_density_matrices(occ_n, D_asii)

        if self.ncomponents < 4:
            self.psit_nX.abs_square(weights=occ_n, out=nt_sR[self.spin])
            return

        psit_nsG = self.psit_nX
        assert isinstance(psit_nsG, PWArray)

        tmp_sR = nt_sR.desc.new(dtype=complex).empty(2)
        p1_R, p2_R = tmp_sR.data
        nt_xR = nt_sR.data

        for f, psit_sG in zips(occ_n, psit_nsG):
            psit_sG.ifft(out=tmp_sR)
            p11_R = p1_R.real**2 + p1_R.imag**2
            p22_R = p2_R.real**2 + p2_R.imag**2
            p12_R = p1_R.conj() * p2_R
            nt_xR[0] += f * (p11_R + p22_R)
            nt_xR[1] += 2 * f * p12_R.real
            nt_xR[2] += 2 * f * p12_R.imag
            nt_xR[3] += f * (p11_R - p22_R)

    def add_to_ked(self, taut_sR) -> None:
        occ_n = self.weight * self.spin_degeneracy * self.myocc_n
        self.psit_nX.add_ked(occ_n, taut_sR[self.spin])

    @trace
    def orthonormalize(self, psit2_nX):
        r"""Orthonormalize wave functions.

        Computes the overlap matrix:::

               / ~ _ *~ _   _   ---  a  * a   a
          S  = | ψ(r) ψ(r) dr + >  (P  ) P  ΔS
           mn  /  m    n        ---  im   jn  ij
                                aij

        With `LSL^\dagger=1`, we update the wave functions and projections
        inplace like this:::

                --  *
          Ψ  <- >  L  Ψ ,
           m    --  mn n
                n

        and:::

           a     --  *  a
          P   <- >  L  P  .
           mi    --  mn ni
                 n

        """
        if self.orthonormalized:
            return
        psit_nX = self.psit_nX
        domain_comm = psit_nX.desc.comm
        P_ani = self.P_ani

        P2_ani = P_ani.new()
        if psit2_nX is None:
            psit2_nX = psit_nX.new()
        dS_aii = self.setups.get_overlap_corrections(
            P_ani.layout.atomdist,
            self.xp,
            dtype=as_real_dtype(P_ani.data.dtype))

        # We are actually calculating S^*:
        S = psit_nX.matrix_elements(psit_nX, domain_sum=False, cc=True)
        P_ani.block_diag_multiply(dS_aii, out_ani=P2_ani)
        P_ani.matrix.multiply(P2_ani, opb='C', symmetric=True, out=S, beta=1.0)
        domain_comm.sum(S.data, 0)

        if domain_comm.rank == 0:
            S.invcholesky()
        domain_comm.broadcast(S.data, 0)
        # S now contains L^*

        S.multiply(psit_nX, out=psit2_nX)
        S.multiply(P_ani, out=P2_ani)
        psit_nX.data[:] = psit2_nX.data
        P_ani.data[:] = P2_ani.data
        self.orthonormalized = True

    @trace
    def subspace_diagonalize(self,
                             Ht,
                             dH,
                             psit2_nX,
                             data_buffer=None,
                             scalapack_parameters=(None, 1, 1, None)):
        """
        If data_buffer is None, psit2_nX will be used as a buffer
        for the wave functions.

        Ht(in, out):::

           ~   ^   ~
           H = T + v

        dH:::

           ~  ~    a  ~  ~
          <𝜓 |p> ΔH  <p |𝜓>
            m  i   ij  j  n
        """
        self.orthonormalize(psit2_nX)
        psit_nX = self.psit_nX
        P_ani = self.P_ani
        P2_ani = P_ani.new()
        domain_comm = psit_nX.desc.comm

        Ht = partial(Ht, out=psit2_nX, spin=self.spin, calculate_energy=True)
        H = psit_nX.matrix_elements(psit_nX,
                                    function=Ht,
                                    domain_sum=False,
                                    cc=True)
        dH(P_ani, out_ani=P2_ani, spin=self.spin)
        P_ani.matrix.multiply(P2_ani, opb='C', symmetric=True,
                              out=H, beta=1.0)
        domain_comm.sum(H.data, 0)
        if domain_comm.rank == 0:
            slcomm, r, c, b = scalapack_parameters
            if r == c == 1:
                slcomm = None
            # print(H.data)
            self._eig_n = as_np(H.eigh(scalapack=(slcomm, r, c, b)))
            # print(self._eig_n)
            H.complex_conjugate()
            # H.data[n, :] now contains the nth eigenvector and eps_n[n]
            # the nth eigenvalue
        else:
            self._eig_n = np.empty(psit_nX.dims)

        domain_comm.broadcast(H.data, 0)
        domain_comm.broadcast(self._eig_n, 0)
        if data_buffer is None:
            H.multiply(psit_nX, out=psit2_nX)
            psit_nX.data[:] = psit2_nX.data
            H.multiply(P_ani, out=P2_ani)
            P_ani.data[:] = P2_ani.data
        else:
            H.multiply(psit_nX, out=psit_nX, data_buffer=data_buffer)
            H.multiply(psit2_nX, out=psit2_nX, data_buffer=data_buffer)
            H.multiply(P_ani, out=P2_ani)
            P_ani.data[:] = P2_ani.data

    def force_contribution(self,
                           potential: Potential,
                           F_av: Array2D) -> None:
        xp = self.xp
        dH_asii = potential.dH_asii
        myeig_n = xp.asarray(self.myeig_n)
        myocc_n = xp.asarray(
            self.weight * self.spin_degeneracy * self.myocc_n)

        if self.ncomponents == 4:
            self._non_collinear_force_contribution(dH_asii, myocc_n, F_av)
            return

        F_anvi = self.pt_aiX.derivative(self.psit_nX)
        for a, F_nvi in F_anvi.items():
            F_nvi = F_nvi.conj()
            F_nvi *= myocc_n[:, np.newaxis, np.newaxis]
            dH_ii = dH_asii[a][self.spin]
            P_ni = self.P_ani[a]
            F_vii = xp.einsum('nvi, nj, jk -> vik', F_nvi, P_ni, dH_ii)
            F_nvi *= myeig_n[:, np.newaxis, np.newaxis]
            dO_ii = xp.asarray(self.setups[a].dO_ii)
            F_vii -= xp.einsum('nvi, nj, jk -> vik', F_nvi, P_ni, dO_ii)
            F_av[a] += 2 * F_vii.real.trace(0, 1, 2)

    def _non_collinear_force_contribution(self,
                                          dH_asii,
                                          myocc_n,
                                          F_av):
        F_ansvi = self.pt_aiX.derivative(self.psit_nX)
        for a, F_nsvi in F_ansvi.items():
            F_nsvi = F_nsvi.conj()
            F_nsvi *= myocc_n[:, np.newaxis, np.newaxis, np.newaxis]
            dH_sii = dH_asii[a]
            dH_ii = dH_sii[0]
            dH_vii = dH_sii[1:]
            dH_ssii = np.array(
                [[dH_ii + dH_vii[2], dH_vii[0] - 1j * dH_vii[1]],
                 [dH_vii[0] + 1j * dH_vii[1], dH_ii - dH_vii[2]]])
            P_nsi = self.P_ani[a]
            F_v = np.einsum('nsvi, stij, ntj -> v', F_nsvi, dH_ssii, P_nsi)
            F_nsvi *= self.myeig_n[:, np.newaxis, np.newaxis, np.newaxis]
            dO_ii = self.setups[a].dO_ii
            F_v -= np.einsum('nsvi, ij, nsj -> v', F_nsvi, dO_ii, P_nsi)
            F_av[a] += 2 * F_v.real

    def collect(self,
                n1: int = 0,
                n2: int = 0) -> PWFDWaveFunctions | None:
        """Collect range of bands to master of band and domain comms."""
        # Also collect projections instead of recomputing XXX
        n2 = n2 if n2 > 0 else self.nbands + n2
        spinors = (2,) if self.ncomponents == 4 else ()
        band_comm = self.psit_nX.comm
        domain_comm = self.psit_nX.desc.comm
        nbands = self.nbands
        mynbands = (nbands + band_comm.size - 1) // band_comm.size
        rank1, b1 = divmod(n1, mynbands)
        rank2, b2 = divmod(n2, mynbands)
        if band_comm.rank == 0:
            if domain_comm.rank == 0:
                psit_nX = self.psit_nX.desc.new(comm=None).empty(
                    (n2 - n1, *spinors))
            rank = rank1
            ba = b1
            na = n1
            while (rank, ba) < (rank2, b2):
                bb = min((rank + 1) * mynbands, nbands) - rank * mynbands
                if rank == rank2 and bb > b2:
                    bb = b2
                nb = na + bb - ba
                if bb > ba:
                    if rank == 0:
                        psit_bX = self.psit_nX[ba:bb].gather()
                        if domain_comm.rank == 0:
                            psit_nX.data[:bb - ba] = psit_bX.data
                    else:
                        if domain_comm.rank == 0:
                            band_comm.receive(psit_nX.data[na - n1:nb - n1],
                                              rank)
                rank += 1
                ba = 0
                na = nb
            if domain_comm.rank == 0:
                wfs = PWFDWaveFunctions.from_wfs(
                    self,
                    psit_nX,
                    atomdist=self.atomdist.gather())
                wfs._eig_n = self.eig_n[n1:n2]
                return wfs
        else:
            rank = band_comm.rank
            ranka, ba = max((rank1, b1), (rank, 0))
            rankb, bb = min((rank2, b2), (rank, self.psit_nX.mydims[0]))
            if (ranka, ba) < (rankb, bb):
                assert ranka == rankb == rank
                band_comm.send(self.psit_nX.data[ba:bb], dest=0)

        return None

    def send(self, rank, comm):
        stuff = (self.kpt_c,
                 self.psit_nX.data,
                 self.spin,
                 self.q,
                 self.k,
                 self.weight)
        send(stuff, rank, comm)

    def receive(self, rank, comm):
        kpt_c, data, spin, q, k, weight = receive(rank, comm)
        psit_nX = self.psit_nX.desc.new(kpt=kpt_c, comm=None).from_data(data)
        return PWFDWaveFunctions(psit_nX,
                                 spin=spin,
                                 q=q,
                                 k=k,
                                 setups=self.setups,
                                 relpos_ac=self.relpos_ac,
                                 atomdist=self.atomdist.gather(),
                                 weight=weight,
                                 ncomponents=self.ncomponents,
                                 qspiral_v=self.qspiral_v)

    def dipole_matrix_elements(self) -> Array3D:
        """Calculate dipole matrix-elements.

        :::

           _    / ~ ~ _ _   ---  a  a  _a
           μ  = | 𝜓 𝜓 rdr + >   P  P  Δμ
            mn  /  m n      ---  im jn  ij
                            aij

        Returns
        -------
        Array3D:
            matrix elements in atomic units.
        """
        cell_cv = self.psit_nX.desc.cell_cv

        dipole_nnv = np.zeros((self.nbands, self.nbands, 3))

        position_av = self.relpos_ac @ cell_cv

        R_aiiv = []
        for setup, position_v in zips(self.setups, position_av):
            Delta_iiL = setup.Delta_iiL
            R_iiv = Delta_iiL[:, :, [3, 1, 2]] * (4 * pi / 3)**0.5
            R_iiv += position_v * setup.Delta_iiL[:, :, :1] * (4 * pi)**0.5
            R_aiiv.append(R_iiv)

        for a, P_ni in self.P_ani.items():
            dipole_nnv += np.einsum('mi, ijv, nj -> mnv',
                                    P_ni, R_aiiv[a], P_ni)

        self.psit_nX.desc.comm.sum(dipole_nnv)

        if isinstance(self.psit_nX, UGArray):
            psit_nR = self.psit_nX
        else:
            assert isinstance(self.psit_nX, PWArray)
            # Find size of fft grid large enough to store square of wfs.
            pw = self.psit_nX.desc
            s1, s2, s3 = np.ptp(pw.indices_cG, axis=1)  # type: ignore
            assert pw.dtype == float
            # Last dimension is special because dtype=float:
            size_c = [2 * s1 + 2,
                      2 * s2 + 2,
                      4 * s3 + 2]
            size_c = [get_efficient_fft_size(N, 2) for N in size_c]
            grid = UGDesc(cell=pw.cell_cv, size=size_c)
            psit_nR = self.psit_nX.ifft(grid=grid)

        for na, psita_R in enumerate(psit_nR):
            for nb, psitb_R in enumerate(psit_nR[:na + 1]):
                d_v = (psita_R * psitb_R).moment()
                dipole_nnv[na, nb] += d_v
                if na != nb:
                    dipole_nnv[nb, na] += d_v

        return dipole_nnv

    def gather_wave_function_coefficients(self) -> np.ndarray | None:
        psit_nX = self.psit_nX.gather()  # gather X
        if psit_nX is not None:
            data_nX = psit_nX.matrix.gather()  # gather n
            if data_nX.dist.comm.rank == 0:
                # XXX PW-gamma-point mode: float or complex matrix.dtype?
                return data_nX.data.view(
                    psit_nX.data.dtype).reshape((-1,) + psit_nX.data.shape[1:])
        return None

    def to_uniform_grid_wave_functions(self,
                                       grid,
                                       basis):
        if isinstance(self.psit_nX, UGArray):
            return self

        grid = grid.new(kpt=self.kpt_c, dtype=self.dtype)
        psit_nR = grid.zeros(self.nbands, self.band_comm)
        self.psit_nX.ifft(out=psit_nR)
        return PWFDWaveFunctions.from_wfs(self, psit_nR)

    def morph(self, desc, relpos_ac, atomdist):
        desc = desc.new(kpt=self.psit_nX.desc.kpt_c)
        psit_nX = self.psit_nX.morph(desc)

        # Save memory:
        self.psit_nX.data = None
        self._P_ani = None
        self._pt_aiX = None

        return PWFDWaveFunctions.from_wfs(self, psit_nX,
                                          relpos_ac=relpos_ac)
