from __future__ import annotations
from dataclasses import dataclass
import functools
from time import ctime

import numpy as np
from ase.units import Hartree
from scipy.special import p_roots

import gpaw.mpi as mpi
from gpaw.response import timer
from gpaw.response.chi0 import Chi0Calculator
from gpaw.response.coulomb_kernels import CoulombKernel
from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.pair import get_gs_and_context


def default_ecut_extrapolation(ecut, extrapolate):
    return ecut * (1 + 0.5 * np.arange(extrapolate))**(-2 / 3)


class GCut:
    def __init__(self, cut_G):
        self._cut_G = cut_G

    @property
    def nG(self):
        return len(self._cut_G)

    def spin_cut(self, array_GG, ns):
        # Strange special case for spin-repeated arrays.
        # Maybe we can get rid of this.
        if self._cut_G is None:
            return array_GG

        cut_sG = np.tile(self._cut_G, ns)
        cut_sG[self.nG:] += len(array_GG) // ns
        array_GG = array_GG.take(cut_sG, 0).take(cut_sG, 1)
        return array_GG

    def cut(self, array, axes=(0,)):
        if self._cut_G is None:
            return array

        for axis in axes:
            array = array.take(self._cut_G, axis)
        return array


def initialize_q_points(kd, qsym):
    bzq_qc = kd.get_bz_q_points(first=True)

    if not qsym:
        ibzq_qc = bzq_qc
        weight_q = np.ones(len(bzq_qc)) / len(bzq_qc)
    else:
        U_scc = kd.symmetry.op_scc
        ibzq_qc = kd.get_ibz_q_points(bzq_qc, U_scc)[0]
        weight_q = kd.q_weights
    return ibzq_qc, weight_q


@dataclass
class RPAIntegral:
    omega_w: np.ndarray
    weight_w: np.ndarray
    ibzq_qc: np.ndarray
    weight_q: np.ndarray
    ecut_i: np.ndarray

    @property
    def nq(self):
        """Number of q-points."""
        return len(self.weight_q)

    @property
    def nw(self):
        """Number of frequencies."""
        nw = len(self.omega_w)
        assert len(self.weight_w) == nw
        return nw

    @property
    def ni(self):
        """Number of plane-wave cutoffs to extrapolate over."""
        return len(self.ecut_i)

    def integrate_frequencies(self, in_xwi):
        out_xi = self.weight_w @ in_xwi
        return out_xi

    def integrate_qpoints(self, in_xqi):
        out_xi = self.weight_q @ in_xqi
        return out_xi


@dataclass
class RPAData:
    integral: RPAIntegral

    def __post_init__(self):
        self.energy_qwi = np.zeros(self.shape)

    @property
    def shape(self):
        return (self.integral.nq, self.integral.nw, self.integral.ni)

    def contribution_from_qpoint(self, q, i=-1):
        return self.integral.integrate_frequencies(self.energy_qwi[q, :, i])

    @property
    def energy_wqi(self):
        return np.swapaxes(self.energy_qwi, 0, 1)

    @property
    def energy_qi(self):
        """Correlation energy contribution, E_c(q)."""
        return self.integral.integrate_frequencies(self.energy_qwi)

    @property
    def energy_wi(self):
        """Correlation energy contribution, E_c(ω)."""
        return self.integral.integrate_qpoints(self.energy_wqi)

    @property
    def energy_i(self):
        """Correlation energy E_c as a function of the plane-wave cutoff."""
        return self.integral.integrate_qpoints(self.energy_qi)


class RPACalculator:
    def __init__(
            self,
            gs,
            *,
            context,
            ecut,
            frequencies,
            weights,
            qsym=True,
            skip_gamma=False,
            truncation=None,
            nblocks=1,
            calculate_q=None
    ):
        self.gs = gs
        self.context = context

        # Normalize RPA integral inputs
        if isinstance(ecut, (float, int)):
            ecut = default_ecut_extrapolation(ecut, extrapolate=6)
        ecut_i = np.asarray(np.sort(ecut)) / Hartree
        # TODO: We should avoid this requirement.
        # thosk notes: it might work now (after some extensive clean-up of the
        # frequency integration), but I have not tested it
        assert len(frequencies) % nblocks == 0
        # We should actually have a kpoint descriptor for the qpoints.
        # We are badly failing at making use of the existing tools by reducing
        # the qpoints to dumb arrays.
        ibzq_qc, weight_q = initialize_q_points(gs.kd, qsym)
        # Collect information about the RPA integral on a single object (a.u.)
        self.integral = RPAIntegral(
            omega_w=frequencies / Hartree,
            weight_w=weights / Hartree / (2 * np.pi),
            ibzq_qc=ibzq_qc,
            weight_q=weight_q,
            ecut_i=ecut_i,
        )

        self.chi0calc = Chi0Calculator(
            self.gs, self.context.with_txt('chi0.txt'),
            nblocks=nblocks,
            wd=FrequencyDescriptor(1j * self.integral.omega_w),
            eta=0.0,
            intraband=False,
            hilbert=False,
            ecut=max(self.integral.ecut_i) * Hartree)
        self.coulomb = CoulombKernel.from_gs(gs, truncation=truncation)

        self.skip_gamma = skip_gamma
        # This is a super weird way of achieving inheritance...
        if calculate_q is None:
            calculate_q = self.calculate_q_rpa
        self.calculate_q = calculate_q

    def calculate(self, *, nbands=None) -> np.ndarray:
        """Calculate the RPA correlation energy as a function of cutoff."""
        data = self.calculate_all_contributions(nbands=nbands)
        return data.energy_i * Hartree  # energies in eV

    def calculate_all_contributions(
            self, *, nbands=None, spin=False) -> RPAData:
        """Calculate RPA correlation energy contributions.

        nbands: int
            Number of bands (defaults to number of plane-waves).
        spin: bool
            Separate spin in response function.
            (Only needed for beyond RPA methods that inherit this function).
        """

        p = functools.partial(self.context.print, flush=False)

        if nbands is None:
            p('Response function bands : Equal to number of plane waves')
        else:
            p('Response function bands : %s' % nbands)
        p('Plane wave cutoffs (eV) :', end='')
        for e in self.integral.ecut_i:
            p(f' {e * Hartree:.3f}', end='')
        p()
        p(self.coulomb.description())
        p('', flush=True)

        self.context.timer.start('RPA')

        data = RPAData(self.integral)
        ecutmax = max(self.integral.ecut_i)
        for q, q_c in enumerate(self.integral.ibzq_qc):
            if np.allclose(q_c, 0.0) and self.skip_gamma:
                p('Not calculating E_c(q) at Gamma', end='\n')
                continue

            chi0_s = [self.chi0calc.create_chi0(q_c)]
            if spin:
                chi0_s.append(self.chi0calc.create_chi0(q_c))

            qpd = chi0_s[0].qpd
            nG = qpd.ngmax

            # First not completely filled band:
            m1 = self.gs.nocc1
            p(f'# {q}  -  {ctime().split()[-2]}')
            p('q = [%1.3f %1.3f %1.3f]' % tuple(q_c))

            for i, ecut in enumerate(self.integral.ecut_i):
                if ecut == ecutmax:
                    # Nothing to cut away:
                    gcut = GCut(None)
                    m2 = nbands or nG
                else:
                    gcut = GCut(np.arange(nG)[qpd.G2_qG[0] <= 2 * ecut])
                    m2 = gcut.nG

                p('E_cut = %d eV / Bands = %d:' % (ecut * Hartree, m2),
                  end='\n', flush=True)
                p('E_c(q) = ', end='', flush=False)
                data.energy_qwi[q, :, i] = self.calculate_q(
                    chi0_s, m1, m2, gcut
                )
                energy = data.contribution_from_qpoint(q, i=i)
                p('%.3f eV' % (energy * Hartree), flush=True)
                m1 = m2

            p()

        e_i = data.energy_i
        p('==========================================================')
        p()
        p('Total correlation energy:')
        for e_cut, e in zip(self.integral.ecut_i, e_i):
            p(f'{e_cut * Hartree:6.0f}:   {e * Hartree:6.4f} eV')
        p()

        if len(e_i) > 1:
            self.extrapolate(e_i, self.integral.ecut_i)

        p('Calculation completed at: ', ctime())
        p()

        self.context.timer.stop('RPA')
        self.context.write_timer()

        return data

    @timer('chi0(q)')
    def calculate_q_rpa(self, chi0_s, m1, m2, gcut):
        chi0 = chi0_s[0]
        self.chi0calc.update_chi0(
            chi0, m1=m1, m2=m2, spins=range(self.chi0calc.gs.nspins))
        qpd = chi0.qpd
        chi0_wGG = chi0.body.get_distributed_frequencies_array()
        wblocks = chi0.body.get_distributed_frequencies_blocks1d()
        # Calculate RPA energy contributions (as a function of w)
        if chi0.qpd.optical_limit:
            chi0_wvv = chi0.chi0_Wvv[wblocks.myslice]
            chi0_wxvG = chi0.chi0_WxvG[wblocks.myslice]
            energy_w = self.calculate_optical_limit_rpa_energies(
                qpd, chi0_wGG, chi0_wvv, chi0_wxvG, gcut
            )
        else:
            energy_w = self.calculate_rpa_energies(qpd, chi0_wGG, gcut)
        return wblocks.all_gather(energy_w)

    def calculate_optical_limit_rpa_energies(
            self, qpd, chi0_wGG, chi0_wvv, chi0_wxvG, gcut):
        """Calculate correlation energy from chi0 in the optical limit."""
        from gpaw.response.gamma_int import GammaIntegral

        gamma_int = GammaIntegral(self.coulomb, qpd=qpd)

        energy_w = []
        for chi0_GG, chi0_vv, chi0_xvG in zip(chi0_wGG, chi0_wvv, chi0_wxvG):
            # Integrate over the optical wave vector volume
            energy = 0.
            for qweight, sqrtV_G, chi0_mapping in gamma_int:
                chi0p_GG = chi0_mapping(chi0_GG, chi0_vv, chi0_xvG)
                energy += qweight * single_rpa_energy(
                    chi0p_GG, gcut.cut(sqrtV_G), gcut)
            energy_w.append(energy)
        return np.array(energy_w)

    def calculate_rpa_energies(self, qpd, chi0_wGG, gcut):
        """Evaluate correlation energy from chi0 at finite q."""
        sqrtV_G = gcut.cut(self.coulomb.sqrtV(qpd, q_v=None))
        return np.array([
            single_rpa_energy(chi0_GG, sqrtV_G, gcut) for chi0_GG in chi0_wGG
        ])

    def extrapolate(self, e_i, ecut_i):
        self.context.print('Extrapolated energies:', flush=False)
        ex_i = []
        for i in range(len(e_i) - 1):
            e1, e2 = e_i[i:i + 2]
            x1, x2 = ecut_i[i:i + 2]**-1.5
            ex = (e1 * x2 - e2 * x1) / (x2 - x1)
            ex_i.append(ex)

            self.context.print('  %4.0f -%4.0f:  %5.3f eV' %
                               (ecut_i[i] * Hartree, ecut_i[i + 1]
                                * Hartree, ex * Hartree), flush=False)
        self.context.print('')

        return e_i * Hartree


def single_rpa_energy(chi0_GG, sqrtV_G, gcut):
    nG = len(sqrtV_G)
    chi0_GG = gcut.cut(chi0_GG, [0, 1])
    e_GG = np.eye(nG) - chi0_GG * sqrtV_G * sqrtV_G[:, np.newaxis]
    e = np.log(np.linalg.det(e_GG)) + nG - np.trace(e_GG)
    return e.real


def get_gauss_legendre_points(nw=16, frequency_max=800.0, frequency_scale=2.0):
    y_w, weights_w = p_roots(nw)
    y_w = y_w.real
    ys = 0.5 - 0.5 * y_w
    ys = ys[::-1]
    w = (-np.log(1 - ys))**frequency_scale
    w *= frequency_max / w[-1]
    alpha = (-np.log(1 - ys[-1]))**frequency_scale / frequency_max
    transform = (-np.log(1 - ys))**(frequency_scale - 1) \
        / (1 - ys) * frequency_scale / alpha
    return w, weights_w * transform / 2


class RPACorrelation(RPACalculator):
    def __init__(self, calc, xc='RPA',
                 nlambda=None,
                 nfrequencies=16, frequency_max=800.0, frequency_scale=2.0,
                 frequencies=None, weights=None,
                 world=mpi.world,
                 txt='-',
                 truncation: str | None = None,
                 **kwargs):
        """Creates the RPACorrelation object

        calc: str or calculator object
            The string should refer to the .gpw file contaning KS orbitals
        xc: str
            Exchange-correlation kernel. This is only different from RPA when
            this object is constructed from a different module - e.g. fxc.py
        skip_gamma: bool
            If True, skip q = [0,0,0] from the calculation
        qsym: bool
            Use symmetry to reduce q-points
        nlambda: int
            Number of lambda points. Only used for numerical coupling
            constant integration involved when called from fxc.py
        nfrequencies: int
            Number of frequency points used in the Gauss-Legendre integration
        frequency_max: float
            Largest frequency point in Gauss-Legendre integration
        frequency_scale: float
            Determines density of frequency points at low frequencies. A slight
            increase to e.g. 2.5 or 3.0 improves convergence wth respect to
            frequency points for metals
        frequencies: list
            List of frequencies for user-specified frequency integration
        weights: list
            list of weights (integration measure) for a user specified
            frequency grid. Must be specified and have the same length as
            frequencies if frequencies is not None
        truncation: str or None
            Coulomb truncation scheme. Can be None, '0D' or '2D'.  If None
            and the system is a molecule then '0D' will be used.
        world: communicator
        nblocks: int
            Number of parallelization blocks. Frequency parallelization
            can be specified by setting nblocks=nfrequencies and is useful
            for memory consuming calculations
        ecut: float or list of floats
            Plane-wave cutoff(s) in eV.
        txt: str
            txt file for saving and loading contributions to the correlation
            energy from different q-points
        """
        gs, context = get_gs_and_context(calc=calc, txt=txt, world=world,
                                         timer=None)

        if frequencies is None:
            frequencies, weights = get_gauss_legendre_points(nfrequencies,
                                                             frequency_max,
                                                             frequency_scale)
            user_spec = False
        else:
            assert weights is not None
            user_spec = True

        if truncation is None and not gs.pbc.any():
            truncation = '0D'

        super().__init__(gs=gs, context=context,
                         frequencies=frequencies, weights=weights,
                         truncation=truncation,
                         **kwargs)

        self.print_initialization(xc, frequency_scale, nlambda, user_spec)

    def print_initialization(self, xc, frequency_scale, nlambda, user_spec):
        p = functools.partial(self.context.print, flush=False)
        p('----------------------------------------------------------')
        p('Non-self-consistent %s correlation energy' % xc)
        p('----------------------------------------------------------')
        p('Started at:  ', ctime())
        p()
        p('Atoms                          :',
          self.gs.atoms.get_chemical_formula(mode='hill'))
        p('Ground state XC functional     :', self.gs.xcname)
        p('Valence electrons              :', self.gs.nvalence)
        p('Number of bands                :', self.gs.bd.nbands)
        p('Number of spins                :', self.gs.nspins)
        p('Number of k-points             :', len(self.gs.kd.bzk_kc))
        p('Number of irreducible k-points :', len(self.gs.kd.ibzk_kc))
        p('Number of irreducible q-points :', len(self.integral.ibzq_qc))
        p()
        for q, weight in zip(self.integral.ibzq_qc, self.integral.weight_q):
            p('    q: [%1.4f %1.4f %1.4f] - weight: %1.3f' %
              (q[0], q[1], q[2], weight))
        p()
        p('----------------------------------------------------------')
        p('----------------------------------------------------------')
        p()
        if nlambda is None:
            p('Analytical coupling constant integration')
        else:
            p('Numerical coupling constant integration using', nlambda,
              'Gauss-Legendre points')
        p()
        p('Frequencies')
        if not user_spec:
            p('    Gauss-Legendre integration with %s frequency points' %
              len(self.integral.omega_w))
            p('    Transformed from [0,oo] to [0,1] using e^[-aw^(1/B)]')
            p('    Highest frequency point at %5.1f eV and B=%1.1f' %
              (self.integral.omega_w[-1] * Hartree, frequency_scale))
        else:
            p('    User specified frequency integration with',
              len(self.integral.omega_w), 'frequency points')
        p()
        p('Parallelization')
        p('    Total number of CPUs          : % s' % self.context.comm.size)
        blockcomm = self.chi0calc.chi0_body_calc.blockcomm
        p('    G-vector decomposition        : % s' % blockcomm.size)
        kncomm = self.chi0calc.chi0_body_calc.kncomm
        p('    K-point/band decomposition    : % s' % kncomm.size)
        self.context.print('')
