from __future__ import annotations
import functools
from io import StringIO
from math import pi, sqrt
import ase.units as units
import numpy as np
from ase.data import chemical_symbols

from gpaw import debug
from gpaw.basis_data import Basis, BasisFunction
from gpaw.gaunt import gaunt, nabla
from gpaw.overlap import OverlapCorrections
from gpaw.setup_data import SetupData, search_for_file
from gpaw.spline import Spline
from gpaw.utilities import pack_density, unpack_hermitian
from gpaw.xc import XC
from gpaw.new import zips
from gpaw.xc.ri.spherical_hse_kernel import RadialHSE
from gpaw.core.atom_arrays import AtomArraysLayout


class WrongMagmomForHundsRuleError(ValueError):
    """
    Custom error for catching bad magnetic moments in Hund's rule calculation
    """


def create_setup(symbol, xc='LDA', lmax=0,
                 type='paw', basis=None, setupdata=None,
                 filter=None, world=None, backwards_compatible=True):
    if isinstance(xc, str):
        xc = XC(xc)

    if isinstance(type, str) and ':' in type:
        from gpaw.hubbard import parse_hubbard_string
        type, hubbard_u = parse_hubbard_string(type)
    else:
        hubbard_u = None

    if setupdata is None:
        if type == 'hgh' or type == 'hgh.sc':
            lmax = 0
            from gpaw.hgh import HGHSetupData, sc_setups, setups
            if type == 'hgh.sc':
                table = sc_setups
            else:
                table = setups
            parameters = table[symbol]
            setupdata = HGHSetupData(parameters)
        elif type == 'ah':
            from gpaw.ah import AppelbaumHamann
            ah = AppelbaumHamann()
            ah.build(basis)
            return ah
        elif type == 'ae':
            from gpaw.ae import HydrogenAllElectronSetup
            assert symbol == 'H'
            ae = HydrogenAllElectronSetup()
            ae.build(basis)
            return ae
        elif type == 'ghost':
            from gpaw.lcao.bsse import GhostSetupData
            setupdata = GhostSetupData(symbol)
        elif type == 'sg15':
            from gpaw.upf import read_sg15
            upfname = f'{symbol}_ONCV_PBE-*.upf'
            try:
                upfpath, source = search_for_file(upfname, world=world)
            except RuntimeError:
                raise OSError('Could not find pseudopotential file %s '
                              'in any GPAW search path.  '
                              'Please install the SG15 setups using, '
                              'e.g., "gpaw install-data".' % upfname)
            setupdata = read_sg15(upfpath)
            if xc.get_setup_name() != 'PBE':
                raise ValueError('SG15 pseudopotentials support only the PBE '
                                 'functional.  This calculation would use '
                                 'the %s functional.' % xc.get_setup_name())
        else:
            setupdata = SetupData.find_and_read_path(symbol,
                                                     xc.get_setup_name(),
                                                     setuptype=type,
                                                     world=world)

    if hasattr(setupdata, 'build'):
        # It is not so nice that we have hubbard_u floating around here.
        # For example, none of the other setup types are aware
        # of hubbard u, so they silently ignore it!
        if isinstance(setupdata, SetupData):
            kwargs = dict(backwards_compatible=backwards_compatible)
        else:
            kwargs = {}
        setup = LeanSetup(
            setupdata.build(xc, lmax, basis, filter, **kwargs),
            hubbard_u=hubbard_u)
        return setup
    else:
        return setupdata


def correct_occ_numbers(f_j,
                        degeneracy_j,
                        jsorted,
                        correction: float,
                        eps=1e-12) -> None:
    """Correct f_j ndarray in-place."""

    if correction > 0:
        # Add electrons to the lowest eigenstates:
        for j in jsorted:
            c = min(correction, degeneracy_j[j] - f_j[j])
            f_j[j] += c
            correction -= c
            if correction < eps:
                break
    elif correction < 0:
        # Add electrons to the highest eigenstates:
        for j in jsorted[::-1]:
            c = min(-correction, f_j[j])
            f_j[j] -= c
            correction += c
            if correction > -eps:
                break


class LocalCorrectionVar:
    """Class holding data for local the calculation of local corr."""
    def __init__(self, s=None):
        """Initialize our data."""
        for work_key in ('nq', 'lcut', 'n_qg', 'nt_qg', 'nc_g', 'nct_g',
                         'rgd2', 'Delta_lq', 'T_Lqp'):
            if s is None or not hasattr(s, work_key):
                setattr(self, work_key, None)
            else:
                setattr(self, work_key, getattr(s, work_key))


class BaseSetup:
    """Mixin-class for setups.

    This makes it possible to inherit the most important methods without
    the cumbersome constructor of the ordinary Setup class.

    Maybe this class will be removed in the future, or it could be
    made a proper base class with attributes and so on."""

    orbital_free = False
    hubbard_u = None  # XXX remove me
    is_pseudo = False

    def print_info(self, text):
        self.data.print_info(text, self)

    def get_basis_description(self):
        return self.basis.get_description()

    def get_partial_waves_for_atomic_orbitals(self):
        """Get those states phit that represent a real atomic state.

        This typically corresponds to the (truncated) partial waves (PAW) or
        a single-zeta basis."""

        # XXX ugly hack for pseudopotentials:
        if not hasattr(self, 'pseudo_partial_waves_j'):
            return []

        # The zip may cut off part of phit_j if there are more states than
        # projectors.  This should be the correct behaviour for all the
        # currently supported PAW/pseudopotentials.
        partial_waves_j = []
        for n, phit in zip(self.n_j, self.pseudo_partial_waves_j):
            if n > 0:
                partial_waves_j.append(phit)

        assert all(n > 0 for n in self.n_j[:len(partial_waves_j)])

        return partial_waves_j

    def calculate_initial_occupation_numbers(self, magmom, hund, charge,
                                             nspins, f_j=None):
        """If f_j is specified, custom occupation numbers will be used.

        Hund rules disabled if so."""
        nao = self.nao
        f_sI = np.zeros((nspins, nao))

        assert (not hund) or f_j is None
        if f_j is None:
            f_j = self.f_j
        f_j = np.array(f_j, float)
        l_j = np.array(self.l_j)

        if hasattr(self, 'data') and hasattr(self.data, 'eps_j'):
            eps_j = np.array(self.data.eps_j)
        else:
            eps_j = np.ones(len(self.n_j))
            # Bound states:
            for j, n in enumerate(self.n_j):
                if n > 0:
                    eps_j[j] = -1.0

        deg_j = 2 * (2 * l_j + 1)

        # Sort after:
        #
        # 1) empty state (f == 0)
        # 2) open shells (d - f)
        # 3) eigenvalues (e)

        states = []
        for j, (f, d, e) in enumerate(zips(f_j, deg_j, eps_j, strict=False)):
            if e < 0.0:
                states.append((f == 0, d - f, e, j))
        states.sort()
        jsorted = [j for _, _, _, j in states]

        # if len(l_j) == 0:
        #     l_j = np.ones(1)

        # distribute the charge to the radial orbitals
        if nspins == 1:
            assert magmom == 0.0
            f_sj = np.array([f_j])
            if not self.orbital_free:
                correct_occ_numbers(f_sj[0], deg_j, jsorted, -charge)
            else:
                # ofdft degeneracy of one orbital is infinite
                f_sj[0, 0] -= charge
        else:
            nval = f_j.sum() - charge
            if np.abs(magmom) > nval:
                raise RuntimeError(
                    'Magnetic moment larger than number ' +
                    f'of valence electrons (|{magmom:g}| > {nval:g})')
            f_sj = 0.5 * np.array([f_j, f_j])
            nup = 0.5 * (nval + magmom)
            ndn = 0.5 * (nval - magmom)
            deg_j //= 2
            correct_occ_numbers(f_sj[0], deg_j, jsorted, nup - f_sj[0].sum())
            correct_occ_numbers(f_sj[1], deg_j, jsorted, ndn - f_sj[1].sum())

        for j, (n1, l1, f, f_s) in enumerate(zip(self.n_j, self.l_j,
                                                 f_j, f_sj.T)):
            if not f_s.any() or n1 < 0:
                continue
            I = 0
            for bf in self.basis.bf_j:
                l = bf.l
                n = bf.n
                if (n, l) == (n1, l1):
                    break
                I += 2 * l + 1
            else:  # no break
                raise ValueError(f'Bad basis for {self.symbol}')

            degeneracy = 2 * l + 1

            if hund:
                # Use Hunds rules:
                # assert f == int(f)
                f = int(f)
                f_sI[0, I:I + min(f, degeneracy)] = 1.0      # spin up
                f_sI[1, I:I + max(f - degeneracy, 0)] = 1.0  # spin down
                if f < degeneracy:
                    magmom -= f
                else:
                    magmom -= 2 * degeneracy - f
            else:
                for s in range(nspins):
                    f_sI[s, I:I + degeneracy] = f_s[s] / degeneracy

        if hund and magmom != 0:
            raise WrongMagmomForHundsRuleError(
                f'Bad magnetic moment {magmom:g} for {self.symbol} atom!')

        # print('fsi=', f_si)
        return f_sI

    def get_hunds_rule_moment(self, charge=0):
        for M in range(10):
            try:
                self.calculate_initial_occupation_numbers(M, True, charge, 2)
            except WrongMagmomForHundsRuleError:
                pass
            else:
                return M
        raise RuntimeError

    def initialize_density_matrix(self, f_sI):
        nspins, nao = f_sI.shape
        ni = self.ni

        D_sii = np.zeros((nspins, ni, ni))
        D_sp = np.zeros((nspins, ni * (ni + 1) // 2))

        I = 0
        for bf in self.basis.bf_j:
            f_sm = f_sI[:, I:I + 2 * bf.l + 1]
            if f_sm.any():
                i = 0
                for n, l in zip(self.n_j, self.l_j):
                    if (n, l) == (bf.n, bf.l):
                        break
                    i += 2 * l + 1
                else:  # no break
                    raise ValueError(f'Bad basis for {self.symbol}')

                if i < ni:
                    for m in range(2 * l + 1):
                        D_sii[:, i + m, i + m] = f_sm[:, m]

            I += 2 * bf.l + 1

        for s in range(nspins):
            D_sp[s] = pack_density(D_sii[s])
        return D_sp

    def get_partial_waves(self):
        """Return spline representation of partial waves and densities."""

        l_j = self.l_j

        # cutoffs
        rcut2 = 2 * max(self.rcut_j)
        gcut2 = self.rgd.ceil(rcut2)

        data = self.data

        # Construct splines:
        nc_g = data.nc_g.copy()
        nct_g = data.nct_g.copy()
        tauc_g = data.tauc_g
        tauct_g = data.tauct_g
        nc = self.rgd.spline(nc_g, rcut2, points=1000)
        nct = self.rgd.spline(nct_g, rcut2, points=1000)
        if tauc_g is None:
            tauc_g = np.zeros(nct_g.shape)
            tauct_g = tauc_g
        tauc = self.rgd.spline(tauc_g, rcut2, points=1000)
        tauct = self.rgd.spline(tauct_g, rcut2, points=1000)
        phi_j = []
        phit_j = []
        for j, (phi_g, phit_g) in enumerate(zips(data.phi_jg, data.phit_jg)):
            l = l_j[j]
            phi_g = phi_g.copy()
            phit_g = phit_g.copy()
            phi_g[gcut2:] = phit_g[gcut2:] = 0.0
            phi_j.append(self.rgd.spline(phi_g, rcut2, l, points=100))
            phit_j.append(self.rgd.spline(phit_g, rcut2, l, points=100))
        return phi_j, phit_j, nc, nct, tauc, tauct

    def four_phi_integrals(self):
        """Calculate four-phi integral.

        Calculate the integral over the product of four all electron
        functions in the augmentation sphere, i.e.::

          /
          | d vr  ( phi_i1 phi_i2 phi_i3 phi_i4
          /         - phit_i1 phit_i2 phit_i3 phit_i4 ),

        where phi_i1 is an all electron function and phit_i1 is its
        smooth partner.
        """
        if hasattr(self, 'I4_pp'):
            return self.I4_pp

        # radial grid
        r2dr_g = self.rgd.r_g**2 * self.rgd.dr_g

        phi_jg = self.data.phi_jg
        phit_jg = self.data.phit_jg

        # compute radial parts
        nj = len(self.l_j)
        R_jjjj = np.empty((nj, nj, nj, nj))
        for j1 in range(nj):
            for j2 in range(nj):
                for j3 in range(nj):
                    for j4 in range(nj):
                        R_jjjj[j1, j2, j3, j4] = np.dot(
                            r2dr_g,
                            phi_jg[j1] * phi_jg[j2] *
                            phi_jg[j3] * phi_jg[j4] -
                            phit_jg[j1] * phit_jg[j2] *
                            phit_jg[j3] * phit_jg[j4])

        # prepare for angular parts
        L_i = []
        j_i = []
        for j, l in enumerate(self.l_j):
            for m in range(2 * l + 1):
                L_i.append(l**2 + m)
                j_i.append(j)
        ni = len(L_i)
        # j_i is the list of j values
        # L_i is the list of L (=l**2+m for 0<=m<2*l+1) values
        # https://gpaw.readthedocs.io/devel/overview.html

        G_LLL = gaunt(max(self.l_j))

        # calculate the integrals
        _np = ni * (ni + 1) // 2  # length for packing
        self.I4_pp = np.empty((_np, _np))
        p1 = 0
        for i1 in range(ni):
            L1 = L_i[i1]
            j1 = j_i[i1]
            for i2 in range(i1, ni):
                L2 = L_i[i2]
                j2 = j_i[i2]
                p2 = 0
                for i3 in range(ni):
                    L3 = L_i[i3]
                    j3 = j_i[i3]
                    for i4 in range(i3, ni):
                        L4 = L_i[i4]
                        j4 = j_i[i4]
                        self.I4_pp[p1, p2] = (np.dot(G_LLL[L1, L2],
                                                     G_LLL[L3, L4]) *
                                              R_jjjj[j1, j2, j3, j4])
                        p2 += 1
                p1 += 1

        # To unpack into I4_iip do:
        # from gpaw.utilities import unpack_hermitian
        # I4_iip = np.empty((ni, ni, _np)):
        # for p in range(_np):
        #     I4_iip[..., p] = unpack_hermitian(I4_pp[:, p])

        return self.I4_pp

    def get_default_nbands(self):
        assert len(self.l_orb_J) == len(self.n_j), (self.l_orb_J, self.n_j)
        return sum([2 * l + 1 for (l, n) in zips(self.l_orb_J, self.n_j)
                    if n > 0])

    def calculate_coulomb_corrections(self, wn_lqg, wnt_lqg, wg_lg, wnc_g,
                                      wmct_g):
        """Calculate "Coulomb" energies."""
        # Can we reduce the excessive parameter passing?
        # Seems so ....
        # Added instance variables
        # T_Lqp = self.local_corr.T_Lqp
        # n_qg = self.local_corr.n_qg
        # Delta_lq = self.local_corr.Delta_lq
        # nt_qg = self.local_corr.nt_qg
        # Local variables derived from instance variables
        _np = self.ni * (self.ni + 1) // 2  # change to inst. att.?
        mct_g = self.local_corr.nct_g + self.Delta0 * self.g_lg[0]  # s.a.
        rdr_g = self.local_corr.rgd2.r_g * \
            self.local_corr.rgd2.dr_g  # change to inst. att.?

        A_q = 0.5 * (np.dot(wn_lqg[0], self.local_corr.nc_g) + np.dot(
            self.local_corr.n_qg, wnc_g))
        A_q -= sqrt(4 * pi) * self.Z * np.dot(self.local_corr.n_qg, rdr_g)
        A_q -= 0.5 * (np.dot(wnt_lqg[0], mct_g) +
                      np.dot(self.local_corr.nt_qg, wmct_g))
        A_q -= 0.5 * (np.dot(mct_g, wg_lg[0]) +
                      np.dot(self.g_lg[0], wmct_g)) * \
            self.local_corr.Delta_lq[0]
        M_p = np.dot(A_q, self.local_corr.T_Lqp[0])

        A_lqq = []
        for l in range(2 * self.local_corr.lcut + 1):
            A_qq = 0.5 * np.dot(self.local_corr.n_qg, np.transpose(wn_lqg[l]))
            A_qq -= 0.5 * np.dot(self.local_corr.nt_qg,
                                 np.transpose(wnt_lqg[l]))
            if l <= self.lmax:
                A_qq -= 0.5 * np.outer(self.local_corr.Delta_lq[l],
                                       np.dot(wnt_lqg[l], self.g_lg[l]))
                A_qq -= 0.5 * np.outer(np.dot(self.local_corr.nt_qg,
                                              wg_lg[l]),
                                       self.local_corr.Delta_lq[l])
                A_qq -= 0.5 * np.dot(self.g_lg[l], wg_lg[l]) * \
                    np.outer(self.local_corr.Delta_lq[l],
                             self.local_corr.Delta_lq[l])
            A_lqq.append(A_qq)

        M_pp = np.zeros((_np, _np))
        L = 0
        for l in range(2 * self.local_corr.lcut + 1):
            for m in range(2 * l + 1):  # m?
                M_pp += np.dot(np.transpose(self.local_corr.T_Lqp[L]),
                               np.dot(A_lqq[l], self.local_corr.T_Lqp[L]))
                L += 1

        return M_p, M_pp  # , V_p

    def calculate_integral_potentials(self, func):
        """Calculates a set of potentials using func."""
        wg_lg = [func(self.g_lg[l], l)
                 for l in range(self.lmax + 1)]
        wn_lqg = [np.array([func(self.local_corr.n_qg[q], l)
                            for q in range(self.nq)])
                  for l in range(2 * self.local_corr.lcut + 1)]
        wnt_lqg = [np.array([func(self.local_corr.nt_qg[q], l)
                             for q in range(self.nq)])
                   for l in range(2 * self.local_corr.lcut + 1)]
        wnc_g = func(self.local_corr.nc_g, l=0)
        wnct_g = func(self.local_corr.nct_g, l=0)
        wmct_g = wnct_g + self.Delta0 * wg_lg[0]
        return wg_lg, wn_lqg, wnt_lqg, wnc_g, wnct_g, wmct_g

    def calculate_yukawa_interaction(self, gamma):
        """Calculate and return the Yukawa based interaction."""

        # Solves the radial screened poisson equation for density n_g
        def Yuk(n_g, l):
            """Solve radial screened poisson for density n_g."""
            return self.local_corr.rgd2.yukawa(n_g, l, gamma) * \
                self.local_corr.rgd2.r_g * self.local_corr.rgd2.dr_g

        return self.calculate_vvx_interactions(Yuk)

    def calculate_erfc_interaction(self, omega):
        """Calculate and return erfc based valence valence
           exchange interactions."""
        hse = RadialHSE(self.local_corr.rgd2, omega).screened_coulomb_dv

        def erfc_interaction(n_g, l):
            return hse(n_g, l)
        return self.calculate_vvx_interactions(erfc_interaction)

    def calculate_vvx_interactions(self, interaction):
        """Calculate valence valence interactions for generic
           interaction."""
        (wg_lg, wn_lqg, wnt_lqg, wnc_g, wnct_g, wmct_g) = \
            self.calculate_integral_potentials(interaction)
        return self.calculate_coulomb_corrections(
            wn_lqg, wnt_lqg, wg_lg, wnc_g, wmct_g)[1]


class LeanSetup(BaseSetup):
    """Setup class with minimal attribute set.

    A setup-like class must define at least the attributes of this
    class in order to function in a calculation."""
    def __init__(self, s, hubbard_u=None):
        """Copies precisely the necessary attributes of the Setup s."""
        # Hubbard U is poked onto the setup in hacky ways.
        # This needs cleaning.
        self.hubbard_u = hubbard_u

        self.N0_q = s.N0_q  # Required for LDA+U.
        self.type = s.type  # required for writing to file
        self.fingerprint = s.fingerprint  # also req. for writing
        self.filename = s.filename

        self.symbol = s.symbol
        self.Z = s.Z
        self.Nv = s.Nv
        self.Nc = s.Nc

        self.ni = s.ni
        self.nao = s.nao

        self.pt_j = s.pt_j

        self.pseudo_partial_waves_j = s.pseudo_partial_waves_j
        self.basis_functions_J = s.basis_functions_J

        self.Nct = s.Nct
        self.nct = s.nct

        self.lmax = s.lmax
        self.ghat_l = s.ghat_l
        self.vbar = s.vbar

        self.Delta_pL = s.Delta_pL
        self.Delta0 = s.Delta0

        self.E = s.E
        self.Kc = s.Kc

        self.M = s.M
        self.M_p = s.M_p
        self.M_pp = s.M_pp
        self.M_wpp = s.M_wpp
        self.K_p = s.K_p
        self.MB = s.MB
        self.MB_p = s.MB_p

        self.dO_ii = s.dO_ii

        self.xc_correction = s.xc_correction

        # Required to calculate initial occupations
        self.f_j = s.f_j
        self.n_j = s.n_j
        self.l_j = s.l_j
        self.l_orb_J = s.l_orb_J
        self.nj = len(s.l_j)
        self.nq = s.nq

        self.data = s.data

        # Below are things which are not really used all that much,
        # i.e. shouldn't generally be necessary.  Maybe we can make a system
        # involving dictionaries for these "optional" parameters

        # Required by print_info
        self.rcutfilter = s.rcutfilter
        self.rcore = s.rcore
        self.basis = s.basis  # we don't need nao if we use this instead

        # XXX figure out better way to store these.
        # Refactoring: We should delete this and use psit_j.  However
        # the code depends on psit_j being the *basis* functions sometimes.
        if hasattr(s, 'pseudo_partial_waves_j'):
            self.pseudo_partial_waves_j = s.pseudo_partial_waves_j
        # Can also get rid of the phit_j splines if need be

        self.N0_p = s.N0_p  # req. by estimate_magnetic_moments
        self.nabla_iiv = s.nabla_iiv  # req. by lrtddft
        self.rxnabla_iiv = s.rxnabla_iiv  # req. by lrtddft and lrtddft2

        # XAS stuff
        self.phicorehole_g = s.phicorehole_g  # should be optional

        # Required to get all electron density
        self.rgd = s.rgd
        self.rcut_j = s.rcut_j

        self.tauct = s.tauct  # required by TPSS, MGGA

        self.Delta_iiL = s.Delta_iiL  # required with external potential

        self.B_ii = s.B_ii  # required for exact inverse overlap operator
        self.dC_ii = s.dC_ii  # required by time-prop tddft with apply_inverse

        # Required by exx
        self.X_p = s.X_p
        self.X_wp = s.X_wp
        self.ExxC = s.ExxC
        self.ExxC_w = s.ExxC_w

        # Required by yukawa rsf
        self.X_pg = s.X_pg

        # Required by electrostatic correction
        self.dEH0 = s.dEH0
        self.dEH_p = s.dEH_p

        # Required by utilities/kspot.py (AllElectronPotential)
        self.g_lg = s.g_lg

        # Probably empty dictionary, required by GLLB
        self.extra_xc_data = s.extra_xc_data

        self.orbital_free = s.orbital_free

        # Stuff required by Yukawa RSF to calculate Mg_pp at runtime
        # the calcualtion of Mg_pp at rt is needed for dscf
        if hasattr(s, 'local_corr'):
            self.local_corr = s.local_corr
        else:
            self.local_corr = LocalCorrectionVar(s)


class Setup(BaseSetup):
    """Attributes:

    ========== =====================================================
    Name       Description
    ========== =====================================================
    ``Z``      Charge
    ``type``   Type-name of setup (eg. 'paw')
    ``symbol`` Chemical element label (eg. 'Mg')
    ``xcname`` Name of xc
    ``data``   Container class for information on the the atom, eg.
               Nc, Nv, n_j, l_j, f_j, eps_j, rcut_j.
               It defines the radial grid by ng and beta, from which
               r_g = beta * arange(ng) / (ng - arange(ng)).
               It stores pt_jg, phit_jg, phi_jg, vbar_g
    ========== =====================================================


    Attributes for making PAW corrections

    ============= ==========================================================
    Name          Description
    ============= ==========================================================
    ``Delta0``    Constant in compensation charge expansion coeff.
    ``Delta_iiL`` Linear term in compensation charge expansion coeff.
    ``Delta_pL``  Packed version of ``Delta_iiL``.
    ``dO_ii``     Overlap coefficients
    ``B_ii``      Projector function overlaps B_ii = <pt_i | pt_i>
    ``dC_ii``     Inverse overlap coefficients
    ``E``         Reference total energy of atom
    ``M``         Constant correction to Coulomb energy
    ``M_p``       Linear correction to Coulomb energy
    ``M_pp``      2nd order correction to Coulomb energy and Exx energy
    ``M_wpp``     2nd order correction to erfc screened Coulomb energy and Exx
                  energy for given w.
    ``Kc``        Core kinetic energy
    ``K_p``       Linear correction to kinetic energy
    ``ExxC``      Core Exx energy
    ``X_p``       Linear correction to Exx energy
    ``MB``        Constant correction due to vbar potential
    ``MB_p``      Linear correction due to vbar potential
    ``dEH0``      Constant correction due to average electrostatic potential
    ``dEH_p``     Linear correction due to average electrostatic potential
    ``I4_iip``    Correction to integrals over 4 all electron wave functions
    ``Nct``       Analytical integral of the pseudo core density ``nct``
    ============= ==========================================================

    It also has the attribute ``xc_correction`` which is an XCCorrection class
    instance capable of calculating the corrections due to the xc functional.


    Splines:

    ========== ============================================
    Name       Description
    ========== ============================================
    ``pt_j``   Projector functions
    ``phit_j`` Pseudo partial waves
    ``vbar``   vbar potential
    ``nct``    Pseudo core density
    ``ghat_l`` Compensation charge expansion functions
    ``tauct``  Pseudo core kinetic energy density
    ========== ============================================
    """
    def __init__(self, data, xc, lmax=0, basis=None, filter=None,
                 backwards_compatible=True):
        self.type = data.name

        if not data.is_compatible(xc):
            raise ValueError('Cannot use %s setup with %s functional' %
                             (data.setupname, xc.get_setup_name()))

        self.symbol = data.symbol
        self.data = data

        self.Nc = data.Nc
        self.Nv = data.Nv
        self.Z = data.Z
        l_j = self.l_j = data.l_j
        self.l_orb_J = data.l_orb_J
        n_j = self.n_j = data.n_j
        self.f_j = data.f_j
        self.eps_j = data.eps_j
        nj = self.nj = len(l_j)
        self.nq = nj * (nj + 1) // 2
        rcut_j = self.rcut_j = data.rcut_j

        self.ExxC = data.ExxC
        self.ExxC_w = data.ExxC_w
        self.X_p = data.X_p
        self.X_wp = data.X_wp

        self.X_pg = data.X_pg

        self.orbital_free = data.orbital_free

        pt_jg = data.pt_jg
        phit_jg = data.phit_jg
        phi_jg = data.phi_jg

        self.fingerprint = data.fingerprint
        self.filename = data.filename

        rgd = self.rgd = data.rgd
        r_g = rgd.r_g
        dr_g = rgd.dr_g

        self.lmax = lmax

        # Attributes for run-time calculation of _Mg_pp
        self.local_corr = LocalCorrectionVar(data)

        rcutmax = max(rcut_j)
        rcut2 = 2 * rcutmax
        gcut2 = self.gcut2 = rgd.ceil(rcut2)

        vbar_g = data.vbar_g

        # if float(data.version) < 0.7 and data.generator_version < 2:
        if data.generator_version < 2:
            # Old-style Fourier-filtered datasets.
            # Find Fourier-filter cutoff radius:
            gcutfilter = rgd.get_cutoff(pt_jg[0])

        elif filter:
            rc = rcutmax
            vbar_g = vbar_g.copy()
            filter(rgd, rc, vbar_g)

            pt_jg = [pt_g.copy() for pt_g in pt_jg]
            for l, pt_g in zips(l_j, pt_jg):
                filter(rgd, rc, pt_g, l)

            for l in range(max(l_j) + 1):
                J = [j for j, lj in enumerate(l_j) if lj == l]
                A_nn = [[rgd.integrate(phit_jg[j1] * pt_jg[j2]) / 4 / pi
                         for j1 in J] for j2 in J]
                B_nn = np.linalg.inv(A_nn)
                pt_ng = np.dot(B_nn, [pt_jg[j] for j in J])
                for n, j in enumerate(J):
                    pt_jg[j] = pt_ng[n]
            gcutfilter = rgd.get_cutoff(pt_jg[0])
        else:
            gcutfilter = rgd.ceil(max(rcut_j))

        if (vbar_g[gcutfilter:] != 0.0).any():
            gcutfilter = rgd.get_cutoff(vbar_g)
            assert r_g[gcutfilter] < 2.0 * max(rcut_j)

        self.rcutfilter = rcutfilter = r_g[gcutfilter]

        ni = 0
        i = 0
        j = 0
        jlL_i = []
        for l, n in zips(l_j, n_j):
            for m in range(2 * l + 1):
                jlL_i.append((j, l, l**2 + m))
                i += 1
            j += 1
        ni = i
        self.ni = ni

        _np = ni * (ni + 1) // 2

        lcut = max(l_j)
        if 2 * lcut < lmax:
            lcut = (lmax + 1) // 2
        self.local_corr.lcut = lcut

        self.B_ii = self.calculate_projector_overlaps(pt_jg)

        self.fcorehole = data.fcorehole
        self.lcorehole = data.lcorehole

        # Construct splines:
        self.vbar = rgd.spline(vbar_g, rcutfilter)

        rcore, nc_g, nct_g, nct = self.construct_core_densities(
            data, backwards_compatible)
        self.rcore = rcore
        self.nct = nct

        # Construct splines for core kinetic energy density:
        tauct_g = data.tauct_g
        if tauct_g is not None:
            self.tauct = rgd.spline(tauct_g, self.rcore)
        else:
            self.tauct = None

        self.pt_j = self.create_projectors(pt_jg, rcutfilter)

        partial_waves = self.create_basis_functions(phit_jg, rcut2, gcut2)
        self.pseudo_partial_waves_j = partial_waves.tosplines()

        if basis is None:
            basis = partial_waves
            basis_functions_J = self.pseudo_partial_waves_j
        else:
            basis_functions_J = basis.tosplines()

        self.basis_functions_J = basis_functions_J
        self.basis = basis

        self.nao = 0
        for bf in self.basis_functions_J:
            l = bf.get_angular_momentum_number()
            self.nao += 2 * l + 1

        rgd2 = self.local_corr.rgd2 = rgd.new(gcut2)
        r_g = rgd2.r_g
        dr_g = rgd2.dr_g
        phi_jg = np.array([phi_g[:gcut2].copy() for phi_g in phi_jg])
        phit_jg = np.array([phit_g[:gcut2].copy() for phit_g in phit_jg])
        self.local_corr.nc_g = nc_g = nc_g[:gcut2].copy()
        self.local_corr.nct_g = nct_g = nct_g[:gcut2].copy()
        vbar_g = vbar_g[:gcut2].copy()

        extra_xc_data = dict(data.extra_xc_data)
        # Cut down the GLLB related extra data
        for key, item in extra_xc_data.items():
            if len(item) == rgd.N:
                extra_xc_data[key] = item[:gcut2].copy()
        self.extra_xc_data = extra_xc_data

        self.phicorehole_g = data.phicorehole_g
        if self.phicorehole_g is not None:
            self.phicorehole_g = self.phicorehole_g[:gcut2].copy()

        self.local_corr.T_Lqp = self.calculate_T_Lqp(lcut, _np, nj, jlL_i)
        #  set the attributes directly?
        (self.g_lg, self.local_corr.n_qg, self.local_corr.nt_qg,
         self.local_corr.Delta_lq, self.Lmax, self.Delta_pL, self.Delta0,
         self.N0_p) = self.get_compensation_charges(phi_jg, phit_jg, _np,
                                                    self.local_corr.T_Lqp)

        # Solves the radial poisson equation for density n_g
        def H(n_g, l):
            return rgd2.poisson(n_g, l) * r_g * dr_g

        (wg_lg, wn_lqg, wnt_lqg, wnc_g, wnct_g, wmct_g) = \
            self.calculate_integral_potentials(H)
        self.wg_lg = wg_lg

        rdr_g = r_g * dr_g
        dv_g = r_g * rdr_g
        A = 0.5 * np.dot(nc_g, wnc_g)
        A -= sqrt(4 * pi) * self.Z * np.dot(rdr_g, nc_g)
        mct_g = nct_g + self.Delta0 * self.g_lg[0]
        # wmct_g = wnct_g + self.Delta0 * wg_lg[0]
        A -= 0.5 * np.dot(mct_g, wmct_g)
        self.M = A
        self.MB = -np.dot(dv_g * nct_g, vbar_g)

        AB_q = -np.dot(self.local_corr.nt_qg, dv_g * vbar_g)
        self.MB_p = np.dot(AB_q, self.local_corr.T_Lqp[0])

        # Correction for average electrostatic potential:
        #
        #   dEH = dEH0 + dot(D_p, dEH_p)
        #
        self.dEH0 = sqrt(4 * pi) * (wnc_g - wmct_g -
                                    sqrt(4 * pi) * self.Z * r_g * dr_g).sum()
        dEh_q = (wn_lqg[0].sum(1) - wnt_lqg[0].sum(1) -
                 self.local_corr.Delta_lq[0] * wg_lg[0].sum())
        self.dEH_p = np.dot(dEh_q, self.local_corr.T_Lqp[0]) * sqrt(4 * pi)

        M_p, M_pp = self.calculate_coulomb_corrections(wn_lqg, wnt_lqg,
                                                       wg_lg, wnc_g, wmct_g)
        self.M_p = M_p
        self.M_pp = M_pp

        self.M_wpp = {}
        for omega in self.ExxC_w:
            self.M_wpp[omega] = self.calculate_erfc_interaction(omega)

        self.Kc = data.e_kinetic_core - data.e_kinetic
        self.M -= data.e_electrostatic
        self.E = data.e_total

        Delta0_ii = unpack_hermitian(self.Delta_pL[:, 0].copy())
        self.dO_ii = data.get_overlap_correction(Delta0_ii)
        self.dC_ii = self.get_inverse_overlap_coefficients(self.B_ii,
                                                           self.dO_ii)

        self.Delta_iiL = np.zeros((ni, ni, self.Lmax))
        for L in range(self.Lmax):
            self.Delta_iiL[:, :, L] = unpack_hermitian(
                self.Delta_pL[:, L].copy())

        self.Nct = data.get_smooth_core_density_integral(self.Delta0)
        self.K_p = data.get_linear_kinetic_correction(self.local_corr.T_Lqp[0])

        self.ghat_l = [rgd2.spline(g_g, rcut2, l, 50)
                       for l, g_g in enumerate(self.g_lg)]

        self.xc_correction = data.get_xc_correction(rgd2, xc, gcut2, lcut)
        self.nabla_iiv = self.get_derivative_integrals(rgd2, phi_jg, phit_jg)
        self.rxnabla_iiv = self.get_magnetic_integrals(rgd2, phi_jg, phit_jg)

    def create_projectors(self, pt_jg, rcut):
        pt_j = []
        for j, pt_g in enumerate(pt_jg):
            l = self.l_j[j]
            pt_j.append(self.rgd.spline(pt_g, rcut, l))
        return pt_j

    def get_inverse_overlap_coefficients(self, B_ii, dO_ii):
        ni = len(B_ii)
        xO_ii = np.dot(B_ii, dO_ii)
        return -np.dot(dO_ii, np.linalg.inv(np.identity(ni) + xO_ii))

    def calculate_T_Lqp(self, lcut, _np, nj, jlL_i):
        Lcut = (2 * lcut + 1)**2
        G_LLL = gaunt(max(self.l_j))[:, :, :Lcut]
        LGcut = G_LLL.shape[2]
        T_Lqp = np.zeros((Lcut, self.nq, _np))
        p = 0
        i1 = 0
        for j1, l1, L1 in jlL_i:
            for j2, l2, L2 in jlL_i[i1:]:
                if j1 < j2:
                    q = j2 + j1 * nj - j1 * (j1 + 1) // 2
                else:
                    q = j1 + j2 * nj - j2 * (j2 + 1) // 2
                T_Lqp[:LGcut, q, p] = G_LLL[L1, L2]
                p += 1
            i1 += 1
        return T_Lqp

    def calculate_projector_overlaps(self, pt_jg):
        """Compute projector function overlaps B_ii = <pt_i | pt_i>."""
        nj = len(pt_jg)
        B_jj = np.zeros((nj, nj))
        for j1, pt1_g in enumerate(pt_jg):
            for j2, pt2_g in enumerate(pt_jg):
                B_jj[j1, j2] = self.rgd.integrate(pt1_g * pt2_g) / (4 * pi)
        B_ii = np.zeros((self.ni, self.ni))
        i1 = 0
        for j1, l1 in enumerate(self.l_j):
            for m1 in range(2 * l1 + 1):
                i2 = 0
                for j2, l2 in enumerate(self.l_j):
                    for m2 in range(2 * l2 + 1):
                        if l1 == l2 and m1 == m2:
                            B_ii[i1, i2] = B_jj[j1, j2]
                        i2 += 1
                i1 += 1
        return B_ii

    def get_compensation_charges(self, phi_jg, phit_jg, _np, T_Lqp):
        lmax = self.lmax
        gcut2 = self.gcut2
        rcut_j = self.rcut_j
        nq = self.nq

        rgd = self.local_corr.rgd2
        r_g = rgd.r_g
        dr_g = rgd.dr_g

        g_lg = self.data.create_compensation_charge_functions(lmax)

        n_qg = np.zeros((nq, gcut2))
        nt_qg = np.zeros((nq, gcut2))
        gcut_q = np.zeros(nq, dtype=int)
        N0_q = np.zeros(nq)
        q = 0  # q: common index for j1, j2
        for j1 in range(self.nj):
            for j2 in range(j1, self.nj):
                n_qg[q] = phi_jg[j1] * phi_jg[j2]
                nt_qg[q] = phit_jg[j1] * phit_jg[j2]

                gcut = rgd.ceil(min(rcut_j[j1], rcut_j[j2]))
                N0_q[q] = sum(n_qg[q, :gcut] * r_g[:gcut]**2 * dr_g[:gcut])
                gcut_q[q] = gcut

                q += 1

        self.gcut_q = gcut_q
        self.N0_q = N0_q

        Delta_lq = np.zeros((lmax + 1, nq))
        for l in range(lmax + 1):
            Delta_lq[l] = np.dot(n_qg - nt_qg, r_g**(2 + l) * dr_g)

        Lmax = (lmax + 1)**2
        Delta_pL = np.zeros((_np, Lmax))
        for l in range(lmax + 1):
            L = l**2
            for m in range(2 * l + 1):
                delta_p = np.dot(Delta_lq[l], T_Lqp[L + m])
                Delta_pL[:, L + m] = delta_p

        Delta0 = np.dot(self.local_corr.nc_g - self.local_corr.nct_g,
                        r_g**2 * dr_g) - self.Z / sqrt(4 * pi)

        # Electron density inside augmentation sphere.  Used for estimating
        # atomic magnetic moment:
        N0_p = N0_q @ T_Lqp[0] * sqrt(4 * pi)

        return (g_lg[:, :gcut2].copy(), n_qg, nt_qg,
                Delta_lq, Lmax, Delta_pL, Delta0, N0_p)

    def get_derivative_integrals(self, rgd, phi_jg, phit_jg):
        """Calculate PAW-correction matrix elements of nabla.

        ::

          /  _       _  d       _     ~   _  d   ~   _
          | dr [phi (r) -- phi (r) - phi (r) -- phi (r)]
          /        1    dx    2         1    dx    2

        and similar for y and z."""
        # lmax needs to be at least 1 for evaluating
        # the Gaunt coefficients from derivatives
        lmax = max(1, max(self.l_j))
        G_LLL = gaunt(lmax)
        Y_LLv = nabla(lmax)

        r_g = rgd.r_g
        dr_g = rgd.dr_g
        nabla_iiv = np.empty((self.ni, self.ni, 3))
        if debug:
            nabla_iiv[:] = np.nan
        i1 = 0
        for j1 in range(self.nj):
            l1 = self.l_j[j1]
            nm1 = 2 * l1 + 1
            i2 = 0
            for j2 in range(self.nj):
                l2 = self.l_j[j2]
                nm2 = 2 * l2 + 1
                f1f2or = np.dot(phi_jg[j1] * phi_jg[j2] -
                                phit_jg[j1] * phit_jg[j2], r_g * dr_g)
                dphidr_g = np.empty_like(phi_jg[j2])
                rgd.derivative_spline(phi_jg[j2], dphidr_g)
                dphitdr_g = np.empty_like(phit_jg[j2])
                rgd.derivative_spline(phit_jg[j2], dphitdr_g)
                f1df2dr = np.dot(phi_jg[j1] * dphidr_g -
                                 phit_jg[j1] * dphitdr_g, r_g**2 * dr_g)
                for v in range(3):
                    Lv = 1 + (v + 2) % 3
                    G_12 = G_LLL[Lv, l1**2:l1**2 + nm1, l2**2:l2**2 + nm2]
                    Y_12 = Y_LLv[l1**2:l1**2 + nm1, l2**2:l2**2 + nm2, v]
                    nabla_iiv[i1:i1 + nm1, i2:i2 + nm2, v] = (
                        sqrt(4 * pi / 3) * (f1df2dr - l2 * f1f2or) * G_12
                        + f1f2or * Y_12)
                i2 += nm2
            i1 += nm1
        if debug:
            assert not np.any(np.isnan(nabla_iiv))
        return nabla_iiv

    def get_magnetic_integrals(self, rgd, phi_jg, phit_jg):
        """Calculate PAW-correction matrix elements of r x nabla.

        ::

          /  _       _          _     ~   _      ~   _
          | dr [phi (r) O  phi (r) - phi (r) O  phi (r)]
          /        1     x    2         1     x    2

                       d      d
          where O  = y -- - z --
                 x     dz     dy

        and similar for y and z."""
        # lmax needs to be at least 1 for evaluating
        # the Gaunt coefficients from derivatives
        lmax = max(1, max(self.l_j))
        G_LLL = gaunt(lmax)
        Y_LLv = nabla(2 * lmax)

        r_g = rgd.r_g
        dr_g = rgd.dr_g
        rxnabla_iiv = np.empty((self.ni, self.ni, 3))
        if debug:
            rxnabla_iiv[:] = np.nan
        i1 = 0
        for j1 in range(self.nj):
            l1 = self.l_j[j1]
            nm1 = 2 * l1 + 1
            i2 = 0
            for j2 in range(self.nj):
                l2 = self.l_j[j2]
                nm2 = 2 * l2 + 1
                f1f2or = np.dot(phi_jg[j1] * phi_jg[j2] -
                                phit_jg[j1] * phit_jg[j2], r_g**2 * dr_g)
                for v in range(3):
                    v1 = (v + 1) % 3
                    v2 = (v + 2) % 3
                    Lv1 = 1 + (v1 + 2) % 3
                    Lv2 = 1 + (v2 + 2) % 3
                    # term from radial wfs does not contribute
                    # term from spherical harmonics derivatives
                    G_12 = np.zeros((nm1, nm2))
                    G_12 += np.dot(G_LLL[Lv1, l1**2:l1**2 + nm1, :],
                                   Y_LLv[:, l2**2:l2**2 + nm2, v2])
                    G_12 -= np.dot(G_LLL[Lv2, l1**2:l1**2 + nm1, :],
                                   Y_LLv[:, l2**2:l2**2 + nm2, v1])
                    rxnabla_iiv[i1:i1 + nm1, i2:i2 + nm2, v] = (
                        sqrt(4 * pi / 3) * f1f2or * G_12)
                i2 += nm2
            i1 += nm1
        if debug:
            assert not np.any(np.isnan(rxnabla_iiv))
        return rxnabla_iiv

    def construct_core_densities(self, setupdata, backwards_compatible=True):
        rcore = self.data.find_core_density_cutoff(setupdata.nc_g)
        nct = self.rgd.spline(
            setupdata.nct_g, rcore,
            points=None if backwards_compatible else 256,
            backwards_compatible=backwards_compatible)
        return rcore, setupdata.nc_g, setupdata.nct_g, nct

    def create_basis_functions(self, phit_jg, rcut2, gcut2):
        # Cutoff for atomic orbitals used for initial guess:
        rcut3 = 8.0  # XXXXX Should depend on the size of the atom!
        gcut3 = self.rgd.ceil(rcut3)

        # We cut off the wave functions smoothly at rcut3 by the
        # following replacement:
        #
        #            /
        #           | f(r),                                   r < rcut2
        #  f(r) <- <  f(r) - a(r) f(rcut3) - b(r) f'(rcut3),  rcut2 < r < rcut3
        #           | 0,                                      r > rcut3
        #            \
        #
        # where a(r) and b(r) are 4. order polynomials:
        #
        #  a(rcut2) = 0,  a'(rcut2) = 0,  a''(rcut2) = 0,
        #  a(rcut3) = 1, a'(rcut3) = 0
        #  b(rcut2) = 0, b'(rcut2) = 0, b''(rcut2) = 0,
        #  b(rcut3) = 0, b'(rcut3) = 1
        #
        r_g = self.rgd.r_g
        x = (r_g[gcut2:gcut3] - rcut2) / (rcut3 - rcut2)
        a_g = 4 * x**3 * (1 - 0.75 * x)
        b_g = x**3 * (x - 1) * (rcut3 - rcut2)

        basis_functions_J = []
        n_J = []
        for j, phit_g in enumerate(phit_jg):
            n = self.n_j[j]
            if n > 0:
                n_J.append(n)
                l = self.l_j[j]
                phit_g = phit_g.copy()
                phit = phit_g[gcut3]
                dphitdr = ((phit - phit_g[gcut3 - 1]) /
                           (r_g[gcut3] - r_g[gcut3 - 1]))
                phit_g[gcut2:gcut3] -= phit * a_g + dphitdr * b_g
                phit_g[gcut3:] = 0.0
                basis_function = self.rgd.spline(phit_g, rcut3, l, points=100)
                basis_functions_J.append(basis_function)
        basis = PartialWaveBasis(self.symbol, basis_functions_J, n_J)
        return basis


class PartialWaveBasis(Basis):  # yuckkk
    def __init__(self, symbol, phit_J, n_J):
        self._basis_functions_J = phit_J
        super().__init__(
            symbol, 'partial-waves',
            bf_j=[BasisFunction(n, phit.get_angular_momentum_number())
                  for n, phit in zip(n_J, phit_J)])

    def tosplines(self):
        return self._basis_functions_J

    def get_description(self):
        template = 'Using partial waves for %s as LCAO basis'
        string = template % self.symbol
        return string


class Setups(list):
    """Collection of Setup objects. One for each distinct atom.

    Non-distinct atoms are those with the same atomic number, setup, and basis.

    Class attributes:

    ``nvalence``    Number of valence electrons.
    ``nao``         Number of atomic orbitals.
    ``Eref``        Reference energy.
    ``core_charge`` Core hole charge.
    """

    def __init__(self, Z_a, setup_types, basis_sets, xc, *,
                 filter=None,
                 world=None,
                 backwards_compatible=True):
        list.__init__(self)
        symbols = [chemical_symbols[Z] for Z in Z_a]
        type_a = types2atomtypes(symbols, setup_types, default='paw')
        basis_a = types2atomtypes(symbols, basis_sets, default=None)

        for a, _type in enumerate(type_a):
            # Make basis files correspond to setup files.
            #
            # If the setup has a name (i.e. non-default _type), then
            # prepend that name to the basis name.
            #
            # Typically people might specify '11' as the setup but just
            # 'dzp' for the basis set.  Here we adjust to
            # obtain, say, '11.dzp' which loads the correct basis set.
            #
            # There will be no way to obtain the original 'dzp' with
            # a custom-named setup except by loading directly from
            # BasisData.
            #
            # Due to the "szp(dzp)" syntax this is complicated!
            # The name has to go as "szp(name.dzp)".
            basis = basis_a[a]
            if isinstance(basis, str):
                if isinstance(_type, str):
                    setupname = _type
                else:
                    setupname = _type.name  # _type is an object like SetupData
                # Drop DFT+U specification from type string if it is there:
                if hasattr(setupname, 'swapcase'):
                    setupname = setupname.split(':')[0]

                # Basis names inherit setup names except default setups
                # and ghost atoms.
                if setupname != 'paw' and setupname != 'ghost':
                    if setupname:
                        if '(' in basis:
                            reduced, name = basis.split('(')
                            assert name.endswith(')')
                            name = name[:-1]
                            fullname = f'{reduced}({setupname}.{name})'
                        else:
                            fullname = f'{setupname}.{basis_a[a]}'
                        basis_a[a] = fullname

        # Construct necessary PAW-setup objects:
        self.setups = {}
        natoms = {}
        Mcumulative = 0
        self.M_a = []
        self.id_a = list(zips(Z_a, type_a, basis_a))
        for id in self.id_a:
            setup = self.setups.get(id)
            if setup is None:
                Z, type, basis = id
                symbol = chemical_symbols[Z]
                setupdata = None
                if not isinstance(type, str):
                    setupdata = type
                # Basis may be None (meaning that the setup decides), a string
                # (meaning we load the basis set now from a file) or an actual
                # pre-created Basis object (meaning we just pass it along)
                if isinstance(basis, str):
                    basis = Basis.find(symbol, basis, world=world)
                setup = create_setup(symbol, xc, 2, type,
                                     basis, setupdata=setupdata,
                                     filter=filter, world=world,
                                     backwards_compatible=backwards_compatible)
                self.setups[id] = setup
                natoms[id] = 0
            natoms[id] += 1
            self.append(setup)
            self.M_a.append(Mcumulative)
            Mcumulative += setup.nao

        # Sum up ...
        self.nvalence = 0       # number of valence electrons
        self.nao = 0            # number of atomic orbitals
        self.Eref = 0.0         # reference energy
        self.core_charge = 0.0  # core hole charge
        for id, setup in self.setups.items():
            n = natoms[id]
            self.Eref += n * setup.E
            self.core_charge += n * (setup.Z - setup.Nv - setup.Nc)
            self.nvalence += n * setup.Nv
            self.nao += n * setup.nao

        self.dS = OverlapCorrections(self)

        self.backwards_compatible = backwards_compatible

    def __str__(self):
        # Write PAW setup information in order of appearance:
        ids = set()
        s = 'species:\n'
        for id in self.id_a:
            if id in ids:
                continue
            ids.add(id)
            setup = self.setups[id]
            output = StringIO()
            setup.print_info(functools.partial(print, file=output))
            txt = output.getvalue()
            txt += '  # ' + setup.get_basis_description().replace('\n',
                                                                  '\n  # ')
            txt = txt.replace('\n', '\n  ')
            s += '  ' + txt + '\n\n'

        s += f'Reference energy: {self.Eref * units.Hartree:.6f}  # eV\n'
        return s

    def set_symmetry(self, symmetry):
        """Find rotation matrices for spherical harmonics."""
        # XXX It is ugly that we set self.atomrotations from here;
        # it would be better to return it to the caller.
        from gpaw.atomrotations import AtomRotations
        self.atomrotations = AtomRotations(self.setups, self.id_a, symmetry)

    def empty_atomic_matrix(self, ns, atom_partition, dtype=float):
        Dshapes_a = [(ns, setup.ni * (setup.ni + 1) // 2)
                     for setup in self]
        return atom_partition.arraydict(Dshapes_a, dtype)

    def estimate_dedecut(self, ecut):
        from gpaw.utilities.ekin import dekindecut, ekin
        dedecut = 0.0
        e = {}
        for id in self.id_a:
            if id not in e:
                G, de, e0 = ekin(self.setups[id])
                e[id] = -dekindecut(G, de, ecut)
            dedecut += e[id]
        return dedecut

    def basis_indices(self):
        return FunctionIndices([setup.basis_functions_J for setup in self])

    def projector_indices(self):
        return FunctionIndices([setup.pt_j for setup in self])

    def create_pseudo_core_densities(self, domain, positions, atomdist,
                                     xp=np):
        spline_aj = []
        for setup in self:
            if setup.nct is None:
                spline_aj.append([])
            else:
                spline_aj.append([setup.nct])
        if self.backwards_compatible and hasattr(domain, 'ecut'):
            integrals = None
        else:
            # 0.0 will skip normalization:
            integrals = [setup.Nct if abs(setup.Nct) > 1e-12 else 0.0
                         for setup in self]
        return domain.atom_centered_functions(
            spline_aj, positions,
            atomdist=atomdist,
            integrals=integrals,
            cut=True, xp=xp)

    def create_pseudo_core_ked(self,
                               domain,
                               positions,
                               atomdist):
        return domain.atom_centered_functions(
            [[setup.tauct] for setup in self],
            positions,
            atomdist=atomdist,
            cut=True)

    def create_local_potentials(self, domain, positions, atomdist, xp=np):
        return domain.atom_centered_functions(
            [[setup.vbar] for setup in self],
            positions,
            atomdist=atomdist,
            xp=xp)

    def create_compensation_charges(self, domain, positions, atomdist=None,
                                    xp=np):
        if self.backwards_compatible and hasattr(domain, 'ecut'):
            integral = None
        else:
            integral = sqrt(4 * pi)
        return domain.atom_centered_functions(
            [setup.ghat_l for setup in self], positions,
            atomdist=atomdist,
            integrals=integral,
            xp=xp)

    def get_overlap_corrections(self, atomdist, xp, dtype=np.float64):
        if atomdist is getattr(self, '_atomdist', None):
            if self.dS_aii.data.dtype == dtype:
                return self.dS_aii
        self._atomdist = atomdist
        dS_aii = AtomArraysLayout([setup.dO_ii.shape for setup in self],
                                  atomdist=atomdist, dtype=dtype).empty()
        for a, dS_ii in dS_aii.items():
            dS_ii[:] = self[a].dO_ii
        self.dS_aii = dS_aii.to_xp(xp)
        return self.dS_aii

    def partial_wave_corrections(self) -> list[list[Spline]]:
        splines: dict[Setup, list[Spline]] = {}
        dphi_aj = []
        for setup in self:
            dphi_j = splines.get(setup)
            if dphi_j is None:
                rcut = max(setup.rcut_j) * 1.1
                gcut = setup.rgd.ceil(rcut)
                dphi_j = []
                for l, phi_g, phit_g in zips(setup.l_j,
                                             setup.data.phi_jg,
                                             setup.data.phit_jg):
                    dphi_g = (phi_g - phit_g)[:gcut]
                    dphi_j.append(setup.rgd.spline(dphi_g, rcut, l,
                                                   points=200))
                splines[setup] = dphi_j
            dphi_aj.append(dphi_j)

        return dphi_aj


class FunctionIndices:
    def __init__(self, f_aj):
        nm_a = [0]
        for f_j in f_aj:
            nm = sum([2 * f.get_angular_momentum_number() + 1 for f in f_j])
            nm_a.append(nm)
        self.M_a = np.cumsum(nm_a)
        self.nm_a = np.array(nm_a[1:])
        self.max = self.M_a[-1]

    def __getitem__(self, a):
        return self.M_a[a], self.M_a[a + 1]


class CachedYukawaInteractions:
    def __init__(self, omega):
        self.omega = omega
        self._cache = {}

    def get_Mg_pp(self, setup):
        if setup not in self._cache:
            Mg_pp = setup.calculate_yukawa_interaction(self.omega)
            self._cache[setup] = Mg_pp
        return self._cache[setup]


def types2atomtypes(symbols, types, default):
    """Map a types identifier to a list with a type id for each atom.

    types can be a single str, or a dictionary mapping chemical
    symbols and/or atom numbers to a type identifier.
    If both a symbol key and atomnumber key relates to the same atom, then
    the atomnumber key is dominant.

    If types is a dictionary and contains the string 'default', this will
    be used as default type, otherwize input arg ``default`` is used as
    default.
    """
    natoms = len(symbols)
    if isinstance(types, str):
        return [types] * natoms

    # If present, 'default' will map to the default type,
    # else use the input default
    type_a = [types.get('default', default)] * natoms

    # First symbols ...
    for symbol, type in types.items():
        # Types are given either by strings or they are objects that
        # have a 'symbol' attribute (SetupData, Pseudopotential, Basis, etc.).
        assert isinstance(type, str) or hasattr(type, 'symbol')
        if isinstance(symbol, str):
            for a, symbol2 in enumerate(symbols):
                if symbol == symbol2:
                    type_a[a] = type

    # and then atom indices
    for a, type in types.items():
        if isinstance(a, int):
            type_a[a] = type

    return type_a


if __name__ == '__main__':
    print("""\
This is not the setup.py you are looking for!  This setup.py defines a
Setup class used to hold the atomic data needed for a specific atom.
For building the GPAW code you must use the setup.py setuptools script
at the root of the code tree.  Just do "cd .." and you will be at the
right place.""")
    raise SystemExit
