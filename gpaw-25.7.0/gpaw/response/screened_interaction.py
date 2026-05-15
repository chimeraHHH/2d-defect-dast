import numpy as np
from math import pi
from gpaw.response.q0_correction import Q0Correction
from ase.units import Ha
from ase.dft.kpoints import monkhorst_pack
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.response.temp import DielectricFunctionCalculator
from gpaw.response.hilbert import GWHilbertTransforms
from gpaw.response.mpa_interpolation import RESolver
from gpaw.cgpaw import evaluate_mpa_poly


class GammaIntegrationMode:
    def __init__(self, gamma_integration):
        if isinstance(gamma_integration, GammaIntegrationMode):
            self.type = gamma_integration.type
            self.reduced = gamma_integration.reduced
            self._N = gamma_integration._N
            return

        defaults = {'sphere': {'type': 'sphere'},
                    'reciprocal': {'type': 'reciprocal'},
                    'reciprocal2D': {'type': 'reciprocal', 'reduced': True},
                    '1BZ': {'type': '1BZ'},
                    '1BZ2D': {'type': '1BZ', 'reduced': True},
                    'WS': {'type': 'WS'}}

        if isinstance(gamma_integration, int):
            raise TypeError("gamma_integration=INT is no longer supported. "
                            "Please start using the new notations, as is given"
                            " in the documentation in gpaw/response/g0w0.py"
                            " of __init__ of class G0W0.")

        if isinstance(gamma_integration, str):
            gamma_integration = defaults[gamma_integration]

        self.type = gamma_integration['type']
        self.reduced = gamma_integration.get('reduced', False)
        self._N = gamma_integration.get('N', 100)

        if self.type not in {'sphere', 'reciprocal', '1BZ', 'WS'}:
            raise TypeError('type in gamma_integration should be one of sphere'
                            ', reciprocal, 1BZ, or WS.')

        if not self.is_numerical:
            if gamma_integration.get('reduced', False):
                raise TypeError('reduced key being True is only supported for '
                                'type reciprocal or 1BZ.')

    def __repr__(self):
        return f'type: {self.type} reduced: {self.reduced}'

    @property
    def is_analytical(self):
        return self.type == 'sphere'

    @property
    def is_numerical(self):
        return self.type in {'reciprocal', '1BZ'}

    @property
    def is_Wigner_Seitz(self):
        return self.type == 'WS'

    @property
    def to_1bz(self):
        return self.type == '1BZ'

    @property
    def N(self):
        return self._N


class QPointDescriptor(KPointDescriptor):

    @staticmethod
    def from_gs(gs):
        kd, atoms = gs.kd, gs.atoms
        # Find q-vectors and weights in the IBZ:
        assert -1 not in kd.bz2bz_ks
        offset_c = 0.5 * ((kd.N_c + 1) % 2) / kd.N_c
        bzq_qc = monkhorst_pack(kd.N_c) + offset_c
        qd = KPointDescriptor(bzq_qc)
        qd.set_symmetry(atoms, kd.symmetry)
        return qd


def initialize_w_calculator(chi0calc, context, *,
                            coulomb,
                            xc='RPA',  # G0W0Kernel arguments
                            mpa=None, E0=Ha, eta=None,
                            integrate_gamma=GammaIntegrationMode('sphere'),
                            q0_correction=False):
    """Initialize a WCalculator from a Chi0Calculator.

    Parameters
    ----------
    chi0calc : Chi0Calculator
    xc : str
        Kernel to use when including vertex corrections.

    Remaining arguments: See WCalculator
    """
    from gpaw.response.g0w0_kernels import G0W0Kernel

    gs = chi0calc.gs
    qd = QPointDescriptor.from_gs(gs)

    xckernel = G0W0Kernel(xc=xc, ecut=chi0calc.chi0_body_calc.ecut,
                          gs=gs, qd=qd,
                          context=context)

    kwargs = dict()
    if mpa:
        wcalc_cls = MPACalculator
        kwargs['mpa'] = mpa
    else:
        wcalc_cls = WCalculator

    return wcalc_cls(gs, context, qd=qd,
                     coulomb=coulomb, xckernel=xckernel,
                     integrate_gamma=integrate_gamma, eta=eta,
                     q0_correction=q0_correction, **kwargs)


class WBaseCalculator():

    def __init__(self, gs, context, *, qd,
                 coulomb, xckernel,
                 integrate_gamma=GammaIntegrationMode('sphere'),
                 eta=None,
                 q0_correction=False):
        """
        Base class for W Calculator including basic initializations and Gamma
        Gamma handling.

        Parameters
        ----------
        gs : ResponseGroundStateAdapter
        context : ResponseContext
        qd : QPointDescriptor
        coulomb : CoulombKernel
        xckernel : G0W0Kernel
        integrate_gamma: GammaIntegrationMode
        q0_correction : bool
            Analytic correction to the q=0 contribution applicable to 2D
            systems.
        """
        self.gs = gs
        self.context = context
        self.qd = qd
        self.coulomb = coulomb
        self.xckernel = xckernel
        self.integrate_gamma = integrate_gamma
        self.eta = eta

        if q0_correction:
            assert self.coulomb.truncation == '2D'
            self.q0_corrector = Q0Correction(
                cell_cv=self.gs.gd.cell_cv,
                bzk_kc=self.gs.kd.bzk_kc,
                N_c=self.qd.N_c)

            npts_c = self.q0_corrector.npts_c
            self.context.print('Applying analytical 2D correction to W:',
                               flush=False)
            self.context.print('    Evaluating Gamma point contribution to W '
                               + 'on a %dx%dx%d grid' % tuple(npts_c))
        else:
            self.q0_corrector = None

    def get_V0sqrtV0(self, chi0):
        """
        Integrated Coulomb kernels.
        """
        V0 = None
        sqrtV0 = None
        if self.integrate_gamma.is_numerical:
            reduced = self.integrate_gamma.reduced
            tofirstbz = self.integrate_gamma.to_1bz
            N = self.integrate_gamma.N
            V0, sqrtV0 = self.coulomb.integrated_kernel(qpd=chi0.qpd,
                                                        reduced=reduced,
                                                        tofirstbz=tofirstbz,
                                                        N=N)
        elif self.integrate_gamma.is_analytical:
            if chi0.optical_limit:
                # The volume of reciprocal cell occupied by a single q-point
                bzvol = (2 * np.pi)**3 / self.gs.volume / self.qd.nbzkpts
                # Radius of a sphere with a volume of the bzvol above
                Rq0 = (3 * bzvol / (4 * np.pi))**(1. / 3.)
                # Analytical integral of Coulomb interaction over the sphere
                # defined above centered at q=0.
                # V0 = 1/|bzvol| int_bzvol dq 4 pi / |q^2|
                V0 = 16 * np.pi**2 * Rq0 / bzvol
                # Analytical integral of square root of Coulomb interaction
                # over the same sphere
                # sqrtV0 = 1/|bzvol| int_bzvol dq sqrt(4 pi / |q^2|)
                sqrtV0 = (4 * np.pi)**(1.5) * Rq0**2 / bzvol / 2
        else:
            raise KeyError('Unknown integrate_gamma option:'
                           f'{self.integrate_gamma}.')
        return V0, sqrtV0

    def apply_gamma_correction(self, W_GG, einv_GG, V0, sqrtV0, sqrtV_G):
        """
        Replacing q=0, (G,G')= (0,0), (0,:), (:,0) with corresponding
        matrix elements calculated with an average of the (diverging)
        Coulomb interaction.
        XXX: Understand and document exact expressions
        """
        W_GG[0, 0] = einv_GG[0, 0] * V0
        W_GG[0, 1:] = einv_GG[0, 1:] * sqrtV_G[1:] * sqrtV0
        W_GG[1:, 0] = einv_GG[1:, 0] * sqrtV0 * sqrtV_G[1:]


class WCalculator(WBaseCalculator):
    def get_HW_model(self, chi0, fxc_mode, only_correlation=True):
        assert only_correlation
        W_wGG = self.calculate_W_WgG(chi0,
                                     fxc_mode=fxc_mode,
                                     only_correlation=True)
        # HT used to calculate convulution between time-ordered G and W
        hilbert_transform = GWHilbertTransforms(chi0.wd.omega_w, self.eta)
        with self.context.timer('Hilbert'):
            W_xwGG = hilbert_transform(W_wGG)

        factor = 1.0 / (self.qd.nbzkpts * 2 * pi * self.gs.volume)
        return FullFrequencyHWModel(chi0.wd, W_xwGG, factor)

    def calculate_W_WgG(self, chi0,
                        fxc_mode='GW',
                        only_correlation=False):
        """Calculate the screened interaction in W_wGG or W_WgG representation.

        Additional Parameters
        ----------
        only_correlation: bool
             if true calculate Wc otherwise calculate full W
        out_dist: str
             specifices output distribution of W array (wGG or WgG)
        """
        W_wGG = self.calculate_W_wGG(chi0, fxc_mode,
                                     only_correlation=only_correlation)

        W_WgG = chi0.body.blockdist.distribute_as(W_wGG, chi0.body.nw, 'WgG')
        return W_WgG

    def calculate_W_wGG(self, chi0, fxc_mode='GW',
                        only_correlation=False):
        """In-place calculation of the screened interaction."""
        dfc = DielectricFunctionCalculator(chi0, self.coulomb,
                                           self.xckernel, fxc_mode)
        self.context.timer.start('Dyson eq.')

        if self.integrate_gamma.is_Wigner_Seitz:
            from gpaw.hybrids.wstc import WignerSeitzTruncatedCoulomb
            wstc = WignerSeitzTruncatedCoulomb(chi0.qpd.gd.cell_cv,
                                               dfc.coulomb.N_c)
            sqrtV_G = wstc.get_potential(chi0.qpd)**0.5
        else:
            sqrtV_G = dfc.sqrtV_G
            V0, sqrtV0 = self.get_V0sqrtV0(chi0)

        einv_wGG = dfc.get_epsinv_wGG(only_correlation=False)
        W_wGG = np.empty_like(einv_wGG)
        for iw, (einv_GG, W_GG) in enumerate(zip(einv_wGG, W_wGG)):
            # If only_correlation = True function spits out
            # W^c = sqrt(V)(epsinv - delta_GG')sqrt(V). However, full epsinv
            # is still needed for q0_corrector.
            einvt_GG = (einv_GG - dfc.I_GG) if only_correlation else einv_GG
            W_GG[:] = einvt_GG * (sqrtV_G *
                                  sqrtV_G[:, np.newaxis])
            if self.q0_corrector is not None and chi0.optical_limit:
                W = dfc.wblocks.a + iw
                self.q0_corrector.add_q0_correction(chi0.qpd, W_GG,
                                                    einv_GG,
                                                    chi0.chi0_WxvG[W],
                                                    chi0.chi0_Wvv[W],
                                                    sqrtV_G)
            elif (self.integrate_gamma.is_analytical and chi0.optical_limit) \
                    or self.integrate_gamma.is_numerical:
                self.apply_gamma_correction(W_GG, einvt_GG,
                                            V0, sqrtV0, dfc.sqrtV_G)

        self.context.timer.stop('Dyson eq.')
        return W_wGG

    def dyson_and_W_new(self, iq, q_c, chi0, ecut, coulomb):
        # assert not self.do_GW_too
        assert ecut == chi0.qpd.ecut
        assert self.fxc_mode == 'GW'
        assert not np.allclose(q_c, 0)

        nW = len(self.wd)
        nG = chi0.qpd.ngmax

        from gpaw.response.wgg import Grid

        WGG = (nW, nG, nG)
        WgG_grid = Grid(
            comm=self.blockcomm,
            shape=WGG,
            cpugrid=(1, self.blockcomm.size, 1))
        assert chi0.chi0_wGG.shape == WgG_grid.myshape

        my_gslice = WgG_grid.myslice[1]

        dielectric_WgG = chi0.chi0_wGG  # XXX
        for iw, chi0_GG in enumerate(chi0.chi0_wGG):
            sqrtV_G = coulomb.sqrtV(chi0.qpd, q_v=None)
            e_GG = np.eye(nG) - chi0_GG * sqrtV_G * sqrtV_G[:, np.newaxis]
            e_gG = e_GG[my_gslice]

            dielectric_WgG[iw, :, :] = e_gG

        wgg_grid = Grid(comm=self.blockcomm, shape=WGG)

        dielectric_wgg = wgg_grid.zeros(dtype=complex)
        WgG_grid.redistribute(wgg_grid, dielectric_WgG, dielectric_wgg)

        assert np.allclose(dielectric_wgg, dielectric_WgG)

        wgg_grid.invert_inplace(dielectric_wgg)

        wgg_grid.redistribute(WgG_grid, dielectric_wgg, dielectric_WgG)
        inveps_WgG = dielectric_WgG

        self.context.timer.start('Dyson eq.')

        for iw, inveps_gG in enumerate(inveps_WgG):
            inveps_gG -= np.identity(nG)[my_gslice]
            thing_GG = sqrtV_G * sqrtV_G[:, np.newaxis]
            inveps_gG *= thing_GG[my_gslice]

        W_WgG = inveps_WgG
        Wp_wGG = W_WgG.copy()
        Wm_wGG = W_WgG.copy()
        return chi0.qpd, Wm_wGG, Wp_wGG  # not Hilbert transformed yet


class HWModel:
    """
        Hilbert Transformed W Model.
    """

    def get_HW(self, omega, occ):
        """
            Get Hilbert transformed W at frequency omega.

            occ: The occupation number for the orbital of the Greens function.
        """
        raise NotImplementedError


class FullFrequencyHWModel(HWModel):
    def __init__(self, wd, HW_swGG, factor):
        self.wd = wd
        self.HW_swGG = HW_swGG
        self.factor = factor

    def get_HW(self, omega, occ):
        # For more information about how fsign and wsign works, see
        # https://backend.orbit.dtu.dk/ws/portalfiles/portal/93075765/hueser_PhDthesis.pdf
        # eq. 2.2 endind up to eq. 2.11
        # Effectively, the symmetry of time ordered W is used,
        # i.e. W(w) = -W(-w). To allow that data is only stored for w>=0.
        # Hence, the interpolation happends always to the positive side, but
        # the information of true w is keps tract using wsign.
        # In addition, whether the orbital in question at G is occupied or
        # unoccupied, which then again affects, which Hilbert transform of
        # W is chosen, is kept track with fsign.
        fsign = np.sign(2 * occ - 1)
        o = abs(omega)
        wsign = np.sign(omega + 1e-15)
        wd = self.wd
        # Pick +i*eta or -i*eta:
        s = (1 + wsign * np.sign(-fsign)).astype(int) // 2
        w = wd.get_floor_index(o, safe=False)

        # Interpolation indexes w + 1, therefore - 2 here
        if w > len(wd) - 2:
            return None, None

        o1 = wd.omega_w[w]
        o2 = wd.omega_w[w + 1]

        C1_GG = self.HW_swGG[s][w]
        C2_GG = self.HW_swGG[s][w + 1]
        p = self.factor * wsign

        sigma_GG = ((o - o1) * C2_GG + (o2 - o) * C1_GG) / (o2 - o1)
        dsigma_GG = wsign * (C2_GG - C1_GG) / (o2 - o1)
        return -1j * p * sigma_GG, -1j * p * dsigma_GG


class MPAHWModel(HWModel):
    def __init__(self, W_nGG, omegat_nGG, eta, factor):
        self.W_nGG = np.ascontiguousarray(W_nGG)
        self.omegat_nGG = np.ascontiguousarray(omegat_nGG)
        self.eta = eta
        self.factor = factor

    def get_HW(self, omega, occ):
        x_GG = np.empty(self.omegat_nGG.shape[1:], dtype=complex)
        dx_GG = np.empty(self.omegat_nGG.shape[1:], dtype=complex)
        evaluate_mpa_poly(x_GG, dx_GG, omega, occ, self.omegat_nGG, self.W_nGG,
                          self.eta, self.factor)

        return x_GG.conj(), dx_GG.conj()  # Why do we have to do a conjugate


class MPACalculator(WBaseCalculator):
    def __init__(self, gs, context, *, eta, mpa, **kwargs):
        super().__init__(gs, context, **kwargs)
        self.eta = eta
        self.mpa = mpa

    def get_HW_model(self, chi0,
                     fxc_mode='GW'):
        """Calculate the MPA parametrization of screened interaction.
        """

        dfc = DielectricFunctionCalculator(chi0,
                                           self.coulomb,
                                           self.xckernel,
                                           fxc_mode)

        self.context.timer.start('Dyson eq.')
        einv_wGG = dfc.get_epsinv_wGG(only_correlation=True)
        einv_WgG = chi0.body.blockdist.distribute_as(einv_wGG, chi0.nw, 'WgG')

        solver = RESolver(chi0.wd.omega_w)
        E_pGG, R_pGG = solver.solve(einv_WgG)
        E_pGG -= 1j * self.eta  # DALV: This is just to match the FF results

        R_pGG = chi0.body.blockdist.distribute_as(R_pGG, self.mpa['npoles'],
                                                  'wGG')
        E_pGG = chi0.body.blockdist.distribute_as(E_pGG,
                                                  self.mpa['npoles'], 'wGG')

        if self.integrate_gamma.is_Wigner_Seitz:
            from gpaw.hybrids.wstc import WignerSeitzTruncatedCoulomb
            wstc = WignerSeitzTruncatedCoulomb(chi0.qpd.gd.cell_cv,
                                               dfc.coulomb.N_c)
            sqrtV_G = wstc.get_potential(chi0.qpd)**0.5
        else:
            sqrtV_G = dfc.sqrtV_G

        W_pGG = pi * R_pGG * sqrtV_G[np.newaxis, :, np.newaxis] \
            * sqrtV_G[np.newaxis, np.newaxis, :]

        assert self.q0_corrector is None
        if (self.integrate_gamma.is_analytical and chi0.optical_limit)\
                or self.integrate_gamma.is_numerical:
            V0, sqrtV0 = self.get_V0sqrtV0(chi0)
            for W_GG, R_GG in zip(W_pGG, R_pGG):
                self.apply_gamma_correction(W_GG, pi * R_GG,
                                            V0, sqrtV0,
                                            dfc.sqrtV_G)

        W_pGG = np.transpose(W_pGG, axes=(0, 2, 1))  # Why the transpose
        E_pGG = np.transpose(E_pGG, axes=(0, 2, 1))

        W_pGG = chi0.body.blockdist.distribute_as(W_pGG, self.mpa['npoles'],
                                                  'WgG')
        E_pGG = chi0.body.blockdist.distribute_as(E_pGG,
                                                  self.mpa['npoles'], 'WgG')

        self.context.timer.stop('Dyson eq.')

        factor = 1.0 / (self.qd.nbzkpts * 2 * pi * self.gs.volume)
        return MPAHWModel(W_pGG, E_pGG, self.eta, factor)
