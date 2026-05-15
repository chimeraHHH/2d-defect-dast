import numpy as np
from functools import cached_property
from gpaw.response.pw_parallelization import Blocks1D
from gpaw.response.gamma_int import GammaIntegral


class DielectricFunctionCalculator:
    def __init__(self, chi0, coulomb, xckernel, mode):
        self.coulomb = coulomb
        self.qpd = chi0.qpd
        self.mode = mode
        self.optical_limit = chi0.optical_limit
        self.chi0 = chi0
        self.xckernel = xckernel
        self.wblocks = Blocks1D(chi0.body.blockdist.blockcomm, len(chi0.wd))
        # Generate fine grid in vicinity of gamma
        if chi0.optical_limit and self.wblocks.nlocal:
            self.gamma_int = GammaIntegral(self.coulomb, self.qpd)
        else:
            self.gamma_int = None

    @cached_property
    def sqrtV_G(self):
        return self.coulomb.sqrtV(qpd=self.qpd, q_v=None)

    @cached_property
    def I_GG(self):
        return np.eye(self.qpd.ngmax)

    @cached_property
    def fxc_GG(self):
        if self.mode == 'GW':
            return self.I_GG
        else:
            return self.xckernel.calculate(self.qpd)

    @cached_property
    def chi0_wGG(self):
        return self.chi0.body.copy_array_with_distribution('wGG')

    def get_epsinv_wGG(self, only_correlation=True):
        """
        Calculates inverse dielectric matrix for all frequencies.
        """
        epsinv_wGG = []
        for w in range(self.wblocks.nlocal):
            epsinv_GG = self.single_frequency_epsinv_GG(w)
            if only_correlation:
                epsinv_GG -= self.I_GG
            epsinv_wGG.append(epsinv_GG)
        return np.asarray(epsinv_wGG)

    def single_frequency_epsinv_GG(self, w):
        """
        Calculates inverse dielectric matrix for single frequency
        """
        _dfc = _DielectricFunctionCalculator(self.sqrtV_G,
                                             self.chi0_wGG[w],
                                             self.mode,
                                             self.fxc_GG)
        if self.optical_limit:
            W = self.wblocks.a + w
            _dfc = _GammaDielectricFunctionCalculator(
                _dfc, self.gamma_int,
                self.chi0.chi0_Wvv[W], self.chi0.chi0_WxvG[W])
        return _dfc.get_epsinv_GG()


class _DielectricFunctionCalculator:
    def __init__(self, sqrtV_G, chi0_GG, mode, fxc_GG=None):
        self.sqrtV_G = sqrtV_G
        self.chiVV_GG = chi0_GG * sqrtV_G * sqrtV_G[:, np.newaxis]

        self.I_GG = np.eye(len(sqrtV_G))

        self.fxc_GG = fxc_GG
        self.chi0_GG = chi0_GG
        self.mode = mode

    def new_with(self, *, sqrtV_G, chi0_GG):
        return _DielectricFunctionCalculator(
            sqrtV_G, chi0_GG, self.mode, fxc_GG=self.fxc_GG)

    def _chiVVfxc_GG(self):
        assert self.mode != 'GW'
        assert self.fxc_GG is not None
        return self.chiVV_GG @ self.fxc_GG

    def eps_GG_gwp(self):
        gwp_inv_GG = np.linalg.inv(self.I_GG - self._chiVVfxc_GG() +
                                   self.chiVV_GG)
        return self.I_GG - gwp_inv_GG @ self.chiVV_GG

    def eps_GG_gws(self):
        # Note how the signs are different wrt. gwp.
        # Nobody knows why.
        gws_inv_GG = np.linalg.inv(self.I_GG + self._chiVVfxc_GG() -
                                   self.chiVV_GG)
        return gws_inv_GG @ (self.I_GG - self.chiVV_GG)

    def eps_GG_plain(self):
        return self.I_GG - self.chiVV_GG

    def eps_GG_w_fxc(self):
        return self.I_GG - self._chiVVfxc_GG()

    def get_eps_GG(self):
        mode = self.mode
        if mode == 'GWP':
            return self.eps_GG_gwp()
        elif mode == 'GWS':
            return self.eps_GG_gws()
        elif mode == 'GW':
            return self.eps_GG_plain()
        elif mode == 'GWG':
            return self.eps_GG_w_fxc()
        raise ValueError(f'Unknown mode: {mode}')

    def get_epsinv_GG(self):
        eps_GG = self.get_eps_GG()
        return np.linalg.inv(eps_GG)


class _GammaDielectricFunctionCalculator:

    def __init__(self, _dfc, gamma_int, chi0_vv, chi0_xvG):
        self._dfc = _dfc
        self.gamma_int = gamma_int

        self.chi0_vv = chi0_vv
        self.chi0_xvG = chi0_xvG

    @property
    def chi0_GG(self):
        return self._dfc.chi0_GG

    def get_epsinv_GG(self):
        # Get average epsinv over small region around Gamma
        epsinv_GG = np.zeros(self.chi0_GG.shape, complex)
        for qweight, sqrtV_G, chi0_mapping in self.gamma_int:
            chi0p_GG = chi0_mapping(self.chi0_GG, self.chi0_vv, self.chi0_xvG)
            _dfc = self._dfc.new_with(sqrtV_G=sqrtV_G, chi0_GG=chi0p_GG)
            epsinv_GG += qweight * _dfc.get_epsinv_GG()
        return epsinv_GG
