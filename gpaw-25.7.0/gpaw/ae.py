"""Bare Coulomb potential for hydrogen."""

import numpy as np
from scipy.special import erf

from gpaw.basis_data import Basis
from gpaw.setup import BaseSetup
from gpaw.spline import Spline


class HydrogenAllElectronSetup(BaseSetup):

    def __init__(self, alpha1=10.0, alpha2=300.0):
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.natoms = 0
        self.E = 0.0
        self.Z = 1
        self.Nc = 0
        self.Nv = 1
        self.nao = None
        self.pt_j = []
        self.ni = 0
        self.l_j = [0]
        self.l_orb_J = [0]
        self.n_j = [1]
        self.tauct = self.nct = Spline.from_data(0, 0.5, [0.0, 0.0, 0.0])
        self.Nct = 0.0
        self.N0_p = np.zeros(0)
        rc = 2.0
        r_g = np.linspace(0, rc, 100)
        r2_g = r_g**2
        self.ghat_l = [Spline.from_data(0, rc, 4 * alpha1**1.5 / np.pi**0.5 *
                                        np.exp(-alpha1 * r2_g))]
        v_g = erf(alpha1**0.5 * r_g) - erf(alpha2**0.5 * r_g)
        v_g[1:] *= (4 * np.pi)**0.5 / r_g[1:]
        v_g[0] = 4 * (alpha1**0.5 - alpha2**0.5)
        self.vbar = Spline.from_data(0, rc, v_g)
        self.Delta_pL = np.zeros((0, 1))
        self.Delta_iiL = np.zeros((0, 0, 1))
        self.Delta0 = -1 / (4 * np.pi)**0.5
        self.lmax = 0
        self.K_p = self.M_p = self.MB_p = self.X_p = np.zeros(0)
        self.M_pp = np.zeros((0, 0))
        self.Kc = 0.0
        self.MB = 0.0
        self.M = -(alpha1 / 2 / np.pi)**0.5
        self.xc_correction = None
        self.dO_ii = np.zeros((0, 0))
        self.type = 'all-electron'
        self.fingerprint = None
        self.symbol = 'H'
        self.ExxC = 0.0

    def get_default_nbands(self):
        return 1

    def build(self, basis):
        if basis is None:
            basis = Basis.find('H', 'sz(dzp)')
        elif isinstance(basis, str):
            basis = Basis.find('H', basis)

        self.basis = basis
        self.basis_functions_J = self.basis.tosplines()
        self.pseudo_partial_waves_j = None  # XXX

        self.f_j = [1.0]
        self.nao = self.basis.nao

    def print_info(self, text):
        text('Hydrogen all-electron potential')
