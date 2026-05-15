import numpy as np
from gpaw.xc import XC
from gpaw.xc.gllb.contribution import Contribution


class C_XC(Contribution):
    def __init__(self, weight, functional):
        Contribution.__init__(self, weight)
        self.xc = XC(functional)

    def get_name(self):
        return 'XC'

    def get_desc(self):
        desc = self.xc.get_description()
        return self.xc.name if desc is None else desc

    def initialize(self, density, hamiltonian, wfs):
        Contribution.initialize(self, density, hamiltonian, wfs)
        self.vt_sg = self.finegd.empty(self.nspins)
        self.e_g = self.finegd.empty()

    def initialize_1d(self, ae):
        Contribution.initialize_1d(self, ae)
        self.v_g = np.zeros(self.ae.N)

    def calculate(self, e_g, n_sg, v_sg):
        self.e_g[:] = 0.0
        self.vt_sg[:] = 0.0
        self.xc.calculate(self.finegd, n_sg, self.vt_sg, self.e_g)
        v_sg += self.weight * self.vt_sg
        e_g += self.weight * self.e_g

    def calculate_energy_and_derivatives(self, setup, D_sp, H_sp, a,
                                         addcoredensity=True):
        E = self.xc.calculate_paw_correction(setup, D_sp, H_sp, True, a)
        E += setup.xc_correction.e_xc0
        return E

    def add_xc_potential_and_energy_1d(self, v_g):
        self.v_g[:] = 0.0
        Exc = self.xc.calculate_spherical(self.ae.rgd,
                                          self.ae.n.reshape((1, -1)),
                                          self.v_g.reshape((1, -1)))
        v_g += self.weight * self.v_g
        return self.weight * Exc

    def add_smooth_xc_potential_and_energy_1d(self, vt_g):
        self.v_g[:] = 0.0
        Exc = self.xc.calculate_spherical(self.ae.rgd,
                                          self.ae.nt.reshape((1, -1)),
                                          self.v_g.reshape((1, -1)))
        vt_g += self.weight * self.v_g
        return self.weight * Exc
