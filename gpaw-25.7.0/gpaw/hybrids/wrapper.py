from typing import Tuple, Dict

import numpy as np

from gpaw.xc import XC
from .coulomb import coulomb_interaction
from .forces import calculate_forces
from .paw import calculate_paw_stuff
from .scf import apply1, apply2
from .symmetry import Symmetry


class HybridXC:
    orbital_dependent = True
    type = 'HYB'

    def __init__(self,
                 xcname: str,
                 fraction: float = None,
                 omega: float = None,
                 yukawa=None):
        from . import parse_name
        if xcname in ['EXX', 'PBE0', 'HSE03', 'HSE06', 'B3LYP', 'YS-PBE0']:
            if fraction is not None or omega is not None or yukawa is not None:
                raise ValueError
            self.name = xcname
            xcname, fraction, omega, yukawa = parse_name(xcname)
        else:
            if fraction is None or omega is None:
                raise ValueError
            self.name = f'{xcname}-{fraction:.3f}-{omega:.3f}'

        self.xc = XC(xcname)
        self.exx_fraction = fraction
        self.omega = omega
        self.yukawa = yukawa

        if xcname == 'null':
            self.description = ''
        else:
            self.description = f'{xcname} + '
        self.description += f'{fraction} * EXX(omega = {omega} bohr^-1)'

        self.vlda_sR = None
        self.v_sknG: Dict[Tuple[int, int], np.ndarray] = {}

        self.ecc = np.nan
        self.evc = np.nan
        self.evv = np.nan
        self.ekin = np.nan

        self.sym = None
        self.coulomb = None

    def get_setup_name(self):
        return 'PBE'

    def initialize(self, dens, ham, wfs):
        self.dens = dens
        self.wfs = wfs
        self.ecc = sum(setup.ExxC for setup in wfs.setups) * self.exx_fraction
        assert wfs.world.size == wfs.gd.comm.size

    def get_description(self):
        return self.description

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac

    def calculate(self, gd, nt_sr, vt_sr):
        energy = self.ecc + self.evv + self.evc
        energy += self.xc.calculate(gd, nt_sr, vt_sr)
        return energy

    def calculate_paw_correction(self, setup, D_sp, dH_sp=None, a=None):
        return self.xc.calculate_paw_correction(setup, D_sp, dH_sp, a=a)

    def get_kinetic_energy_correction(self):
        return self.ekin

    def apply_orbital_dependent_hamiltonian(self, kpt, psit_xG,
                                            Htpsit_xG=None, dH_asp=None):
        wfs = self.wfs
        if self.coulomb is None:
            self.coulomb = coulomb_interaction(
                self.omega, wfs.gd, wfs.kd, yukawa=self.yukawa)
            self.description += '\n' + self.coulomb.get_description()
            self.sym = Symmetry(wfs.kd)

        paw_s = calculate_paw_stuff(wfs, self.dens)  # ???????

        if kpt.f_n is None:
            # Just use LDA_X for first step:
            if self.vlda_sR is None:
                # First time:
                self.vlda_sR = self.calculate_lda_potential()
            pd = kpt.psit.pd
            for psit_G, Htpsit_G in zip(psit_xG, Htpsit_xG):
                Htpsit_G += pd.fft(self.vlda_sR[kpt.s] *
                                   pd.ifft(psit_G, kpt.k), kpt.q)
        else:
            self.vlda_sR = None
            if kpt.psit.array.base is psit_xG.base:
                if (kpt.s, kpt.k) not in self.v_sknG:
                    assert not any(s == kpt.s for s, k in self.v_sknG)
                    evc, evv, ekin, v_knG = apply1(
                        kpt, Htpsit_xG,
                        wfs,
                        self.coulomb, self.sym,
                        paw_s[kpt.s])
                    if kpt.s == 0:
                        self.evc = 0.0
                        self.evv = 0.0
                        self.ekin = 0.0
                    scale = 2 / wfs.nspins * self.exx_fraction
                    self.evc += evc * scale
                    self.evv += evv * scale
                    self.ekin += ekin * scale
                    self.v_sknG = {(kpt.s, k): v_nG
                                   for k, v_nG in v_knG.items()}
                v_nG = self.v_sknG.pop((kpt.s, kpt.k))
            else:
                v_nG = apply2(kpt, psit_xG, Htpsit_xG, wfs,
                              self.coulomb, self.sym,
                              paw_s[kpt.s])
            Htpsit_xG += v_nG * self.exx_fraction

    def calculate_lda_potential(self):
        from gpaw.xc import XC
        lda = XC('LDA_X')
        nt_sr = self.dens.nt_sg
        vt_sr = np.zeros_like(nt_sr)
        vlda_sR = self.dens.gd.zeros(self.wfs.nspins)
        lda.calculate(self.dens.finegd, nt_sr, vt_sr)
        for vt_R, vt_r in zip(vlda_sR, vt_sr):
            vt_R[:], _ = self.dens.pd3.restrict(vt_r, self.dens.pd2)
        return vlda_sR * self.exx_fraction

    def summary(self, log):
        log(self.description)

    def add_forces(self, F_av):
        paw_s = calculate_paw_stuff(self.wfs, self.dens)
        F_av += calculate_forces(self.wfs,
                                 self.coulomb,
                                 self.sym,
                                 paw_s) * self.exx_fraction

    def correct_hamiltonian_matrix(self, kpt, H_nn):
        return

    def rotate(self, kpt, U_nn):
        pass  # 1 / 0

    def add_correction(self, kpt, psit_xG, Htpsit_xG, P_axi, c_axi, n_x,
                       calculate_change=False):
        pass  # 1 / 0

    def read(self, reader):
        pass

    def write(self, writer):
        pass

    def set_grid_descriptor(self, gd):
        pass
