from math import pi
import numpy as np

from gpaw.density import Density
from gpaw.pw.descriptor import PWDescriptor, PWMapping
from gpaw.pw.lfc import PWLFC


class PseudoCoreKineticEnergyDensityLFC(PWLFC):
    def add(self, tauct_R):
        tauct_R += self.pd.ifft(1.0 / self.pd.gd.dv *
                                self.expand().sum(1).view(complex))

    def derivative(self, dedtaut_R, dF_aiv):
        PWLFC.derivative(self, self.pd.fft(dedtaut_R), dF_aiv)


class ReciprocalSpaceDensity(Density):
    def __init__(self, ecut,
                 gd, finegd, nspins, collinear, charge, redistributor,
                 background_charge=None):
        Density.__init__(self, gd, finegd, nspins, collinear, charge,
                         redistributor=redistributor,
                         background_charge=background_charge)
        ecut0 = 0.5 * pi**2 / (gd.h_cv**2).sum(1).max()
        ecut = min(ecut, ecut0)
        self.pd2 = PWDescriptor(ecut, gd)
        self.pd3 = PWDescriptor(4 * ecut, finegd)

        self.map23 = PWMapping(self.pd2, self.pd3)

        self.nct_q = None
        self.nt_Q = None
        self.rhot_q = None

    def initialize(self, setups, timer, magmom_av, hund):
        Density.initialize(self, setups, timer, magmom_av, hund)

        spline_aj = []
        for setup in setups:
            if setup.nct is None:
                spline_aj.append([])
            else:
                spline_aj.append([setup.nct])
        self.nct = PWLFC(spline_aj, self.pd2)

        self.ghat = PWLFC([setup.ghat_l for setup in setups], self.pd3,
                          )  # blocksize=256, comm=self.xc_redistributor.comm)

    def set_positions(self, spos_ac, atom_partition):
        Density.set_positions(self, spos_ac, atom_partition)
        self.nct_q = self.pd2.zeros()
        self.nct.add(self.nct_q, 1.0 / self.nspins)
        self.nct_G = self.pd2.ifft(self.nct_q)

    def interpolate_pseudo_density(self, comp_charge=None):
        """Interpolate pseudo density to fine grid."""
        if comp_charge is None:
            comp_charge, _Q_aL = self.calculate_multipole_moments()

        if self.nt_xg is None:
            self.nt_xg = self.finegd.empty(self.ncomponents)
            self.nt_sg = self.nt_xg[:self.nspins]
            self.nt_vg = self.nt_xg[self.nspins:]
            self.nt_Q = self.pd2.empty()

        self.nt_Q[:] = 0.0

        x = 0
        for nt_G, nt_g in zip(self.nt_xG, self.nt_xg):
            nt_g[:], nt_Q = self.pd2.interpolate(nt_G, self.pd3)
            if x < self.nspins:
                self.nt_Q += nt_Q
            x += 1

    def interpolate(self, in_xR, out_xR=None):
        """Interpolate array(s)."""
        if out_xR is None:
            out_xR = self.finegd.empty(in_xR.shape[:-3])

        a_xR = in_xR.reshape((-1,) + in_xR.shape[-3:])
        b_xR = out_xR.reshape((-1,) + out_xR.shape[-3:])

        for in_R, out_R in zip(a_xR, b_xR):
            out_R[:] = self.pd2.interpolate(in_R, self.pd3)[0]

        return out_xR

    distribute_and_interpolate = interpolate

    def calculate_pseudo_charge(self):
        self.rhot_q = self.pd3.zeros()
        Q_aL = self.Q.calculate(self.D_asp)
        self.ghat.add(self.rhot_q, Q_aL)
        self.map23.add_to2(self.rhot_q, self.nt_Q)
        self.background_charge.add_fourier_space_charge_to(self.pd3,
                                                           self.rhot_q)

    def get_pseudo_core_kinetic_energy_density_lfc(self):
        return PseudoCoreKineticEnergyDensityLFC(
            [[setup.tauct] for setup in self.setups], self.pd2)

    def calculate_dipole_moment(self):
        pd = self.pd3
        N_c = pd.tmp_Q.shape

        m0_q, m1_q, m2_q = (i_G == 0
                            for i_G in np.unravel_index(pd.Q_qG[0], N_c))
        rhot_q = self.pd3.gather(self.rhot_q)
        if pd.comm.rank == 0:
            irhot_q = rhot_q.imag
            rhot_cs = [irhot_q[m1_q & m2_q],
                       irhot_q[m0_q & m2_q],
                       irhot_q[m0_q & m1_q]]
            d_c = [np.dot(rhot_s[1:], 1.0 / np.arange(1, len(rhot_s)))
                   for rhot_s in rhot_cs]
            d_v = -np.dot(d_c, pd.gd.cell_cv) / pi * pd.gd.dv
        else:
            d_v = np.empty(3)
        pd.comm.broadcast(d_v, 0)
        return d_v
