import numpy as np
from gpaw.arraydict import ArrayDict
from gpaw.density import Density
from gpaw.external import NoExternalPotential
from gpaw.hamiltonian import Hamiltonian
from gpaw.pw.lfc import PWLFC
from gpaw.pw.poisson import (ChargedReciprocalSpacePoissonSolver,
                             ReciprocalSpacePoissonSolver)
from gpaw.typing import Array3D


class ReciprocalSpaceHamiltonian(Hamiltonian):
    def __init__(self, gd, finegd, pd2, pd3, nspins, collinear,
                 setups, timer, xc, world, xc_redistributor,
                 vext=None,
                 psolver=None, redistributor=None, realpbc_c=None,
                 charge=0.0):

        assert redistributor is not None  # XXX should not be like this

        if vext is None:
            vext = NoExternalPotential()

        Hamiltonian.__init__(self, gd, finegd, nspins, collinear, setups,
                             timer, xc, world, vext=vext,
                             redistributor=redistributor)

        self.vbar = PWLFC([[setup.vbar] for setup in setups], pd2)
        self.pd2 = pd2
        self.pd3 = pd3
        self.xc_redistributor = xc_redistributor
        self.charge = charge

        self.vHt_q = pd3.empty()

        if psolver is None:
            psolver = {}
        elif not isinstance(psolver, dict):
            psolver = psolver.todict()

        if psolver.get('name') == 'nointeraction':
            self.poisson = ReciprocalSpacePoissonSolver(pd3, charge, 0.0)
        else:
            if charge == 0.0 or realpbc_c.any():
                self.poisson = ReciprocalSpacePoissonSolver(pd3, charge)
            else:
                self.poisson = ChargedReciprocalSpacePoissonSolver(pd3, charge)

            if 'dipolelayer' in psolver:
                direction = psolver['dipolelayer']
                assert len(psolver) == 1
                from gpaw.dipole_correction import DipoleCorrection
                self.poisson = DipoleCorrection(self.poisson, direction)
                self.poisson.check_direction(gd, realpbc_c)
            else:
                assert not psolver

        self.npoisson = 0

        self.vbar_Q = None
        self.vt_Q = None
        self.estress = None

    def __str__(self):
        s = Hamiltonian.__str__(self)
        if self.charge != 0.0:
            s += f'Poisson solver:\n  {self.poisson}\n'
        return s

    @property
    def xc_gd(self):
        if self.xc_redistributor is None:
            return self.finegd
        return self.xc_redistributor.aux_gd

    def set_positions(self, spos_ac, atom_partition):
        Hamiltonian.set_positions(self, spos_ac, atom_partition)
        self.vbar_Q = self.pd2.zeros()
        self.vbar.add(self.vbar_Q)

    def update_pseudo_potential(self, dens):
        ebar = self.pd2.integrate(self.vbar_Q, dens.nt_Q,
                                  global_integral=False)
        with self.timer('Poisson'):
            epot = self.poisson.solve(self.vHt_q, dens)
            epot /= self.finegd.comm.size

        self.vt_Q = self.vbar_Q.copy()

        dens.map23.add_to1(self.vt_Q, self.vHt_q)

        # vt_sG[:] = pd2.ifft(vt_Q)
        eext = self.vext.update_potential_pw(self, dens)

        self.timer.start('XC 3D grid')

        nt_xg = dens.nt_xg

        # If we have a redistributor, we want to do the
        # good old distribute-calculate-collect:
        redist = self.xc_redistributor
        if redist is not None:
            nt_xg = redist.distribute(nt_xg)

        vxct_xg = np.zeros_like(nt_xg)
        exc = self.xc.calculate(self.xc_gd, nt_xg, vxct_xg)
        exc /= self.finegd.comm.size
        if redist is not None:
            vxct_xg = redist.collect(vxct_xg)

        for x, (vt_G, vxct_g) in enumerate(zip(self.vt_xG, vxct_xg)):
            vxc_G, vxc_Q = self.pd3.restrict(vxct_g, self.pd2)
            if x < self.nspins:
                vt_G += vxc_G
                self.vt_Q += vxc_Q / self.nspins
            else:
                vt_G += vxc_G

        self.timer.stop('XC 3D grid')

        energies = np.array([epot, ebar, eext, exc])
        self.estress = self.gd.comm.sum_scalar(epot + ebar)
        return energies

    def calculate_atomic_hamiltonians(self, density):
        def getshape(a):
            return sum(2 * l + 1
                       for l, _ in enumerate(self.setups[a].ghat_l)),
        W_aL = ArrayDict(self.atomdist.aux_partition, getshape, float)

        self.vext.update_atomic_hamiltonians_pw(self, W_aL, density)
        return self.atomdist.to_work(self.atomdist.from_aux(W_aL))

    def calculate_kinetic_energy(self, density):
        ekin = 0.0
        for vt_G, nt_G in zip(self.vt_xG, density.nt_xG):
            ekin -= self.gd.integrate(vt_G, nt_G, global_integral=False)
        ekin += self.gd.integrate(self.vt_sG, density.nct_G,
                                  global_integral=False).sum()
        return ekin

    def restrict(self, in_xR, out_xR=None):
        """Restrict array."""
        if out_xR is None:
            out_xR = self.gd.empty(in_xR.shape[:-3])

        a_xR = in_xR.reshape((-1,) + in_xR.shape[-3:])
        b_xR = out_xR.reshape((-1,) + out_xR.shape[-3:])

        for in_R, out_R in zip(a_xR, b_xR):
            out_R[:] = self.pd3.restrict(in_R, self.pd2)[0]

        return out_xR

    restrict_and_collect = restrict

    def calculate_forces2(self, dens, ghat_aLv, nct_av, vbar_av):
        self.vext.derivative_pw(self, ghat_aLv, dens)
        dens.nct.derivative(self.vt_Q, nct_av)
        self.vbar.derivative(dens.nt_Q, vbar_av)

    def get_electrostatic_potential(self, dens: Density) -> Array3D:
        self.poisson.solve(self.vHt_q, dens)
        vHt_R = self.pd3.ifft(self.vHt_q, distribute=False)
        self.pd3.comm.broadcast(vHt_R, 0)
        return vHt_R
