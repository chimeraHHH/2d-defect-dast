"""PAW Poisson-solvers.

Adds smooth compensation charges to the pseudo density.
"""
from __future__ import annotations

import numpy as np
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor as RGD
from gpaw.atom.shapefunc import shape_functions
from gpaw.core import PWArray, PWDesc
from gpaw.core.atom_arrays import AtomArrays, AtomDistribution
from gpaw.gpu import cupy as cp
from gpaw.setup import Setups
from gpaw.spline import Spline


class PAWPoissonSolver:
    def __init__(self,
                 poisson_solver):
        self.poisson_solver = poisson_solver

    def dipole_layer_correction(self) -> None:
        return self.poisson_solver.dipole_layer_correction()

    def move(self,
             relpos_ac: np.ndarray,
             atomdist: AtomDistribution) -> None:
        raise NotImplementedError

    def solve(self,
              nt_g: PWArray,
              Q_aL: AtomArrays,
              vt0_g: PWArray,
              vHt_h: PWArray | None = None) -> tuple[float,
                                                     PWArray,
                                                     AtomArrays]:
        raise NotImplementedError


class SlowPAWPoissonSolver(PAWPoissonSolver):
    """Solve Poisson-equation on very fine grid."""
    def __init__(self,
                 pwg: PWDesc,
                 # cutoff_a,
                 setups: Setups,
                 poisson_solver,
                 relpos_ac: np.ndarray,
                 atomdist: AtomDistribution,
                 xp=np):
        super().__init__(poisson_solver)
        self.xp = xp
        self.pwg = pwg
        self.pwg0 = pwg.new(comm=None)  # not distributed
        self.pwh = poisson_solver.pw
        self.ghat_aLh = setups.create_compensation_charges(
            self.pwh, relpos_ac, atomdist, xp)
        self.h_g, self.g_r = self.pwh.map_indices(self.pwg0)
        if xp is cp:
            self.h_g = cp.asarray(self.h_g)
            self.g_r = [cp.asarray(g) for g in self.g_r]

    def move(self,
             relpos_ac: np.ndarray,
             atomdist: AtomDistribution) -> None:
        self.ghat_aLh.move(relpos_ac, atomdist)

    def solve(self,
              nt0_g: PWArray,
              Q_aL: AtomArrays,
              vt0_g: PWArray,
              vHt_h: PWArray | None = None) -> tuple[float,
                                                     PWArray,
                                                     AtomArrays]:
        charge_h = self.pwh.zeros(xp=self.xp)
        self.ghat_aLh.add_to(charge_h, Q_aL)
        pwg = self.pwg

        if pwg.comm.rank == 0:
            for rank, g in enumerate(self.g_r):
                if rank == 0:
                    charge_h.data[self.h_g] += nt0_g.data[g]
                else:
                    pwg.comm.send(nt0_g.data[g], rank)
        else:
            data = self.xp.empty(len(self.h_g), complex)
            pwg.comm.receive(data, 0)
            charge_h.data[self.h_g] += data

        if vHt_h is None:
            vHt_h = self.pwh.zeros(xp=self.xp)

        e_coulomb = self.poisson_solver.solve(vHt_h, charge_h)

        if pwg.comm.rank == 0:
            for rank, g in enumerate(self.g_r):
                if rank == 0:
                    vt0_g.data[g] += vHt_h.data[self.h_g]
                else:
                    data = self.xp.empty(len(g), complex)
                    pwg.comm.receive(data, rank)
                    vt0_g.data[g] += data
        else:
            pwg.comm.send(vHt_h.data[self.h_g], 0)

        V_aL = self.ghat_aLh.integrate(vHt_h)

        return e_coulomb, vHt_h, V_aL

    def force_contribution(self, Q_aL, vHt_h, nt_g):
        force_av = self.xp.zeros((len(Q_aL), 3))

        F_avL = self.ghat_aLh.derivative(vHt_h)
        for a, dF_vL in F_avL.items():
            force_av[a] += dF_vL @ Q_aL[a]

        return force_av

    def stress_contribution(self, vHt_h, Q_aL):
        return self.ghat_aLh.stress_contribution(vHt_h, Q_aL)


class SimplePAWPoissonSolver(PAWPoissonSolver):
    """For testing only!"""
    def __init__(self,
                 pwg: PWDesc,
                 cutoff_a,
                 poisson_solver,
                 relpos_ac: np.ndarray,
                 atomdist: AtomDistribution,
                 xp=np):
        self.xp = xp
        self.pwg = pwg
        self.pwg0 = pwg.new(comm=None)  # not distributed
        self.poisson_solver = poisson_solver
        d = 0.005
        rgd = RGD(d, int(10.0 / d))
        cache: dict[float, list[Spline]] = {}
        ghat_al = []
        for rc in cutoff_a:
            if rc in cache:
                ghat_l = cache[rc]
            else:
                g_lg = shape_functions(rgd, 'gauss', rc, lmax=2)
                ghat_l = [rgd.spline(g_g, l=l) for l, g_g in enumerate(g_lg)]
                cache[rc] = ghat_l
            ghat_al.append(ghat_l)

        self.ghat_aLg = pwg.atom_centered_functions(
            ghat_al, relpos_ac, atomdist=atomdist, xp=xp)

    def solve(self,
              nt0_g: PWArray,
              Q_aL: AtomArrays,
              vt0_g: PWArray,
              vHt_g: PWArray | None = None):

        charge_g = self.pwg.empty(xp=self.xp)
        charge_g.scatter_from(nt0_g)
        self.ghat_aLg.add_to(charge_g, Q_aL)
        pwg = self.pwg
        if vHt_g is None:
            vHt_g = pwg.empty(xp=self.xp)
        e_coulomb = self.poisson_solver.solve(vHt_g, charge_g)
        vHt0_g = vHt_g.gather()
        if pwg.comm.rank == 0:
            vt0_g.data += vHt0_g.data
        V_aL = self.ghat_aLg.integrate(vHt_g)
        return e_coulomb, vHt_g, V_aL

    def force_contribution(self, Q_aL, vHt_g, nt_g):
        force_av = self.xp.zeros((len(Q_aL), 3))

        F_avL = self.ghat_aLg.derivative(vHt_g)
        for a, dF_vL in F_avL.items():
            force_av[a] += dF_vL @ Q_aL[a]
        return force_av
