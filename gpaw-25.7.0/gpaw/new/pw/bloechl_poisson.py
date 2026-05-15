"""Fast PAW Poisson-solver.

See equations (25-28) in
P. E. Blöchl: https://sci-hub.st/10.1103/PhysRevB.50.17953
"""
from __future__ import annotations

from math import pi

import numpy as np
from ase.neighborlist import primitive_neighbor_list
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor as RGD
from gpaw.atom.shapefunc import shape_functions
from gpaw.core import PWArray, PWDesc
from gpaw.core.atom_arrays import AtomArrays, AtomDistribution
from gpaw.lcao.overlap import (FourierTransformer, LazySphericalHarmonics,
                               LazySphericalHarmonicsDerivative,
                               ManySiteOverlapCalculator,
                               TwoSiteOverlapCalculator)
from gpaw.spline import Spline
from scipy.special import erf
from gpaw.new.pw.paw_poisson import PAWPoissonSolver


def vg(r_r: np.ndarray, rc: float) -> np.ndarray:
    v_lr = np.empty((3, len(r_r)))
    x_r = r_r / rc
    v_lr[0] = 4 * pi * erf(x_r)
    v_lr[0, 0] = 8 * pi**0.5 / rc
    v_lr[0, 1:] /= r_r[1:]

    v_lr[1] = v_lr[0] / 3 - 8 * pi**0.5 / 3 * np.exp(-x_r**2) / rc
    v_lr[1, 0] = 16 * pi**0.5 / 9 / rc**3
    v_lr[1, 1:] /= r_r[1:]**2

    v_lr[2] = (v_lr[0] / 5 -
               8 * pi**0.5 / 5 * (1 + 2 * x_r**2 / 3) * np.exp(-x_r**2) / rc)
    v_lr[2, 0] = 32 * pi**0.5 / 75 / rc**5
    v_lr[2, 1:] /= r_r[1:]**4

    return v_lr


def tci(rcut, I_a, gtilde_Il, vhat_Il, ghat_Il):
    transformer = FourierTransformer(rcut=rcut, N=2**10)
    tsoc = TwoSiteOverlapCalculator(transformer)

    msoc = ManySiteOverlapCalculator(tsoc, I_a, I_a)
    gtilde_Ilq = msoc.transform(gtilde_Il)
    vhat_Ilq = msoc.transform(vhat_Il)
    ghat_Ilq = msoc.transform(ghat_Il)
    l_Il = [[gtilde.l for gtilde in gtilde_l] for gtilde_l in gtilde_Il]
    expansions1 = msoc.calculate_expansions(l_Il, gtilde_Ilq,
                                            l_Il, vhat_Ilq)
    expansions2 = msoc.calculate_expansions(l_Il, vhat_Ilq,
                                            l_Il, ghat_Ilq)
    return expansions1, expansions2


class BloechlPAWPoissonSolver(PAWPoissonSolver):
    def __init__(self,
                 pwg: PWDesc,
                 cutoff_a: np.ndarray,
                 poisson_solver,
                 relpos_ac: np.ndarray,
                 atomdist: AtomDistribution,
                 xp=np,
                 test=0):
        super().__init__(poisson_solver)
        self.xp = xp
        self.pwg = pwg
        self.pwg0 = pwg.new(comm=None)  # not distributed
        self.relpos_ac = relpos_ac
        self.cutoff_a = np.asarray(cutoff_a)
        self.r2 = self.cutoff_a.max() * 2.0
        self.rcut = 7 * self.r2
        d = 0.0051
        rgd = RGD(d, int(self.rcut / d))
        g0_lg = shape_functions(rgd, 'gauss', self.r2, lmax=2)
        P = 25
        if test:
            ghat_al = []
            for r1 in cutoff_a:
                g_lg = shape_functions(rgd, 'gauss', r1, lmax=2)
                ghat_al.append(
                    [rgd.spline(g_g, l=l, points=P)
                     for l, g_g in enumerate(g_lg)])
        else:
            ghat_l = [rgd.spline(g_g, l=l, points=P)
                      for l, g_g in enumerate(g0_lg)]
            ghat_al = [ghat_l] * len(self.cutoff_a)
        cache: dict[float, tuple[int, list[Spline], list[Spline]]] = {}
        gtilde_Il = []
        ghat_Il = []
        vhat_Il = []
        vhat_al = []
        self.I_a = []
        for r1 in cutoff_a:
            if r1 in cache:
                I, gtilde_l, vhat_l = cache[r1]
            else:
                g_lg = shape_functions(rgd, 'gauss', r1, lmax=2)
                gtilde_l = [rgd.spline(g_g, l=l, points=P)
                            for l, g_g in enumerate(g_lg)]
                v_lg = vg(rgd.r_g, r1) - vg(rgd.r_g, self.r2)
                if test:
                    vhat_l = [rgd.spline(v_g * 0, l=l, points=P)
                              for l, v_g in enumerate(v_lg)]
                else:
                    vhat_l = [rgd.spline(v_g, l=l, points=P)
                              for l, v_g in enumerate(v_lg)]
                I = len(cache)
                cache[r1] = I, gtilde_l, vhat_l
                gtilde_Il.append(gtilde_l)
                vhat_Il.append(vhat_l)
                ghat_Il.append(ghat_l)
            self.I_a.append(I)
            vhat_al.append(vhat_l)

        self.ghat_aLg = pwg.atom_centered_functions(
            ghat_al, relpos_ac, atomdist=atomdist, xp=xp)
        self.vhat_aLg = pwg.atom_centered_functions(
            vhat_al, relpos_ac, atomdist=atomdist, xp=xp)

        self.expansions = tci(self.rcut, self.I_a, gtilde_Il, vhat_Il, ghat_Il)

        self._neighbors = None
        self._force_av: np.ndarray | None = None
        self._stress_vv: np.ndarray | None = None

    def get_neighbors(self):
        if self._neighbors is None:
            pw = self.pwg
            i, j, d, D = primitive_neighbor_list(
                'ijdD', pw.pbc, pw.cell, self.relpos_ac,
                2 * self.rcut,
                use_scaled_positions=True,
                self_interaction=True)
            comm = self.pwg.comm
            x = slice(comm.rank, None, comm.size)
            self._neighbors = i[x], j[x], d[x], D[x]
        return self._neighbors

    def dipole_layer_correction(self):
        return self.poisson_solver.dipole_layer_correction()

    def move(self, relpos_ac, atomdist):
        self.relpos_ac = relpos_ac
        self.ghat_aLg.move(relpos_ac, atomdist)
        self.vhat_aLg.move(relpos_ac, atomdist)
        self._neighbors = None
        self._force_av = None
        self._stress_vv = None

    def solve(self,
              nt0_g: PWArray,
              Q_aL: AtomArrays,
              vt0_g: PWArray,
              vHt_g: PWArray | None = None):
        nt_g = self.pwg.empty(xp=self.xp)
        nt_g.scatter_from(nt0_g)
        charge_g = nt_g.copy()
        self.ghat_aLg.add_to(charge_g, Q_aL)
        pwg = self.pwg
        comm = pwg.comm

        if vHt_g is None:
            vHt_g = pwg.empty(xp=self.xp)

        e_coulomb1 = self.poisson_solver.solve(vHt_g, charge_g)

        vhat_g = pwg.empty(xp=self.xp)  # MYPY
        vhat_g.data[:] = 0.0  # MYPY

        self.vhat_aLg.add_to(vhat_g, Q_aL)
        vhat0_g = vhat_g.gather()
        if comm.rank == 0:
            vt0_g.data += vhat0_g.data
            e_coulomb2 = vhat0_g.integrate(nt0_g)
        else:
            e_coulomb2 = 0.0

        if comm.rank == 0:
            vHt_g.data[0] = -vhat_g.data[0]
        V_aL = self.ghat_aLg.integrate(vHt_g)
        self.vhat_aLg.integrate(nt_g, V_aL, add_to=True)

        Q_all_aL = Q_aL.to_cpu().gather(broadcast=True)
        dV_all_aL = Q_all_aL.new()
        dV_all_aL.data[:] = 0.0

        e_coulomb3 = 0.0
        for a1, a2, d, d_v in zip(*self.get_neighbors()):
            rlY_lm = LazySphericalHarmonics(d_v)
            ex1, ex2 = self.expansions
            I1 = self.I_a[a1]
            I2 = self.I_a[a2]
            v_LL = (ex1.tsoe_II[I1, I2].evaluate(d, rlY_lm) +
                    ex2.tsoe_II[I1, I2].evaluate(d, rlY_lm))
            vQ2_L = v_LL @ Q_all_aL[a2]
            dV_all_aL[a1] += vQ2_L / 2
            dV_all_aL[a2] += Q_all_aL[a1] @ v_LL / 2
            e_coulomb3 -= float(Q_all_aL[a1] @ vQ2_L)
        e_coulomb3 *= -0.5

        comm.sum(dV_all_aL.data)
        dV_aL = Q_aL.new(xp=np)
        dV_aL.scatter_from(dV_all_aL)
        V_aL.data += self.xp.asarray(dV_aL.data)

        vHt0_g = vHt_g.gather()
        if comm.rank == 0:
            vt0_g.data += vHt0_g.data

        e_coulomb = comm.sum_scalar(e_coulomb1 / comm.size +
                                    e_coulomb2 +
                                    e_coulomb3)
        return e_coulomb, vHt_g, V_aL

    def force_contribution(self, Q_aL, vHt_g, nt_g):
        force_av = self.xp.zeros((len(Q_aL), 3))

        F_avL = self.ghat_aLg.derivative(vHt_g)
        Fhat_avL = self.vhat_aLg.derivative(nt_g)
        for a, dF_vL in F_avL.items():
            force_av[a] += (dF_vL + Fhat_avL[a]) @ Q_aL[a]
        pair_pot_force_av, _ = self._force_and_stress(Q_aL)
        return force_av + pair_pot_force_av

    def _force_and_stress(self,
                          Q_aL) -> tuple[np.ndarray, np.ndarray]:
        if self._force_av is not None and self._stress_vv is not None:
            return self._force_av, self._stress_vv
        xp = self.xp
        force_av = xp.zeros((len(Q_aL), 3))
        stress_vv = xp.zeros((3, 3))
        Q_aL = Q_aL.gather(broadcast=True)
        for a1, a2, d, d_v in zip(*self.get_neighbors()):
            if d == 0.0:
                continue
            rlY_lm = LazySphericalHarmonics(d_v)
            drlYdR_lmv = LazySphericalHarmonicsDerivative(d_v)
            ex1, ex2 = self.expansions
            I1 = self.I_a[a1]
            I2 = self.I_a[a2]
            n_v = d_v / d
            v_vLL = xp.asarray(
                ex1.tsoe_II[I1, I2].derivative(d, n_v, rlY_lm, drlYdR_lmv) +
                ex2.tsoe_II[I1, I2].derivative(d, n_v, rlY_lm, drlYdR_lmv))
            f_v = (v_vLL @ Q_aL[a2]) @ Q_aL[a1] / 2
            force_av[a1] += f_v
            force_av[a2] -= f_v
            stress_vv += xp.outer(xp.asarray(d_v), f_v)
        self._force_av = force_av
        self._stress_vv = stress_vv
        return force_av, stress_vv

    def stress_contribution(self, vHt_g, Q_aL):
        _, pair_pot_stress_vv = self._force_and_stress(Q_aL)
        return self.ghat_aLg.stress_contribution(vHt_g, Q_aL)
