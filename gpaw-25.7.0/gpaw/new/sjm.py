from __future__ import annotations

import numpy as np
from ase.units import Bohr
from gpaw.core import UGArray, PWDesc, PWArray
from gpaw.jellium import create_background_charge
from gpaw.new.environment import Environment, FixedPotentialJellium, Jellium
from gpaw.new.poisson import PoissonSolverWrapper
from gpaw.new.pw.poisson import PWPoissonSolver
from gpaw.new.solvation import SolvationEnvironment, Solvation


class SJM(Solvation):
    name = 'sjm'

    def __init__(self,
                 *,
                 cavity,
                 dielectric,
                 interactions,
                 jelliumregion: dict | None = None,
                 target_potential: float | None,  # eV
                 excess_electrons: float = 0.0,
                 tol: float = 0.001):  # eV
        super().__init__(cavity, dielectric, interactions)
        self.jelliumregion = jelliumregion or {}
        self.target_potential = target_potential
        self.excess_electrons = excess_electrons
        self.tol = tol

    def build(self,
              setups,
              grid,
              relpos_ac,
              log,
              comm) -> SJMEnvironment:
        solvation = super().build(
            setups=setups, grid=grid, relpos_ac=relpos_ac,
            log=log, comm=comm)
        h = grid.cell_cv[2, 2] * Bohr
        z1 = relpos_ac[:, 2].max() * h + 3.0
        z2 = self.jelliumregion.get('top', h - 1.0)
        background = create_background_charge(charge=self.excess_electrons,
                                              z1=z1,
                                              z2=z2)
        background.set_grid_descriptor(grid._gd)
        if self.target_potential is None:
            jellium = Jellium(background,
                              natoms=len(relpos_ac),
                              grid=grid)
        else:
            jellium = FixedPotentialJellium(
                background,
                natoms=len(relpos_ac),
                grid=grid,
                workfunction=self.target_potential,
                tolerance=self.tol)
        return SJMEnvironment(solvation, jellium)

    def todict(self):
        dct = super().todict()
        dct.update(
            jelliumregion=self.jelliumregion,
            target_potential=self.target_potential,
            excess_electrons=self.excess_electrons,
            tol=self.tol)
        return dct


class SJMEnvironment(Environment):
    def __init__(self,
                 solvation: SolvationEnvironment,
                 jellium: Jellium):
        super().__init__(solvation.natoms)
        self.solvation = solvation
        self.jellium = jellium
        self.charge = jellium.charge
        self.dielectric = solvation.dielectric

    def create_poisson_solver(self, **kwargs):
        ps = self.solvation.create_poisson_solver(**kwargs).solver
        return SJMPoissonSolver(ps, self.solvation.dielectric)

    def post_scf_convergence(self,
                             ibzwfs,
                             nelectrons,
                             occ_calc,
                             mixer,
                             log) -> bool:
        converged = self.jellium.post_scf_convergence(
            ibzwfs, nelectrons, occ_calc, mixer, log)
        self.charge = self.jellium.charge
        return converged

    def update1(self, nt_r):
        self.solvation.update1(nt_r)
        self.jellium.update1(nt_r)

    def update1pw(self, nt_g):
        nt_r = self.jellium.grid.empty()
        nt_r.scatter_from(nt_g.ifft(grid=self.jellium.grid.new(comm=None))
                          if nt_g is not None else None)
        self.solvation.update1(nt_r)
        self.jellium.update1pw(nt_g)

    def update2(self, nt_r, vHt_r, vt_sr) -> float:
        return self.solvation.update2(nt_r, vHt_r, vt_sr)


class SJMPoissonSolver(PoissonSolverWrapper):
    def __init__(self, solver, dielectric):
        super().__init__(solver)

    def solve(self,
              vHt_r,
              rhot_r) -> float:
        self.solver.solve(vHt_r.data, rhot_r.data)
        eps_r = vHt_r.desc.from_data(self.solver.dielectric.eps_gradeps[0])
        eps0_r = eps_r.gather()
        vHt0_r = vHt_r.gather()
        if eps0_r is not None:
            saw_tooth_z = modified_saw_tooth(eps0_r)
            s1, s2 = saw_tooth_z[[2, 10]]
            v1, v2 = vHt0_r.data[:, :, [2, 10]].mean(axis=(0, 1))
            vHt0_r.data -= (v2 - v1) / (s2 - s1) * saw_tooth_z[np.newaxis,
                                                               np.newaxis]
            vHt0_r.data -= vHt0_r.data[:, :, -1].mean()
        vHt_r.scatter_from(vHt0_r)
        return np.nan


def modified_saw_tooth(eps_r: UGArray) -> np.ndarray:
    a_z = 1.0 / eps_r.data.mean(axis=(0, 1))
    saw_tooth_z = np.add.accumulate(a_z)
    saw_tooth_z -= 0.5 * a_z  # +0.5 from z=0.0 ???
    return saw_tooth_z


class SJMPWPoissonSolver(PWPoissonSolver):
    def __init__(self, pw, dielectric):
        super().__init__(pw)
        self.dielectric = dielectric
        self.saw_tooth_g = saw_tooth(pw, 0.25)

    def solve(self, vHt_g, rhot_g):
        energy = super().solve(vHt_g, rhot_g)
        dipole = rhot_g.moment()[2]
        slope = 4 * np.pi * dipole / rhot_g.desc.volume
        vHt_g.data += slope * self.saw_tooth_g.data
        # Shift potential so that it is zero above the slab:
        shift = 0.5 * slope * rhot_g.desc.cell_cv[2, 2]
        v0 = vHt_g.boundary_value(2)
        vHt_g.data[0] -= shift + v0
        return energy


def saw_tooth_sympy():
    """Fourier-transform."""
    from sympy import Symbol, integrate, sin, var
    z = var('z')
    G = Symbol('G', positive=True)
    b = Symbol('b', positive=True)
    m = integrate(sin(G * z) * z, (z, 0, b))
    print(m)  # -b*cos(G*b)/G + sin(G*b)/G**2


def saw_tooth(pw: PWDesc, width: float = 0.5) -> PWArray:
    """Saw-tooth in reciprocal space with a slope of 1."""
    assert np.allclose(pw.cell_cv[:2, 2], 0.0)
    assert np.allclose(pw.cell_cv[2, :2], 0.0)

    m0_g, m1_g = pw.indices_cG[:2, pw.ng1:pw.ng2] == 0
    mask_g = m0_g & m1_g
    Gz_i = pw.G_plus_k_Gv[mask_g, 2]
    if pw.comm.rank == 0.0:
        assert Gz_i[0] == 0.0
        Gz_i[0] = 1.0
    L = pw.cell_cv[2, 2]
    b = L / 2
    st_i = -(np.sin(b * Gz_i) / Gz_i -
             b * np.cos(b * Gz_i)) / Gz_i * (2j / L)
    if pw.comm.rank == 0.0:
        st_i[0] = 0.0

    # Make the saw-tooth more smooth (fold with Gaussian):
    alpha = width**-2
    st_i *= np.exp(-Gz_i**2 / (4 * alpha))

    # Shift by half the cell height:
    st_i *= np.exp(1j * Gz_i * b)

    st_g = pw.zeros()
    st_g.data[mask_g] = st_i
    return st_g
