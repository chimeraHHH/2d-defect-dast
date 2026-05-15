from __future__ import annotations
import numpy as np
from gpaw.new.poisson import PoissonSolver
from gpaw.core import UGArray, UGDesc, PWArray
from ase.units import Ha
from gpaw.new.ibzwfs import IBZWaveFunctions


class Environment:
    """Environment object.

    Used for jellium, solvation, solvated jellium model, ...
    """
    def __init__(self, natoms: int):
        self.natoms = natoms
        self.charge = 0.0

    def create_poisson_solver(self, *, grid, xp, solver) -> PoissonSolver:
        return solver.build(grid=grid, xp=xp)

    def post_scf_convergence(self,
                             ibzwfs: IBZWaveFunctions,
                             nelectrons: float,
                             occ_calc,
                             mixer,
                             log) -> bool:
        """Allow for environment to "converge"."""
        return True

    def update1(self, nt_r) -> None:
        """Hook called right before solving the Poisson equation."""
        pass

    def update1pw(self, nt_g) -> None:
        """PW-mode hook called right before solving the Poisson equation."""
        pass

    def update2(self, nt_r, vHt_r, vt_sr) -> float:
        """Calculate environment energy."""
        return 0.0

    def forces(self, nt_r, vHt_r):
        return np.zeros((self.natoms, 3))


class Jellium(Environment):
    def __init__(self,
                 jellium,
                 natoms: int,
                 grid: UGDesc):
        super().__init__(natoms)
        self.grid = grid
        self.charge = jellium.charge
        self.mask_r = grid.from_data(jellium.mask_g / jellium.volume)
        self.mask_g: PWArray | str = 'undefined'

    def update1(self, nt_r: UGArray) -> None:
        nt_r.data -= self.mask_r.data * self.charge

    def update1pw(self, nt_g: PWArray | None) -> None:
        if self.mask_g == 'undefined':
            mask_r = self.mask_r.gather()
            if nt_g is not None:
                self.mask_g = mask_r.fft(pw=nt_g.desc)
            else:
                self.mask_g = 'ready'
        if nt_g is None:
            return
        assert not isinstance(self.mask_g, str)
        nt_g.data -= self.mask_g.data * self.charge


class FixedPotentialJellium(Jellium):
    def __init__(self,
                 jellium,
                 natoms: int,
                 grid: UGDesc,
                 workfunction: float,  # eV
                 tolerance: float = 0.001):  # eV
        """Adjust jellium charge to get the desired Fermi-level."""
        super().__init__(jellium, natoms, grid)
        self.workfunction = workfunction / Ha
        self.tolerance = tolerance / Ha
        # Charge, Fermi-level history:
        self.history: list[tuple[float, float]] = []

    def post_scf_convergence(self,
                             ibzwfs: IBZWaveFunctions,
                             nelectrons: float,
                             occ_calc,
                             mixer,
                             log) -> bool:
        fl1 = ibzwfs.fermi_level
        log(f'charge: {self.charge:.6f} |e|, Fermi-level: {fl1 * Ha:.3f} eV')
        fl = -self.workfunction
        if abs(fl1 - fl) <= self.tolerance:
            return True
        self.history.append((self.charge, fl1))
        if len(self.history) == 1:
            area = abs(np.linalg.det(self.grid.cell_cv[:2, :2]))
            dc = -(fl1 - fl) * area * 0.02
        else:
            (c2, fl2), (c1, fl1) = self.history[-2:]
            c = c2 + (fl - fl2) / (fl1 - fl2) * (c1 - c2)
            dc = c - c1
            if abs(dc) > abs(c2 - c1):
                dc *= abs((c2 - c1) / dc)
        self.charge += dc
        nelectrons += dc
        ibzwfs.calculate_occs(occ_calc, nelectrons)
        mixer.reset()
        return False
