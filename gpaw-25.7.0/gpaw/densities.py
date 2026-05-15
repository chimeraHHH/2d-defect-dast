from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import numpy as np
from ase.units import Bohr

from gpaw.core.atom_arrays import AtomArrays, AtomArraysLayout
from gpaw.core.uniform_grid import UGArray
from gpaw.setup import Setups
from gpaw.spherical_harmonics import Y
from gpaw.spline import Spline
from gpaw.typing import Array1D, Array3D, Vector, Array2D
from gpaw.new import zips as zip

if TYPE_CHECKING:
    from gpaw.new.calculation import DFTCalculation


class Densities:
    def __init__(self,
                 nt_sR: UGArray,
                 D_asii: AtomArrays,
                 relpos_ac: Array2D,
                 setups: Setups):
        self.nt_sR = nt_sR
        self.D_asii = D_asii
        self.relpos_ac = relpos_ac
        self.setups = setups

    @classmethod
    def from_calculation(cls, calculation: DFTCalculation):
        density = calculation.density
        return cls(density.nt_sR,
                   density.D_asii,
                   calculation.relpos_ac,
                   calculation.setups)

    def pseudo_densities(self,
                         grid_spacing: float = None,  # Ang
                         grid_refinement: int = None,
                         add_compensation_charges: bool = True
                         ) -> UGArray:
        nt_sR = self._pseudo_densities(grid_spacing, grid_refinement)

        ncomponents = nt_sR.dims[0]
        ndensities = ncomponents % 3

        if add_compensation_charges:
            cc_asL = AtomArraysLayout(
                [(ncomponents, setup.Delta_iiL.shape[2])
                 for setup in self.setups],
                atomdist=self.D_asii.layout.atomdist).empty()

            for a, D_sii in self.D_asii.items():
                Q_sL = np.einsum('sij, ijL -> sL',
                                 D_sii.real, self.setups[a].Delta_iiL)
                delta = (self.setups[a].Delta0 +
                         self.setups[a].Nv / (4 * pi)**0.5)
                Q_sL[:ndensities, 0] += delta / ndensities
                cc_asL[a] = Q_sL

            ghat_aLR = self.setups.create_compensation_charges(
                nt_sR.desc,
                self.relpos_ac,
                self.D_asii.layout.atomdist)
            ghat_aLR.add_to(nt_sR, cc_asL)

        return nt_sR.scaled(Bohr, Bohr**-3)

    def _pseudo_densities(self,
                          grid_spacing: float = None,  # Ang
                          grid_refinement: int = None,
                          ) -> UGArray:
        nt_sR = self.nt_sR.to_pbc_grid()
        grid = nt_sR.desc
        if grid_spacing is not None:
            assert grid_refinement is None
            grid = grid.uniform_grid_with_grid_spacing(
                grid_spacing / Bohr)
        elif grid_refinement is not None and grid_refinement > 1:
            grid = grid.new(size=grid.size * grid_refinement)
        else:
            return nt_sR.copy()

        return nt_sR.interpolate(grid=grid)

    def all_electron_densities(self,
                               *,
                               grid_spacing: float = None,  # Ang
                               grid_refinement: int = None,
                               skip_core: bool = False,
                               ) -> UGArray:
        n_sR = self._pseudo_densities(grid_spacing, grid_refinement)
        ncomponents = n_sR.dims[0]
        nspins = ncomponents % 3
        grid = n_sR.desc

        electrons_as = np.zeros((len(self.relpos_ac), ncomponents))
        splines = {}
        for a, D_sii in self.D_asii.items():
            D_sii = D_sii.real
            relpos_c = self.relpos_ac[a]
            setup = self.setups[a]
            if setup not in splines:
                phi_j, phit_j, nc, nct = setup.get_partial_waves()[:4]
                if skip_core:
                    nc = Spline.from_data(0, 10.0, [0.0, 0.0])
                rcut = max(setup.rcut_j)
                splines[setup] = (rcut, phi_j, phit_j, nc, nct)
            rcut, phi_j, phit_j, nc, nct = splines[setup]

            # Expected integral of PAW correction:
            electrons_s = np.zeros(ncomponents)
            if skip_core:
                electrons_s[:nspins] = -setup.Nct / nspins
            else:
                electrons_s[:nspins] = (setup.Nc - setup.Nct) / nspins
            electrons_s += (4 * pi)**0.5 * np.einsum('sij, ij -> s',
                                                     D_sii,
                                                     setup.Delta_iiL[:, :, 0])

            # Add PAW correction:
            R_v = relpos_c @ grid.cell_cv
            electrons_s -= add(R_v, n_sR, phi_j, phit_j, nc, nct, rcut, D_sii)
            electrons_as[a] = electrons_s

        if not skip_core:
            # Add missing charge to grid-points closest to atoms:
            grid.comm.sum(electrons_as)
            R_ac = np.around(grid.size * self.relpos_ac).astype(int)
            R_ac %= grid.size
            for R_c, electrons_s in zip(R_ac, electrons_as):
                R_c -= grid.start_c
                if (R_c >= 0).all() and (R_c < grid.mysize_c).all():
                    for n_R, e in zip(n_sR.data, electrons_s):
                        n_R[tuple(R_c)] += e / grid.dv

        return n_sR.scaled(Bohr, Bohr**-3)

    def spin_contamination(self, majority_spin=None):
        """Calculate the spin contamination.

        Spin contamination is defined as the integral over the
        spin density difference, where it is negative (i.e. the
        minority spin density is larger than the majority spin density.
        """
        n_sR = self.all_electron_densities()
        m0, m1 = n_sR.integrate()
        if majority_spin is None:
            majority_spin = int(m1 > m0)
        d_R = n_sR[0].data - n_sR[1].data
        if majority_spin == 0:
            d_R *= -1.0
        d_R = np.where(d_R > 0, d_R, 0.0)
        return n_sR.desc.from_data(d_R).integrate()


def add(R_v: Vector,
        a_sR: UGArray,
        phi_j: list[Spline],
        phit_j: list[Spline],
        nc: Spline,
        nct: Spline,
        rcut: float,
        D_sii: Array3D) -> Array1D:
    """Add PAW corrections to real-space grid.

    Returns number of elctrons added.
    """
    ug = a_sR.desc
    R_Rv = ug.xyz()
    lmax = max(phi.l for phi in phi_j)
    ncomponents = a_sR.dims[0]
    nspins = ncomponents % 3
    electrons_s = np.zeros(ncomponents)
    start_c = 0 - ug.pbc
    stop_c = 1 + ug.pbc
    for u0 in range(start_c[0], stop_c[0]):
        for u1 in range(start_c[1], stop_c[1]):
            for u2 in range(start_c[2], stop_c[2]):
                d_Rv = R_Rv - (R_v + (u0, u1, u2) @ ug.cell_cv)
                d_R = (d_Rv**2).sum(3)**0.5
                mask_R = d_R < rcut
                npoints = mask_R.sum()
                if npoints == 0:
                    continue

                a_sr = np.zeros((ncomponents, npoints))
                d_rv = d_Rv[mask_R]
                d_r = d_R[mask_R]
                Y_Lr = [Y(L, *d_rv.T) for L in range((lmax + 1)**2)]
                phi_jr = [phi.map(d_r) for phi in phi_j]
                phit_jr = [phit.map(d_r) for phit in phit_j]
                l_j = [phi.l for phi in phi_j]

                i1 = 0
                for l1, phi1_r, phit1_r in zip(l_j, phi_jr, phit_jr):
                    i2 = 0
                    i1b = i1 + 2 * l1 + 1
                    D_smi = D_sii[:, i1:i1b]
                    for l2, phi2_r, phit2_r in zip(l_j, phi_jr, phit_jr):
                        i2b = i2 + 2 * l2 + 1
                        D_smm = D_smi[:, :, i2:i2b]
                        b_sr = np.einsum(
                            'smn, mr, nr -> sr',
                            D_smm,
                            Y_Lr[l1**2:(l1 + 1)**2],
                            Y_Lr[l2**2:(l2 + 1)**2]) * (
                            phi1_r * phi2_r - phit1_r * phit2_r)
                        a_sr += b_sr
                        i2 = i2b
                    i1 = i1b

                dn_r = nc.map(d_r) - nct.map(d_r)
                a_sr[:nspins] += dn_r * ((4 * pi)**-0.5 / nspins)
                electrons_s += a_sr.sum(1) * a_sR.desc.dv
                a_sR.data[:, mask_R] += a_sr
    return electrons_s
