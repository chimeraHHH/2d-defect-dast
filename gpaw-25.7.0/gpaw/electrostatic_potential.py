from __future__ import annotations

from math import pi, sqrt
from typing import TYPE_CHECKING

import numpy as np
from ase.units import Bohr, Ha

from gpaw.atom.shapefunc import shape_functions
from gpaw.core.arrays import DistributedArrays
from gpaw.core.atom_arrays import AtomArrays
from gpaw.core.uniform_grid import UGArray
from gpaw.setup import Setups
from gpaw.typing import Array1D, ArrayLike2D
from gpaw.utilities import pack_density

if TYPE_CHECKING:
    from gpaw.new.calculation import DFTCalculation


class ElectrostaticPotential:
    def __init__(self,
                 vHt_x: DistributedArrays,
                 W_aL: AtomArrays,
                 Q_aL: AtomArrays,
                 D_asii: AtomArrays,
                 relpos_ac: ArrayLike2D,
                 setups: Setups):
        self.vHt_x = vHt_x
        self.W_aL = W_aL
        self.Q_aL = Q_aL
        self.D_asii = D_asii
        self.relpos_ac = relpos_ac
        self.setups = setups

        # Caching of interpolated pseudo-potential:
        self._grid_spacing = -1.0
        self._vHt_R: UGArray | None = None

    @classmethod
    def from_calculation(cls, calculation: DFTCalculation):
        density = calculation.density
        potential, _, W_aL = calculation.pot_calc.calculate(density)
        Q_aL = density.calculate_compensation_charge_coefficients()
        return cls(potential.vHt_x,
                   W_aL,
                   Q_aL,
                   density.D_asii,
                   calculation.relpos_ac,
                   calculation.setups)

    def atomic_potentials(self) -> Array1D:
        W_aL = self.W_aL.gather(broadcast=True)
        return W_aL.data[::9] * (Ha / (4 * pi)**0.5)

    def atomic_corrections(self):
        """Calculate PAW correction to average electrostatic potential."""
        dEH_a = np.zeros(len(self.setups))
        for a, D_sii in self.D_asii.items():
            setup = self.setups[a]
            D_p = pack_density(D_sii.sum(0))
            dEH_a[a] = setup.dEH0 + setup.dEH_p @ D_p
        self.D_asii.layout.atomdist.comm.sum(dEH_a)
        return dEH_a * Ha * Bohr**3

    def pseudo_potential(self,
                         grid_spacing: float = 0.05,  # Ang
                         ) -> UGArray:
        return self._pseudo_potential(grid_spacing / Bohr).scaled(Bohr, Ha)

    def _pseudo_potential(self,
                          grid_spacing: float,  # Bohr
                          ) -> UGArray:
        if grid_spacing == self._grid_spacing:
            assert self._vHt_R is not None
            return self._vHt_R

        vHt_x = self.vHt_x
        if isinstance(self.vHt_x, UGArray):
            vHt_x = self.vHt_x.to_pbc_grid()
        grid = vHt_x.desc.uniform_grid_with_grid_spacing(grid_spacing / Bohr)
        self._vHt_R = vHt_x.interpolate(grid=grid)
        self._grid_spacing = grid_spacing
        return self._vHt_R

    def all_electron_potential(self,
                               grid_spacing: float = 0.05,  # Ang
                               rcgauss: float = 0.02,  # Ang
                               npoints: int = 200) -> UGArray:
        """Interpolate electrostatic potential.

        Return value in eV.

        ae: bool
            Add PAW correction to get the all-electron potential.
        rcgauss: float
            Width of gaussian (in Angstrom) used to represent the nuclear
            charge.
        """
        vHt_R = self._pseudo_potential(grid_spacing / Bohr).copy()

        dv_a = []
        for a, D_sii in self.D_asii.items():
            setup = self.setups[a]
            c = setup.xc_correction
            rgd = c.rgd
            params = setup.data.shape_function.copy()
            params['lmax'] = 0
            ghat_g = shape_functions(rgd, **params)[0]
            Z_g = shape_functions(rgd, 'gauss', rcgauss, lmax=0)[0] * setup.Z
            D_p = pack_density(D_sii.sum(axis=0))
            D_q = D_p @ c.B_pqL[:, :, 0]
            dn_g = D_q @ (c.n_qg - c.nt_qg) * sqrt(4 * pi)
            dn_g += 4 * pi * (c.nc_g - c.nct_g)
            dn_g -= Z_g
            dn_g -= self.Q_aL[a][0] * ghat_g * sqrt(4 * pi)
            dv_g = rgd.poisson(dn_g) / sqrt(4 * pi)
            dv_g[1:] /= rgd.r_g[1:]
            dv_g[0] = dv_g[1]
            dv_g[-1] = 0.0
            dv_a.append([rgd.spline(dv_g, points=npoints)])

        dv_aR = vHt_R.desc.atom_centered_functions(dv_a, self.relpos_ac)
        dv_aR.add_to(vHt_R)
        return vHt_R.scaled(Bohr, Ha)
