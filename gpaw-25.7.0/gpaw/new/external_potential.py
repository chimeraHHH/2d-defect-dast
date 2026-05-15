from __future__ import annotations

import numpy as np
from ase.units import Ha, Bohr

from gpaw.typing import Array1D, Array2D, ArrayLike1D
from gpaw.core import UGArray
from gpaw.new.density import Density


def create_external_potential(params: dict) -> ExternalPotential:
    if not params:
        return ExternalPotential()
    params = params.copy()
    name = params.pop('name')
    if name == 'BField':
        return BField(**params)
    if name == 'ConstantElectricField':
        return ConstantElectricField(**params)
    raise ValueError


class ExternalPotential:
    def update_potential(self,
                         vt_sR: UGArray,
                         density) -> float:
        return 0.0

    def add_paw_correction(self, Delta_p: Array1D, dH_sp: Array2D) -> float:
        return 0.0


class ConstantElectricField(ExternalPotential):
    def __init__(self, strength, direction=[0, 0, 1], tolerance=1e-7):
        """External constant electric field.

        strength: float
            Field strength in V/Ang.
        direction: vector
            Polarization direction.
        """
        self.strength = strength * Bohr / Ha
        self.direction_v = np.array(direction, dtype=float)
        self.direction_v /= np.linalg.norm(self.direction_v)
        self.field_v = self.strength * self.direction_v
        self.tolerance = tolerance
        self.name = 'ConstantElectricField'

    def __str__(self):
        return ('Constant electric field: '
                '({:.3f}, {:.3f}, {:.3f}) V/Ang'
                .format(*(self.field_v * Ha / Bohr)))

    def update_potential(self,
                         vt_sR: UGArray,
                         density) -> float:
        grid = vt_sR.desc
        L_c = grid.cell_cv @ self.direction_v
        (axis,) = np.where(abs(L_c) > self.tolerance)[0]
        # assert not grid.pbc_c[axis]
        relpos_r = np.linspace(grid.start_c[axis],
                               grid.end_c[axis],
                               grid.mysize_c[axis],
                               endpoint=False) / grid.size_c[axis]
        v_r = L_c[axis] * (relpos_r - 0.5) * self.strength
        if grid.start_c[axis] == 0:
            v_r[0] = 0.0
        vt_sR.data += v_r.reshape([1] +
                                  [-1 if c == axis else 1 for c in range(3)])
        return 0.0


class BField(ExternalPotential):
    def __init__(self, field: ArrayLike1D):
        """Constant magnetic field.

        field:
            B-field vector in units of Ha/bohr-magnoton.
        """
        self.field_v = np.array(field) / Ha
        assert self.field_v.shape == (3,)

    def update_potential(self,
                         vt_sR: UGArray,
                         density: Density) -> float:
        magmom_v, _ = density.calculate_magnetic_moments()
        eext = -self.field_v @ magmom_v
        ncomponents = len(vt_sR)
        if ncomponents == 2:
            assert (self.field_v[:2] == 0.0).all()
            vt_sR.data[0] -= self.field_v[2]
            vt_sR.data[1] += self.field_v[2]
        elif ncomponents == 4:
            vt_sR.data[1:] = -self.field_v.reshape((3, 1, 1, 1))
        else:
            1 / 0
        return eext

    def add_paw_correction(self, Delta_p: Array1D, dH_sp: Array2D) -> float:
        if len(dH_sp) == 2:
            c = (4 * np.pi)**0.5 * self.field_v[2]
            dH_sp[0] -= c * Delta_p
            dH_sp[1] += c * Delta_p
        else:
            c_vp = (4 * np.pi)**0.5 * self.field_v[:, np.newaxis]
            dH_sp[1:] -= c_vp * Delta_p
        return 0.0
