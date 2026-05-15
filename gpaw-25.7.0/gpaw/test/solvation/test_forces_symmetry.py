from ase import Atoms
import pytest
from ase.units import Pascal, m
from gpaw.solvation import (
    SolvationGPAW,
    EffectivePotentialCavity,
    Power12Potential,
    LinearDielectric,
    KB51Volume,
    GradientSurface,
    VolumeInteraction,
    SurfaceInteraction,
    LeakedDensityInteraction)
import numpy as np

h = 0.2
d = 2.5
min_vac = 4.0
u0 = .180
epsinf = 80.
T = 298.15


def test_solvation_forces_symmetry():
    xy_cell = np.ceil((min_vac * 2.) / h / 8.) * 8. * h
    z_cell = np.ceil((min_vac * 2. + d) / h / 8.) * 8. * h
    atoms = Atoms(
        'NaCl', positions=(
            (xy_cell / 2., xy_cell / 2., z_cell / 2. - d / 2.),
            (xy_cell / 2., xy_cell / 2., z_cell / 2. + d / 2.)
        )
    )
    atoms.set_cell((xy_cell, xy_cell, z_cell))

    atoms.calc = SolvationGPAW(
        mode='fd',
        xc='PBE',
        h=h,
        setups={'Na': '1'},
        cavity=EffectivePotentialCavity(
            effective_potential=Power12Potential(u0=u0),
            temperature=T,
            volume_calculator=KB51Volume(),
            surface_calculator=GradientSurface()),
        dielectric=LinearDielectric(epsinf=epsinf),
        # parameters chosen to give ~ 1eV for each interaction
        interactions=[
            VolumeInteraction(pressure=-1e9 * Pascal),
            SurfaceInteraction(surface_tension=100. * 1e-3 * Pascal * m),
            LeakedDensityInteraction(voltage=10.)])
    F = atoms.calc.get_forces(atoms)

    difference = F[0][2] + F[1][2]
    print(difference)
    assert difference == pytest.approx(.0, abs=0.02)
    F[0][2] = F[1][2] = 0.0
    print(np.abs(F))
    assert np.abs(F) == pytest.approx(.0, abs=1e-10)
