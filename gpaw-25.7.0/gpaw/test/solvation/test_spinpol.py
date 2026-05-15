import pytest
from gpaw.utilities.adjust_cell import adjust_cell
from ase.build import molecule
from ase.units import Pascal, m
from gpaw.solvation import (
    SolvationGPAW,
    EffectivePotentialCavity,
    Power12Potential,
    LinearDielectric,
    SurfaceInteraction,
    VolumeInteraction,
    LeakedDensityInteraction,
    GradientSurface,
    KB51Volume)
import numpy as np


def test_solvation_spinpol():
    h = 0.3
    vac = 3.0
    u0 = .180
    epsinf = 80.
    T = 298.15
    atomic_radii = {'H': 1.09}

    atoms = molecule('CN')
    adjust_cell(atoms, vac, h)
    atoms2 = atoms.copy()
    atoms2.set_initial_magnetic_moments(None)

    atomss = (atoms, atoms2)
    Es = []
    Fs = []

    for atoms in atomss:
        atoms.calc = SolvationGPAW(
            mode='fd', xc='LDA', h=h, charge=-1,
            cavity=EffectivePotentialCavity(
                effective_potential=Power12Potential(atomic_radii, u0),
                temperature=T,
                surface_calculator=GradientSurface(),
                volume_calculator=KB51Volume()
            ),
            dielectric=LinearDielectric(epsinf=epsinf),
            interactions=[
                SurfaceInteraction(
                    surface_tension=100. * 1e-3 * Pascal * m
                ),
                VolumeInteraction(
                    pressure=-1.0 * 1e9 * Pascal
                ),
                LeakedDensityInteraction(
                    voltage=1.0
                )
            ]
        )
        Es.append(atoms.get_potential_energy())
        Fs.append(atoms.get_forces())

    # compare to expected difference of a gas phase calc
    print('difference E: ', Es[0] - Es[1])
    assert Es[0] == pytest.approx(Es[1], abs=0.0002)
    print('difference F: ', np.abs(Fs[0] - Fs[1]).max())
    assert Fs[0] == pytest.approx(Fs[1], abs=0.003)

    # XXX add test case where spin matters, e.g. charge=0 for CN?
