import numpy as np
import pytest
from ase.build import molecule

from gpaw import GPAW
from gpaw.solvation import (EffectivePotentialCavity, LinearDielectric,
                            Power12Potential, SolvationGPAW)
from gpaw.utilities.adjust_cell import adjust_cell


def test_solvation_vacuum():
    SKIP_REF_CALC = True

    energy_eps = 0.0005 / 8
    forces_eps = 3e-2

    h = 0.3
    vac = 3.0
    u0 = 0.180
    T = 298.15

    atomic_radii = {'H': 1.09}

    atoms = molecule('H2O')
    adjust_cell(atoms, vac, h)

    convergence = {
        'energy': energy_eps * 0.1,
        'forces': forces_eps * 0.1,
        'density': 10.0,
        'eigenstates': 10.0}

    if not SKIP_REF_CALC:
        atoms.calc = GPAW(mode='fd', xc='LDA', h=h, convergence=convergence)
        Eref = atoms.get_potential_energy()
        print(Eref)
        Fref = atoms.get_forces()
        print(Fref)
    else:
        Eref = -11.9929
        Fref = np.array([[0.0, 0.0, -6.07500],
                         [0.0, 1.60924, 0.05999],
                         [0.0, -1.60924, 0.05999]])

    atoms.calc = SolvationGPAW(
        mode='fd',
        xc='LDA',
        h=h,
        convergence=convergence,
        cavity=EffectivePotentialCavity(
            effective_potential=Power12Potential(atomic_radii=atomic_radii,
                                                 u0=u0),
            temperature=T),
        dielectric=LinearDielectric(epsinf=1.0))
    Etest = atoms.get_potential_energy()
    if atoms.calc.old:
        Eeltest = atoms.calc.get_electrostatic_energy()
    else:
        Eeltest = Etest - atoms.calc.environment.interaction_energy()
    Ftest = atoms.get_forces()
    assert Etest == pytest.approx(
        Eref, abs=energy_eps * atoms.calc.get_number_of_electrons())
    assert Ftest == pytest.approx(Fref, abs=forces_eps)
    assert Eeltest == pytest.approx(Etest, abs=0.0)
