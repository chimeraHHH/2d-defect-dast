import pytest

from ase import Atoms

from gpaw import GPAW
from gpaw.external import StepPotentialz


@pytest.mark.old_gpaw_only
def test_He():
    a = 3
    b = 3 * a
    atoms = Atoms('He', positions=[(a, a, a)],
                  cell=(2 * a, 2 * a, b))

    c00 = GPAW(mode='fd', charge=1)
    atoms.calc = c00
    atoms.get_potential_energy()

    # apply potential where the atom is
    constant = -5
    c10 = GPAW(mode='fd',
               charge=1,
               external=StepPotentialz(b / 2, value_left=constant))
    atoms.calc = c10
    atoms.get_potential_energy()

    # the total energy is shifted for charged systems
    assert c00.get_potential_energy() == pytest.approx(
        c10.get_potential_energy() + constant, 1e-4)

    # apply potential where there is no atom
    c01 = GPAW(mode='fd',
               charge=1,
               external=StepPotentialz(b / 2, value_right=constant))
    atoms.calc = c01
    atoms.get_potential_energy()

    # the total energy should be the same as without the potential
    assert c00.get_potential_energy() == pytest.approx(
        c01.get_potential_energy(), 1e-4)
