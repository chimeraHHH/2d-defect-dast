import pytest
from .base_calc import calculator
from ase.build import fcc111


@pytest.fixture
def atoms():
    pytest.skip('https://gitlab.com/gpaw/gpaw/-/issues/1381')
    atoms = fcc111('H', size=(1, 1, 1), a=2.5)
    atoms.center(axis=2, vacuum=5)
    atoms.cell[2][2] = 10
    atoms.calc = calculator()
    return atoms
