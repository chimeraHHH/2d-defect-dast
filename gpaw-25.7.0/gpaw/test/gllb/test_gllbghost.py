import pytest
from ase.build import molecule
from gpaw import GPAW


@pytest.mark.libxc
@pytest.mark.gllb
def test_gllbghost():
    atoms = molecule('H2')
    atoms.center(vacuum=2)
    calc = GPAW(mode='lcao', basis='dzp', setups={0: 'paw', 1: 'ghost'},
                xc='GLLBSC')
    atoms.calc = calc
    atoms.get_potential_energy()
