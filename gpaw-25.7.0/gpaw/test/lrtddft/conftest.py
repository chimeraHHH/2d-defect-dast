import pytest
from ase import Atom, Atoms
from gpaw import GPAW


@pytest.fixture
def H2struct():
    R = 0.7  # approx. experimental bond length
    a = 3.0
    c = 4.0
    return Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
                  Atom('H', (a / 2, a / 2, (c + R) / 2))],
                 cell=(a, a, c))


@pytest.fixture
def H2(H2struct):
    H2 = H2struct.copy()
    H2.calc = GPAW(mode='fd',
                   xc='PBE',
                   poissonsolver={'name': 'fd'},
                   nbands=3,
                   spinpol=False)
    H2.get_potential_energy()
    return H2
