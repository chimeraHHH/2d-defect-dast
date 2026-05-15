from ase import Atoms
from gpaw import GPAW
import pytest


@pytest.mark.mgga
def test_symm_mgga():
    a = 5.47
    b = a / 2
    si = Atoms('Si2',
               scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
               cell=[[0, b, b], [b, 0, b], [b, b, 0]],
               pbc=True)
    k = 2
    xc = 'M06-L'
    energies = []
    for symmetry in [True, False]:
        si.calc = GPAW(mode={'name': 'pw', 'ecut': 200},
                       kpts={'size': (k, k, k), 'gamma': True},
                       symmetry={} if symmetry else 'off',
                       xc=xc)
        e = si.get_potential_energy()
        energies.append(e)
    e1, e2 = energies
    assert e1 == pytest.approx(e2, abs=0.001)
