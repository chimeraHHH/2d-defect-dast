"""Make sure we get an exception when an atom is too close to the boundary."""
from ase import Atoms
from gpaw import GPAW
from gpaw.grid_descriptor import GridBoundsError
from gpaw.utilities import AtomsTooClose
import pytest


@pytest.mark.parametrize('mode', ['fd', 'pw'])
def test_too_close_to_boundary(mode, gpaw_new):
    if mode == 'pw' and not gpaw_new:
        return
    a = 4.0
    x = 0.1
    hydrogen = Atoms('H', [(x, x, x)],
                     cell=(a, a, a),
                     pbc=(1, 1, 0))
    hydrogen.calc = GPAW(mode=mode)
    with pytest.raises((GridBoundsError, AtomsTooClose)):
        hydrogen.get_potential_energy()
