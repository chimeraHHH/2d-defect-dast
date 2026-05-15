from ase import Atom, Atoms
from gpaw import GPAW
from gpaw.mixer import BroydenMixer
import pytest


def test_broydenmixer(in_tmp_dir):
    a = 2.7
    bulk = Atoms([Atom('Li')], pbc=True, cell=(a, a, a))
    k = 2
    g = 16
    calc = GPAW(mode='fd', gpts=(g, g, g), kpts=(k, k, k), nbands=2,
                mixer=BroydenMixer())
    bulk.calc = calc
    e = bulk.get_potential_energy()
    calc.write('Li.gpw')
    GPAW('Li.gpw')

    energy_tolerance = 0.0001
    assert e == pytest.approx(-1.20258, abs=energy_tolerance)
