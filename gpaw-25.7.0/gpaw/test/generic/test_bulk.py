from ase import Atom, Atoms
from gpaw import GPAW
import pytest
import numpy as np


def test_generic_bulk():
    bulk = Atoms([Atom('Li')], pbc=True)
    k = 4
    g = 8
    calc = GPAW(mode='fd', gpts=(g, g, g), kpts=(k, k, k), nbands=2)
    bulk.calc = calc
    a = np.linspace(2.6, 2.8, 5)
    e = []
    for x in a:
        bulk.set_cell((x, x, x))
        e1 = bulk.get_potential_energy()
        e.append(e1)

    fit = np.polyfit(a, e, 2)
    a0 = np.roots(np.polyder(fit, 1))[0]
    e0 = np.polyval(fit, a0)
    print('a,e =', a0, e0)
    assert a0 == pytest.approx(2.641, abs=0.001)
    assert e0 == pytest.approx(-1.98357, abs=0.0002)

    energy_tolerance = 0.0002
    assert e1 == pytest.approx(-1.96157, abs=energy_tolerance)
