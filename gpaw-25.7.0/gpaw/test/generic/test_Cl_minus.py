from ase import Atoms
from gpaw import GPAW
import pytest


def test_generic_Cl_minus():
    s = Atoms('Cl')
    s.center(vacuum=3)
    c = GPAW(mode='fd',
             xc={'name': 'PBE', 'stencil': 1},
             nbands=-4,
             charge=-1,
             h=0.3)
    s.calc = c

    e = s.get_potential_energy()
    niter = c.get_number_of_iterations()

    print(e, niter)
    energy_tolerance = 0.004
    assert e == pytest.approx(-2.8967, abs=energy_tolerance)
