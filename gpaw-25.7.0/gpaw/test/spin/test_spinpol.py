import pytest
from ase import Atom, Atoms
from gpaw import GPAW, FermiDirac


def test_spin_spinpol():
    a = 4.0
    n = 16
    hydrogen = Atoms([Atom('H')], cell=(a, a, a), pbc=True)
    hydrogen.center()
    hydrogen.calc = GPAW(
        mode='fd',
        gpts=(n, n, n),
        nbands=1,
        convergence={'energy': 1e-5},
        occupations=FermiDirac(0.0))
    e1 = hydrogen.get_potential_energy()

    hydrogen.calc = hydrogen.calc.new(hund=True)
    e2 = hydrogen.get_potential_energy()

    de = e1 - e2
    print(de)
    assert de == pytest.approx(0.7871, abs=1.e-4)

    energy_tolerance = 0.0006
    assert e1 == pytest.approx(-0.499854, abs=energy_tolerance)
    assert e2 == pytest.approx(-1.287, abs=energy_tolerance)
