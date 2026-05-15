from ase import Atom, Atoms
from ase.units import Bohr
from gpaw import GPAW
import pytest


def test_xc_nonselfconsistentLDA(in_tmp_dir):
    a = 7.5 * Bohr
    n = 16
    atoms = Atoms([Atom('He', (0.0, 0.0, 0.0))], cell=(a, a, a), pbc=True)
    calc = GPAW(mode='fd', gpts=(n, n, n), nbands=1, xc='LDA')
    atoms.calc = calc
    e1 = atoms.get_potential_energy()
    e1ref = calc.get_reference_energy()
    de12 = calc.get_xc_difference({'name': 'PBE', 'stencil': 1})
    calc = GPAW(
        mode='fd', gpts=(n, n, n), nbands=1, xc={'name': 'PBE', 'stencil': 1})
    atoms.calc = calc
    e2 = atoms.get_potential_energy()
    e2ref = calc.get_reference_energy()
    de21 = calc.get_xc_difference('LDA')
    print(e1ref + e1 + de12, e2ref + e2)
    print(e1ref + e1, e2ref + e2 + de21)
    print(de12, de21)
    assert e1ref + e1 + de12 == pytest.approx(e2ref + e2, abs=0.02)
    assert e1ref + e1 == pytest.approx(e2ref + e2 + de21, abs=0.025)

    calc.write('PBE.gpw')

    de21b = GPAW('PBE.gpw').get_xc_difference('LDA')
    print(de21, de21b)
    assert de21 == pytest.approx(de21b, abs=9e-8)

    energy_tolerance = 0.0007
    assert e1 == pytest.approx(-0.0961003634812, abs=energy_tolerance)
    assert e2 == pytest.approx(-0.0790249564625, abs=energy_tolerance)
