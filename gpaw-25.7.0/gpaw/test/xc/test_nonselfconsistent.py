from ase import Atoms
from ase.units import Bohr
from gpaw import GPAW
import pytest


def xc(name):
    return {'name': name, 'stencil': 1}


def test_xc_nonselfconsistent(in_tmp_dir):
    a = 7.5 * Bohr
    n = 16
    atoms = Atoms('He', [(0.0, 0.0, 0.0)], cell=(a, a, a), pbc=True)
    params = dict(mode='fd', gpts=(n, n, n), nbands=1)
    atoms.calc = GPAW(**params, xc=xc('PBE'))
    e1 = atoms.get_potential_energy()
    e1ref = atoms.calc.get_reference_energy()
    de12 = atoms.calc.get_xc_difference(xc('revPBE'))
    atoms.calc = GPAW(**params, xc=xc('revPBE'))
    e2 = atoms.get_potential_energy()
    e2ref = atoms.calc.get_reference_energy()
    de21 = atoms.calc.get_xc_difference(xc('PBE'))
    print(e1ref + e1 + de12 - (e2ref + e2))
    print(e1ref + e1 - (e2ref + e2 + de21))
    print(de12, de21)
    assert e1ref + e1 + de12 == pytest.approx(e2ref + e2, abs=8e-4)
    assert e1ref + e1 == pytest.approx(e2ref + e2 + de21, abs=3e-3)

    atoms.calc.write('revPBE.gpw')

    de21b = GPAW('revPBE.gpw').get_xc_difference(xc('PBE'))
    assert de21 == pytest.approx(de21b, abs=9e-8)

    energy_tolerance = 0.0005
    assert e1 == pytest.approx(-0.07904951, abs=energy_tolerance)
    assert e2 == pytest.approx(-0.08147563, abs=energy_tolerance)
