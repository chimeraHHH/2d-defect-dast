from ase import Atoms
from gpaw import GPAW, PoissonSolver
import pytest
from ase.units import Bohr, Hartree


def test_xc_revPBE():
    a = 7.5 * Bohr
    n = 16
    atoms = Atoms('He', [(0.0, 0.0, 0.0)], cell=(a, a, a), pbc=True)
    params = dict(mode='fd',
                  gpts=(n, n, n),
                  nbands=1,
                  xc={'name': 'PBE', 'stencil': 1},
                  poissonsolver=PoissonSolver('fd'))
    atoms.calc = GPAW(**params)
    e1 = atoms.get_potential_energy()
    e1a = atoms.calc.get_reference_energy()
    params['xc'] = {'name': 'revPBE', 'stencil': 1}
    atoms.calc = GPAW(**params)
    e2 = atoms.get_potential_energy()
    e2a = atoms.calc.get_reference_energy()

    assert e1a == pytest.approx(-2.893 * Hartree, abs=8e-3)
    assert e2a == pytest.approx(-2.908 * Hartree, abs=9e-3)
    assert e1 == pytest.approx(e2, abs=4e-3)

    energy_tolerance = 0.0005
    assert e1 == pytest.approx(-0.0790449962, abs=energy_tolerance)
    assert e2 == pytest.approx(-0.08147563, abs=energy_tolerance)
