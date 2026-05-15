from ase import Atom, Atoms
from gpaw import GPAW
import pytest


@pytest.mark.legacy
def test_eigen_cg():
    a = 4.05
    d = a / 2**0.5
    bulk = Atoms([Atom('Al', (0, 0, 0)),
                  Atom('Al', (0.5, 0.5, 0.5))],
                 pbc=True)
    bulk.set_cell((d, d, a), scale_atoms=True)
    h = 0.25
    base_params = dict(
        mode='fd',
        h=h,
        nbands=2 * 8,
        kpts=(2, 2, 2),
        convergence={'energy': 1e-5})
    calc = GPAW(**base_params)
    bulk.calc = calc
    e0 = bulk.get_potential_energy()
    calc = GPAW(**base_params, eigensolver='cg')
    bulk.calc = calc
    e1 = bulk.get_potential_energy()
    assert e0 == pytest.approx(e1, abs=5.e-5)

    energy_tolerance = 0.001
    assert e0 == pytest.approx(-6.97626, abs=energy_tolerance)
    assert e1 == pytest.approx(-6.97627, abs=energy_tolerance)
