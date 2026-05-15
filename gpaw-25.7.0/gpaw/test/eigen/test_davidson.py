from ase import Atom, Atoms
from gpaw import GPAW
from gpaw.eigensolvers.davidson import Davidson
from gpaw.mpi import size
import pytest


def test_eigen_davidson():
    a = 4.05
    d = a / 2**0.5
    bulk = Atoms([Atom('Al', (0, 0, 0)),
                  Atom('Al', (0.5, 0.5, 0.5))], pbc=True)
    bulk.set_cell((d, d, a), scale_atoms=True)
    h = 0.25
    base_params = dict(
        mode='fd',
        h=h,
        nbands=2 * 8,
        kpts=(2, 2, 2))
    base_convergence = {'eigenstates': 7.2e-9, 'energy': 1e-5}
    calc = GPAW(**base_params, convergence=base_convergence)
    bulk.calc = calc
    e0 = bulk.get_potential_energy()
    calc = GPAW(**base_params,
                convergence={**base_convergence, 'bands': 5},
                eigensolver='davidson')
    bulk.calc = calc
    e1 = bulk.get_potential_energy()
    assert e0 == pytest.approx(e1, abs=5.0e-5)

    energy_tolerance = 0.0004
    assert e0 == pytest.approx(-6.97626, abs=energy_tolerance)
    assert e1 == pytest.approx(-6.976265, abs=energy_tolerance)

    # band parallelization
    if size % 2 == 0:
        calc = GPAW(**base_params,
                    convergence={**base_convergence, 'bands': 5},
                    parallel={'band': 2},
                    eigensolver=Davidson(niter=3))
        bulk.calc = calc
        e3 = bulk.get_potential_energy()
        assert e0 == pytest.approx(e3, abs=5.0e-5)
