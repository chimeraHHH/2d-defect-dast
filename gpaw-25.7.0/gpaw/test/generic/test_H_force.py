from ase import Atoms
from gpaw.test import calculate_numerical_forces
from gpaw import GPAW, Mixer, FermiDirac, Davidson
import pytest


def test_generic_H_force():
    a = 4.0
    n = 16
    atoms = Atoms('H',
                  positions=[[1.234, 2.345, 3.456]],
                  cell=(a, a, a),
                  pbc=True)
    calc = GPAW(mode='fd',
                nbands=1,
                gpts=(n, n, n),
                txt=None,
                eigensolver=Davidson(4),
                mixer=Mixer(0.3, 3, 1),
                convergence={'energy': 1e-7},
                occupations=FermiDirac(0.0))
    atoms.calc = calc
    e1 = atoms.get_potential_energy()
    f1 = atoms.get_forces()[0]
    f2 = calculate_numerical_forces(atoms, 0.001)[0]
    print(f1, f2)
    assert f1 == pytest.approx(f2, abs=0.00025)

    energy_tolerance = 0.001
    force_tolerance = 0.004
    assert e1 == pytest.approx(-0.5318, abs=energy_tolerance)
    f1_ref = [-0.29138, -0.3060, -0.3583]
    for i in range(3):
        assert f1[i] == pytest.approx(f1_ref[i], abs=force_tolerance)
