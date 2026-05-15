from ase import Atoms
from gpaw.test import calculate_numerical_forces
from gpaw import GPAW, FermiDirac, PoissonSolver
import pytest
from gpaw.xc.tools import vxc


def test_generic_8Si():
    a = 5.404
    bulk = Atoms(symbols='Si8',
                 positions=[(0, 0, 0.1 / a),
                            (0, 0.5, 0.5),
                            (0.5, 0, 0.5),
                            (0.5, 0.5, 0),
                            (0.25, 0.25, 0.25),
                            (0.25, 0.75, 0.75),
                            (0.75, 0.25, 0.75),
                            (0.75, 0.75, 0.25)],
                 pbc=True)
    bulk.set_cell((a, a, a), scale_atoms=True)
    n = 20
    calc = GPAW(mode='fd',
                gpts=(n, n, n),
                nbands='150%',
                occupations=FermiDirac(width=0.01),
                poissonsolver=PoissonSolver('fd', nn='M', relax='J'),
                kpts=(2, 2, 2),
                convergence={'energy': 1e-8}
                )
    bulk.calc = calc
    f1 = bulk.get_forces()[0, 2]
    e1 = bulk.get_potential_energy()
    v_xc = vxc(calc.gs_adapter())
    print(v_xc)
    niter1 = calc.get_number_of_iterations()

    f2 = calculate_numerical_forces(bulk, 0.001, [0], [2])[0, 0]
    print((f1, f2, f1 - f2))
    assert f1 == pytest.approx(f2, abs=0.005)

    # Volume per atom:
    vol = a**3 / 8
    de = calc.get_electrostatic_corrections() / vol
    print(de)
    assert abs(de[0] - -2.190) < 0.001

    print((e1, f1, niter1))
    energy_tolerance = 0.0025
    force_tolerance = 0.01
    assert e1 == pytest.approx(-46.6628, abs=energy_tolerance)
    assert f1 == pytest.approx(-1.38242356123, abs=force_tolerance)
