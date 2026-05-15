import pytest
from ase import Atoms
from gpaw import GPAW
from gpaw.pipekmezey.pipek_mezey_wannier import PipekMezey


@pytest.mark.pipekmezey
def test_pipekmezey_spin(in_tmp_dir):

    atoms = Atoms('O2',
                  positions=[[0, 0, 0],
                             [0, 0, 1.207]])
    atoms.center(vacuum=5)

    calc = GPAW(mode='fd',
                h=0.24,
                convergence={'density': 1e-4,
                             'eigenstates': 1e-4},
                hund=True)

    calc.atoms = atoms
    calc.calculate()

    # Spin 0
    PM = PipekMezey(calc=calc,
                    spin=0,
                    seed=42)
    PM.localize()

    P = PM.get_function_value()

    assert P == pytest.approx(5.973, abs=0.001)

    # Spin 1
    PM = PipekMezey(calc=calc,
                    spin=1,
                    seed=42)
    PM.localize()

    P = PM.get_function_value()

    assert P == pytest.approx(3.315, abs=0.001)
