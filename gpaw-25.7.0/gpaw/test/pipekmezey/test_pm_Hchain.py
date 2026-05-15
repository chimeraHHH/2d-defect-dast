import pytest
from ase import Atoms
from ase.dft.kpoints import monkhorst_pack
from gpaw import GPAW, PW, mpi
from gpaw.pipekmezey.pipek_mezey_wannier import PipekMezey


@pytest.mark.pipekmezey
def test_pipekmezey_chain(in_tmp_dir):

    atoms = Atoms('H4',
                  positions=[[0, 0, 0],
                             [0, 0, 0.74],
                             [0, 0, 1.48],
                             [0, 0, 2.22]])
    atoms.cell = [10, 10, 2.98]
    atoms.center()
    atoms.pbc = (False, False, True)

    kpts = monkhorst_pack((1, 1, 4))

    calc = GPAW(mode=PW(200),
                h=0.24,
                kpts=kpts,
                convergence={'density': 1e-4,
                             'eigenstates': 1e-4},
                symmetry={'point_group': False,
                          'time_reversal': False},
                parallel={'domain': mpi.world.size})

    calc.atoms = atoms
    calc.calculate()

    PM = PipekMezey(calc=calc, seed=42)
    PM.localize()

    P = PM.get_function_value()

    assert P == pytest.approx(0.6803, abs=0.002)
