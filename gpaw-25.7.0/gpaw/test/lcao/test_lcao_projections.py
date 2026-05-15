import numpy as np
import pytest
from ase.build import molecule

from gpaw import GPAW
from gpaw.lcao.projected_wannier import get_lcao_projections_HSP
from gpaw.poisson import FDPoissonSolver


@pytest.mark.legacy
def test_lcao_lcao_projections():
    atoms = molecule('C2H2')
    atoms.center(vacuum=3.0)
    calc = GPAW(mode='fd',
                gpts=(32, 32, 48),
                poissonsolver=FDPoissonSolver(),
                eigensolver='rmm-diis')
    atoms.calc = calc
    atoms.get_potential_energy()

    V_qnM, H_qMM, S_qMM, P_aqMi = get_lcao_projections_HSP(
        calc, bfs=None, spin=0, projectionsonly=False)

    # Test H and S
    eig = sorted(np.linalg.eigvals(np.linalg.solve(S_qMM[0], H_qMM[0])).real)
    eig_ref = np.array([-17.87913099255579, -13.24870583835115,
                        -11.431118704888123, -7.125564231198733,
                        -7.1255642311987195, 0.5929813570452659,
                        0.5929813570454503, 3.925217670277378,
                        7.451140780537926, 26.734705668744386])
    print(eig)
    assert np.allclose(eig, eig_ref)
