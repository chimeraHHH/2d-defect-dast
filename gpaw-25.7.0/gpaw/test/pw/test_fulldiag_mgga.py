import pytest
from ase.build import bulk
from numpy.testing import assert_almost_equal

from gpaw import GPAW, PW


@pytest.mark.mgga
def test_full_hamiltonian(in_tmp_dir):
    si = bulk('Si')
    si.calc = GPAW(mode=PW(250),
                   xc='TPSS',
                   kpts={'size': (3, 3, 3), 'gamma': True},
                   parallel={'domain': 1},
                   convergence={'energy': 1e-8},
                   txt='si.txt')
    si.get_potential_energy()
    evals1 = si.calc.get_eigenvalues()
    si.calc.diagonalize_full_hamiltonian()
    evals2 = si.calc.get_eigenvalues()
    assert_almost_equal(evals1, evals2[:evals1.size])
