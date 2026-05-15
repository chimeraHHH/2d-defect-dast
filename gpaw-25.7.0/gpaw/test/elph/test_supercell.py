"""Basic test of elph/supercell/

"""
import numpy as np
import pytest

from gpaw.elph import Supercell
from gpaw.mpi import world

SUPERCELL = (2, 1, 1)

g00 = np.array([[[[-1.12259936e-06, 1.25832645e-02],
                  [1.25832645e-02, 4.04316008e-02]],
                 [[3.46964784e-02, 1.25839117e-02],
                  [1.37267189e-02, 4.76311583e-07]]],
                [[[3.46964784e-02, 1.37267189e-02],
                  [1.25839117e-02, 4.76311583e-07]],
                 [[3.80642006e-07, -1.37265345e-02],
                  [-1.37265345e-02, -4.04323116e-02]]]])


@pytest.mark.slow
@pytest.mark.skipif(world.size > 2,
                    reason='world.size > 2')
@pytest.mark.elph
def test_supercell(module_tmp_path, supercell_cache):
    # Generate supercell_cache
    supercell_cache

    # read supercell matrix
    g_xsNNMM, basis_info = Supercell.load_supercell_matrix()
    assert g_xsNNMM.shape == (6, 1, 2, 2, 2, 2)
    print(g_xsNNMM[0, 0])
    assert g_xsNNMM[0, 0] == pytest.approx(g00, rel=1e-2, abs=1e-4)
