from gpaw.grid_descriptor import GridDescriptor
import pytest
import numpy as np


def test_symmetrize_fractional_translations():
    gd = GridDescriptor(N_c=np.array([5, 9, 6]))
    a_g = gd.empty(10)
    op_scc = [np.identity(3), -np.identity(3)]
    ft_sc = np.array([[0, 0, 0], [0.5, 0, 0]])
    with pytest.raises(ValueError, match=r"^The specified number"):
        gd.symmetrize(a_g, op_scc, ft_sc)

    newN_c = gd.get_nearest_compatible_grid(ft_sc)
    assert np.array_equal(newN_c, np.array([6, 9, 6]))
    ft2_sc = np.array([[1 / 20, 0, 0], [1 / 30, 1 / 4, 0]])
    newN_c = gd.get_nearest_compatible_grid(ft2_sc)
    assert np.array_equal(newN_c, np.array([60, 8, 6]))
