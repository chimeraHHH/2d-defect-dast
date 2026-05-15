import pytest
import numpy as np
from gpaw.grid_descriptor import GridDescriptor


def test_fd_ops_gd():
    gd = GridDescriptor([4, 4, 4])
    a = gd.empty(dtype=complex)
    a[:] = 1.0
    assert gd.integrate(a.real, a.real) == 1.0

    def mic(r_v, pbc_c):
        "Mnimal image convention in an [1.0] * 3 unit cell"
        r_c = r_v % 1.0
        r_c -= (2 * r_c).astype(int)
        return np.where(pbc_c, r_c, r_v)

    pbc_c = [1, 1, 0]
    gd = GridDescriptor([10] * 3, cell_cv=[1.] * 3, pbc_c=pbc_c)
    # Point outside box in non-periodic direction should stay
    r_v = np.array([0.01, 2.49, 0.01])
    dr_cG = gd.get_grid_point_distance_vectors(r_v)
    assert dr_cG[:, 0, 0, 0] == pytest.approx(
        mic(np.dot(gd.h_cv, gd.beg_c) - r_v, pbc_c), abs=1e-15)
    # Point outside box in periodic direction should be folded inside
    r_v = np.array([2.49, 0.01, 0.01])
    dr_cG = gd.get_grid_point_distance_vectors(r_v)
    assert dr_cG[:, 0, 0, 0] == pytest.approx(
        mic(np.dot(gd.h_cv, gd.beg_c) - r_v, pbc_c), abs=1e-15)


@pytest.mark.serial
def test_nonorthogonal_distances():
    """test distance vectors in non-orthogonal cells"""
    cell_cv = np.eye(3)
    cell_cv[0, 1] = 1

    gd = GridDescriptor([2, 4, 1], cell_cv)
    r_vG = gd.get_grid_point_distance_vectors((0, 0, 0))

    assert r_vG[1, 1].reshape(4) == pytest.approx([-0.5, -0.25, -1., -0.75])
