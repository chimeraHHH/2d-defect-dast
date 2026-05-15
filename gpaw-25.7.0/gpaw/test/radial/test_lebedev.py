import pytest
from gpaw.sphere.lebedev import run, weight_n, Y_nL, R_nv


@pytest.mark.ci
def test_radial_lebedev(in_tmp_dir):
    weight0_n, Y0_nL, R0_nv = run()
    assert abs(weight0_n - weight_n).sum() < 1e-13
    assert abs(R0_nv - R_nv).sum() < 1e-13
    assert abs(Y0_nL - Y_nL).sum() < 1e-13
