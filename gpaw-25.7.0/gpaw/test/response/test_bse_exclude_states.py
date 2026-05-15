import pytest
from gpaw import GPAW
from gpaw.response.bse import BSE


@pytest.mark.response
def test_bse_exclude_states(in_tmp_dir, gpw_files):
    eshift = 0.8
    bse = BSE(gpw_files['si_gw_a0_all'],
              ecut=50.,
              valence_bands=range(1, 4),
              conduction_bands=range(4, 7),
              deps_max=6,
              eshift=eshift,
              nbands=8)
    bse_matrix = bse.get_bse_matrix()
    w_T, v_Rt, exclude_S = bse.diagonalize_bse_matrix(bse_matrix)

    calc = GPAW(gpw_files['si_gw_a0_all'])
    nk = calc.wfs.kd.nbzkpts
    nval = 3
    ncond = 3
    n_pairs = nk * nval * ncond
    assert len(exclude_S) == 27
    assert len(w_T) == n_pairs - len(exclude_S)
    assert w_T[0] == pytest.approx(0.1008, abs=0.001)
    assert w_T[11] == pytest.approx(0.1262, abs=0.001)
    assert w_T[29] == pytest.approx(0.1888, abs=0.001)
