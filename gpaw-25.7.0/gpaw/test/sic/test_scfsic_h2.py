import pytest

from gpaw import GPAW


@pytest.mark.old_gpaw_only
@pytest.mark.sic
def test_sic_scfsic_h2(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files["h2_sic_scfsic"])
    H2 = calc.atoms
    H2.calc = calc
    calc_H = GPAW(gpw_files["h_magmom"])
    H = calc_H.atoms
    H.calc = calc_H
    e1 = H.get_potential_energy()
    e2 = H2.get_potential_energy()
    de = 2 * e1 - e2
    # Used to be a commented out abs=0.1
    assert de == pytest.approx(4.5, abs=0.4)
    # Test forces ...
