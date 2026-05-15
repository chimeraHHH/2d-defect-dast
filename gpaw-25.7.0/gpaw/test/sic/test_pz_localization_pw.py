import pytest

from gpaw import GPAW


@pytest.mark.old_gpaw_only
@pytest.mark.sic
def test_pz_localization_pw(in_tmp_dir, gpw_files):
    """
    Test Perdew-Zunger and Kohn-Sham localizations in PW mode
    :param in_tmp_dir:
    :return:
    """
    calc = GPAW(gpw_files["h2o_pz_localization_pw"])
    H2O = calc.atoms
    H2O.calc = calc
    e = H2O.get_potential_energy()
    assert e == pytest.approx(-10.118236, abs=0.1)
