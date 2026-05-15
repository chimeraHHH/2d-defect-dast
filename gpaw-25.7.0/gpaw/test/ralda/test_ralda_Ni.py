import pytest
from gpaw.xc.fxc import FXCCorrelation


@pytest.fixture
def ni_gpw(gpw_files, scalapack):
    return gpw_files['ni_pw_kpts333']


@pytest.mark.rpa
@pytest.mark.response
@pytest.mark.parametrize('params, ref_energy', [
    (dict(xc='RPA'), -7.827),
    (dict(xc='rALDA', unit_cells=[2, 1, 1]), -7.501),
    (dict(xc='rAPBE', unit_cells=[2, 1, 1]), -7.456),
])
def test_Ni(in_tmp_dir, ni_gpw, params, ref_energy):
    fxc = FXCCorrelation(ni_gpw, nfrequencies=8, skip_gamma=True,
                         ecut=[50], **params)
    E_fxc = fxc.calculate()
    assert E_fxc == pytest.approx(ref_energy, abs=0.01)
