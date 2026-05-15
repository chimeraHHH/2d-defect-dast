import pytest

from ase.units import Hartree
from gpaw.response.df import DielectricFunction
from gpaw.response.qpd import SingleCylQPWDescriptor


@pytest.mark.response
def test_response_gw_MoS2_cut(gpw_files):
    ecut_sphere = 50.0

    DFs = DielectricFunction(calc=gpw_files['mos2_pw'],
                             frequencies={'type': 'nonlinear',
                                          'domega0': 0.5},
                             ecut=ecut_sphere,
                             truncation='2D',
                             hilbert=False)
    dfs1, dfs2 = DFs.get_dielectric_function()

    ecut_cyl = {
        'class': SingleCylQPWDescriptor,
        'kwargs': {'ecut_xy': ecut_sphere / Hartree,
                   'ecut_z': 0.5 * ecut_sphere / Hartree}
    }

    DFc = DielectricFunction(calc=gpw_files['mos2_pw'],
                             frequencies={'type': 'nonlinear',
                                          'domega0': 0.5},
                             ecut=ecut_cyl,
                             truncation='2D',
                             hilbert=False)
    dfc1, dfc2 = DFc.get_dielectric_function()

    assert dfc1 == pytest.approx(dfs1, rel=1e-6)
    assert dfc2 == pytest.approx(dfs2, rel=5e-2)
