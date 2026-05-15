import numpy as np
import pytest
from ase.io.ulm import ulmopen
from ase.parallel import parprint

from gpaw import GPAW


@pytest.mark.mgga
def test_pw_fe_stress_mgga(gpw_files, gpaw_new):
    if gpaw_new and ulmopen(gpw_files['fe_pw_distorted']).version < 4:
        pytest.skip('Unsupported new-GPAW + old gpw-file combo')

    fe = GPAW(gpw_files['fe_pw_distorted']).get_atoms()

    # Trigger nasty bug (fixed in !486):
    if not gpaw_new:
        fe.calc.wfs.pt.blocksize = fe.calc.wfs.pd.maxmyng - 1

    s_analytical = fe.get_stress()
    # Calculated numerical stress once, store here to speed up test
    # numerical stresses:
    # revTPSS stress: [0.03113369 -0.05080607 -0.03739338
    #                  -0.03096389  0.21181234  0.0114693]
    s_numerical = np.array([0.03113369, -0.05080607, -0.03739338,
                            -0.03096389, 0.21181234, 0.0114693])
    # s_numerical = fe.calc.calculate_numerical_stress(fe, 1e-5)
    s_err = s_numerical - s_analytical

    parprint('Analytical stress:\n', s_analytical)
    parprint('Numerical stress:\n', s_numerical)
    parprint('Error in stress:\n', s_err)
    assert np.all(abs(s_err) < 1e-4)
