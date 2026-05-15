import numpy as np
import pytest
from ase.io.ulm import ulmopen
from ase.parallel import parprint

from gpaw import GPAW
from gpaw.test import calculate_numerical_forces


@pytest.mark.mgga
def test_pw_si_stress_mgga(gpw_files, gpaw_new):
    if gpaw_new and ulmopen(gpw_files['si_pw_distorted']).version < 4:
        pytest.skip('Unsupported new-GPAW + old gpw-file combo')

    si = GPAW(gpw_files['si_pw_distorted']).get_atoms()

    # Trigger nasty bug (fixed in !486):
    if not gpaw_new:
        si.calc.wfs.pt.blocksize = si.calc.wfs.pd.maxmyng - 1

    s_analytical = si.get_stress()
    # Calculated numerical stress once, store here to speed up test
    s_numerical = np.array([-0.01140242, -0.04084746, -0.0401058,
                            -0.02119629, 0.13584242, 0.00911572])
    if 0:
        s_numerical = si.calc.calculate_numerical_stress(si, 1e-5)

    s_err = s_numerical - s_analytical

    parprint('Analytical stress:\n', s_analytical)
    parprint('Numerical stress:\n', s_numerical)
    parprint('Error in stress:\n', s_err)
    assert np.all(abs(s_err) < 1e-5)

    # Check y-component of second atom:
    f = si.get_forces()[1, 1]
    fref = -2.066952082010687
    if 0:
        fref = calculate_numerical_forces(si, 0.001, [1], [1])[0, 0]
    print(f, fref, f - fref)
    assert abs(f - fref) < 0.0005
