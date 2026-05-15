import numpy as np
import pytest

from gpaw.mpi import world
from gpaw.response.g0w0 import G0W0

reference_kn = [[0.69806561, 2.58472004, 2.58472066,
                 2.58469313, 3.60859825, 3.60859883],
                [0.87642735, 1.02930988, 4.52049808,
                 4.85337269, 4.85355968, 9.60323838],
                [0.96375991, 2.57490687, 2.57494555,
                 4.59771405, 4.59774543, 8.67625318]]


@pytest.mark.response
def test_hubbard_GW(in_tmp_dir, gpw_files, gpaw_new):
    # This tests checks the actual numerical accuracy which is asserted below
    if gpaw_new and world.size > 1:
        pytest.skip('Parallelization bug for new-gpaw')
    gw = G0W0(gpw_files['ag_plusU_pw'], 'gw',
              integrate_gamma='sphere',
              frequencies={'type': 'nonlinear',
                           'domega0': 0.1, 'omegamax': None},
              nbands=19,  # Carefully selected to avoid slicing degenerate band
              ecut=52.8,  # This too
              eta=0.2)
    results = gw.calculate()

    qp_kn = results['qp'][0]

    assert np.allclose(qp_kn, reference_kn, atol=1e-4, rtol=1e-4)
