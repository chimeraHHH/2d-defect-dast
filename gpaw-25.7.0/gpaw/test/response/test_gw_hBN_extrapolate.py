""" Tests extrapolation to infinite energy cutoff + block parallelization.
It takes ~10 s on one core"""

import pytest
from gpaw.response.g0w0 import G0W0
import numpy as np


@pytest.mark.response
def test_response_gw_hBN_extrapolate(in_tmp_dir, scalapack, gpw_files):
    ecuts = [20, 25, 30]
    common = dict(truncation='2D',
                  q0_correction=True,
                  kpts=[0],
                  eta=0.2,
                  bands=(3, 5),
                  nblocksmax=True)
    gw = G0W0(gpw_files['hbn_pw'],
              'gw-hBN',
              ecut=30,
              frequencies={'type': 'nonlinear',
                           'domega0': 0.1},
              ecut_extrapolation=ecuts,
              **common)

    results = gw.calculate()
    e_qp = results['qp'][0, 0]

    ev = -1.4528
    ec = 3.69469
    assert e_qp[0] == pytest.approx(ev, abs=0.01)
    assert e_qp[1] == pytest.approx(ec, abs=0.01)

    # The individual sigma matrix elements should be exactly the same
    # when running with ecut-extrapolation, and when calculating
    # with particular ecuts individually (given that nbands is not specified,
    # and it is also chosen utilizing ecut).
    for ie, ecut in enumerate(ecuts):
        gw = G0W0(gpw_files['hbn_pw'],
                  f'gw-hBN-separate-ecut{ecut}',
                  ecut=ecut,
                  frequencies={'type': 'nonlinear',
                               'omegamax': 43.2,  # We need same grid as above
                               'domega0': 0.1},
                  ecut_extrapolation=False,
                  **common)
        res = gw.calculate()
        assert np.allclose(res['dsigma_eskn'][0], results['dsigma_eskn'][ie])
        assert np.allclose(res['sigma_eskn'][0], results['sigma_eskn'][ie])
