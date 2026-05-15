import pytest
import numpy as np
from gpaw.response.g0w0 import G0W0


@pytest.mark.response
@pytest.mark.parametrize('fxc_mode, ref_gap', [
    ('GWP', 4.667170),
    ('GWS', 4.988230),
    ('GWG', 4.894904)])
def test_fxc_mode(in_tmp_dir, gpw_files, fxc_mode, ref_gap, scalapack):
    gw = G0W0(gpw_files['bn_pw'],
              bands=(3, 5),
              nbands=9,
              nblocks=1,
              xc='rALDA',
              ecut=40,
              fxc_mode=fxc_mode)

    result = gw.calculate()

    calculated_gap = np.min(result['qp'][0, :, 1])\
        - np.max(result['qp'][0, :, 0])
    assert calculated_gap == pytest.approx(ref_gap, abs=1e-4)
