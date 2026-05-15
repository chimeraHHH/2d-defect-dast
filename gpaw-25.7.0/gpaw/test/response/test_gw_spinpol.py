"""Test to make sure spin-polarized GW calculations don't break."""

# General modules
import pytest
from gpaw.response.g0w0 import G0W0
from gpaw.mpi import world


@pytest.mark.response
def test_gw_spinpol(in_tmp_dir, gpw_files):
    if world.size > 1:
        nblocks = 2
    else:
        nblocks = 1

    gw = G0W0(gpw_files['h2_bcc_afm'],
              nbands=4,  # keep consistent with gpw nbands
              ecut=100,
              kpts=[(0, 0, 0)],
              nblocks=nblocks,
              relbands=(-1, 1))
    result = gw.calculate()

    # Make sure gaps in both spin-channels are the same and don't change.
    # test values do not necessarily reflect those in literature. They simply
    # ensure the value does not change.
    lda_sn = result['eps'][:, 0]
    lda_gap_s = lda_sn[:, 1] - lda_sn[:, 0]
    assert abs(lda_gap_s - 10.5095).max() < 0.01
    qp_sn = result['qp'][:, 0]
    qp_gap_s = qp_sn[:, 1] - qp_sn[:, 0]
    assert abs(qp_gap_s - 9.8156).max() < 0.01
