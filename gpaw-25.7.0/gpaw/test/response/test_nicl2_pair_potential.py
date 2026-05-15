import pytest

import numpy as np

from gpaw import GPAW
from gpaw.mpi import world
from gpaw.response import ResponseContext, ResponseGroundStateAdapter
from gpaw.response.pw_parallelization import block_partition
from gpaw.response.matrix_elements import TransversePairPotentialCalculator

from gpaw.test.response.test_parallel_kptpair_extraction import \
    initialize_extractor, initialize_transitions, initialize_integral


@pytest.mark.response
@pytest.mark.kspair
@pytest.mark.old_gpaw_only
def test_nicl2_pair_potential(gpw_files):
    """Test that the transverse pair potential vanishes in vacuum."""

    # ---------- Inputs ---------- #

    wfs = 'nicl2_pw_evac'
    q_qc = [[0., 0., 0.],
            [1. / 3., 1. / 3., 0.]]
    rshewmin = 1e-8
    nblocks = world.size // 2 if world.size % 2 == 0 else 1

    # ---------- Script ---------- #

    context = ResponseContext()
    calc = GPAW(gpw_files[wfs], parallel=dict(domain=1))
    gs = ResponseGroundStateAdapter(calc)

    # Set up extractor and transitions
    tcomm, kcomm = block_partition(context.comm, nblocks)
    extractor = initialize_extractor(gs, context, tcomm, kcomm)
    transitions = initialize_transitions(gs, '+-', nbands=20)

    # Set up calculator
    pair_potential_calc = TransversePairPotentialCalculator(gs, context,
                                                            rshewmin=rshewmin)

    # Loop over k-point pairs
    for q_c in q_qc:
        integral = initialize_integral(extractor, context, q_c)
        for kptpair, weight in integral.weighted_kpoint_pairs(transitions):
            if weight is None:
                assert kptpair is None
                continue
            _, _, ut1_mytR, ut2_mytR = \
                pair_potential_calc.extract_pseudo_waves(kptpair)
            wt_mytR = \
                pair_potential_calc._evaluate_pseudo_matrix_element(
                    ut1_mytR, ut2_mytR)

            # Average out the in-plane degrees of freedom
            wt_mytz = np.average(wt_mytR, axis=(1, 2))

            # Gather all the transitions
            wt_tz = kptpair.tblocks.all_gather(wt_mytz)

            # import matplotlib.pyplot as plt
            # for wt_z in wt_tz:
            #     plt.subplot(1, 2, 1)
            #     plt.plot(np.arange(len(wt_z)), wt_z.real)
            #     plt.subplot(1, 2, 2)
            #     plt.plot(np.arange(len(wt_z)), wt_z.imag)
            # plt.show()

            # Find the maximum absolute value of the pair potential and make
            # sure that the pair potential is much smaller than that close to
            # the cell boundary for all the transitions
            abswt_tz = np.abs(wt_tz)
            max_wt = np.max(abswt_tz)
            assert np.all(abswt_tz[:, :10] < max_wt / 25.)
            assert np.all(abswt_tz[:, -10:] < max_wt / 25.)
