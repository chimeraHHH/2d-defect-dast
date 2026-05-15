import pytest
from itertools import product

import numpy as np

from gpaw import GPAW
from gpaw.mpi import world
from gpaw.response import ResponseContext, ResponseGroundStateAdapter
from gpaw.response.pw_parallelization import block_partition
from gpaw.response.kspair import KohnShamKPointPairExtractor
from gpaw.response.pair_transitions import PairTransitions
from gpaw.response.pair_integrator import KPointPairPointIntegral
from gpaw.response.symmetry import QSymmetryAnalyzer

from gpaw.test.response.test_chiks import (generate_system_s, generate_qrel_q,
                                           get_q_c, generate_nblocks_n)
from gpaw.test.gpwfile import response_band_cutoff

pytestmark = pytest.mark.skipif(world.size == 1, reason='world.size == 1')


# ---------- Actual tests ---------- #


@pytest.mark.response
@pytest.mark.kspair
@pytest.mark.parametrize('system,qrel,nblocks', product(generate_system_s(),
                                                        generate_qrel_q(),
                                                        generate_nblocks_n()))
def test_parallel_extract_kptdata(in_tmp_dir, gpw_files,
                                  system, qrel, nblocks):
    """Test that the KohnShamKPointPair data extracted from a serial and a
    parallel calculator object is identical."""

    # ---------- Inputs ---------- #

    wfs, spincomponent = system
    q_c = get_q_c(wfs, qrel)

    # ---------- Script ---------- #

    # Initialize serial ground state adapter
    serial_gs = ResponseGroundStateAdapter.from_gpw_file(gpw_files[wfs])
    assert not serial_gs.is_parallelized()

    # Initialize parallel ground state adapter
    calc = GPAW(gpw_files[wfs], parallel=dict(domain=1))
    nbands = response_band_cutoff[wfs]
    parallel_gs = ResponseGroundStateAdapter(calc)
    assert parallel_gs.is_parallelized()

    # Set up extractors and integrals
    context = ResponseContext()
    tcomm, kcomm = block_partition(context.comm, nblocks)
    serial_extractor = initialize_extractor(
        serial_gs, context, tcomm, kcomm)
    assert serial_extractor.gs.world.size == 1
    parallel_extractor = initialize_extractor(
        parallel_gs, context, tcomm, kcomm)
    assert parallel_extractor.gs.world.size > 1
    serial_integral = initialize_integral(serial_extractor, context, q_c)
    parallel_integral = initialize_integral(parallel_extractor, context, q_c)

    # Set up transitions
    transitions = initialize_transitions(serial_gs, spincomponent, nbands)
    parallel_transitions = initialize_transitions(
        parallel_gs, spincomponent, nbands)
    assert np.allclose(parallel_transitions.n1_t, transitions.n1_t)
    assert np.allclose(parallel_transitions.n2_t, transitions.n2_t)
    assert np.allclose(parallel_transitions.s1_t, transitions.s1_t)
    assert np.allclose(parallel_transitions.s2_t, transitions.s2_t)

    # Extract and compare kptpairs
    ni = serial_integral.ni  # Number of iterations in kptpair generator
    assert parallel_integral.ni == ni
    serial_kptpairs = serial_integral.weighted_kpoint_pairs(transitions)
    parallel_kptpairs = parallel_integral.weighted_kpoint_pairs(transitions)
    for _ in range(ni):
        kptpair1, _ = next(serial_kptpairs)
        kptpair2, _ = next(parallel_kptpairs)
        compare_kptpairs(kptpair1, kptpair2)


# ---------- Test functionality ---------- #


def compare_kptpairs(kptpair1, kptpair2):
    if kptpair1 is None:
        # Due to k-point distribution, all ranks don't necessarily have a
        # kptpair to integrate
        assert kptpair2 is None
        return
    assert kptpair1.K1 == kptpair2.K1
    assert kptpair1.K2 == kptpair2.K2
    assert np.allclose(kptpair1.deps_myt, kptpair2.deps_myt)
    assert np.allclose(kptpair1.df_myt, kptpair2.df_myt)

    compare_ikpts(kptpair1.ikpt1, kptpair2.ikpt1)
    compare_ikpts(kptpair1.ikpt2, kptpair2.ikpt2)


def compare_ikpts(ikpt1, ikpt2):
    assert ikpt1.ik == ikpt2.ik
    assert np.allclose(ikpt1.Ph.array, ikpt2.Ph.array)
    assert np.allclose(ikpt1.psit_hG, ikpt2.psit_hG)
    assert np.all(ikpt1.h_myt == ikpt2.h_myt)


def initialize_extractor(gs, context, tcomm, kcomm):
    return KohnShamKPointPairExtractor(gs, context,
                                       transitions_blockcomm=tcomm,
                                       kpts_blockcomm=kcomm)


def initialize_integral(extractor, context, q_c):
    _, generator = QSymmetryAnalyzer().analyze(
        np.asarray(q_c), extractor.gs.kpoints, context)
    return KPointPairPointIntegral(extractor, generator)


def initialize_transitions(gs, spincomponent, nbands):
    bandsummation = 'pairwise'
    return PairTransitions.from_transitions_domain_arguments(
        spincomponent, nbands, gs.nocc1, gs.nocc2, gs.nspins, bandsummation)
