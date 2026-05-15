"""
Calculate the magnetic response in iron using ALDA.

Fast test, where the kernel is scaled to fulfill the Goldstone theorem.
"""

# Workflow modules
from pathlib import Path
import pytest
import numpy as np

# Script modules
from gpaw.test import findpeak
from gpaw.mpi import world

from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.chiks import ChiKSCalculator
from gpaw.response.susceptibility import (ChiFactory, spectral_decomposition,
                                          read_eigenmode_lineshapes)
from gpaw.response.localft import LocalGridFTCalculator, LocalPAWFTCalculator
from gpaw.response.fxc_kernels import FXCKernel, AdiabaticFXCCalculator
from gpaw.response.dyson import HXCKernel
from gpaw.response.goldstone import FMGoldstoneScaling
from gpaw.response.pair_functions import read_pair_function


def set_up_fxc_calculators(gs, context):
    fxckwargs_and_identifiers = []

    # Set up grid calculator (without file buffer)
    localft_calc = LocalGridFTCalculator(gs, context)
    fxckwargs_grid = {'fxc_calculator': AdiabaticFXCCalculator(localft_calc),
                      'hxc_scaling': FMGoldstoneScaling()}
    fxckwargs_and_identifiers.append((fxckwargs_grid, 'grid'))

    # Set up paw calculator (with file buffer)
    localft_calc = LocalPAWFTCalculator(gs, context, rshelmax=0)
    fxckwargs_paw = {'fxc_calculator': AdiabaticFXCCalculator(localft_calc),
                     'fxc_file': Path('paw_ALDA_fxc.npz'),
                     'hxc_scaling': FMGoldstoneScaling()}
    fxckwargs_and_identifiers.append((fxckwargs_paw, 'paw'))

    return fxckwargs_and_identifiers


def get_test_values(identifier):
    test_mw1 = 0.  # meV
    test_Ipeak1 = 7.48  # a.u.
    test_Ipeak3 = 19.83  # a.u.
    if identifier == 'grid':
        test_fxcs = 1.059
        test_mw2 = 363.  # meV
        test_Ipeak2 = 3.47  # a.u.
        test_Ipeak4 = 9.20  # a.u.
    elif identifier == 'paw':
        test_fxcs = 1.131
        test_mw2 = 352.  # meV
        test_Ipeak2 = 3.35  # a.u.
        test_Ipeak4 = 9.14  # a.u.
    else:
        raise ValueError(f'Invalid identifier {identifier}')

    return (test_fxcs, test_mw1, test_mw2,
            test_Ipeak1, test_Ipeak2, test_Ipeak3, test_Ipeak4)


@pytest.mark.kspair
@pytest.mark.response
def test_response_iron_sf_gssALDA(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    nbands = 6
    q_qc = [[0.0, 0.0, 0.0], [0.0, 0.0, 1. / 4.]]  # Two q-points along G-N
    frq_qw = [np.linspace(-0.080, 0.120, 26), np.linspace(0.250, 0.450, 26)]
    fxc = 'ALDA'
    ecut = 300
    eta = 0.1
    if world.size > 1:
        nblocks = 2
    else:
        nblocks = 1

    # ---------- Script ---------- #

    gs = ResponseGroundStateAdapter.from_gpw_file(gpw_files['fe_pw'])
    chiks_calc = ChiKSCalculator(gs,
                                 nbands=nbands,
                                 ecut=ecut,
                                 gammacentered=True,
                                 nblocks=nblocks)
    fxckwargs_and_identifiers = set_up_fxc_calculators(
        gs, chiks_calc.context)

    chi_factory = ChiFactory(
        chiks_calc,
        # Use the first fxc_calculator for the ChiFactory
        fxc_calculator=fxckwargs_and_identifiers[0][0]['fxc_calculator'])

    for q in range(2):
        complex_frequencies = frq_qw[q] + 1.j * eta

        # Calculate chi using the various fxc calculators
        for f, (fxckwargs, identifier) in enumerate(fxckwargs_and_identifiers):
            if f == 0:
                # The first kernel is used directly through the ChiFactory
                chiks, chi = chi_factory('+-', q_qc[q], complex_frequencies,
                                         fxc=fxc,
                                         hxc_scaling=fxckwargs['hxc_scaling'])
            else:  # We wish to test storing the remaining kernels in files
                assert 'fxc_file' in fxckwargs
                fxc_file = fxckwargs['fxc_file']
                if q == 0:  # Calculate and save kernel
                    assert not fxc_file.is_file()
                    fxc_calculator = fxckwargs['fxc_calculator']
                    fxc_kernel = fxc_calculator(fxc, '+-', chiks.qpd)
                    fxc_kernel.save(fxc_file)
                else:  # Reuse kernel from previous calculation
                    assert fxc_file.is_file()
                    fxc_kernel = FXCKernel.from_file(fxc_file)

                # Calculate many-body susceptibility
                chi = chi_factory.dyson_solver(
                    chiks, HXCKernel(
                        hartree_kernel=chi_factory.get_hartree_kernel(
                            '+-', chiks.qpd),
                        xc_kernel=fxc_kernel),
                    hxc_scaling=fxckwargs['hxc_scaling'])

            chi.write_macroscopic_component(identifier + '_iron_dsus'
                                            + '_%d.csv' % (q + 1))
            Amaj, _ = spectral_decomposition(chi, pos_eigs=10, neg_eigs=0)
            Amaj.write_eigenmode_lineshapes(identifier + '_Amaj_modes'
                                            + '_%d.csv' % (q + 1), nmodes=1)

        chi_factory.context.write_timer()

    world.barrier()

    # plot_comparison('grid', 'paw')

    # Compare results to test values
    for fxckwargs, identifier in fxckwargs_and_identifiers:
        fxcs = fxckwargs['hxc_scaling'].lambd
        (_, _, mw1, Ipeak1, _, _, mw2, Ipeak2,
         _, _, mw3, Ipeak3, _, _, mw4, Ipeak4) = extract_data(identifier)

        print(fxcs, mw1, mw2, Ipeak1, Ipeak2, mw3, Ipeak3, mw4, Ipeak4)

        (test_fxcs, test_mw1, test_mw2,
         test_Ipeak1, test_Ipeak2,
         test_Ipeak3, test_Ipeak4) = get_test_values(identifier)

        # fxc scaling:
        assert fxcs == pytest.approx(test_fxcs, abs=0.005)

        # Magnon peak:
        assert mw1 == pytest.approx(test_mw1, abs=20.)
        assert mw2 == pytest.approx(test_mw2, abs=50.)
        assert mw3 == pytest.approx(mw1, abs=10.)
        assert mw4 == pytest.approx(mw2, abs=10.)

        # Scattering function intensity:
        assert Ipeak1 == pytest.approx(test_Ipeak1, abs=0.5)
        assert Ipeak2 == pytest.approx(test_Ipeak2, abs=0.5)
        assert Ipeak3 == pytest.approx(test_Ipeak3, abs=0.5)
        assert Ipeak4 == pytest.approx(test_Ipeak4, abs=1.1)


def extract_data(identifier):
    # Read data
    w1_w, chi1_w = read_pair_function(identifier + '_iron_dsus_1.csv')
    w2_w, chi2_w = read_pair_function(identifier + '_iron_dsus_2.csv')
    w3_w, s3_wm = read_eigenmode_lineshapes(identifier + '_Amaj_modes_1.csv')
    w4_w, s4_wm = read_eigenmode_lineshapes(identifier + '_Amaj_modes_2.csv')
    s3_w = s3_wm[:, 0]
    s4_w = s4_wm[:, 0]

    # Spectral function
    S1_w = -chi1_w.imag
    S2_w = -chi2_w.imag

    # Identify peaks
    wpeak1, Ipeak1 = findpeak(w1_w, S1_w)
    wpeak2, Ipeak2 = findpeak(w2_w, S2_w)
    wpeak3, Ipeak3 = findpeak(w3_w, s3_w)
    wpeak4, Ipeak4 = findpeak(w4_w, s4_w)

    # Peak positions in meV
    mw1 = wpeak1 * 1000
    mw2 = wpeak2 * 1000
    mw3 = wpeak3 * 1000
    mw4 = wpeak4 * 1000

    return (w1_w, S1_w, mw1, Ipeak1, w2_w, S2_w, mw2, Ipeak2,
            w3_w, s3_w, mw3, Ipeak3, w4_w, s4_w, mw4, Ipeak4)


def plot_comparison(identifier1, identifier2):
    (w11_w, S11_w, _, _, w12_w, S12_w, _, _,
     w13_w, s13_w, _, _, w14_w, s14_w, _, _) = extract_data(identifier1)
    (w21_w, S21_w, _, _, w22_w, S22_w, _, _,
     w23_w, s23_w, _, _, w24_w, s24_w, _, _) = extract_data(identifier2)

    import matplotlib.pyplot as plt
    plt.subplot(2, 2, 1)
    plt.plot(w11_w, S11_w)
    plt.plot(w21_w, S21_w)
    plt.subplot(2, 2, 2)
    plt.plot(w12_w, S12_w)
    plt.plot(w22_w, S22_w)
    plt.subplot(2, 2, 3)
    plt.plot(w13_w, s13_w)
    plt.plot(w23_w, s23_w)
    plt.subplot(2, 2, 4)
    plt.plot(w14_w, s14_w)
    plt.plot(w24_w, s24_w)
    plt.show()
