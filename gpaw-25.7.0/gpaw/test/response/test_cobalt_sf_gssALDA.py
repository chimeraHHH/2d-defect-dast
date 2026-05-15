# General modules
import pytest
import numpy as np

# Script modules
from gpaw import GPAW
from gpaw.test import findpeak
from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.chiks import ChiKSCalculator
from gpaw.response.susceptibility import (ChiFactory, spectral_decomposition,
                                          EigendecomposedSpectrum,
                                          read_full_spectral_weight,
                                          read_eigenmode_lineshapes)
from gpaw.response.fxc_kernels import AdiabaticFXCCalculator
from gpaw.response.goldstone import FMGoldstoneScaling
from gpaw.response.pair_functions import read_susceptibility_array
from gpaw.test.gpwfile import response_band_cutoff


@pytest.mark.kspair
@pytest.mark.response
def test_response_cobalt_sf_gssALDA(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    fxc = 'ALDA'
    q_qc = [[0.0, 0.0, 0.0], [1. / 4., 0.0, 0.0]]  # Two q-points along G-M
    # We make sure to have exactly 49 frequency points, so that one rank will
    # have no block distributed frequencies when world.size == 8
    frq_w = np.linspace(-0.6, 1.2, 49)
    eta = 0.2

    rshelmax = 0
    hxc_scaling = FMGoldstoneScaling()
    ecut = 250
    reduced_ecut = 100  # ecut for eigenmode analysis
    pos_eigs = 2  # majority modes
    neg_eigs = 0  # minority modes
    nblocks = 'max'

    # ---------- Script ---------- #

    # Initialize objects to calculat Chi
    context = ResponseContext()
    calc = GPAW(gpw_files['co_pw'], parallel=dict(domain=1))
    nbands = response_band_cutoff['co_pw']
    gs = ResponseGroundStateAdapter(calc)
    chiks_calc = ChiKSCalculator(gs, context,
                                 nbands=nbands,
                                 ecut=ecut,
                                 gammacentered=True,
                                 nblocks=nblocks)
    fxc_calculator = AdiabaticFXCCalculator.from_rshe_parameters(
        gs, context, rshelmax=rshelmax)
    chi_factory = ChiFactory(chiks_calc, fxc_calculator)

    for q, q_c in enumerate(q_qc):
        complex_frequencies = frq_w + 1.j * eta
        chiks, chi = chi_factory('+-', q_c, complex_frequencies,
                                 fxc=fxc, hxc_scaling=hxc_scaling)

        # Check that the dissipative part of the susceptibility can be
        # reconstructed from the EigendecomposedSpectrum
        chid = chi.copy_dissipative_part()
        spectrum = EigendecomposedSpectrum.from_chi(chi)
        assert np.allclose(-np.pi * spectrum.A_wGG, chid.array)

        # Perform a spectral decomposition of the magnetic excitations in the
        # Kohn-Sham and many-body susceptibilities, writing the full spectral
        # weight of both the majority and minority spectra
        Aksmaj, Aksmin = spectral_decomposition(chiks)
        Aksmaj.write_full_spectral_weight(f'Aksmaj_q{q}.dat')
        Aksmin.write_full_spectral_weight(f'Aksmin_q{q}.dat')
        Amaj, Amin = spectral_decomposition(chi,
                                            pos_eigs=pos_eigs,
                                            neg_eigs=neg_eigs)
        Amaj.write_full_spectral_weight(f'Amaj_q{q}.dat')
        Amin.write_full_spectral_weight(f'Amin_q{q}.dat')
        # Write also the majority magnon modes of the many-body system
        Amaj.write_eigenmode_lineshapes(f'Amaj_modes_q{q}.dat', nmodes=2)

        # Perform an analogous spectral decomposition of the many-body
        # susceptibility in a reduced plane-wave basis and write the decomposed
        # spectrum to a file
        chi = chi.copy_with_reduced_ecut(reduced_ecut)
        chi.write_diagonal(f'chiwG_q{q}.npz')
        Amaj, Amin = spectral_decomposition(chi,
                                            pos_eigs=pos_eigs,
                                            neg_eigs=neg_eigs)
        Amaj.write(f'Amaj_q{q}.npz')
        Amin.write(f'Amin_q{q}.npz')
        assert f'{fxc},+-' in chi_factory.fxc_kernel_cache

    context.write_timer()
    context.comm.barrier()

    # Read data
    wksmaj0_w, Aksmaj0_w = read_full_spectral_weight('Aksmaj_q0.dat')
    wksmin0_w, Aksmin0_w = read_full_spectral_weight('Aksmin_q0.dat')
    wksmaj1_w, Aksmaj1_w = read_full_spectral_weight('Aksmaj_q1.dat')
    wksmin1_w, Aksmin1_w = read_full_spectral_weight('Aksmin_q1.dat')
    wmaj0_w, Amaj0_w = read_full_spectral_weight('Amaj_q0.dat')
    wmin0_w, Amin0_w = read_full_spectral_weight('Amin_q0.dat')
    wmaj1_w, Amaj1_w = read_full_spectral_weight('Amaj_q1.dat')
    wmin1_w, Amin1_w = read_full_spectral_weight('Amin_q1.dat')
    wa0_w, a0_wm = read_eigenmode_lineshapes('Amaj_modes_q0.dat')
    wa1_w, a1_wm = read_eigenmode_lineshapes('Amaj_modes_q1.dat')
    w0_w, _, chi0_wG = read_susceptibility_array('chiwG_q0.npz')
    w1_w, _, chi1_wG = read_susceptibility_array('chiwG_q1.npz')
    Amaj0 = EigendecomposedSpectrum.from_file('Amaj_q0.npz')
    Amaj1 = EigendecomposedSpectrum.from_file('Amaj_q1.npz')

    # Find acoustic magnon mode
    wpeak00, Ipeak00 = findpeak(w0_w, -chi0_wG[:, 0].imag)
    wpeak01, Ipeak01 = findpeak(w1_w, -chi1_wG[:, 0].imag)
    # Find optical magnon mode
    wpeak10, Ipeak10 = findpeak(w0_w, -chi0_wG[:, 1].imag)
    wpeak11, Ipeak11 = findpeak(w1_w, -chi1_wG[:, 1].imag)

    # Extract the eigenmodes from the eigendecomposed spectrum in the reduced
    # plane-wave basis, restricting the frequencies to be non-negative
    wmask = frq_w >= 0.
    wm0 = Amaj0.get_eigenmode_frequency(nmodes=pos_eigs, wmask=wmask)
    ar0_wm = Amaj0.get_eigenmodes_at_frequency(wm0, nmodes=pos_eigs)
    wm1 = Amaj1.get_eigenmode_frequency(nmodes=pos_eigs, wmask=wmask)
    ar1_wm = Amaj1.get_eigenmodes_at_frequency(wm1, nmodes=pos_eigs)

    # Find peaks in eigenmodes (in full and reduced basis)
    mpeak00, Speak00 = findpeak(wa0_w, a0_wm[:, 0])
    mpeak01, Speak01 = findpeak(wa1_w, a1_wm[:, 0])
    mpeak10, Speak10 = findpeak(wa0_w, a0_wm[:, 1])
    mpeak11, Speak11 = findpeak(wa1_w, a1_wm[:, 1])
    mrpeak00, Srpeak00 = findpeak(Amaj0.omega_w, ar0_wm[:, 0])
    mrpeak01, Srpeak01 = findpeak(Amaj1.omega_w, ar1_wm[:, 0])
    mrpeak10, Srpeak10 = findpeak(Amaj0.omega_w, ar0_wm[:, 1])
    mrpeak11, Srpeak11 = findpeak(Amaj1.omega_w, ar1_wm[:, 1])

    # Calculate the majority spectral enhancement at the acoustic magnon maxima
    w0 = np.argmin(np.abs(Amaj0.omega_w - wpeak00))
    w1 = np.argmin(np.abs(Amaj1.omega_w - wpeak01))
    enh0 = Amaj0_w[w0] / Aksmaj0_w[w0]
    enh1 = Amaj1_w[w1] / Aksmaj1_w[w1]

    # Calculate the minority spectral enhancement at 600 meV (corresponding to
    # -600 meV on original frequency grid)
    min_enh0 = Amin0_w[0] / Aksmin0_w[0]
    min_enh1 = Amin1_w[0] / Aksmin1_w[0]

    if context.comm.rank == 0:
        # import matplotlib.pyplot as plt
        # # Plot the magnon lineshapes
        # # q=0
        # plt.subplot(2, 3, 1)
        # plt.plot(w0_w, -chi0_wG[:, 0].imag)
        # plt.axvline(wpeak00, c='0.5', linewidth=0.8)
        # plt.plot(w0_w, -chi0_wG[:, 1].imag)
        # plt.axvline(wpeak10, c='0.5', linewidth=0.8)
        # plt.plot(Amaj0.omega_w, Amaj0.s_we[:, 0])
        # plt.plot(Amaj0.omega_w, Amaj0.s_we[:, 1])
        # plt.plot(wa0_w, a0_wm[:, 0])
        # plt.plot(wa0_w, a0_wm[:, 1])
        # plt.plot(Amaj0.omega_w, ar0_wm[:, 0])
        # plt.plot(Amaj0.omega_w, ar0_wm[:, 1])
        # # q=1
        # plt.subplot(2, 3, 4)
        # plt.plot(w1_w, -chi1_wG[:, 0].imag)
        # plt.axvline(wpeak01, c='0.5', linewidth=0.8)
        # plt.plot(w1_w, -chi1_wG[:, 1].imag)
        # plt.axvline(wpeak11, c='0.5', linewidth=0.8)
        # plt.plot(Amaj1.omega_w, Amaj1.s_we[:, 0])
        # plt.plot(Amaj1.omega_w, Amaj1.s_we[:, 1])
        # plt.plot(wa1_w, a1_wm[:, 0])
        # plt.plot(wa1_w, a1_wm[:, 1])
        # plt.plot(Amaj1.omega_w, ar1_wm[:, 0])
        # plt.plot(Amaj1.omega_w, ar1_wm[:, 1])
        # # Plot full spectral weight of majority excitations
        # # q=0
        # plt.subplot(2, 3, 2)
        # plt.plot(wmaj0_w, Amaj0_w)
        # plt.plot(wksmaj0_w, Aksmaj0_w)
        # plt.plot(Amaj0.omega_w, Amaj0.A_w)
        # plt.axvline(wpeak00, c='0.5', linewidth=0.8)
        # # q=1
        # plt.subplot(2, 3, 5)
        # plt.plot(wmaj1_w, Amaj1_w)
        # plt.plot(wksmaj1_w, Aksmaj1_w)
        # plt.plot(Amaj1.omega_w, Amaj1.A_w)
        # plt.axvline(wpeak01, c='0.5', linewidth=0.8)
        # # Plot full spectral weight of minority excitations
        # # q=0
        # plt.subplot(2, 3, 3)
        # plt.plot(wmin0_w, Amin0_w)
        # plt.plot(wksmin0_w, Aksmin0_w)
        # # q=1
        # plt.subplot(2, 3, 6)
        # plt.plot(wmin1_w, Amin1_w)
        # plt.plot(wksmin1_w, Aksmin1_w)
        # plt.show()

        # Print values
        print(Amaj0.omega_w[wm0], Amaj1.omega_w[wm1])
        print(hxc_scaling.lambd)
        print(wpeak00, wpeak01, wpeak10, wpeak11)
        print(Ipeak00, Ipeak01, Ipeak10, Ipeak11)
        print(mpeak00, mpeak01, mpeak10, mpeak11)
        print(Speak00, Speak01, Speak10, Speak11)
        print(mrpeak00, mrpeak01, mrpeak10, mrpeak11)
        print(Srpeak00, Srpeak01, Srpeak10, Srpeak11)
        print(enh0, enh1, min_enh0, min_enh1)

    # Test that the mode extraction frequency is indeed non-negative
    assert Amaj0.omega_w[wm0] >= 0.
    assert Amaj1.omega_w[wm1] >= 0.

    # Test kernel scaling
    assert hxc_scaling.lambd == pytest.approx(0.9675, abs=0.005)

    # Test magnon frequencies
    assert wpeak00 == pytest.approx(-0.0281, abs=0.005)
    assert wpeak01 == pytest.approx(0.218, abs=0.01)
    assert wpeak10 == pytest.approx(0.508, abs=0.01)
    assert wpeak11 == pytest.approx(0.637, abs=0.01)

    # Test magnon amplitudes
    assert Ipeak00 == pytest.approx(2.897, abs=0.01)
    assert Ipeak01 == pytest.approx(2.245, abs=0.01)
    assert Ipeak10 == pytest.approx(1.090, abs=0.01)
    assert Ipeak11 == pytest.approx(1.023, abs=0.01)

    # Test magnon frequency consistency
    assert mpeak00 == pytest.approx(wpeak00, abs=0.005)
    assert mpeak01 == pytest.approx(wpeak01, abs=0.01)
    assert mpeak10 == pytest.approx(wpeak10, abs=0.01)
    assert mpeak11 == pytest.approx(wpeak11, abs=0.01)
    assert mrpeak00 == pytest.approx(wpeak00, abs=0.005)
    assert mrpeak01 == pytest.approx(wpeak01, abs=0.01)
    assert mrpeak10 == pytest.approx(wpeak10, abs=0.01)
    assert mrpeak11 == pytest.approx(wpeak11, abs=0.01)

    # Test magnon mode eigenvalues at extrema
    assert Speak00 == pytest.approx(8.409, abs=0.02)
    assert Speak01 == pytest.approx(6.734, abs=0.02)
    assert Speak10 == pytest.approx(3.800, abs=0.02)
    assert Speak11 == pytest.approx(3.683, abs=0.02)
    assert Srpeak00 == pytest.approx(6.402, abs=0.02)
    assert Srpeak01 == pytest.approx(5.087, abs=0.02)
    assert Srpeak10 == pytest.approx(2.837, abs=0.02)
    assert Srpeak11 == pytest.approx(2.692, abs=0.02)

    # Test enhancement factors
    assert enh0 == pytest.approx(36.77, abs=0.1)
    assert enh1 == pytest.approx(24.10, abs=0.1)
    assert min_enh0 == pytest.approx(1.141, abs=0.01)
    assert min_enh1 == pytest.approx(1.162, abs=0.01)
