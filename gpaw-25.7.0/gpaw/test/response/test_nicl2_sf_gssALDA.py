import numpy as np
import pytest

from gpaw.mpi import world
from gpaw.test import findpeak

from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.chiks import ChiKSCalculator
from gpaw.response.fxc_kernels import AdiabaticFXCCalculator
from gpaw.response.dyson import HXCKernel
from gpaw.response.goldstone import FMGoldstoneScaling
from gpaw.response.susceptibility import ChiFactory
from gpaw.response.pair_functions import read_pair_function


pytestmark = pytest.mark.skipif(world.size < 4,
                                reason='too slow for world.size < 4')


@pytest.mark.kspair
@pytest.mark.response
@pytest.mark.old_gpaw_only
def test_nicl2_magnetic_response(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    q_qc = [[0., 0., 0.],
            [1. / 3., 1. / 3., 0.]]
    fxc = 'ALDA'
    rshelmax = 0
    rshewmin = None
    bg_density = 0.004
    ecut = 200
    frq_w = np.linspace(-0.15, 0.075, 16)
    eta = 0.12
    zd = ComplexFrequencyDescriptor.from_array(frq_w + 1.j * eta)
    nblocks = 4

    # ---------- Script ---------- #

    # Magnetic response calculation
    gs = ResponseGroundStateAdapter.from_gpw_file(gpw_files['nicl2_pw'])
    chiks_calc = ChiKSCalculator(gs,
                                 ecut=ecut,
                                 gammacentered=True,
                                 nblocks=nblocks)

    # Calculate the magnetic response with and without a background density
    hxc_scaling = FMGoldstoneScaling()
    fxc_calculator = AdiabaticFXCCalculator.from_rshe_parameters(
        gs, chiks_calc.context,
        rshelmax=rshelmax,
        rshewmin=rshewmin)
    chi_factory = ChiFactory(chiks_calc, fxc_calculator)
    bgd_hxc_scaling = FMGoldstoneScaling()
    bgd_fxc_calculator = AdiabaticFXCCalculator.from_rshe_parameters(
        gs, chiks_calc.context,
        bg_density=bg_density,
        rshelmax=rshelmax,
        rshewmin=rshewmin)

    # Check that the pseudo-density is nonnegative
    nt_sr, gd = gs.get_pseudo_density(gridrefinement=2)
    assert np.all(nt_sr >= 0.)

    # Plot pseudo spin-density to check exponential localization into vacuum
    # if world.rank == 0:
    #     import matplotlib.pyplot as plt
    #     N_c = gd.N_c
    #     # Plot the spin-densities above one of the Cl atoms
    #     plt.plot(range(N_c[2]), nt_sr[0, 2 * N_c[0] // 3, N_c[1] // 3])
    #     plt.plot(range(N_c[2]), nt_sr[1, 2 * N_c[0] // 3, N_c[1] // 3])
    #     plt.yscale('log')
    #     plt.show()

    filestr = 'nicl2_macro_tms'
    bgd_filestr = 'nicl2_macro_tms_bgd'
    for q, q_c in enumerate(q_qc):
        chiks, chi = chi_factory('+-', q_c, zd,
                                 fxc=fxc,
                                 hxc_scaling=hxc_scaling,
                                 txt=filestr + '_q%d.txt' % q)
        chi.write_macroscopic_component(filestr + '_q%d.csv' % q)
        if q == 0:
            # Calculate kernel with background charge
            bgd_hxc_kernel = HXCKernel(
                hartree_kernel=chi_factory.get_hartree_kernel(
                    '+-', chiks.qpd),
                xc_kernel=bgd_fxc_calculator(fxc, '+-', chiks.qpd))
        bgd_chi = chi_factory.dyson_solver(
            chiks, bgd_hxc_kernel, bgd_hxc_scaling)
        bgd_chi.write_macroscopic_component(bgd_filestr + '_q%d.csv' % q)

    chiks_calc.context.write_timer()
    world.barrier()

    # Compare new results to test values
    check_magnons(filestr, hxc_scaling,
                  test_fxcs=0.7130,
                  test_mw0=-10.3,  # meV
                  test_mw1=-40.9,  # meV
                  test_Ipeak0=0.2306,  # a.u.
                  test_Ipeak1=0.0956,  # a.u.
                  )
    check_magnons(bgd_filestr, bgd_hxc_scaling,
                  test_fxcs=1.0826,
                  test_mw0=-10.2,  # meV
                  test_mw1=-48.0,  # meV
                  test_Ipeak0=0.2321,  # a.u.
                  test_Ipeak1=0.0956,  # a.u.
                  )


def check_magnons(filestr, hxc_scaling, *,
                  test_fxcs, test_mw0, test_mw1, test_Ipeak0, test_Ipeak1):
    # Identify magnon peaks and extract kernel scaling
    w0_w, chi0_w = read_pair_function(filestr + '_q0.csv')
    w1_w, chi1_w = read_pair_function(filestr + '_q1.csv')

    wpeak0, Ipeak0 = findpeak(w0_w, -chi0_w.imag / np.pi)
    wpeak1, Ipeak1 = findpeak(w1_w, -chi1_w.imag / np.pi)
    mw0 = wpeak0 * 1e3  # meV
    mw1 = wpeak1 * 1e3  # meV

    assert hxc_scaling.lambd is not None
    fxcs = hxc_scaling.lambd

    if world.rank == 0:
        # import matplotlib.pyplot as plt
        # plt.plot(w0_w, -chi0_w.imag / np.pi)
        # plt.plot(w1_w, -chi1_w.imag / np.pi)
        # plt.show()

        print(fxcs, mw0, mw1, Ipeak0, Ipeak1)

    # Test fxc scaling
    assert fxcs == pytest.approx(test_fxcs, abs=0.005)

    # Test magnon peaks
    assert mw0 == pytest.approx(test_mw0, abs=5.0)
    assert mw1 == pytest.approx(test_mw1, abs=10.0)

    # Test peak intensities
    assert Ipeak0 == pytest.approx(test_Ipeak0, abs=0.01)
    assert Ipeak1 == pytest.approx(test_Ipeak1, abs=0.01)
