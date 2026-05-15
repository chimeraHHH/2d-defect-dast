"""Calculate the Heisenberg exchange constants in Fe and Co using the MFT.
Test with unrealisticly loose parameters to catch if the numerics change.
"""

# General modules
import pytest
import numpy as np

# Script modules
from gpaw import GPAW

from gpaw.response import ResponseGroundStateAdapter, ResponseContext
from gpaw.response.chiks import ChiKSCalculator
from gpaw.response.localft import LocalFTCalculator, LocalPAWFTCalculator
from gpaw.response.site_data import (AtomicSites,
                                     calculate_site_magnetization,
                                     calculate_site_zeeman_energy)
from gpaw.response.mft import (IsotropicExchangeCalculator,
                               HeisenbergExchangeCalculator,
                               calculate_single_particle_site_magnetization,
                               calculate_pair_site_magnetization,
                               calculate_single_particle_site_zeeman_energy,
                               calculate_pair_site_zeeman_energy,
                               calculate_exchange_parameters)
from gpaw.response.site_kernels import (SphericalSiteKernels,
                                        CylindricalSiteKernels,
                                        ParallelepipedicSiteKernels)
from gpaw.response.heisenberg import (calculate_single_site_magnon_energies,
                                      calculate_fm_magnon_energies)
from gpaw.test.gpwfile import response_band_cutoff
from gpaw.test.response.test_chiks import generate_qrel_q, get_q_c


@pytest.mark.response
@pytest.mark.kspair
def test_Fe_bcc(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    # MFT calculation
    ecut = 50
    # Do the high symmetry points of the bcc lattice
    q_qc = np.array([[0, 0, 0],           # Gamma
                     [0.5, -0.5, 0.5],    # H
                     [0.0, 0.0, 0.5],     # N
                     ])
    # Define site kernels to test
    # Test a single site of spherical and cylindrical geometries
    rc_pa = np.array([[1.0], [1.5], [2.0]])
    hc_pa = np.array([[1.0], [1.5], [2.0]])
    ez_pav = np.array([[[1., 0., 0.]], [[0., 1., 0.]], [[0., 0., 1.]]])

    # ---------- Script ---------- #

    # Extract the ground state fixture
    calc = GPAW(gpw_files['fe_pw'], parallel=dict(domain=1))
    nbands = response_band_cutoff['fe_pw']
    atoms = calc.atoms

    # Set up site kernels with a single site
    positions = atoms.get_positions()
    sitekernels = SphericalSiteKernels(positions, rc_pa)
    sitekernels.append(CylindricalSiteKernels(positions, ez_pav,
                                              rc_pa, hc_pa))
    # Set up a kernel to fill out the entire unit cell
    sitekernels.append(ParallelepipedicSiteKernels(positions,
                                                   [[atoms.get_cell()]]))

    # Initialize the Heisenberg exchange calculator
    gs = ResponseGroundStateAdapter(calc)
    context = ResponseContext()
    chiks_calc = ChiKSCalculator(gs, context,
                                 ecut=ecut, nbands=nbands, gammacentered=True)
    localft_calc = LocalFTCalculator.from_rshe_parameters(gs, context)
    isoexch_calc = IsotropicExchangeCalculator(chiks_calc, localft_calc)

    # Allocate array for the exchange constants
    nq = len(q_qc)
    nsites = sitekernels.nsites
    npartitions = sitekernels.npartitions
    J_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)

    # Calcualate the exchange constant for each q-point
    for q, q_c in enumerate(q_qc):
        J_qabp[q] = isoexch_calc(q_c, sitekernels)

    # Since we only have a single site, reduce the array
    J_qp = J_qabp[:, 0, 0, :]

    # Calculate the magnon energies
    mm = 2.21
    mm_ap = mm * np.ones((1, npartitions))  # Magnetic moments
    mw_qp = calculate_fm_magnon_energies(J_qabp, q_qc, mm_ap)[:, 0, :]

    # Compare results to test values
    test_J_pq = np.array(
        [[2.1907596825086455, 1.172424411323134, 1.6060583789867644],
         [2.612428039019977, 1.2193926800088601, 1.7635196888465006],
         [6.782367391186284, 0.2993922109834177, 1.9346016211386057],
         [1.5764800860123762, 0.8365204592352894, 1.1648584638500161],
         [2.4230224513213234, 1.2179759558303274, 1.6691805687218078],
         [5.35668502504496, 0.3801778545994659, 1.6948968244858478],
         [2.523580017606111, 1.21779750159267, 1.7637120466695273]])
    test_mw_pq = np.array(
        [[0.0, 0.9215703811633589, 0.5291414511510236],
         [0.0, 1.2606654832679791, 0.7682428508357253],
         [0.0, 5.866945864436984, 4.38711834393455],
         [0.0, 0.6696467210652369, 0.3725082553505521],
         [0.0, 1.0905398149239784, 0.682209848506349],
         [0.0, 4.503626398593207, 3.313835475619106],
         [0.0, 1.181703634401304, 0.6876633221145555]])

    # Exchange constants
    assert J_qp.imag == pytest.approx(0.0)
    assert J_qp.T.real == pytest.approx(test_J_pq, rel=2e-3)

    # Magnon energies
    assert mw_qp.T == pytest.approx(test_mw_pq, rel=2e-3)


@pytest.mark.response
@pytest.mark.kspair
def test_Co_hcp(in_tmp_dir, gpw_files):
    # ---------- Inputs ---------- #

    # MFT calculation
    ecut = 100
    # Do high symmetry points of the hcp lattice
    q_qc = np.array([[0, 0, 0],              # Gamma
                     [0.5, 0., 0.],          # M
                     [0., 0., 0.5]           # A
                     ])

    # Use spherical site kernels in a radius range which should yield
    # stable results
    rc_pa = np.array([[1.0, 1.0], [1.1, 1.1], [1.2, 1.2]])

    # Unfortunately, the usage of symmetry leads to such extensive repetition
    # of random noise, that one cannot trust individual values of J very well.
    # This is improved when increasing the number of k-points, but the problem
    # never completely vanishes
    J_atol = 1.e-2
    J_rtol = 5.e-2
    # However, derived physical values have an increased error cancellation due
    # to their collective nature.
    mw_rtol = 25e-3  # relative tolerance of absolute results
    mw_ctol = 5.e-2  # relative tolerance on kernel and eta self-consistency

    # ---------- Script ---------- #

    # Extract the ground state fixture
    calc = GPAW(gpw_files['co_pw'], parallel=dict(domain=1))
    nbands = response_band_cutoff['co_pw']
    atoms = calc.get_atoms()

    # Set up spherical site kernels
    positions = atoms.get_positions()
    sitekernels = SphericalSiteKernels(positions, rc_pa)

    # Set up a site kernel to fill out the entire unit cell
    cell_cv = atoms.get_cell()
    cc_v = np.sum(cell_cv, axis=0) / 2.  # Unit cell center
    ucsitekernels = ParallelepipedicSiteKernels([cc_v], [[cell_cv]])

    # Initialize the exchange calculator with and without symmetry
    gs = ResponseGroundStateAdapter(calc)
    context = ResponseContext()
    chiks_calc0 = ChiKSCalculator(gs, context, qsymmetry=False,
                                  ecut=ecut, nbands=nbands, gammacentered=True)
    localft_calc = LocalPAWFTCalculator(gs, context)
    isoexch_calc0 = IsotropicExchangeCalculator(chiks_calc0, localft_calc)
    chiks_calc1 = ChiKSCalculator(gs, context,
                                  ecut=ecut, nbands=nbands, gammacentered=True)
    isoexch_calc1 = IsotropicExchangeCalculator(chiks_calc1, localft_calc)

    # Allocate array for the spherical site exchange constants
    nq = len(q_qc)
    nsites = sitekernels.nsites
    npartitions = sitekernels.npartitions
    J_qabp = np.empty((nq, nsites, nsites, npartitions), dtype=complex)

    # Allocate array for the unit cell site exchange constants
    Juc_qs = np.empty((nq, 2), dtype=complex)

    # Calcualate the exchange constants for each q-point
    for q, q_c in enumerate(q_qc):
        J_qabp[q] = isoexch_calc0(q_c, sitekernels)
        chiksr_buffer = isoexch_calc0._chiksr
        Juc_qs[q, 0] = isoexch_calc0(q_c, ucsitekernels)[0, 0, 0]
        assert isoexch_calc0._chiksr is chiksr_buffer, \
            'Two subsequent IsotropicExchangeCalculator calls with the same '\
            'q_c, should reuse, not update, the chiks buffer'

        Juc_qs[q, 1] = isoexch_calc1(q_c, ucsitekernels)[0, 0, 0]

    # Calculate the magnon energy
    mom = atoms.get_magnetic_moment()
    mm_ap = mom / 2.0 * np.ones((nsites, npartitions))
    mw_qnp = calculate_fm_magnon_energies(J_qabp, q_qc, mm_ap)
    mw_qnp = np.sort(mw_qnp, axis=1)  # Make sure the eigenvalues are sorted
    mwuc_qs = calculate_single_site_magnon_energies(Juc_qs, q_qc, mom)

    # Compare results to test values
    print(J_qabp[..., 1], mw_qnp[..., 1], mwuc_qs[:, 0])
    test_J_qab = np.array([[[1.23106207 - 0.j, 0.25816335 - 0.j],
                            [0.25816335 + 0.j, 1.23106207 + 0.j]],
                           [[0.88823839 + 0.j, 0.07345416 - 0.04947835j],
                            [0.07345416 + 0.04947835j, 0.88823839 + 0.j]],
                           [[1.09349955 - 0.j, 0.00000010 - 0.01176761j],
                            [0.00000010 + 0.01176761j, 1.09349955 - 0.j]]])
    test_mw_qn = np.array([[0., 0.64793939],
                           [0.64304039, 0.86531921],
                           [0.48182997, 0.51136436]])
    test_mwuc_q = np.array([0., 0.69678659, 0.44825874])

    # Exchange constants
    assert J_qabp[..., 1] == pytest.approx(test_J_qab, abs=J_atol, rel=J_rtol)

    # Magnon energies
    assert np.all(np.abs(mw_qnp[0, 0, :]) < 1.e-8)  # Goldstone theorem
    assert np.allclose(mwuc_qs[0, :], 0.)  # Goldstone
    assert mw_qnp[1:, 0, 1] == pytest.approx(test_mw_qn[1:, 0], rel=mw_rtol)
    assert mw_qnp[:, 1, 1] == pytest.approx(test_mw_qn[:, 1], rel=mw_rtol)
    assert mwuc_qs[1:, 0] == pytest.approx(test_mwuc_q[1:], rel=mw_rtol)

    # Check self-consistency of results
    # We should be in a radius range, where the magnon energies don't change
    assert np.allclose(mw_qnp[1:, 0, ::2],
                       test_mw_qn[1:, 0, np.newaxis], rtol=mw_ctol)
    assert np.allclose(mw_qnp[:, 1, ::2],
                       test_mw_qn[:, 1, np.newaxis], rtol=mw_ctol)
    # Check that symmetry toggle do not change the magnon energies
    assert np.allclose(mwuc_qs[1:, 0], mwuc_qs[1:, 1], rtol=mw_ctol)


@pytest.mark.response
@pytest.mark.kspair
@pytest.mark.parametrize('qrel', generate_qrel_q())
def test_Co_site_magnetization_sum_rule(in_tmp_dir, gpw_files, qrel):
    # Set up ground state adapter and basic parameters
    calc = GPAW(gpw_files['co_pw'], parallel=dict(domain=1))
    gs = ResponseGroundStateAdapter(calc)
    sites = get_co_sites(gs)
    context = 'Co_sum_rule.txt'
    nbands = response_band_cutoff['co_pw']

    # Get wave vector to test
    q_c = get_q_c('co_pw', qrel)

    # Calculate site magnetization
    magmom_ar = calculate_site_magnetization(gs, sites)

    # ----- Single-particle site magnetization ----- #

    # Test that the single-particle site magnetization matches a conventional
    # calculation based on the density
    sp_magmom_ar = calculate_single_particle_site_magnetization(
        gs, sites, context=context)
    assert sp_magmom_ar == pytest.approx(magmom_ar, rel=5e-3)

    # ----- Two-particle site magnetization ----- #

    magmom_abr = calculate_pair_site_magnetization(
        gs, sites, context=context, q_c=q_c, nbands=nbands)

    # Test that the site pair magnetization is a positive-valued diagonal
    # real array
    tp_magmom_ra = magmom_abr.diagonal()
    assert np.all(tp_magmom_ra.real > 0)
    assert np.all(np.abs(tp_magmom_ra.imag) / tp_magmom_ra.real < 1e-6)
    assert np.all(np.abs(np.diagonal(np.fliplr(  # off-diagonal elements
        magmom_abr))) / tp_magmom_ra.real < 5e-2)

    # Test that the magnetic moments on the two Co atoms are identical
    tp_magmom_ar = magmom_abr.diagonal().T.real
    assert tp_magmom_ar[0] == pytest.approx(tp_magmom_ar[1], rel=1e-4)

    # Test that the result more or less matches a conventional calculation at
    # close-packing
    assert np.average(tp_magmom_ar, axis=0)[-1] == pytest.approx(
        np.average(magmom_ar, axis=0)[-1], rel=5e-2)

    # Test values against reference
    print(np.average(tp_magmom_ar, axis=0)[::2])
    assert np.average(tp_magmom_ar, axis=0)[::2] == pytest.approx(
        np.array([3.91823444e-04, 1.45641911e-01, 6.85939109e-01,
                  1.18813171e+00, 1.49761591e+00, 1.58954270e+00]), rel=5e-2)

    # import matplotlib.pyplot as plt
    # from ase.units import Bohr
    # rc_r = sites.rc_ap[0] * Bohr
    # plt.plot(rc_r, magmom_ar[0], '-o', mec='k')
    # plt.plot(rc_r, sp_magmom_ar[0], '-o', mec='k', zorder=0)
    # plt.plot(rc_r, tp_magmom_ar[0], '-o', mec='k', zorder=1)
    # plt.xlabel(r'$r_\mathrm{c}$ [$\mathrm{\AA}$]')
    # plt.ylabel(r'$m$ [$\mu_\mathrm{B}$]')
    # plt.title(str(q_c))
    # plt.show()


@pytest.mark.response
@pytest.mark.kspair
@pytest.mark.parametrize('qrel', generate_qrel_q())
def test_Co_site_zeeman_energy_sum_rule(in_tmp_dir, gpw_files, qrel):
    # Set up ground state adapter and atomic site data
    calc = GPAW(gpw_files['co_pw'], parallel=dict(domain=1))
    gs = ResponseGroundStateAdapter(calc)
    sites = get_co_sites(gs)
    context = ResponseContext('Co_sum_rule.txt')
    nbands = response_band_cutoff['co_pw']

    # Get wave vector to test
    q_c = get_q_c('co_pw', qrel)

    # Calculate the site Zeeman energy
    EZ_ar = calculate_site_zeeman_energy(gs, sites)

    # ----- Single-particle site Zeeman energy ----- #

    # Test that the results match a conventional calculation
    sp_EZ_ar = calculate_single_particle_site_zeeman_energy(
        gs, sites, context=context)
    assert sp_EZ_ar == pytest.approx(EZ_ar, rel=5e-3)

    # ----- Two-particle site Zeeman energy ----- #

    EZ_abr = calculate_pair_site_zeeman_energy(
        gs, sites, context=context, q_c=q_c, nbands=nbands)

    # Test that the pair site Zeeman energy is a positive-valued diagonal
    # real array
    tp_EZ_ra = EZ_abr.diagonal()
    assert np.all(tp_EZ_ra.real > 0)
    assert np.all(np.abs(tp_EZ_ra.imag) / tp_EZ_ra.real < 1e-4)
    assert np.all(np.abs(np.diagonal(np.fliplr(  # off-diagonal elements
        EZ_abr))) / tp_EZ_ra.real < 5e-2)

    # Test that the Zeeman energy on the two Co atoms is identical
    tp_EZ_ar = EZ_abr.diagonal().T.real
    assert tp_EZ_ar[0] == pytest.approx(tp_EZ_ar[1], rel=1e-4)

    # Test values against reference
    print(np.average(tp_EZ_ar, axis=0)[::2])
    assert np.average(tp_EZ_ar, axis=0)[::2] * 2. == pytest.approx(
        np.array([3.68344584e-04, 3.13780575e-01, 1.35409600e+00,
                  2.14237563e+00, 2.52032513e+00, 2.61406726e+00]), rel=5e-2)

    # Test ability to distribute band and spin transitions
    if context.comm.size > 1:
        assert calculate_pair_site_zeeman_energy(
            gs, sites, context=context,
            q_c=q_c, nbands=nbands, nblocks='max') == pytest.approx(EZ_abr)

    # import matplotlib.pyplot as plt
    # from ase.units import Bohr
    # rc_r = sites.rc_ap[0] * Bohr
    # plt.plot(rc_r, EZ_ar[0], '-o', mec='k')
    # plt.plot(rc_r, sp_EZ_ar[0], '-o', mec='k', zorder=0)
    # plt.plot(rc_r, tp_EZ_ar[0], '-o', mec='k', zorder=1)
    # plt.xlabel(r'$r_\mathrm{c}$ [$\mathrm{\AA}$]')
    # plt.ylabel(r'$E_\mathrm{Z}$ [eV]')
    # plt.title(str(q_c))
    # plt.show()


def get_Co_exchange_reference(qrel):
    if qrel == 0.:
        return np.array([0.186, 0.731, 1.048, 1.14, 1.145])
    elif qrel == 0.25:
        return np.array([0.166, 0.656, 0.943, 1.029, 1.035])
    elif qrel == 0.5:
        return np.array([0.151, 0.594, 0.856, 0.936, 0.943])
    raise ValueError(qrel)


@pytest.mark.response
@pytest.mark.kspair
@pytest.mark.parametrize('qrel', generate_qrel_q())
def test_Co_exchange(in_tmp_dir, gpw_files, qrel):
    # Set up ground state adapter and atomic site data
    calc = GPAW(gpw_files['co_pw'], parallel=dict(domain=1))
    gs = ResponseGroundStateAdapter(calc)
    context = ResponseContext('Co_exchange.txt')
    sites = get_co_sites(gs)
    nbands = response_band_cutoff['co_pw']

    # Get wave vector to test
    q_c = get_q_c('co_pw', qrel)

    # Calculate the exchange parameters
    J_abr = calculate_exchange_parameters(
        gs, sites, q_c, context=context, nbands=nbands, nblocks=1)

    # Test that J is hermitian in (a,b)
    assert np.transpose(J_abr, (1, 0, 2)).conj() == pytest.approx(J_abr)
    # Test the Co site symmetry
    J_ar = J_abr.diagonal().T
    assert J_ar[0] == pytest.approx(J_ar[1], rel=1e-4)
    # Test against reference values
    Jref_r = get_Co_exchange_reference(qrel)
    assert J_ar[0, 2::2] == pytest.approx(Jref_r, abs=2e-3)

    # Test ability to distribute band and spin transitions
    if context.comm.size > 1:
        assert calculate_exchange_parameters(
            gs, sites, q_c, context=context, nbands=nbands,
            nblocks='max') == pytest.approx(J_abr)

    # Calculate the magnon energy at the Gamma-point
    if qrel == 0.:
        mom = calc.get_atoms().get_magnetic_moment()
        mm_ar = mom / 2. * np.ones(J_ar.shape)
        mw_nr = calculate_fm_magnon_energies(
            np.array([J_abr]), np.array([q_c]), mm_ar)[0]
        mw_nr = np.sort(mw_nr, axis=0)  # Make sure the eigenvalues are sorted

        assert mw_nr[0] == pytest.approx(0.)  # Goldstone theorem
        assert mw_nr[1, 2::2] == pytest.approx(np.array(
            [0.092, 0.358, 0.504, 0.539, 0.539]), abs=1e-3)

        # import matplotlib.pyplot as plt
        # from ase.units import Bohr
        # rc_r = sites.rc_ap[0] * Bohr
        # for n, mw_r in enumerate(mw_nr):
        #     plt.plot(rc_r, mw_r, '-o', mec='k', label=str(n))
        # plt.legend()
        # plt.xlabel(r'$r_\mathrm{c}$ [$\mathrm{\AA}$]')
        # plt.ylabel(r'$\hbar\omega$ [eV]')
        # plt.title(str(q_c))
        # plt.show()


@pytest.mark.response
@pytest.mark.kspair
@pytest.mark.parallel
def test_heisenberg_distribution_over_transitions(in_tmp_dir, gpw_files):
    # Set up ground state adapter and atomic site data
    calc = GPAW(gpw_files['co_pw'], parallel=dict(domain=1))
    gs = ResponseGroundStateAdapter(calc)
    sites = get_co_sites(gs)

    # To properly test the distributions over transitions, we need a value of
    # nbands, which produces a number of band and spin transitions, which isn't
    # divisible by the number of blocks.
    nbands = response_band_cutoff['co_pw'] - 3
    context = ResponseContext('distributed.txt')
    calc = HeisenbergExchangeCalculator(
        gs, sites, context=context, nbands=nbands, nblocks='max')
    assert context.comm.size % 2 == 0
    assert len(calc.transitions) % 2 > 0
    J_abr = calc(q_c=[0, 0, 0]).array

    # Test that the result doesn't depend on nblocks
    context.new_txt_and_timer('undistributed.txt')
    ref_calc = HeisenbergExchangeCalculator(
        gs, sites, context=context, nbands=nbands, nblocks=1)
    assert J_abr == pytest.approx(ref_calc(q_c=[0, 0, 0]).array)
    assert np.all(np.abs(J_abr) > 1e-10)  # check that some actual work is done


# ---------- Test functionality ---------- #


def get_co_sites(gs):
    from gpaw.response.site_data import get_site_radii_range
    # Set up site radii
    rmin_a, _ = get_site_radii_range(gs)
    # Make sure that the two sites do not overlap
    nn_dist = min(2.5071, np.sqrt(2.5071**2 / 3 + 4.0695**2 / 4))
    rc_r = np.linspace(rmin_a[0], nn_dist / 2, 11)
    return AtomicSites(indices=[0, 1], radii=[rc_r, rc_r])
