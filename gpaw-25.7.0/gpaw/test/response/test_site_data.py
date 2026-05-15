import pytest

import numpy as np

from ase.spacegroup import crystal
from ase.units import Bohr

from gpaw import GPAW
from gpaw.sphere.integrate import integrate_lebedev

from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.site_data import (AtomicSites, AtomicSiteData,
                                     calculate_site_magnetization,
                                     calculate_site_zeeman_energy,
                                     get_site_radii_range,
                                     maximize_site_magnetization)
from gpaw.response.localft import add_spin_polarization


@pytest.mark.response
def test_Fe_site_magnetization(gpw_files):
    # Set up ground state adapter
    calc = GPAW(gpw_files['fe_pw'], parallel=dict(domain=1))
    gs = ResponseGroundStateAdapter(calc)

    # Extract valid site radii range
    rmin_a, rmax_a = get_site_radii_range(gs)
    rmin = rmin_a[0]  # Only one magnetic atom in the unit cell
    rmax = rmax_a[0]
    # We expect rmax to be equal to the nearest neighbour distance
    # subtracted with the augmentation sphere radius. For a bcc lattice,
    # nn_dist = sqrt(3) a / 2:
    augr = gs.get_aug_radii()[0]
    rmax_expected = np.sqrt(3) * 2.867 / 2. - augr * Bohr
    assert abs(rmax - rmax_expected) < 1e-6
    # Test that an error is raised outside the valid range
    with pytest.raises(AssertionError):
        AtomicSiteData(
            gs, AtomicSites(indices=[0],  # Too small radii
                            radii=[np.linspace(rmin * 0.8, rmin, 5)]))
    with pytest.raises(AssertionError):
        AtomicSiteData(
            gs, AtomicSites(indices=[0],  # Too large radii
                            radii=[np.linspace(rmax, rmax * 1.2, 5)]))
    # Define atomic sites to span the valid range
    rc_r = np.linspace(rmin_a[0], rmax_a[0], 100)
    # Add the radius of the augmentation sphere explicitly
    rc_r = np.append(rc_r, [augr * Bohr])
    sites = AtomicSites(indices=[0], radii=[rc_r])
    site_data = AtomicSiteData(gs, sites)

    # Calculate site magnetization
    magmom_ar = site_data.calculate_magnetic_moments()
    magmom_r = magmom_ar[0]

    # Test that a cutoff at the augmentation sphere radius reproduces
    # the local magnetic moment of the GPAW calculation
    magmom_at_augr = calc.get_atoms().get_magnetic_moments()[0]
    assert abs(magmom_r[-1] - magmom_at_augr) < 4e-2

    # Do a manual calculation of the magnetic moment using the
    # all-electron partial waves
    # Calculate all-electron m(r)
    micro_setup = site_data.micro_setup_a[0]
    m_ng = np.array([micro_setup.rgd.zeros()
                     for n in range(micro_setup.Y_nL.shape[0])])
    for n, Y_L in enumerate(micro_setup.Y_nL):
        n_sg = np.dot(Y_L, micro_setup.n_sLg)
        add_spin_polarization(micro_setup.rgd, n_sg, m_ng[n, :])
    # Integrate with varrying radii
    m_g = integrate_lebedev(m_ng)
    ae_magmom_r = np.array([
        micro_setup.rgd.integrate_trapz(m_g, rcut=rcut / Bohr)
        for rcut in rc_r])
    # Test that values match approximately inside the augmentation sphere
    inaug_r = rc_r <= augr * Bohr
    assert magmom_r[inaug_r] == pytest.approx(ae_magmom_r[inaug_r], abs=3e-2)

    # import matplotlib.pyplot as plt
    # plt.plot(rc_r[:-1], magmom_r[:-1])
    # plt.plot(rc_r[:-1], ae_magmom_r[:-1], zorder=0)
    # plt.axvline(augr * Bohr, c='0.5', linestyle='--')
    # plt.xlabel(r'$r_\mathrm{c}$ [$\mathrm{\AA}$]')
    # plt.ylabel(r'$m$ [$\mu_\mathrm{B}$]')
    # plt.show()


@pytest.mark.response
def test_Co_site_data(gpw_files):
    # Set up ground state adapter
    calc = GPAW(gpw_files['co_pw'], parallel=dict(domain=1))
    gs = ResponseGroundStateAdapter(calc)

    # Extract valid site radii range
    rmin_a, rmax_a = get_site_radii_range(gs)
    # The valid ranges should be equal due to symmetry
    assert abs(rmin_a[1] - rmin_a[0]) < 1e-8
    assert abs(rmax_a[1] - rmax_a[0]) < 1e-8
    rmin = rmin_a[0]
    rmax = rmax_a[0]
    # We expect rmax to be equal to the nearest neighbour distance
    # subtracted with the augmentation sphere radius. For the hcp-lattice,
    # nn_dist = min(a, sqrt(a^2/3 + c^2/4)):
    augr_a = gs.get_aug_radii()
    assert abs(augr_a[1] - augr_a[0]) < 1e-8
    augr = augr_a[0]
    rmax_expected = min(2.5071, np.sqrt(2.5071**2 / 3 + 4.0695**2 / 4))
    rmax_expected -= augr * Bohr
    assert abs(rmax - rmax_expected) < 1e-6

    # Use radii spanning the entire valid range
    rc_r = np.linspace(rmin, rmax, 101)
    # Add the radius of the augmentation sphere explicitly
    rc_r = np.append(rc_r, [augr * Bohr])
    nr = len(rc_r)
    # Varry the site radii together and independently
    rc1_r = list(rc_r) + list(rc_r) + [augr * Bohr] * nr
    rc2_r = list(rc_r) + [augr * Bohr] * nr + list(rc_r)
    sites = AtomicSites(indices=[0, 1], radii=[rc1_r, rc2_r])

    # Calculate site magnetization
    magmom_ar = calculate_site_magnetization(gs, sites)

    # Test that the magnetization inside the augmentation sphere matches
    # the local magnetic moment of the GPAW calculation
    magmom_at_augr_a = calc.get_atoms().get_magnetic_moments()
    assert magmom_ar[:, -1] == pytest.approx(magmom_at_augr_a, abs=2e-2)

    # Test consistency of varrying radii
    assert magmom_ar[0, :nr] == pytest.approx(magmom_ar[1, :nr])
    assert magmom_ar[0, nr:2 * nr] == pytest.approx(magmom_ar[0, :nr])
    assert magmom_ar[0, 2 * nr:] == pytest.approx([magmom_ar[0, -1]] * nr)
    assert magmom_ar[1, nr:2 * nr] == pytest.approx([magmom_ar[1, -1]] * nr)
    assert magmom_ar[1, 2 * nr:] == pytest.approx(magmom_ar[1, :nr])

    # Calculate the maximized site magnetization
    rm_a, mm_a = maximize_site_magnetization(gs)
    # Test radius consistency
    assert rm_a[0] == pytest.approx(rm_a[1])  # Co site symmetry
    assert np.average(rm_a) == pytest.approx(1.133357)  # reference value
    # Test moment consistency
    assert mm_a[0] == pytest.approx(mm_a[1])  # Co site symmetry
    assert np.average(mm_a) == pytest.approx(1.6362)  # reference value
    assert np.max(magmom_ar) < np.average(mm_a) < np.max(magmom_ar) * 1.01

    # Calculate the atomic Zeeman energy
    rc_r = rc_r[:-1]
    sites = AtomicSites(indices=[0, 1], radii=[rc_r, rc_r])
    EZ_ar = calculate_site_zeeman_energy(gs, sites)
    print(EZ_ar[0, ::20])

    # Test that the Zeeman energy comes out as expected
    assert EZ_ar[0] == pytest.approx(EZ_ar[1])
    assert EZ_ar[0, ::20] * 2 == pytest.approx([0.02638351, 1.41476112,
                                                2.49540004, 2.79727200,
                                                2.82727948, 2.83670767],
                                               rel=1e-3)

    # import matplotlib.pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.plot(rc_r, magmom_ar[0, :nr - 1])
    # plt.axvline(np.average(rm_a), linestyle=':')
    # plt.axvline(augr * Bohr, c='0.5', linestyle='--')
    # plt.xlabel(r'$r_\mathrm{c}$ [$\mathrm{\AA}$]')
    # plt.ylabel(r'$m$ [$\mu_\mathrm{B}$]')
    # plt.subplot(1, 2, 2)
    # plt.plot(rc_r, EZ_ar[0])
    # plt.axvline(np.average(rm_a), linestyle=':')
    # plt.axvline(augr * Bohr, c='0.5', linestyle='--')
    # plt.xlabel(r'$r_\mathrm{c}$ [$\mathrm{\AA}$]')
    # plt.ylabel(r'$E_\mathrm{Z}$ [eV]')
    # plt.show()


@pytest.mark.response
def test_valid_site_radii_symmetry():
    # Set up Cr2O3 crystal
    cellpar = [4.95721, 4.95721, 13.59170, 90, 90, 120]
    Cr_c = [0, 0, 0.34734]
    O_c = [0.30569, 0.0, 0.25]
    spos_ac = [Cr_c, O_c]
    atoms = crystal('CrO',
                    spacegroup=167,
                    cellpar=cellpar,
                    basis=spos_ac,
                    primitive_cell=True,
                    pbc=True)
    # from ase.visualize import view
    # view(atoms)

    # Set up calculator with a specific grid spacing and generate adapter
    spacing = 0.1
    calc = GPAW(mode='fd', h=spacing)
    calc.initialize(atoms)
    gs = DummyAdapter(calc)

    # Generate valid site radii range
    rmin_A, rmax_A = get_site_radii_range(gs)
    # Test that the minimum radius corresponds loosely to the specified
    # spacing. The correspondance would be exact for a cubic cell.
    assert rmin_A == pytest.approx(np.ones(len(atoms)) * spacing / 2, rel=0.2)
    # Test that all Cr and O atoms result in symmetrically equivalent maximum
    # cutoff radii
    CrO_dist = 1.966
    Cr_aug_radius = 2.3 * Bohr
    O_aug_radius = 1.3 * Bohr
    is_Cr = np.array([c == 'Cr' for c in atoms.get_chemical_symbols()])
    refmax_A = np.empty(len(atoms))
    refmax_A[is_Cr] = CrO_dist - O_aug_radius
    refmax_A[~is_Cr] = CrO_dist - Cr_aug_radius
    assert rmax_A == pytest.approx(refmax_A, abs=0.001)


class DummyAdapter(ResponseGroundStateAdapter):
    def __init__(self, calc):
        from gpaw.response.groundstate import PAWDatasetCollection
        self.atoms = calc.atoms
        self.gd = calc.wfs.gd
        self.pawdatasets = PAWDatasetCollection(calc.setups)
