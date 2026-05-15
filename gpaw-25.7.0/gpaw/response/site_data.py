import numpy as np

from ase.units import Bohr, Hartree
from ase.neighborlist import natural_cutoffs, build_neighbor_list

from gpaw.sphere.integrate import (integrate_lebedev,
                                   radial_truncation_function,
                                   spherical_truncation_function_collection,
                                   default_spherical_drcut,
                                   find_volume_conserving_lambd)
from gpaw.response import (ResponseGroundStateAdapter,
                           ResponseGroundStateAdaptable)
from gpaw.response.localft import (add_spin_polarization,
                                   add_LSDA_zeeman_energy)


class AtomicSites:
    """Object defining a set of spherical atomic sites."""

    def __init__(self, indices, radii):
        """Construct the AtomicSites.

        Parameters
        ----------
        indices : 1D array-like
            Atomic index A for each site index a.
        radii : 2D array-like
            Atomic radius rc for each site index a and partitioning p.
        """
        self.A_a = np.asarray(indices)
        assert self.A_a.ndim == 1
        assert len(np.unique(self.A_a)) == len(self.A_a)

        # Parse the input atomic radii
        rc_ap = np.asarray(radii)
        assert rc_ap.ndim == 2
        assert rc_ap.shape[0] == len(self.A_a)
        # Convert radii to internal units (Å to Bohr)
        self.rc_ap = rc_ap / Bohr

        self.npartitions = self.rc_ap.shape[1]
        self.shape = rc_ap.shape

    def __len__(self):
        return len(self.A_a)


def calculate_site_magnetization(
        gs: ResponseGroundStateAdaptable,
        sites: AtomicSites):
    """Calculate the site magnetization.

    Returns
    -------
    magmom_ap : np.ndarray
        Magnetic moment in μB of site a under partitioning p, calculated
        directly from the ground state density.
    """
    return AtomicSiteData(gs, sites).calculate_magnetic_moments()


def calculate_site_zeeman_energy(
        gs: ResponseGroundStateAdaptable,
        sites: AtomicSites):
    """Calculate the site Zeeman energy.

    Returns
    -------
    EZ_ap : np.ndarray
        Local Zeeman energy in eV of site a under partitioning p, calculated
        directly from the ground state density.
    """
    site_data = AtomicSiteData(gs, sites)
    return site_data.calculate_zeeman_energies() * Hartree  # Ha -> eV


def get_site_radii_range(gs):
    """Get the range of valid site radii for the atoms of a given ground state.

    Returns
    -------
    rmin_A : np.ndarray
        Minimum cutoff radius in Å for each atom A.
    rmax_A : np.ndarray
        Maximum cutoff radius in Å for each atom A.
    """
    rmin_A, rmax_A = AtomicSiteData.valid_site_radii_range(gs)
    return rmin_A * Bohr, rmax_A * Bohr  # Bohr -> Å


def maximize_site_magnetization(gs, indices=None):
    """Find the allowed site radii which maximize the site magnetization.

    Assumes that m(rc) is maximized for some rc belonging to the interior of
    the allowed cutoff radii for each atom. Physically, m(rc) has such a
    maximum only if the spin-polarization of the interstitial region is
    anti-parallel to the site in its near vicinity.

    Returns
    -------
    rmax_a : np.ndarray
        Cutoff radius in Å, maximizing the site magnetization for each site a.
    mmax_a : np.ndarray
        Site magnetization in μB at its maximum for each site a.
    """
    # Calculate the site magnetization as a function of radius
    rmin_A, rmax_A = get_site_radii_range(gs)
    if indices is None:
        indices = range(len(rmin_A))
    rc_ar = [np.linspace(rmin_A[A], rmax_A[A], 201) for A in indices]
    magmom_ar = calculate_site_magnetization(gs, AtomicSites(indices, rc_ar))
    # Maximize the site magnetization
    rmax_a = np.empty(len(indices), dtype=float)
    mmax_a = np.empty(len(indices), dtype=float)
    for a, (rc_r, magmom_r) in enumerate(zip(rc_ar, magmom_ar)):
        rmax_a[a], mmax_a[a] = maximize(rc_r, magmom_r)
    return rmax_a, mmax_a


def maximize(x_x, f_x):
    """Maximize f(x) on the given interval (returning xmax and f(xmax)).

    If there is no local maximum on the interior of the interval,
    we return np.nan.
    """
    from gpaw.test import findpeak
    xmax = f_x.argmax()
    if xmax == 0 or xmax == len(x_x) - 1:
        return np.nan, np.nan
    return findpeak(x_x, f_x)


class AtomicSiteData:
    r"""Data object for a set of spherical atomic sites."""

    def __init__(self, gs: ResponseGroundStateAdaptable, sites: AtomicSites):
        """Extract atomic site data from a given ground state."""
        gs = ResponseGroundStateAdapter.from_input(gs)
        assert self.in_valid_site_radii_range(gs, sites), \
            'Please provide site radii in the valid range, see '\
            'gpaw.response.site_data.get_site_radii_range()'
        self.sites = sites

        # Extract the scaled positions and micro_setups for each atomic site
        self.spos_ac = gs.spos_ac[sites.A_a]
        self.micro_setup_a = [gs.micro_setups[A] for A in sites.A_a]

        # Extract pseudo density on the fine real-space grid
        self.finegd = gs.finegd
        self.nt_sr = gs.nt_sr

        # Set up the atomic truncation functions which define the sites based
        # on the coarse real-space grid
        self.gd = gs.gd
        self.drcut = default_spherical_drcut(self.gd)
        self.lambd_ap = np.array(
            [[find_volume_conserving_lambd(rcut, self.drcut)
              for rcut in rc_p] for rc_p in sites.rc_ap])
        self.stfc = spherical_truncation_function_collection(
            self.finegd, self.spos_ac, sites.rc_ap, self.drcut, self.lambd_ap)

    @staticmethod
    def valid_site_radii_range(gs):
        """For each atom in gs, determine the valid site radii range in Bohr.

        The lower bound is determined by the spherical truncation width, when
        truncating integrals on the real-space grid.
        The upper bound is determined by the distance to the nearest
        augmentation sphere.
        """
        atoms = gs.atoms
        drcut = default_spherical_drcut(gs.gd)
        rmin_A = np.array([drcut / 2] * len(atoms))

        # Find neighbours based on covalent radii
        cutoffs = natural_cutoffs(atoms, mult=2)
        neighbourlist = build_neighbor_list(
            atoms, cutoffs, self_interaction=False, bothways=True)
        # Determine rmax for each atom
        augr_A = gs.get_aug_radii()
        rmax_A = []
        for A in range(len(atoms)):
            pos = atoms.positions[A]
            # Calculate the distance to the augmentation sphere of each
            # neighbour
            aug_distances = []
            for An, offset in zip(*neighbourlist.get_neighbors(A)):
                posn = atoms.positions[An] + offset @ atoms.get_cell()
                dist = np.linalg.norm(posn - pos) / Bohr  # Å -> Bohr
                aug_dist = dist - augr_A[An]
                assert aug_dist > 0.
                aug_distances.append(aug_dist)
            # In order for PAW corrections to be valid, we need a sphere of
            # radius rcut not to overlap with any neighbouring augmentation
            # spheres
            rmax_A.append(min(aug_distances))
        rmax_A = np.array(rmax_A)

        return rmin_A, rmax_A

    @staticmethod
    def in_valid_site_radii_range(gs, sites):
        rmin_A, rmax_A = AtomicSiteData.valid_site_radii_range(gs)
        for a, A in enumerate(sites.A_a):
            if not np.all(
                    np.logical_and(
                        sites.rc_ap[a] > rmin_A[A] - 1e-8,
                        sites.rc_ap[a] < rmax_A[A] + 1e-8)):
                return False
        return True

    def calculate_magnetic_moments(self):
        """Calculate the magnetic moments at each atomic site."""
        magmom_ap = self.integrate_local_function(add_spin_polarization)
        return magmom_ap

    def calculate_zeeman_energies(self):
        r"""Calculate the local Zeeman energy E_Z for each atomic site."""
        EZ_ap = self.integrate_local_function(add_LSDA_zeeman_energy)
        return EZ_ap

    def integrate_local_function(self, add_f):
        r"""Integrate a local function f[n](r) = f(n(r)) over the atomic sites.

        For every site index a and partitioning p, the integral is defined via
        a smooth truncation function θ(|r-r_a|<rc_ap):

               /
        f_ap = | dr θ(|r-r_a|<rc_ap) f(n(r))
               /
        """
        out_ap = np.zeros(self.sites.shape, dtype=float)
        self._integrate_pseudo_contribution(add_f, out_ap)
        self._integrate_paw_correction(add_f, out_ap)
        return out_ap

    def _integrate_pseudo_contribution(self, add_f, out_ap):
        """Calculate the pseudo contribution to the atomic site integrals.

        For local functions of the density, the pseudo contribution is
        evaluated by a numerical integration on the real-space grid:

        ̰       /
        f_ap = | dr θ(|r-r_a|<rc_ap) f(ñ(r))
               /
        """
        # Evaluate the local function on the real-space grid
        ft_r = self.finegd.zeros()
        add_f(self.finegd, self.nt_sr, ft_r)

        # Integrate θ(|r-r_a|<rc_ap) f(ñ(r))
        ftdict_ap = {a: np.empty(self.sites.npartitions)
                     for a in range(len(self.sites))}
        self.stfc.integrate(ft_r, ftdict_ap)

        # Add pseudo contribution to the output array
        for a in range(len(self.sites)):
            out_ap[a] += ftdict_ap[a]

    def _integrate_paw_correction(self, add_f, out_ap):
        """Calculate the PAW correction to an atomic site integral.

        The PAW correction is evaluated on the atom centered radial grid, using
        the all-electron and pseudo densities generated from the partial waves:

                /
        Δf_ap = | r^2 dr θ(r<rc_ap) [f(n_a(r)) - f(ñ_a(r))]
                /
        """
        for a, (micro_setup, rc_p, lambd_p) in enumerate(zip(
                self.micro_setup_a, self.sites.rc_ap, self.lambd_ap)):
            # Evaluate the PAW correction and integrate angular components
            df_ng = micro_setup.evaluate_paw_correction(add_f)
            df_g = integrate_lebedev(df_ng)
            for p, (rcut, lambd) in enumerate(zip(rc_p, lambd_p)):
                # Evaluate the smooth truncation function
                theta_g = radial_truncation_function(
                    micro_setup.rgd.r_g, rcut, self.drcut, lambd)
                # Integrate θ(r) Δf(r) on the radial grid
                out_ap[a, p] += micro_setup.rgd.integrate_trapz(df_g * theta_g)
