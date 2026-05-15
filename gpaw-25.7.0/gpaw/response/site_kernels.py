"""Compute site-kernels. Used for computing Heisenberg exchange constants.
Specifically, one maps a DFT calculations onto a Heisenberg lattice model,
where the site kernels define the lattice sites and magnetic moments."""

import numpy as np
from scipy.special import jv
from gpaw.response.pair_functions import get_pw_coordinates
from ase.units import Bohr


class SiteKernels:
    """Factory for calculating sublattice site kernels

                1  /
    K_aGG'(q) = ‾‾ | dr e^(-i[G-G'+q].r) θ(r∊V_a)
                V0 /

    where V_a denotes the integration volume of site a, centered at the site
    position τ_a, and V0 denotes the unit cell volume."""

    def __init__(self, positions, partitions):
        """Construct the site kernel factory.

        Parameters
        ----------
        positions : np.ndarray
            Site positions in a.u. (Bohr). Shape: (nsites, 3).
        partitions : list
            List (len=npartitions) of lists (len=nsites) with site geometries
            where the geometry arguments are given in atomic units.
            For more information on site geometries, see calculate_site_kernels
        """
        assert isinstance(partitions, list)
        assert all([isinstance(geometries, list)
                    and len(geometries) == positions.shape[0]
                    for geometries in partitions])
        assert positions.shape[1] == 3

        self.positions = positions
        self.partitions = partitions

    @property
    def npartitions(self):
        return len(self.partitions)

    @property
    def nsites(self):
        return self.positions.shape[0]

    @property
    def shape(self):
        return self.npartitions, self.nsites

    @property
    def geometry_shapes(self):
        """If all sites of a given partition has the same geometry, the
        partition is said to have a geometry shape. Otherwise, the
        geometry shape is None."""

        geometry_shapes = []
        for geometries in self.partitions:
            # Record the geometry shape, if they are all the same
            if all([shape == geometries[0][0] for shape, _ in geometries]):
                geometry_shapes.append(geometries[0][0])
            else:
                geometry_shapes.append(None)

        return geometry_shapes

    def calculate(self, qpd):
        """Generate the site kernels of each partition.

        Returns
        -------
        K_aGG : np.ndarray (dtype=complex)
            Site kernels of sites a and plane wave components G and G'.
        """
        for geometries in self.partitions:
            # We yield one set of site kernels at a time, because they can be
            # memory intensive
            yield calculate_site_kernels(qpd, self.positions, geometries)

    def __add__(self, sitekernels):
        """Add the sites from two SiteKernels instances to a new joint
        SiteKernels instance with nsites = nsites1 + nsites2."""
        assert isinstance(sitekernels, SiteKernels)
        assert self.npartitions == sitekernels.npartitions

        # Join positions
        nsites = self.nsites + sitekernels.nsites
        positions = np.append(self.positions,
                              sitekernels.positions).reshape(nsites, 3)

        # Join partitions
        partitions = [geometries1 + geometries2
                      for geometries1, geometries2
                      in zip(self.partitions, sitekernels.partitions)]

        return SiteKernels(positions, partitions)

    def append(self, sitekernels):
        """Append the partitions of another array with identical site
        positions such that npartitions = npartitions1 + npartitions2."""
        assert isinstance(sitekernels, SiteKernels)
        assert self.nsites == sitekernels.nsites
        assert np.allclose(self.positions, sitekernels.positions)

        self.partitions += sitekernels.partitions

    def copy(self):
        return SiteKernels(self.positions.copy(), self.partitions.copy())


class SphericalSiteKernels(SiteKernels):

    def __init__(self, positions, radii):
        """Construct a site kernel factory with spherical kernels.

        Parameters
        ----------
        positions : np.ndarray
            Site positions in Angstrom (Å). Shape: (nsites, 3).
        radii : list or np.ndarray
            Spherical radii of the sites in Angstrom (Å).
            Shape: (npartitions, nsites), where the individual spherical radii
            can be varried for each spatial partitioning.
        """
        positions = np.asarray(positions)

        # Parse the input spherical radii
        rc_pa = np.asarray(radii)
        assert len(rc_pa.shape) == 2
        assert rc_pa.shape[1] == positions.shape[0]

        # Convert radii to internal units (Å to Bohr)
        positions = positions / Bohr
        rc_pa = rc_pa / Bohr

        # Generate partitions as list of lists of geometries
        partitions = [[('sphere', (rc,)) for rc in rc_a]
                      for rc_a in rc_pa]

        SiteKernels.__init__(self, positions, partitions)


class CylindricalSiteKernels(SiteKernels):

    def __init__(self, positions, directions, radii, heights):
        """Construct a site kernel factory with cylindrical kernels.

        Parameters
        ----------
        positions : np.ndarray
            Site positions in Angstrom (Å). Shape: (nsites, 3).
        directions : np.ndarray
            Normalized directions of the cylindrical axes.
            Shape: (npartitions, nsites, 3), where the direction of each
            individual cylinder can be varried (along with the radius and
            height) for each spatial partitioning.
        radii : np.ndarray
            Cylindrical radii of the sites in Angstrom (Å).
            Shape: (npartitions, nsites).
        heights : list or np.ndarray
            Cylinder heights in Angstrom (Å). Shape: (npartitions, nsites).
        """
        positions = np.asarray(positions)

        # Parse the cylinder geometry arguments
        ez_pav = np.asarray(directions)
        rc_pa = np.asarray(radii)
        hc_pa = np.asarray(heights)
        nsites = positions.shape[0]
        npartitions = ez_pav.shape[0]
        assert ez_pav.shape == (npartitions, nsites, 3)
        assert np.allclose(np.linalg.norm(ez_pav, axis=-1), 1., atol=1.e-8)
        assert rc_pa.shape == (npartitions, nsites)
        assert hc_pa.shape == (npartitions, nsites)

        # Convert to internal units (Å to Bohr)
        positions = positions / Bohr
        rc_pa = rc_pa / Bohr
        hc_pa = hc_pa / Bohr

        # Generate partitions as list of lists of geometries
        partitions = [[('cylinder', (ez_v, rc, hc))
                       for ez_v, rc, hc in zip(ez_av, rc_a, hc_a)]
                      for ez_av, rc_a, hc_a in zip(ez_pav, rc_pa, hc_pa)]

        SiteKernels.__init__(self, positions, partitions)


class ParallelepipedicSiteKernels(SiteKernels):

    def __init__(self, positions, cells):
        """Construct a site kernel factory with parallelepipedic kernels.

        Parameters
        ----------
        positions : np.ndarray
            Site positions in Angstrom (Å). Shape: (nsites, 3).
        cells : np.ndarray
            Cell vectors of the parallelepiped in Angstrom (Å).
            Shape: (npartitions, nsites, 3, 3), where the parallelepipedic
            cell of each site can be varried independently for each spatial
            partitioning. The second to last entry is the vector index and
            the last entry indexes the cartesian components.
        """
        positions = np.asarray(positions)

        # Parse the parallelepipeds' cells
        cell_pacv = np.asarray(cells)
        assert len(cell_pacv.shape) == 4
        assert cell_pacv.shape[1:] == (positions.shape[0], 3, 3)

        # Convert to internal units (Å to Bohr)
        positions = positions / Bohr
        cell_pacv = cell_pacv / Bohr

        # Generate partitions as list of lists of geometries
        partitions = [[('parallelepiped', (cell_cv,)) for cell_cv in cell_acv]
                      for cell_acv in cell_pacv]

        SiteKernels.__init__(self, positions, partitions)


def calculate_site_kernels(qpd, positions, geometries):
    """Calculate the sublattice site kernel:

                1  /
    K_aGG'(q) = ‾‾ | dr e^(-i[G-G'+q].r) θ(r∊V_a)
                V0 /

    where V_a denotes the integration volume of site a, centered at the site
    position τ_a, and V0 denotes the unit cell volume.

    In the calculation, the kernel is split in two contributions:

    1) The Fourier component of the site position:

    τ_a(Q) = e^(-iQ.τ_a)

    2) The site centered geometry factor

           /
    Θ(Q) = | dr e^(-iQ.r) θ(r+τ_a∊V_a)
           /

    where Θ(Q) only depends on the geometry of the integration volume.
    With this:

                1
    K_aGG'(q) = ‾‾ τ_a(G-G'+q) Θ(G-G'+q)
                V0

    Parameters
    ----------
    qpd : SingleQPWDescriptor
        Plane wave descriptor corresponding to the q wave vector of interest.
    positions : np.ndarray
        Site positions. Array of shape (nsites, 3).
    geometries : list
        List of site geometries. A site geometry is a tuple of the integration
        volume shape (str) and arguments (tuple): (shape, args). Valid shapes
        are 'sphere', 'cylinder' and 'parallelepiped'. The integration volume
        arguments specify the size and orientation of the integration region.

    Returns
    -------
    K_aGG : np.ndarray (dtype=complex)
        Site kernels of sites a and plane wave components G and G'.
    """
    assert positions.shape[0] == len(geometries)
    assert positions.shape[1] == 3

    # Extract unit cell volume
    V0 = qpd.gd.volume

    # Construct Q=G-G'+q
    Q_GGv = construct_wave_vectors(qpd)

    # Allocate site kernel array
    nsites = len(geometries)
    K_aGG = np.zeros((nsites,) + Q_GGv.shape[:2], dtype=complex)

    # Calculate the site kernel for each site individually
    for a, (tau_v, (shape, args)) in enumerate(zip(positions, geometries)):

        # Compute the site centered geometry factor
        _geometry_factor = create_geometry_factor(shape)  # factory pattern
        Theta_GG = _geometry_factor(Q_GGv, *args)

        # Compute the Fourier component of the site position
        tau_GG = np.exp(-1.j * Q_GGv @ tau_v)

        # Update data
        K_aGG[a, :, :] = 1 / V0 * tau_GG * Theta_GG

    return K_aGG


def construct_wave_vectors(qpd):
    """Construct wave vectors Q=G1-G2+q corresponding to the q-vector of
    interest."""
    G_Gv, q_v = get_plane_waves_and_reduced_wave_vector(qpd)

    # Allocate arrays for G, G' and q respectively
    nG = len(G_Gv)
    G1_GGv = np.tile(G_Gv[:, np.newaxis, :], [1, nG, 1])
    G2_GGv = np.tile(G_Gv[np.newaxis, :, :], [nG, 1, 1])
    q_GGv = np.tile(q_v[np.newaxis, np.newaxis, :], [nG, nG, 1])

    # Contruct the wave vector G1 - G2 + q
    Q_GGv = G1_GGv - G2_GGv + q_GGv

    return Q_GGv


def get_plane_waves_and_reduced_wave_vector(qpd):
    """Get the reciprocal lattice vectors and reduced wave vector of the plane
    wave representation corresponding to the q-vector of interest."""
    # Get the reduced wave vector
    q_c = qpd.q_c

    # Get the reciprocal lattice vectors in relative coordinates
    G_Gc = get_pw_coordinates(qpd)

    # Convert to cartesian coordinates
    B_cv = 2.0 * np.pi * qpd.gd.icell_cv  # Coordinate transform matrix
    q_v = np.dot(q_c, B_cv)  # Unit = Bohr^(-1)
    G_Gv = np.dot(G_Gc, B_cv)

    return G_Gv, q_v


def create_geometry_factor(shape):
    """Creator component of the geometry factor factory pattern."""
    if shape == 'sphere':
        return spherical_geometry_factor
    elif shape == 'cylinder':
        return cylindrical_geometry_factor
    elif shape == 'parallelepiped':
        return parallelepipedic_geometry_factor

    raise ValueError('Invalid site kernel shape:', shape)


def spherical_geometry_factor(Q_Qv, rc):
    """Calculate the site centered geometry factor for a spherical site kernel:

           /
    Θ(Q) = | dr e^(-iQ.r) θ(|r|<r_c)
           /

           4πr_c
         = ‾‾‾‾‾ [sinc(|Q|r_c) - cos(|Q|r_c)]
           |Q|^2

                    3 [sinc(|Q|r_c) - cos(|Q|r_c)]
         = V_sphere ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
                              (|Q|r_c)^2

    where the dimensionless geometry factor satisfies:

    Θ(Q)/V_sphere --> 1 for |Q|r_c --> 0.

    Parameters
    ----------
    Q_Qv : np.ndarray
        Wave vectors to evaluate the site centered geometry factor at. The
        cartesian coordinates needs to be the last dimension of the array (v),
        but the preceeding index/indices Q can have any tensor structure, such
        that Q_Qv.shape = (..., 3).
    rc : float
        Radius of the sphere.
    """
    assert Q_Qv.shape[-1] == 3
    assert isinstance(rc, float) and rc > 0.

    # Calculate the sphere volume
    Vsphere = 4 * np.pi * rc**3. / 3

    # Calculate |Q|r_c
    Qrc_Q = np.linalg.norm(Q_Qv, axis=-1) * rc

    # Allocate array with ones to provide the correct dimensionless geometry
    # factor in the |Q|r_c --> 0 limit.
    # This is done to avoid division by zero.
    Theta_Q = np.ones(Q_Qv.shape[:-1], dtype=float)

    # Calculate the dimensionless geometry factor
    Qrcs = Qrc_Q[Qrc_Q > 1.e-8]
    Theta_Q[Qrc_Q > 1.e-8] = 3. * (sinc(Qrcs) - np.cos(Qrcs)) / Qrcs**2.

    Theta_Q *= Vsphere

    return Theta_Q


def cylindrical_geometry_factor(Q_Qv, ez_v, rc, hc):
    """Calculate site centered geometry factor for a cylindrical site kernel:

           /
    Θ(Q) = | dr e^(-iQ.r) θ(ρ<r_c) θ(|z|/2<h_c)
           /

            4πr_c
         = ‾‾‾‾‾‾‾ J_1(Q_ρ r_c) sin(Q_z h_c / 2)
           Q_ρ Q_z

                      2 J_1(Q_ρ r_c)
         = V_cylinder ‾‾‾‾‾‾‾‾‾‾‾‾‾‾ sinc(Q_z h_c / 2)
                         Q_ρ r_c

    where z denotes the cylindrical axis, ρ the radial axis and the
    dimensionless geometry factor satisfy:

    Θ(Q)/V_cylinder --> 1 for Q --> 0.

    Parameters
    ----------
    Q_Qv : np.ndarray
        Wave vectors to evaluate the site centered geometry factor at. The
        cartesian coordinates needs to be the last dimension of the array (v),
        but the preceeding index/indices Q can have any tensor structure, such
        that Q_Qv.shape = (..., 3).
    ez_v : np.ndarray
        Normalized direction of the cylindrical axis.
    rc : float
        Radius of the cylinder.
    hc : float
        Height of the cylinder.
    """
    assert Q_Qv.shape[-1] == 3
    assert ez_v.shape == (3,)
    assert abs(np.linalg.norm(ez_v) - 1.) < 1.e-8
    assert isinstance(rc, float) and rc > 0.
    assert isinstance(hc, float) and hc > 0.

    # Calculate cylinder volume
    Vcylinder = np.pi * rc**2. * hc

    # Calculate Q_z h_c and Q_ρ r_c
    Qzhchalf_Q = np.abs(Q_Qv @ ez_v) * hc / 2.
    Qrhorc_Q = np.linalg.norm(np.cross(Q_Qv, ez_v), axis=-1) * rc

    # Allocate array with ones to provide the correct dimensionless geometry
    # factor in the Q_ρ r_c --> 0 limit.
    # This is done to avoid division by zero.
    Theta_Q = np.ones(Q_Qv.shape[:-1], dtype=float)

    # Calculate the dimensionless geometry factor
    Qrhorcs = Qrhorc_Q[Qrhorc_Q > 1.e-8]
    Theta_Q[Qrhorc_Q > 1.e-8] = 2. * jv(1, Qrhorcs) / Qrhorcs
    Theta_Q *= Vcylinder * sinc(Qzhchalf_Q)

    return Theta_Q


def parallelepipedic_geometry_factor(Q_Qv, cell_cv):
    """Calculate the site centered geometry factor for a parallelepipedic site
    kernel:

           /
    Θ(Q) = | dr e^(-iQ.r) θ(r∊V_parallelepiped)
           /

         = |det[a1, a2, a3]| sinc(Q.a1 / 2) sinc(Q.a2 / 2) sinc(Q.a3 / 2)

         = V_parallelepiped sinc(Q.a1 / 2) sinc(Q.a2 / 2) sinc(Q.a3 / 2)

    where a1, a2 and a3 denotes the parallelepipedic cell vectors.

    Parameters
    ----------
    Q_Qv : np.ndarray
        Wave vectors to evaluate the site centered geometry factor at. The
        cartesian coordinates needs to be the last dimension of the array (v),
        but the preceeding index/indices Q can have any tensor structure, such
        that Q_Qv.shape = (..., 3).
    cell_cv : np.ndarray, shape=(3, 3)
        Cell vectors of the parallelepiped, where v denotes the cartesian
        coordinates.
    """
    assert Q_Qv.shape[-1] == 3
    assert cell_cv.shape == (3, 3)

    # Calculate the parallelepiped volume
    Vparlp = abs(np.linalg.det(cell_cv))
    assert Vparlp > 1.e-8  # Not a valid parallelepiped if volume vanishes

    # Calculate the site-kernel
    a1, a2, a3 = cell_cv
    Theta_Q = Vparlp * sinc(Q_Qv @ a1 / 2) * sinc(Q_Qv @ a2 / 2) * \
        sinc(Q_Qv @ a3 / 2)

    return Theta_Q


def sinc(x):
    """np.sinc(x) = sin(pi*x) / (pi*x), hence the division by pi"""
    return np.sinc(x / np.pi)
