from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn
from dataclasses import dataclass

from gpaw.spline import Spline
from gpaw.ffbt import rescaled_fourier_bessel_transform
from gpaw.gaunt import gaunt, super_gaunt
from gpaw.spherical_harmonics import Y
from gpaw.atom.radialgd import RadialGridDescriptor
from gpaw.sphere.rshe import RealSphericalHarmonicsExpansion
from gpaw.response.pw_parallelization import Blocks1D


# Important note: The test suite monkeypatches this value to 2**10 so
# you may get different results in tests and production until we
# implement a better solution (e.g. generating setup-dependent radial_points).
#
# The motivation for lowering to 2**10 in tests is that many tests
# take 3-4 times longer if we do not.
#
# See https://gitlab.com/gpaw/gpaw/-/issues/984
DEFAULT_RADIAL_POINTS = 2**12


@dataclass
class LeanPAWDataset:
    rgd: RadialGridDescriptor
    l_j: np.ndarray
    rcut_j: np.ndarray
    phit_jg: np.ndarray
    phi_jg: np.ndarray
    # Number of radial points in spline interpolation
    radial_points: int | None = None

    def __post_init__(self):
        if self.radial_points is None:
            # We assign this late due to monkeypatch in testing
            self.radial_points = DEFAULT_RADIAL_POINTS

        # Number of basis functions
        self.ni = np.sum([2 * l + 1 for l in self.l_j])
        # Maximum angular momentum index l
        self.lmax = np.max(self.l_j)
        # Grid cutoff to create spline representation
        self.gcut2 = self.rgd.ceil(2 * max(self.rcut_j))

        # Set up cache
        self.dn_g_cache = {}
        self.dn_kspline_cache = {}

    def dn_kspline(self, j1: int, j2: int, l: int):
        """Get spline representation of Δn_jj'l(k)."""
        if (j1, j2, l) not in self.dn_kspline_cache:
            dn_g = self.get_pair_density_correction(j1, j2)
            self.dn_kspline_cache[j1, j2, l] = \
                self.rescaled_fourier_bessel_transform(dn_g, l)
        return self.dn_kspline_cache[j1, j2, l]

    def get_pair_density_correction(self, j1: int, j2: int):
        """Get Δn_jj'(r), while keeping the newest correction cached."""
        if (j1, j2) not in self.dn_g_cache:
            self.dn_g_cache = {}  # keep only one density in cache
            self.dn_g_cache[j1, j2] = self.calculate_pair_density_correction(
                j1, j2)
        return self.dn_g_cache[j1, j2]

    def calculate_pair_density_correction(self, j1, j2):
        """Calculate the pair density PAW correction for two partial waves.
                               ˷     ˷
        Δn (r) = φ (r) φ (r) - φ (r) φ (r)
          jj'     j     j'      j     j'
        """
        # (Real) radial functions for the partial waves
        phi_jg = self.phi_jg
        phit_jg = self.phit_jg
        return phi_jg[j1] * phi_jg[j2] - phit_jg[j1] * phit_jg[j2]

    def rescaled_fourier_bessel_transform(self, f_g, l):
        """Calculate the rescaled Fourier Bessel transform f_l(k)

                    rc
                4π  ⌠  2
        f (k) = ‾‾‾ ⎪ r dr j (kr) f(r)
         l      k^l ⌡       l
                    0
        """
        # First, we make a spline representation of the radial function f(r),
        # which is rescaled by a factor of r^-l by the radial grid descriptor:
        spline = self.rgd.spline(
            f_g[:self.gcut2], l=l, points=self.radial_points)
        # Once rescaled by r^-l, we can use the gpaw.ffbt module to Fast
        # Fourier Bessel Transform the radial spline r^-l f(r) and obtain
        # f_l(k) in a reciprocal spline representation
        kspline = rescaled_fourier_bessel_transform(
            spline, N=4 * self.radial_points)
        # Since this procedures hinges on a series of hardcoded parameters, we
        # return a self-testing version of the spline. If someone wants to run
        # calculations without dynamic testing of the FFBT methodology (i.e.
        # in an "unprotected" mode), simply return the bare `kspline` here.
        return SelfTestingKSpline(self.rgd, f_g, kspline)


class SelfTestingKSpline(Spline):
    """Self-testing reciprocal spline representation, f_l(k)."""

    def __init__(self, rgd, f_g, spline: Spline):
        # Store original real-space representation of the radial function f(r)
        self.rgd = rgd
        self.f_g = f_g
        super().__init__(spline.spline)

    def map(self, k_G):
        self.test_spline_representation(k_G)
        return super().map(k_G)

    def test_spline_representation(self, k_G):
        """Test validity of the FFBT implementation on input domain.

        At present, LeanPAWDataset's FFBT implementation relies on a range of
        hardcoded parameters, which are not guaranteed to work for all cases.

        In particular, the uniform radial grid used for the FFBT is defined
        through the `rcut` and `N` parameters in
        `rescaled_fourier_bessel_transform()`
        where the former is hardcoded inside the function itself.

        Furthermore, the `points` parameter to `rgd.spline()` controls the
        fidelity of the interpolation between nonlinear and equidistant radial
        grids needed to make use of the FFBT algorithm.

        To make a generally reliable implementation, one would need to control
        all of these parameters based on the setup, e.g. the nonlinear radial
        grid spacing. In doing so, one should be mindful that the `rcut`
        parameter defines the reciprocal grid spacing of the kspline and that
        `N` controls the range of the reciprocal space domain.

        For now, we simply check that the requested plane waves are within the
        computed k-range of the FFBT and check that the resulting transforms
        match a manual calculation at a few selected K-vectors.
        """
        kmax = np.max(k_G)
        assert kmax <= self.get_cutoff()

        # Manual calculation at finite k
        k_k = np.array([kmax, np.average(k_G)])
        f_k = 4 * np.pi * fourier_bessel_transform(
            np.array(k_k), self.l, self.rgd, self.f_g)
        # Manual calculation at k=0
        if self.l == 0:  # Vanishes for l>0
            k_k = np.append(k_k, [0.])
            f_k = np.append(f_k, [self.rgd.integrate(self.f_g)])
        # FFBT calculation
        myf_k = k_k**self.l * super().map(k_k)
        assert np.allclose(myf_k, f_k, rtol=1e-2, atol=1e-3), \
            f'FFBT mismatch: {myf_k}, {f_k}'


def calculate_pair_density_correction(qG_Gv: np.ndarray, *,
                                      pawdata: LeanPAWDataset):
    r"""Calculate the atom-centered PAW correction to the pair density.
                                                      ˍ
    The atom-centered pair density correction tensor, Q_aii', is defined as the
    atom-centered Fourier transform

    ˍ             /                                     ˷         ˷
    Q_aii'(G+q) = | dr e^-i(G+q)r [φ_ai^*(r) φ_ai'(r) - φ_ai^*(r) φ_ai'(r)]
                  /

    evaluated with the augmentation sphere center at the origin. The full pair
    density correction tensor is then given by
                                  ˍ
    Q_aii'(G+q) = e^(-i[G+q].R_a) Q_aii'(G+q)

    Expanding the plane wave coefficient into real spherical harmonics and
    spherical Bessel functions, the correction can split into angular and
    radial contributions

                    l
                __  __
                \   \      l  m ˰   m,m_i,m_i'
    Q_aii'(K) = /   /  (-i)  Y (K) g
                ‾‾  ‾‾        l     l,l_i,l_i'
                l  m=-l
                            rc
                            /                    a     a      ˷a    ˷a
                       × 4π | r^2 dr j_l(|K|r) [φ (r) φ (r) - φ (r) φ (r)]
                            /                    j_i   j_i'    j_i   j_i'
                            0

    where K = G+q and g denotes the Gaunt coefficients.

    For more information, see [PRB 103, 245110 (2021)]. In particular, it
    should be noted that the partial waves themselves are defined via real
    spherical harmonics and radial functions φ_j from the PAW setups:

     a       m_i ˰   a
    φ (r) = Y   (r) φ (r)
     i       l_i     j_i
    """
    ni = pawdata.ni  # Number of partial waves
    l_j = pawdata.l_j  # l-index for each radial function index j
    G_LLL = gaunt(pawdata.lmax)

    # Initialize correction tensor
    npw = qG_Gv.shape[0]
    Qbar_Gii = np.zeros((npw, ni, ni), dtype=complex)

    # K-vector norm
    k_G = np.linalg.norm(qG_Gv, axis=1)

    # Loop of radial function indices for partial waves i and i'
    i1_counter = 0
    for j1, l1 in enumerate(l_j):
        i2_counter = 0
        for j2, l2 in enumerate(l_j):
            # Sample l according to the Gaunt coefficient selection rules, see
            # e.g. gpaw.test.test_gaunt
            for l in range(abs(l1 - l2), l1 + l2 + 1, 2):
                # Calculate the spherical Fourier-Bessel transform
                #                  rc
                #              4π  /
                # Δn_jj'l(k) = ‾‾‾ | r^2 dr j_l(kr) Δn_jj'(r)
                #              k^l /
                #                  0
                # To evaluate the radial integral efficiently, we rely on the
                # Fast Fourier Bessel Transform (FFBT) algorithm, see gpaw.ffbt
                # The FFBT produces a radial spline representation of
                # Δn_jj'l(k), which we map to the K-vector norms in question:
                dn_G = pawdata.dn_kspline(j1, j2, l).map(k_G)

                # Angular part of the integral
                f_G = (-1j)**l * dn_G
                for m in range(l**2, (l + 1)**2):
                    # Calculate the solid harmonic
                    #        m ˰
                    # |K|^l Y (K)
                    #        l
                    klY_G = Y(m, *qG_Gv.T)

                    # Generate m-indices for each radial function
                    for m1 in range(2 * l1 + 1):
                        for m2 in range(2 * l2 + 1):
                            # Set up the i=(l,m) index for each partial wave
                            i1 = i1_counter + m1
                            i2 = i2_counter + m2
                            # Extract Gaunt coefficients
                            gaunt_coeff = G_LLL[l1**2 + m1, l2**2 + m2, m]
                            if (gaunt_coeff == 0):
                                continue

                            # Add contribution to the PAW correction
                            Qbar_Gii[:, i1, i2] += gaunt_coeff * klY_G * f_G

            # Add to i and i' counters
            i2_counter += 2 * l2 + 1
        i1_counter += 2 * l1 + 1
    return Qbar_Gii


def calculate_matrix_element_correction(qG_Gv, pawdata,
                                        rshe: RealSphericalHarmonicsExpansion):
    r"""Calculate the atom-centered correction to a generalized matrix element.

    For matrix elements corresponding to the expectation value of a plane wave
    coefficient e^-i(G+q)r and a known functional of the (spin-)density
    f[n](r), the PAW correction tensor is given by

    F_aii'(G+q) = <φ_ai| e^-i(G+q)r f[n](r) |φ_ai'>
                     ˷                         ˷
                  - <φ_ai| e^-i(G+q)r f[n](r) |φ_ai'>
                                  ˍ
                = e^(-i[G+q].R_a) F_aii'(G+q)
          ˍ
    where F_aii'(G+q) is the atom-centered PAW correction tensor.

    Expanding the functional f[n](r) in the atom-centered frame in real
    spherical harmonics (corresponding to the input rshe),

                  l
              __  __
         →    \   \   m ˰   m
    f[n](r) = /   /  Y (r) f (r)
              ‾‾  ‾‾  l     l
              l  m=-l

    expansion of the plane-wave coefficient in real spherical harmonics and
    spherical Bessel functions j_l(Kr) yields the following expression for the
    atom-centered correction tensor [publication in preparation]:

                       l        l'
                   __  __   __  __
    ˍ              \   \    \   \      l'  m'˰   m_i,m_i',m,m'
    F_aii'(K) = 4π /   /    /   /  (-i)   Y (K) G
                   ‾‾  ‾‾   ‾‾  ‾‾         l'    l_i,l_i',l,l'
                   l  m=-l  l' m'=-l'

                                rc
                                /  2            a     a      ˷a    ˷a      m
                              × | r dr j (Kr) [φ (r) φ (r) - φ (r) φ (r)] f (r)
                                /       l'      j_i   j_i'    j_i   j_i'   l
                                0

    where K=G+q and G_LLLL denotes the super Gaunt coefficients, which yield
    the integrals over four spherical harmonics.
    """
    rgd = rshe.rgd
    assert rgd is pawdata.xc_correction.rgd
    ni = pawdata.ni  # Number of partial waves
    l_j = pawdata.l_j  # l-index for each radial function index j
    lmax = max(l_j)
    assert max(rshe.l_M) <= 2 * lmax
    G_LLLL = super_gaunt(lmax)
    # (Real) radial functions for the partial waves
    phi_jg = pawdata.phi_jg
    phit_jg = pawdata.phit_jg
    # Truncate the radial functions to span only the radial grid coordinates
    # which need correction
    assert np.allclose(rgd.r_g, pawdata.rgd.r_g[:rgd.N])
    phi_jg = np.array(phi_jg)[:, :rgd.N]
    phit_jg = np.array(phit_jg)[:, :rgd.N]

    # Initialize correction tensor
    npw = qG_Gv.shape[0]
    Fbar_Gii = np.zeros((npw, ni, ni), dtype=complex)

    # K-vector norm and direction
    k_G = np.linalg.norm(qG_Gv, axis=1)
    Kd_Gv = qG_Gv.copy()
    Kd_Gv[k_G > 1e-10] /= k_G[k_G > 1e-10, np.newaxis]

    # Loop of radial function indices for partial waves i and i'
    i1_counter = 0
    for j1, l1 in enumerate(l_j):
        i2_counter = 0
        for j2, l2 in enumerate(l_j):
            # Calculate the radial partial wave correction
            #                              ˷      ˷
            # Δn_jj'(r) = φ_j(r) φ_j'(r) - φ_j(r) φ_j'(r)
            dn_g = phi_jg[j1] * phi_jg[j2] - phit_jg[j1] * phit_jg[j2]

            # Loop through the angular components in the real spherical
            # harmonics expansion of f[n](r)
            for l, L, f_g in zip(rshe.l_M, rshe.L_M, rshe.f_gM.T):
                dnf_g = dn_g * f_g
                # Apply Gaunt coefficient selection rules to loop through
                # the l' coefficients of the plane-wave expansion
                lpmin = np.min(abs(
                    np.arange(abs(l1 - l2), l1 + l2 + 1) - l))
                for lp in range(lpmin, l1 + l2 + l + 1):
                    if not (l1 + l2 + l + lp) % 2 == 0:
                        continue
                    # Calculate radial part of the correction
                    dnf_G = parallel_fourier_bessel_transform(
                        k_G, lp, rgd, dnf_g)

                    # Calculate angular part of the correction
                    x_G = 4 * np.pi * (-1j)**lp * dnf_G
                    # Loop through available m-indices for the partial waves
                    # and generate the composite L=(l,m) index as well as the
                    # partial wave index i
                    for m1 in range(2 * l1 + 1):
                        L1 = l1**2 + m1
                        i1 = i1_counter + m1
                        for m2 in range(2 * l2 + 1):
                            L2 = l2**2 + m2
                            i2 = i2_counter + m2
                            # Loop through m' indices of the plane-wave
                            # expansion and generate the L' composite index
                            for mp in range(2 * lp + 1):
                                Lp = lp**2 + mp
                                # If the angular integral (super gaunt
                                # coefficient) is finite,
                                coeff = G_LLLL[L1, L2, L, Lp]
                                if abs(coeff) > 1e-10:
                                    # Calculate spherical harmonic and add
                                    # contribution to the PAW correction
                                    Y_G = Y(Lp, *Kd_Gv.T)
                                    Fbar_Gii[:, i1, i2] += coeff * Y_G * x_G

            # Add to i and i' counters
            i2_counter += 2 * l2 + 1
        i1_counter += 2 * l1 + 1
    return Fbar_Gii


def parallel_fourier_bessel_transform(k_G, *args, comm=None):
    """Distribute FBT plane-wave components over a given communicator."""
    # NB: If we need to do something similar elsewhere, we can generalize this
    # function to a decorator!
    if comm is None:
        from gpaw.mpi import world as comm
    Gblocks = Blocks1D(comm, len(k_G))
    f_myG = fourier_bessel_transform(k_G[Gblocks.myslice], *args)
    return Gblocks.all_gather(f_myG)


def fourier_bessel_transform(k_G, l, rgd, f_g):
    """Perform a spherical Fourier-Bessel transform of a radial function f(r).

    Computes the transform

            max
           r
           ⌠  2
    f(k) = ⎪ r dr j (kr) f(r)
           ⌡       l
           0

    on the supplied radial grid.
    """
    # Vectorize calculation of spherical Bessel functions
    l_Gg = l * np.ones((len(k_G), rgd.N), dtype=int)
    kr_Gg = k_G[:, np.newaxis] * rgd.r_g[np.newaxis]
    jl_Gg = spherical_jn(l_Gg, kr_Gg)  # so slow...
    # Integrate the radial grid using linear interpolation
    f_G = rgd.integrate_trapz(jl_Gg * f_g[np.newaxis])
    return f_G


class PWPAWCorrectionData:
    def __init__(self, Q_aGii, qpd, pawdatasets, pos_av, atomrotations):
        # Sometimes we loop over these in ways that are very dangerous.
        # It must be list, not dictionary.
        assert isinstance(Q_aGii, list)
        assert len(Q_aGii) == len(pos_av) == len(pawdatasets.by_atom)

        self.Q_aGii = Q_aGii

        self.qpd = qpd
        self.pawdatasets = pawdatasets
        self.pos_av = pos_av
        self.atomrotations = atomrotations

    def _new(self, Q_aGii):
        return PWPAWCorrectionData(Q_aGii, qpd=self.qpd,
                                   pawdatasets=self.pawdatasets,
                                   pos_av=self.pos_av,
                                   atomrotations=self.atomrotations)

    def remap(self, M_vv, G_Gv, sym, sign):
        Q_aGii = []
        for a, Q_Gii in enumerate(self.Q_aGii):
            x_G = self._get_x_G(G_Gv, M_vv, self.pos_av[a])
            U_ii = self.atomrotations.get_by_a(a).R_sii[sym]

            Q_Gii = np.einsum('ij,kjl,ml->kim',
                              U_ii,
                              Q_Gii * x_G[:, None, None],
                              U_ii,
                              optimize='optimal')
            if sign == -1:
                Q_Gii = Q_Gii.conj()
            Q_aGii.append(Q_Gii)

        return self._new(Q_aGii)

    def _get_x_G(self, G_Gv, M_vv, pos_v):
        # This doesn't really belong here.  Or does it?  Maybe this formula
        # is only used with PAW corrections.
        return np.exp(1j * (G_Gv @ (pos_v - M_vv @ pos_v)))

    def remap_by_symop(self, symop, G_Gv, M_vv):
        return self.remap(M_vv, G_Gv, symop.symno, symop.sign)

    def multiply(self, P_ani, band):
        assert isinstance(P_ani, list)
        assert len(P_ani) == len(self.Q_aGii)

        C1_aGi = [Qa_Gii @ P1_ni[band].conj()
                  for Qa_Gii, P1_ni in zip(self.Q_aGii, P_ani)]
        return C1_aGi

    def reduce_ecut(self, G2G):
        # XXX actually we should return this with another PW descriptor.
        return self._new([Q_Gii.take(G2G, axis=0) for Q_Gii in self.Q_aGii])

    def almost_equal(self, otherpawcorr, G_G):
        for a, Q_Gii in enumerate(otherpawcorr.Q_aGii):
            e = abs(self.Q_aGii[a] - Q_Gii[G_G]).max()
            if e > 1e-12:
                return False
        return True


def get_pair_density_paw_corrections(pawdatasets, qpd, spos_ac, atomrotations):
    r"""Calculate and bundle paw corrections to the pair densities as a
    PWPAWCorrectionData object.

    The pair density PAW correction tensor is given by:

                  /
    Q_aii'(G+q) = | dr e^(-i[G+q].r) [φ_ai^*(r-R_a) φ_ai'(r-R_a)
                  /                     ˷             ˷
                                      - φ_ai^*(r-R_a) φ_ai'(r-R_a)]
    """
    qG_Gv = qpd.get_reciprocal_vectors(add_q=True)
    pos_av = spos_ac @ qpd.gd.cell_cv

    # Calculate pair density PAW correction tensor
    Qbar_xGii = {}
    for species_index, pawdata in pawdatasets.by_species.items():
        # Calculate atom-centered correction tensor
        Qbar_Gii = calculate_pair_density_correction(qG_Gv, pawdata=pawdata)
        # Add dependency on the atomic position (phase factor)
        Qbar_xGii[species_index] = Qbar_Gii

    Q_aGii = []
    for a, (pos_v, pawdata) in enumerate(zip(pos_av, pawdatasets.by_atom)):
        x_G = np.exp(-1j * (qG_Gv @ pos_v))
        species_index = pawdatasets.id_by_atom[a]
        Qbar_Gii = Qbar_xGii[species_index]
        Q_aGii.append(x_G[:, np.newaxis, np.newaxis] * Qbar_Gii)

    return PWPAWCorrectionData(Q_aGii, qpd=qpd,
                               pawdatasets=pawdatasets,
                               pos_av=pos_av,
                               atomrotations=atomrotations)


def get_matrix_element_paw_corrections(qpd, pawdata_a, rshe_a, spos_ac):
    r"""Calculate the PAW correction to a generalized matrix element.

    For a given functional of the electron (spin-)density f[n](r), the PAW
    correction is given by
                                  ˍ
    F_aii'(G+q) = e^(-i[G+q].R_a) F_aii'(G+q)
          ˍ
    where F_aii'(G+q) is the atom-centered correction (see above).
    """
    qG_Gv = qpd.get_reciprocal_vectors(add_q=True)

    F_aGii = []
    for pawdata, rshe, spos_c in zip(pawdata_a.by_atom, rshe_a, spos_ac):
        # Calculate atom-centered PAW correction
        Fbar_Gii = calculate_matrix_element_correction(
            qG_Gv, pawdata, rshe)

        # XXX Can time be saved by doing some of the processing per species
        # rather than per atom?

        # Add dependency on the atomic position (phase factor)
        pos_v = spos_c @ qpd.gd.cell_cv
        x_G = np.exp(-1j * (qG_Gv @ pos_v))
        F_aGii.append(x_G[:, np.newaxis, np.newaxis] * Fbar_Gii)

    return F_aGii
