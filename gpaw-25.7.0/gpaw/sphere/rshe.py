import numpy as np

from gpaw.atom.radialgd import RadialGridDescriptor
from gpaw.sphere.integrate import integrate_lebedev


class RealSphericalHarmonicsExpansion:
    """Expansion in real spherical harmonics of a function f(r)."""

    def __init__(self,
                 rgd: RadialGridDescriptor,
                 f_gM, Y_nL, L_M=None):
        """Construct the expansion

        Parameters
        ----------
        f_gM : np.array
            f as a function of radial index g and reduced spherical harmonic
            index M.
        Y_nL : np.array
            Real spherical harmonics on the angular Lebedev quadrature as a
            function of the composite spherical harmonics index L=(l,m).
        L_M : np.array
            L index for every reduced expansion index M.
        """
        self.rgd = rgd
        self.f_gM = f_gM
        self.Y_nL = Y_nL

        if L_M is None:
            # Assume that all the composite indices L=(l,m) are represented
            assert f_gM.shape[1] == self.nL
            L_M = np.arange(self.nL)
        self.L_M = L_M

    @classmethod
    def from_spherical_grid(cls, rgd, f_ng, Y_nL):
        r"""Expand the function f(r) in real spherical harmonics.

                / ^    ^     ^
        f (r) = |dr Y (r) f(rr)
         lm     /    lm

        Note that the Lebedev quadrature, which is used to perform the angular
        integral above, is exact up to polynomial order l=11. This implies that
        expansion coefficients up to l=5 are exact.

        Parameters
        ----------
        f_ng : np.array
            f as a function of angular index n (on the Lebedev quadrature) and
            radial index g.
        Y_nL : np.array
            Real spherical harmonics on the angular Lebedev quadrature as a
            function of the composite spherical harmonics index L=(l,m).
        """
        # Include coefficients up to l = 5, where nL = (l + 1)**2
        nL = min(Y_nL.shape[1], 36)

        # Integrate Y_lm(r) * f(r) on the angular grid
        f_gL = integrate_lebedev(
            Y_nL[:, np.newaxis, :nL] * f_ng[..., np.newaxis])

        return cls(rgd, f_gL, Y_nL)

    def reduce_expansion(self, L_M):
        """
        Produce a new expansion with only the spherical harmonic indices L_M.
        """
        # Translate requested indices L_M to the internal index M
        M_M = []
        for L in L_M:
            lookup = np.where(self.L_M == L)[0]
            assert len(lookup) == 1
            M_M.append(lookup[0])

        return RealSphericalHarmonicsExpansion(
            self.rgd, self.f_gM[:, M_M], self.Y_nL, L_M=L_M)

    @property
    def nL(self):
        return self.Y_nL.shape[1]

    @property
    def nM(self):
        return len(self.L_M)

    @property
    def lmax(self):
        flmax = np.sqrt(self.nL)
        lmax = int(flmax)
        assert abs(flmax - lmax) < 1e-8
        return lmax

    @property
    def l_L(self):
        l_L = []
        for l in range(self.lmax + 1):
            l_L += [l] * (2 * l + 1)
        return l_L

    @property
    def l_M(self):
        return [self.l_L[L] for L in self.L_M]

    def evaluate_on_quadrature(self):
        """Evaluate the function f(r) on the angular Lebedev quadrature."""
        Y_nM = self.Y_nL[:, self.L_M]
        return Y_nM @ self.f_gM.T


def calculate_reduced_rshe(rgd, f_ng, Y_nL, lmax=-1, wmin=None):
    """Expand a function f(r) in real spherical harmonics with a reduced number
    of expansion coefficients."""
    rshe = RealSphericalHarmonicsExpansion.from_spherical_grid(rgd, f_ng, Y_nL)
    L_M, info_string = assess_rshe_reduction(f_ng, rshe, lmax=lmax, wmin=wmin)
    rshe = rshe.reduce_expansion(L_M)
    return rshe, info_string


def assess_rshe_reduction(f_ng, rshe, lmax=-1, wmin=None):
    """Assess how to reduce the number of expansion coefficients.

    The composite index L=(l,m) is reduced to an index M, which iterates the
    expansion coefficients which contribute with a weight larger than wmin to
    the surface norm square of the function f(r) on average. The M index is
    further restricted to include coefficients only up to lmax.
    """
    # We do not expand beyond l=5
    if lmax == -1:
        lmax = 5
    assert lmax in range(6)

    # We assume to start with a full expansion
    assert rshe.nM == rshe.nL
    f_gL = rshe.f_gM

    # Filter away (l,m)-coefficients based on their average weight in
    # completing the surface norm square f(r)
    fsns_g = integrate_lebedev(f_ng ** 2)  # surface norm square
    mask_g = fsns_g > 1e-12  # Base filter on finite surface norm squares only
    fw_gL = f_gL[mask_g] ** 2 / fsns_g[mask_g, np.newaxis]  # weight of each L
    rshew_L = np.average(fw_gL, axis=0)  # Average over the radial grid

    # Take rshe coefficients up to l <= lmax (<= 5) which contribute with
    # at least wmin to the surface norm square on average
    nL = min(rshe.nL, (lmax + 1)**2)
    L_L = np.arange(nL)
    if wmin is not None:
        assert isinstance(wmin, float) and wmin > 0.
        L_M = np.where(rshew_L[L_L] >= wmin)[0]
    else:
        L_M = L_L

    info_string = get_reduction_info_string(L_M, fw_gL, rshew_L)

    return L_M, info_string


def get_reduction_info_string(L_M, fw_gL, rshew_L):
    """Construct info string about the reduced expansion."""
    isl = []
    isl.append('{:6}  {:10}  {:10}  {:8}'.format('(l,m)', 'max weight',
                                                 'avg weight', 'included'))
    for L, (fw_g, rshew) in enumerate(zip(fw_gL.T, rshew_L)):
        included = L in L_M
        isl.append('\n' + get_rshe_coefficient_info_string(
            L, included, rshew, fw_g))

    avg_cov = np.average(np.sum(fw_gL[:, L_M], axis=1))
    isl.append(f'\nIn total: {avg_cov} of the surface norm square is '
               'covered on average')

    tot_avg_cov = np.average(np.sum(fw_gL, axis=1))
    isl.append(f'\nIn total: {tot_avg_cov} of the surface norm '
               'square could be covered on average')

    return ''.join(isl)


def get_rshe_coefficient_info_string(L, included, rshew, fw_g):
    """Construct info string about the weight of a given coefficient."""
    l = int(np.sqrt(L))
    m = L - l * (l + 1)
    included = 'yes' if included else 'no'
    info_string = '{:6}  {:1.8f}  {:1.8f}  {:8}'.format(f'({l},{m})',
                                                        np.max(fw_g),
                                                        rshew, included)
    return info_string
