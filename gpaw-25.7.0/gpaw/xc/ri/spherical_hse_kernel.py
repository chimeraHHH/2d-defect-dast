"""

Implementation of spherical harmonic expansion of screened Coulomb kernel.
Based on

    János G Ángyán et al 2006 J. Phys. A: Math. Gen. 39 8613


"""


import numpy as np
from scipy.special import erfc, comb, factorial2
from math import factorial


def safeerfc(x):
    taylor = (1
              - 2 * x / np.pi**0.5
              + 2 * x**3 / (3 * np.pi**0.5)
              - x**5 / (5 * np.pi)**0.5)
    return np.where(x < 1e-4, taylor, erfc(x))


def Dnk(n, k, Xi):
    # Eq. 28
    if k == 0:
        sum = 0
        for m in range(1, n + 1):
            sum += 2**(-m) * Xi**(-2 * m) / factorial2(2 * n - 2 * m + 1)
        return (safeerfc(Xi)
                + np.exp(-Xi**2)
                / (np.pi**0.5) * 2**(n + 1) * Xi**(2 * n + 1) * sum)
    # Eq. 29
    sum = 0
    for m in range(1, k + 1):
        sum += (comb(m - k - 1, m - 1) * 2**(k - m) * Xi**(2 * (k - m))
                / factorial2(2 * n + 2 * k - 2 * m + 1))

    return (np.exp(-Xi**2)
            * 2**(n + 1) * (2 * n + 1) * Xi**(2 * n + 1)
            / np.pi**0.5 / factorial(k)
            / (2 * n + 2 * k + 1) * sum)


def Phinj(n, j, Xi, xi):
    # Eq. 30
    sum = 0
    for k in range(j):
        sum += Dnk(n, k, Xi) / (Xi**(n + 1)) * xi**(n + 2 * k)
    return sum


def Hn(n, Xi, xi):
    """

    Helper function (Eq. 24)

    """
    return 1 / (2 * (xi * Xi)**(n + 1)) * ((Xi**(2 * n + 1) + xi**(2 * n + 1)) * safeerfc(Xi + xi) - (Xi**(2 * n + 1) - xi**(2 * n + 1)) * safeerfc(Xi - xi))  # noqa: E501


def Fn(n, Xi, xi):
    """

        Helper function (Eq. 22).

        It appears, that the article has a typo, because the summation
        starts at p=1, but correct results require to start at p=0.

    """
    prefactor = 2 / np.pi**0.5
    result = 0.0

    for p in range(0, n + 1):
        result += (-1 / (4 * Xi * xi))**(p + 1) * factorial(n + p) / (factorial(p) * factorial(n - p)) * ((-1)**(n - p) * np.exp(-(xi + Xi)**2) - np.exp(-(xi - Xi)**2))  # noqa: E501
    taylor = np.exp(-Xi**2 - xi**2) * 2**(n + 1) * (3 + 2 * n + 2 * xi**2 * Xi**2) * xi**n * Xi**n / (np.pi**0.5 * factorial2(2 * n + 3))  # noqa: E501

    return np.where((Xi * xi)**(2 * n + 1) < 1e-6, taylor, prefactor * result)


def Phi(n, mu, R, r):
    """

        The official spherical kernel expansion

    """
    Rg = np.maximum.reduce([R, r])
    Rl = np.minimum.reduce([R, r])

    # Scaling as given by Eq. 16 and the text above.
    Xi = mu * Rg
    xi = mu * Rl

    # Eq. 21
    result = Fn(n, Xi, xi) + Hn(n, Xi, xi)
    for m in range(1, n + 1):
        result += (Fn(n - m, Xi, xi)
                   * (Xi**(2 * m) + xi**(2 * m)) / (xi * Xi)**m)

    result = np.where(xi < [1e-3, 1e-2, 1e-1, 1e-1, 1e-1, 1e-1][n],
                      Phinj(n, 2, Xi, xi), result)
    result *= mu

    return result


"""

Implementation of spherical harmonic expansion ends. GPAW spesific stuff
remains below.

"""


class RadialHSE:
    def __init__(self, rgd, omega):
        self.rgd = rgd
        self.omega = omega

        self.r1_gg = np.zeros((rgd.N, rgd.N))
        self.r2_gg = np.zeros((rgd.N, rgd.N))
        self.d_gg = np.zeros((rgd.N, rgd.N))
        r_g = rgd.r_g.copy()
        r_g[0] = r_g[1]  # XXX
        self.r1_gg[:] = r_g[None, :]
        self.r2_gg[:] = r_g[:, None]
        self.d_gg[:] = rgd.dr_g[None, :] * rgd.r_g[None, :]**2 * 4 * np.pi
        self.V_lgg = {}

    def screened_coulomb(self, n_g, l):
        # Buffer different l-values for optimal performance
        if l not in self.V_lgg:
            kernel_gg = np.reshape(Phi(l, self.omega, self.r1_gg.ravel(),
                                       self.r2_gg.ravel()),
                                   self.d_gg.shape) / (2 * l + 1)
            self.V_lgg[l] = self.d_gg * kernel_gg
        vr_g = (self.V_lgg[l] @ n_g) * self.rgd.r_g
        return vr_g

    def screened_coulomb_dv(self, n_g, l):
        return self.screened_coulomb(n_g, l) * self.rgd.r_g * self.rgd.dr_g
