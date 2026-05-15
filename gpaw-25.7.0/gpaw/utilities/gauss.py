import numpy as np
from numpy import sqrt, pi, exp, abs
from scipy.special import erf

import gpaw.cgpaw as cgpaw
from gpaw import debug
from gpaw.utilities.tools import coordinates
from gpaw.utilities import is_contiguous


def Y_L(L, x, y, z, r2):
    if L == 0:
        return 0.28209479177387814
    elif L == 1:
        return 0.48860251190291992 * y
    elif L == 2:
        return 0.48860251190291992 * z
    elif L == 3:
        return 0.48860251190291992 * x
    elif L == 4:
        return 1.0925484305920792 * x * y
    elif L == 5:
        return 1.0925484305920792 * y * z
    elif L == 6:
        return 0.31539156525252005 * (3 * z * z - r2)
    elif L == 7:
        return 1.0925484305920792 * x * z
    elif L == 8:
        return 0.54627421529603959 * (x * x - y * y)


def gauss_L(a, L, x, y, z, r2, exp_ar2):
    if L == 0:
        return sqrt(a**3 * 4) / pi * exp_ar2
    elif L == 1:
        return sqrt(a**5 * 5.333333333333333) / pi * y * exp_ar2
    elif L == 2:
        return sqrt(a**5 * 5.333333333333333) / pi * z * exp_ar2
    elif L == 3:
        return sqrt(a**5 * 5.333333333333333) / pi * x * exp_ar2
    elif L == 4:
        return sqrt(a**7 * 4.2666666666666666) / pi * x * y * exp_ar2
    elif L == 5:
        return sqrt(a**7 * 4.2666666666666666) / pi * y * z * exp_ar2
    elif L == 6:
        return sqrt(
            a**7 * 0.35555555555555557) / pi * (3 * z * z - r2) * exp_ar2
    elif L == 7:
        return sqrt(a**7 * 4.2666666666666666) / pi * x * z * exp_ar2
    elif L == 8:
        return sqrt(a**7 * 1.0666666666666667) / pi * (x * x - y * y) * exp_ar2


def gausspot_L(a, L, x, y, z, r, r2, erf_sar, exp_ar2):
    if L == 0:
        return 2.0 * 1.7724538509055159 * erf_sar / r
    elif L == 1:
        return 1.1547005383792515 * (1.7724538509055159 * erf_sar -
                                     2 * sqrt(a) * r * exp_ar2) / r / r2 * y
    elif L == 2:
        return 1.1547005383792515 * (1.7724538509055159 * erf_sar -
                                     2 * sqrt(a) * r * exp_ar2) / r / r2 * z
    elif L == 3:
        return 1.1547005383792515 * (1.7724538509055159 * erf_sar -
                                     2 * sqrt(a) * r * exp_ar2) / r / r2 * x
    elif L == 4:
        return 0.5163977794943222 * (
            5.3173615527165481 * erf_sar -
            (6 + 4 *
             (sqrt(a) * r)**2) * sqrt(a) * r * exp_ar2) / r / r2**2 * x * y
    elif L == 5:
        return 0.5163977794943222 * (
            5.3173615527165481 * erf_sar -
            (6 + 4 *
             (sqrt(a) * r)**2) * sqrt(a) * r * exp_ar2) / r / r2**2 * y * z
    elif L == 6:
        return 0.14907119849998599 * (5.3173615527165481 * erf_sar -
                                      (6 + 4 *
                                       (sqrt(a) * r)**2) * sqrt(a) * r *
                                      exp_ar2) / r / r2**2 * (3 * z * z - r2)
    elif L == 7:
        return 0.5163977794943222 * (
            5.3173615527165481 * erf_sar -
            (6 + 4 *
             (sqrt(a) * r)**2) * sqrt(a) * r * exp_ar2) / r / r2**2 * x * z
    elif L == 8:
        return 0.2581988897471611 * (5.3173615527165481 * erf_sar -
                                     (6 + 4 * (sqrt(a) * r)**2) * sqrt(a) * r *
                                     exp_ar2) / r / r2**2 * (x * x - y * y)


# end of computer generated code


class Gaussian:
    r"""Class offering several utilities related to the generalized gaussians.

    Generalized gaussians are defined by::

                       _____                           2
                      /  1       l!         l+3/2  -a r   l  m
       g (x,y,z) =   / ----- --------- (4 a)      e      r  Y (x,y,z),
        L          \/  4 pi  (2l + 1)!                       l

    where a is the inverse width of the gaussian, and Y_l^m is a real
    spherical harmonic.
    The gaussians are centered in the middle of input grid-descriptor."""
    def __init__(self, gd, a=19., center=None):
        self.gd = gd
        self.xyz, self.r2 = coordinates(gd, center)
        self.r = np.sqrt(self.r2)
        self.set_width(a, center)
        self.exp_ar2 = exp(-self.a * self.r2)
        self.erf_sar = erf(sqrt(self.a) * self.r)

    def set_width(self, a, center):
        """Set exponent of exp-function to -a on the boundary."""
        if center is None:
            self.a = 4 * a * (self.gd.icell_cv**2).sum(1).max()
        else:
            cell_center = self.gd.cell_cv.sum(1) / 2
            r_min = (cell_center - np.abs(center - cell_center)).min()
            self.a = a / r_min**2

    def get_gauss(self, L):
        a = self.a
        x, y, z = tuple(self.xyz)
        r2 = self.r2
        exp_ar2 = self.exp_ar2
        return gauss_L(a, L, x, y, z, r2, exp_ar2)

    def get_gauss_pot(self, L):
        a = self.a
        x, y, z = tuple(self.xyz)
        r2 = self.r2
        r = self.r
        erf_sar = self.erf_sar
        exp_ar2 = self.exp_ar2
        return gausspot_L(a, L, x, y, z, r, r2, erf_sar, exp_ar2)

    def get_moment(self, n, L):
        r2 = self.r2
        x, y, z = tuple(self.xyz)
        return self.gd.integrate(n * Y_L(L, x, y, z, r2))

    def remove_moment(self, n, L, q=None):
        # Determine multipole moment
        if q is None:
            q = self.get_moment(n, L)

        # Don't do anything if moment is less than the tolerance
        if abs(q) < 1e-7:
            return 0.

        # Remove moment from input density
        n -= q * self.get_gauss(L)

        # Return correction
        return q * self.get_gauss_pot(L)


def gaussian_wave(r_vG, r0_v, sigma, k_v=None, A=None, dtype=float,
                  out_G=None):
    r"""Generates function values for atom-centered Gaussian waves.

    ::

                         _ _
        _            / -|r-r0|^2 \           _ _
      f(r) = A * exp( ----------- ) * exp( i k.r )
                     \ 2 sigma^2 /

    If the parameter A is not specified, the Gaussian wave is normalized::

                                                  oo
           /    ____        \ -3/2               /       _  2  2
      A = (    /    '        )        =>    4 pi | dr |f(r)|  r  = 1
           \ \/  pi   sigma /                    /
                                                   0

    Parameters:

    r_vG: ndarray
        Set of coordinates defining the grid positions.
    r0_v: ndarray
        Set of coordinates defining the center of the Gaussian envelope.
    sigma: float
        Specifies the spatial width of the Gaussian envelope.
    k_v: ndarray or None
        Set of reciprocal lattice coordinates defining the wave vector.
        An argument of None is interpreted as the gamma point i.e. k_v=0.
    A: float, complex or None
        Specifies the amplitude of the Gaussian wave. Normalizes if None.
    dtype: type, defaults to float
        Specifies the output data type. Only returns the real-part if float.
    out_G: ndarray or None
        Optional pre-allocated buffer to fill in values. Allocates if None.

    """
    if k_v is None:
        k_v = np.zeros(r0_v.shape)

    if A is None:
        # 4*pi*int(exp(-r^2/(2*sigma^2))^2 * r^2, r=0...infinity)
        # = sigma^3*pi^(3/2) = 1/A^2 -> A = (sqrt(Pi)*sigma)^(-3/2)
        A = 1 / (sigma * np.pi**0.5)**1.5

    if debug:
        assert is_contiguous(r_vG, float)
        assert is_contiguous(r0_v, float)
        assert is_contiguous(k_v, float)
        assert r_vG.ndim >= 2 and r_vG.shape[0] > 0
        assert r0_v.ndim == 1 and r0_v.shape[0] > 0
        assert k_v.ndim == 1 and k_v.shape[0] > 0
        assert (r_vG.shape[0], ) == r0_v.shape == k_v.shape
        assert sigma > 0

    if out_G is None:
        out_G = np.empty(r_vG.shape[1:], dtype=dtype)
    elif debug:
        assert is_contiguous(out_G)
        assert out_G.shape == r_vG.shape[1:]

    # slice_v2vG = [slice(None)] + [np.newaxis]*3
    # gw = lambda r_vG, r0_v, sigma, k_v, A=1/(sigma*np.pi**0.5)**1.5: \
    #    * np.exp(-np.sum((r_vG-r0_v[slice_v2vG])**2, axis=0)/(2*sigma**2)) \
    #    * np.exp(1j*np.sum(np.r_vG*k_v[slice_v2vG], axis=0)) * A
    cgpaw.utilities_gaussian_wave(A, r_vG, r0_v, sigma, k_v, out_G)
    return out_G
