from __future__ import annotations
import numpy as np
from gpaw.typing import Array1D


def sampling_branches(w_dist: Array1D,
                      parallel_lines: int = 2,
                      varpi: float = 1,
                      eta0: float = 1e-5,
                      eta_rest: float = 0.1) -> Array1D:
    """
        w_dist         Array of points in the positive real axis.
        parallel_lines How many (1-2) parallel lines to the real frequency axis
                       the sampling has.
        varpi          Distance of the second line to the real axis.
        eta0           Imaginary part of the first point of the first line.
        eta_rest       Imaginary part of the rest of the points of the first
                       line.
    """
    if parallel_lines not in [1, 2]:
        raise ValueError('parallel_lines must be either 1 or 2.')

    if len(w_dist) == 1:
        assert eta0 >= 0
        assert parallel_lines == 2
        w_grid = np.concatenate([w_dist + 1j * eta0, w_dist + 1j * varpi])
        return w_grid

    if parallel_lines == 1:  # only one branch
        assert varpi >= 0
        w_grid = w_dist + 1j * varpi
        return w_grid

    # parallel lines == 2
    assert eta0 >= 0
    assert eta_rest >= 0
    assert varpi > eta0 and varpi > eta_rest
    w_grid = np.concatenate((np.array([w_dist[0] + 1j * eta0]),
                            w_dist[1:] + 1j * eta_rest, w_dist + 1j * varpi))
    assert len(w_grid.shape) == 1
    return w_grid


def frequency_distribution(npoles: int,
                           wrange: tuple[float, float] | Array1D,
                           alpha: float = 1) -> Array1D:

    if npoles == 1:
        w_grid = np.array([0.])
        return w_grid

    assert wrange[0].real >= 0
    if not wrange[1].real > wrange[0].real:
        raise ValueError('Frequency range inverted.')

    if alpha == 0:  # homogeneous distribution
        w_grid = np.linspace(wrange[0], wrange[1], 2 * npoles)
        return w_grid

    ws = wrange[1] - wrange[0]
    w_grid = wrange[0] + ws * semi_homogenous_partition(npoles)**alpha
    return w_grid


def mpa_frequency_sampling(npoles: int,
                           wrange: tuple[float, float],
                           varpi: float,
                           eta0: float,
                           eta_rest: float,
                           parallel_lines: int = 2,
                           alpha: float = 1) -> Array1D:
    """
    This function creates a frequency grid in the complex plane.
    The grid can have 1 or 2 branches with points non homogeneously
    distributed along the real frequency axis.
    See Fig. 1 and Eq. (18) of Ref. [1], and Eq. (11) of Ref. [2]
    [1] DA. Leon et al, PRB 104, 115157 (2021)
    [2] DA. Leon et al, PRB 107, 155130 (2023)

    Parameters
    ----------
    npoles : numper of poles (half the number of frequency points)
    wrange : [w_start, w_end]. An array of two complex numbers defining
             the sampling range
    varpi : immaginary part of the second branch
    eta0 : damping factor for the first point of the first branch
    eta_rest : damping for the rest of the points of the first branch
    parallel_lines : Either 1 or 2, how many lines there are parallel
                     to the real axis.
    alpha : exponent of the distribution of points along the real axis
    ______________________________________________________________________
                  Example: double parallel sampling with 9 poles
    ----------------------------------------------------------------------
                            complex frequency plane
    ----------------------------------------------------------------------
    |     (wrange[0], varpi) ... . . . . .   . (wrange[1], varpi)        |
    |                                                                    |
    |                         .. . . . . .   . (wrange[1], eta_rest)     |
    |      (wrange[0], eta0) .                                           |
    |____________________________________________________________________|
    """
    _wrange = np.array(wrange)
    grid_p = frequency_distribution(npoles, _wrange, alpha)
    grid_w = sampling_branches(grid_p, parallel_lines=parallel_lines,
                               varpi=varpi, eta0=eta0, eta_rest=eta_rest)
    return grid_w


def semi_homogenous_partition(npoles: int) -> Array1D:
    """
    Returns a semi-homogenous partition with n-poles between 0 and 1
    according to Eq. (18) of Ref. [1], and Eq. (11) of Ref. [2]
    [1] DA. Leon et al, PRB 104, 115157 (2021)
    [2] DA. Leon et al, PRB 107, 155130 (2023)
    """
    small_cases = {1: np.array([0.0]),
                   2: np.array([0.0, 1.0]),
                   3: np.array([0.0, 0.5, 1.0])}
    if npoles < 4:
        return small_cases[npoles]
    # Calculate the grid spacing
    # Round up to the next power of 2. This will determine the minimum spacing
    dw = 1 / 2**np.ceil(np.log2(npoles))

    # Get the previous power of two, by searching down,
    # e.g. lp(4) = 2, lp(7)=4, lp(8)=4, lp(9)=8
    lp = 2**int(np.log2(npoles - 1))

    # There are usually 2 kinds of intervals in a semi homogenous grid,
    # they are always in order such that smallest intervals are closer to zero.
    # The interval sizes are dw, 2*dw.
    # There is an exception to this rule when npoles == power of two + 1,
    # because then there would be only one type of interval with the rule
    # below. To keep the grid inhomogenous, one adds the third interval 4 * dw.
    if npoles == lp + 1:
        np1 = 2
        np3 = 1
    else:
        np1 = (npoles - (lp + 1)) * 2
        np3 = 0
    # The number of intervals is always one less, than the number of points
    # in the grid. Therefore, We can deduce np2 from np1 and np3.
    np2 = npoles - 1 - np1 - np3

    # Build up the intervals onto an array
    dw_n = np.repeat([0, 1, 2, 4], [1, np1, np2, np3])

    # Sum over the intervals to build the point grid
    w_grid = np.cumsum(dw_n) * dw
    return w_grid
