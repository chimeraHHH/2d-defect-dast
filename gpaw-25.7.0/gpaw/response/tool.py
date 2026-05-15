import sys

import numpy as np
from scipy.optimize import leastsq

from ase.units import Ha
import gpaw.mpi as mpi
from gpaw.response.integrators import Domain
from gpaw.response.qpd import SingleQPWDescriptor
from gpaw.response.pair import KPointPairFactory, get_gs_and_context


def check_degenerate_bands(filename, etol):

    from gpaw import GPAW
    calc = GPAW(filename, txt=None)
    print('Number of Electrons   :', calc.get_number_of_electrons())
    nibzkpt = calc.get_ibz_k_points().shape[0]
    nbands = calc.get_number_of_bands()
    print('Number of Bands       :', nbands)
    print('Number of ibz-kpoints :', nibzkpt)
    e_kn = np.array([calc.get_eigenvalues(k) for k in range(nibzkpt)])
    f_kn = np.array([calc.get_occupation_numbers(k) for k in range(nibzkpt)])
    for k in range(nibzkpt):
        for n in range(1, nbands):
            if (f_kn[k, n - 1] - f_kn[k, n] > 1e-5)\
               and (np.abs(e_kn[k, n] - e_kn[k, n - 1]) < etol):
                print(k, n, e_kn[k, n], e_kn[k, n - 1])


def get_orbitals(calc):
    """Get LCAO orbitals on 3D grid by lcao_to_grid method."""

    bfs_a = [setup.basis_functions_J for setup in calc.wfs.setups]

    from gpaw.lfc import BasisFunctions
    bfs = BasisFunctions(calc.wfs.gd, bfs_a, calc.wfs.kd.comm, cut=True)
    bfs.set_positions(calc.spos_ac)

    nLCAO = calc.get_number_of_bands()
    orb_MG = calc.wfs.gd.zeros(nLCAO)
    C_M = np.identity(nLCAO)
    bfs.lcao_to_grid(C_M, orb_MG, q=-1)

    return orb_MG


def get_bz_transitions(filename, q_c, bzk_kc,
                       spins='all',
                       ecut=50, txt=sys.stdout):
    """
    Get transitions in the Brillouin zone from kpoints bzk_kv
    contributing to the linear response at wave vector q_c.
    """

    ecut /= Ha

    gs, context = get_gs_and_context(filename, txt=txt, world=mpi.world,
                                     timer=None)

    kptpair_factory = KPointPairFactory(gs=gs, context=context)
    qpd = SingleQPWDescriptor.from_q(q_c, ecut, gs.gd)
    bzk_kv = np.dot(bzk_kc, qpd.gd.icell_cv) * 2 * np.pi

    if spins == 'all':
        spins = range(kptpair_factory.gs.nspins)
    else:
        for spin in spins:
            assert spin in range(kptpair_factory.gs.nspins)

    domain = Domain(bzk_kv, spins)
    return kptpair_factory, qpd, domain


def get_chi0_integrand(kptpair_factory, qpd, n_n, m_m, point):
    """
    Calculates the pair densities, occupational differences
    and energy differences of transitions from certain kpoint
    and spin.
    """

    k_v = point.kpt_c
    optical_limit = qpd.optical_limit
    k_c = np.dot(qpd.gd.cell_cv, k_v) / (2 * np.pi)
    K = kptpair_factory.gs.kpoints.kptfinder.find(k_c)

    pair_calc = kptpair_factory.pair_calculator()
    kptpair = kptpair_factory.get_kpoint_pair(
        qpd, point.spin, K, n_n[0], n_n[-1] + 1,
        m_m[0], m_m[-1] + 1)

    pairden_paw_corr = pair_calc.gs.pair_density_paw_corrections
    pawcorr = pairden_paw_corr(qpd)

    df_nm = kptpair.get_occupation_differences()
    eps_n = kptpair.kpt1.eps_n
    eps_m = kptpair.kpt2.eps_n

    if optical_limit:
        n_nmP = pair_calc.get_optical_pair_density(
            qpd, kptpair, n_n, m_m, pawcorr=pawcorr)

        return n_nmP, df_nm, eps_n, eps_m
    else:
        n_nmG = pair_calc.get_pair_density(
            qpd, kptpair, n_n, m_m, pawcorr=pawcorr)

        return n_nmG, df_nm, eps_n, eps_m


def get_degeneracy_matrix(eps_n, tol=1.e-3):
    """
    Generate a matrix that can sum over degenerate values.
    """
    degmat = []
    eps_N = []
    nn = len(eps_n)
    nstart = 0
    while nstart < nn:
        deg = [0] * nstart + [1]
        eps_N.append(eps_n[nstart])
        for n in range(nstart + 1, nn):
            if abs(eps_n[nstart] - eps_n[n]) < tol:
                deg += [1]
                nstart += 1
            else:
                break
        deg += [0] * (nn - len(deg))
        degmat.append(deg)
        nstart += 1

    return np.array(degmat), np.array(eps_N)


def get_individual_transition_strengths(n_nmG, df_nm, G1, G2):
    return (df_nm * n_nmG[:, :, G1] * n_nmG[:, :, G2].conj()).real


def find_peaks(x, y, threshold=None):
    """ Find peaks for a certain curve.

    Usage:
    threshold = (xmin, xmax, ymin, ymax)

    """

    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert x.ndim == 1 and y.ndim == 1
    assert x.shape[0] == y.shape[0]

    if threshold is None:
        threshold = (x.min(), x.max(), y.min(), y.max())

    if not isinstance(threshold, tuple):
        threshold = (threshold, )

    if len(threshold) == 1:
        threshold += (x.max(), y.min(), y.max())
    elif len(threshold) == 2:
        threshold += (y.min(), y.max())
    elif len(threshold) == 3:
        threshold += (y.max(),)
    else:
        pass

    xmin = threshold[0]
    xmax = threshold[1]
    ymin = threshold[2]
    ymax = threshold[3]

    peak = {}
    npeak = 0
    for i in range(1, x.shape[0] - 1):
        if (y[i] >= ymin and y[i] <= ymax and
            x[i] >= xmin and x[i] <= xmax):
            if y[i] > y[i - 1] and y[i] > y[i + 1]:
                peak[npeak] = np.array([x[i], y[i]])
                npeak += 1

    peakarray = np.zeros([npeak, 2])
    for i in range(npeak):
        peakarray[i] = peak[i]

    return peakarray


def lorz_fit(x, y, npeak=1, initpara=None):
    """ Fit curve using Lorentzian function

    Note: currently only valid for one and two lorentizian

    The lorentzian function is defined as::

                      A w
        lorz = --------------------- + y0
                (x-x0)**2 + w**2

    where A is the peak amplitude, w is the width, (x0,y0) the peak position

    Parameters:

    x, y: ndarray
        Input data for analyze
    p: ndarray
        Parameters for curving fitting function. [A, x0, y0, w]
    p0: ndarray
        Parameters for initial guessing. similar to p

    """

    def residual(p, x, y):

        err = y - lorz(x, p, npeak)
        return err

    def lorz(x, p, npeak):

        if npeak == 1:
            return p[0] * p[3] / ((x - p[1])**2 + p[3]**2) + p[2]
        if npeak == 2:
            return (p[0] * p[3] / ((x - p[1])**2 + p[3]**2) + p[2]
                    + p[4] * p[7] / ((x - p[5])**2 + p[7]**2) + p[6])
        else:
            raise ValueError('Larger than 2 peaks not supported yet!')

    if initpara is None:
        if npeak == 1:
            initpara = np.array([1., 0., 0., 0.1])
        if npeak == 2:
            initpara = np.array([1., 0., 0., 0.1,
                                 3., 0., 0., 0.1])
    p0 = initpara

    result = leastsq(residual, p0, args=(x, y), maxfev=2000)

    yfit = lorz(x, result[0], npeak)

    return yfit, result[0]


def linear_fit(x, y, initpara=None):
    def residual(p, x, y):
        err = y - linear(x, p)
        return err

    def linear(x, p):
        return p[0] * x + p[1]

    if initpara is None:
        initpara = np.array([1.0, 1.0])

    p0 = initpara
    result = leastsq(residual, p0, args=(x, y), maxfev=2000)
    yfit = linear(x, result[0])

    return yfit, result[0]


def plot_setfont():
    import matplotlib.pyplot as plt
    params = {'axes.labelsize': 18,
              'text.fontsize': 18,
              'legend.fontsize': 18,
              'xtick.labelsize': 18,
              'ytick.labelsize': 18,
              'text.usetex': True}
    #          'figure.figsize': fig_size}
    plt.rcParams.update(params)


def plot_setticks(x=True, y=True):
    import matplotlib.pyplot as plt
    plt.minorticks_on()
    ax = plt.gca()
    if x:
        ax.xaxis.set_major_locator(plt.AutoLocator())
        x_major = ax.xaxis.get_majorticklocs()
        dx_minor = (x_major[-1] - x_major[0]) / (len(x_major) - 1) / 5.
        ax.xaxis.set_minor_locator(plt.MultipleLocator(dx_minor))
    else:
        plt.minorticks_off()

    if y:
        ax.yaxis.set_major_locator(plt.AutoLocator())
        y_major = ax.yaxis.get_majorticklocs()
        dy_minor = (y_major[-1] - y_major[0]) / (len(y_major) - 1) / 5.
        ax.yaxis.set_minor_locator(plt.MultipleLocator(dy_minor))
    else:
        plt.minorticks_off()
