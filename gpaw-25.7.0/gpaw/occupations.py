"""Smearing functions and occupation number calculators."""
from __future__ import annotations

import warnings
from math import inf, nan, pi
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, cast

import numpy as np
from ase.units import Ha
from scipy.special import erf

from gpaw.band_descriptor import BandDescriptor
from gpaw.mpi import MPIComm, broadcast_float, serial_comm
from gpaw.typing import Array1D, Array2D


class ParallelLayout(NamedTuple):
    """Collection of parallel stuff."""
    bd: BandDescriptor
    kpt_comm: MPIComm
    domain_comm: MPIComm


def fermi_dirac(eig: np.ndarray,
                fermi_level: float,
                width: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fermi-Dirac distribution function.

    >>> f, _, _ = fermi_dirac(np.array([-1.0, 0.0, 1.0]), 0.0, 0.01)
    >>> f.round(5)
    array([1. , 0.5, 0. ])

    """
    x = (eig - fermi_level) / width
    x = np.clip(x, -100, 100)
    y = np.exp(x)
    z = y + 1.0
    f = 1.0 / z
    dfde = (f - f**2) / width
    y *= x
    y /= z
    y -= np.log(z)
    e_entropy = y * width
    return f, dfde, e_entropy


def marzari_vanderbilt(eig: np.ndarray,
                       fermi_level: float,
                       width: float) -> Tuple[np.ndarray,
                                              np.ndarray,
                                              np.ndarray]:
    """Marzari-Vanderbilt distribution (cold smearing).

    See: :doi:`10.1103/PhysRevLett.82.3296`
    """
    x = (eig - fermi_level) / width
    expterm = np.exp(-(x + (1 / np.sqrt(2)))**2)
    f = expterm / np.sqrt(2 * np.pi) + 0.5 * (1 - erf(1. / np.sqrt(2) + x))
    dfde = expterm * (2 + np.sqrt(2) * x) / np.sqrt(np.pi) / width
    s = expterm * (1 + np.sqrt(2) * x) / (2 * np.sqrt(np.pi))
    e_entropy = -s * width
    return f, dfde, e_entropy


def methfessel_paxton(eig: np.ndarray,
                      fermi_level: float,
                      width: float,
                      order: int = 0) -> Tuple[np.ndarray,
                                               np.ndarray,
                                               np.ndarray]:
    """Methfessel-Paxton distribution."""
    x = (eig - fermi_level) / width
    f = 0.5 * (1 - erf(x))
    for i in range(order):
        f += (coff_function(i + 1) *
              hermite_poly(2 * i + 1, x) * np.exp(-x**2))
    dfde = 1 / np.sqrt(pi) * np.exp(-x**2)
    for i in range(order):
        dfde += (coff_function(i + 1) *
                 hermite_poly(2 * i + 2, x) * np.exp(-x**2))
    dfde *= 1.0 / width
    e_entropy = (0.5 * coff_function(order) *
                 hermite_poly(2 * order, x) * np.exp(-x**2))
    e_entropy = -e_entropy * width
    return f, dfde, e_entropy


def coff_function(n):
    return (-1)**n / (np.prod(np.arange(1, n + 1)) *
                      4**n * np.sqrt(np.pi))


def hermite_poly(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return 2 * x
    else:
        return (2 * x * hermite_poly(n - 1, x) -
                2 * (n - 1) * hermite_poly(n - 2, x))


class OccupationNumberCalculator:
    """Base class for all occupation number calculators."""
    name = 'unknown'
    extrapolate_factor: float

    def __init__(self,
                 parallel_layout: ParallelLayout = None):
        """Object for calculating fermi level(s) and occupation numbers.

        If fixmagmom=True then the fixed_magmom_value attribute must be set
        and two fermi levels will be calculated.
        """
        if parallel_layout is None:
            parallel_layout = ParallelLayout(BandDescriptor(1),
                                             serial_comm,
                                             serial_comm)
        self.bd = parallel_layout.bd
        self.kpt_comm = parallel_layout.kpt_comm
        self.domain_comm = parallel_layout.domain_comm

    @property
    def parallel_layout(self) -> ParallelLayout:
        return ParallelLayout(self.bd, self.kpt_comm, self.domain_comm)

    def todict(self):
        return {'name': self.name}

    def copy(self,
             parallel_layout: ParallelLayout = None,
             bz2ibzmap: List[int] = None) -> 'OccupationNumberCalculator':
        return create_occ_calc(
            self.todict(),
            parallel_layout=parallel_layout or self.parallel_layout)

    def calculate(self,
                  nelectrons: float,
                  eigenvalues: list[list[float]],
                  weights: list[float],
                  fermi_levels_guess: list[float] = None,
                  fix_fermi_level: bool = False) -> tuple[Array2D,
                                                          list[float],
                                                          float]:
        """Calculate occupation numbers and fermi level(s) from eigenvalues.

        nelectrons:
            Number of electrons.
        eigenvalues: ndarray, shape=(nibzkpts, nbands)
            Eigenvalues in Hartree.
        weights: ndarray, shape=(nibzkpts,)
            Weights of k-points in IBZ (must sum to 1).
        parallel:
            Parallel distribution of eigenvalues.
        fermi_level_guesses:
            Optional guess(es) at fermi level(s).

        Returns a tuple containing:

        * occupation numbers (in the range 0 to 1)
        * fermi-level in Hartree
        * entropy as -S*T in Hartree

        >>> occ = ZeroWidth()
        >>> occ.calculate(1, [[0, 1]], [1])
        (array([[1., 0.]]), [0.5], 0.0)
        """

        eig_qn = [np.asarray(eig_n) for eig_n in eigenvalues]
        weight_q = np.asarray(weights)

        if fermi_levels_guess is None:
            fermi_levels_guess = [nan]

        f_qn = np.empty((len(weight_q), len(eig_qn[0])))

        result = np.empty(2)

        if self.domain_comm.rank == 0:
            # Let the master domain do the work and broadcast results:
            result[:] = self._calculate(
                nelectrons, eig_qn, weight_q, f_qn,
                fermi_levels_guess[0], fix_fermi_level)

        self.domain_comm.broadcast(result, 0)
        self.domain_comm.broadcast(f_qn, 0)

        fermi_level, e_entropy = result
        return f_qn, [float(fermi_level)], float(e_entropy)

    def _calculate(self,
                   nelectrons: float,
                   eig_qn: List[Array1D],
                   weight_q: Array1D,
                   f_qn: Array2D,
                   fermi_level_guess: float,
                   fix_fermi_level: bool = False) -> tuple[float, float]:
        raise NotImplementedError


class FixMagneticMomentOccupationNumberCalculator(OccupationNumberCalculator):
    """Base class for all occupation number objects."""
    name = 'fixmagmom'

    def __init__(self,
                 occ: OccupationNumberCalculator,
                 magmom: float):
        """Object for calculating fermi level(s) and occupation numbers.

        If fixmagmom=True then the fixed_magmom_value attribute must be set
        and two fermi levels will be calculated.
        """
        self.occ = occ
        self.fixed_magmom_value = magmom
        self.extrapolate_factor = occ.extrapolate_factor

    def __str__(self):
        return (f'Fixed magnetic moment: {self.fixed_magmom_value:.3f}\n' +
                str(self.occ))

    def todict(self):
        dct = self.occ.todict()
        dct['fixmagmom'] = True
        return dct

    def calculate(self,
                  nelectrons: float,
                  eigenvalues: List[List[float]],
                  weights: List[float],
                  fermi_levels_guess: List[float] = None,
                  fix_fermi_level: bool = False) -> tuple[Array2D,
                                                          List[float],
                                                          float]:

        magmom = self.fixed_magmom_value

        if fermi_levels_guess is None:
            fermi_levels_guess = [nan, nan]

        f1_qn, fermi_levels1, e_entropy1 = self.occ.calculate(
            (nelectrons + magmom) / 2,
            eigenvalues[::2],
            weights[::2],
            fermi_levels_guess[:1],
            fix_fermi_level)

        f2_qn, fermi_levels2, e_entropy2 = self.occ.calculate(
            (nelectrons - magmom) / 2,
            eigenvalues[1::2],
            weights[1::2],
            fermi_levels_guess[1:],
            fix_fermi_level)

        f_qn = []
        for f1_n, f2_n in zip(f1_qn, f2_qn):
            f_qn += [f1_n, f2_n]

        return (np.array(f_qn),
                fermi_levels1 + fermi_levels2,
                e_entropy1 + e_entropy2)


class SmoothDistribution(OccupationNumberCalculator):
    """Base class for Fermi-Dirac and other smooth distributions."""
    def __init__(self, width: float, parallel_layout: ParallelLayout = None):
        """Smooth distribution.

        width: float
            Width of distribution in eV.
        fixmagmom: bool
            Fix spin moment calculations.  A separate Fermi level for
            spin up and down electrons is found.
        """

        self._width = width
        OccupationNumberCalculator.__init__(self, parallel_layout)

    def todict(self):
        return {'name': self.name, 'width': self._width}

    def _calculate(self,
                   nelectrons,
                   eig_qn,
                   weight_q,
                   f_qn,
                   fermi_level_guess,
                   fix_fermi_level):
        # Guess can be nan or inf:
        if not np.isfinite(fermi_level_guess) or self._width == 0.0:
            zero = ZeroWidth(self.parallel_layout)
            fermi_level_guess, _ = zero._calculate(
                nelectrons, eig_qn, weight_q, f_qn)
            if self._width == 0.0 or np.isinf(fermi_level_guess):
                return fermi_level_guess, 0.0

        x = fermi_level_guess

        data = np.empty(3)

        def func(x, data=data):
            """calculate excess electrons (and update occupation numbers)."""
            data[:] = 0.0
            for eig_n, weight, f_n in zip(eig_qn, weight_q, f_qn):
                f_n[:], dfde_n, e_entropy_n = self.distribution(eig_n, x)
                data += [weight * x_n.sum()
                         for x_n in [f_n, dfde_n, e_entropy_n]]
            self.bd.comm.sum(data)
            self.kpt_comm.sum(data)
            f, dfde = data[:2]
            df = f - nelectrons
            return df, dfde

        if fix_fermi_level:
            fermi_level = x
            func(fermi_level)
        else:
            fermi_level, niter = findroot(func, x)

        e_entropy = data[2]

        return fermi_level, e_entropy


class FermiDiracCalculator(SmoothDistribution):
    name = 'fermi-dirac'
    extrapolate_factor = -0.5

    def distribution(self,
                     eig_n: np.ndarray,
                     fermi_level: float) -> Tuple[np.ndarray,
                                                  np.ndarray,
                                                  np.ndarray]:
        return fermi_dirac(eig_n, fermi_level, self._width)

    def __str__(self):
        return f'Fermi-Dirac:\n  width: {self._width:.4f}  # eV\n'


class MarzariVanderbiltCalculator(SmoothDistribution):
    name = 'marzari-vanderbilt'
    # According to Nicola Marzari, one should not extrapolate M-V energies
    # https://lists.quantum-espresso.org/pipermail/users/2005-October/003170.html
    extrapolate_factor = 0.0

    def distribution(self, eig_n, fermi_level):
        return marzari_vanderbilt(eig_n, fermi_level, self._width)

    def __str__(self):
        return f'Marzari-Vanderbilt:\n  width: {self._width:.4f}  # eV\n'


class MethfesselPaxtonCalculator(SmoothDistribution):
    name = 'methfessel_paxton'

    def __init__(self, width, order=0, parallel_layout: ParallelLayout = None):
        SmoothDistribution.__init__(self, width, parallel_layout)
        self.order = order
        self.extrapolate_factor = -1.0 / (self.order + 2)

    def todict(self):
        dct = SmoothDistribution.todict(self)
        dct['order'] = self.order
        return dct

    def __str__(self):
        return ('Methfessel-Paxton:\n'
                f'  width: {self._width:.4f}  # eV\n'
                f'  order: {self.order}\n')

    def distribution(self, eig_n, fermi_level):
        return methfessel_paxton(eig_n, fermi_level, self._width, self.order)


def findroot(func: Callable[[float], Tuple[float, float]],
             x: float,
             tol: float = 1e-10) -> Tuple[float, int]:
    """Function used for locating Fermi level.

    The function should return a (value, derivative) tuple:

    >>> x, _ = findroot(lambda x: (x, 1.0), 1.0)
    >>> assert abs(x) < 1e-10
    """

    assert np.isfinite(x), x

    xmin = -np.inf
    xmax = np.inf

    # Try 10 step using the gradient:
    niter = 0
    while True:
        f, dfdx = func(x)
        if abs(f) < tol:
            return x, niter
        if f < 0.0 and x > xmin:
            xmin = x
        elif f > 0.0 and x < xmax:
            xmax = x
        dx = -f / max(dfdx, 1e-18)
        if niter == 10 or abs(dx) > 0.3 or not (xmin < x + dx < xmax):
            break  # try bisection
        x += dx
        niter += 1

    # Bracket the solution:
    if not np.isfinite(xmin):
        xmin = x
        fmin = f
        step = 0.01
        while fmin > tol:
            xmin -= step
            fmin = func(xmin)[0]
            step *= 2

    if not np.isfinite(xmax):
        xmax = x
        fmax = f
        step = 0.01
        while fmax < 0:
            xmax += step
            fmax = func(xmax)[0]
            step *= 2

    # Bisect:
    while True:
        x = (xmin + xmax) / 2
        f = func(x)[0]
        if abs(f) < tol:
            return x, niter
        if f > 0:
            xmax = x
        else:
            xmin = x
        niter += 1
        assert niter < 1000


def collect_eigelvalues(eig_qn: np.ndarray,
                        weight_q: np.ndarray,
                        bd: BandDescriptor,
                        kpt_comm: MPIComm) -> Tuple[np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray]:
    """Collect eigenvalues from bd.comm and kpt_comm."""
    nkpts_r = np.zeros(kpt_comm.size, int)
    nkpts_r[kpt_comm.rank] = len(weight_q)
    kpt_comm.sum(nkpts_r)
    nk = cast(int, nkpts_r.sum())
    weight_k = np.zeros(nk)
    k1 = nkpts_r[:kpt_comm.rank].sum()
    k2 = k1 + len(weight_q)
    weight_k[k1:k2] = weight_q
    kpt_comm.sum(weight_k, 0)

    eig_kn: Array2D = np.zeros((0, 0))
    k = 0
    for rank, nkpts in enumerate(nkpts_r):
        for q in range(nkpts):
            if rank == kpt_comm.rank:
                eig_n = eig_qn[q]
                eig_n = bd.collect(eig_n)
            if bd.comm.rank == 0:
                if kpt_comm.rank == 0:
                    if k == 0:
                        eig_kn = np.empty((nk, len(eig_n)))
                    if rank == 0:
                        eig_kn[k] = eig_n
                    else:
                        kpt_comm.receive(eig_kn[k], rank)
                elif rank == kpt_comm.rank:
                    kpt_comm.send(eig_n, 0)
            k += 1
    return eig_kn, weight_k, nkpts_r


def distribute_occupation_numbers(f_kn: np.ndarray,  # input
                                  f_qn: np.ndarray,  # output
                                  nkpts_r: np.ndarray,
                                  bd: BandDescriptor,
                                  kpt_comm: MPIComm) -> None:
    """Distribute occupation numbers over bd.comm and kpt_comm."""
    k = 0
    for rank, nkpts in enumerate(nkpts_r):
        for q in range(nkpts):
            if kpt_comm.rank == 0:
                if rank == 0:
                    if bd.comm.size == 1:
                        f_qn[q] = f_kn[k]
                    else:
                        bd.distribute(None if f_kn is None else f_kn[k],
                                      f_qn[q])
                elif f_kn is not None:
                    kpt_comm.send(f_kn[k], rank)
            elif rank == kpt_comm.rank:
                if bd.comm.size == 1:
                    kpt_comm.receive(f_qn[q], 0)
                else:
                    if bd.comm.rank == 0:
                        f_n = bd.empty(global_array=True)
                        kpt_comm.receive(f_n, 0)
                    else:
                        f_n = None
                    bd.distribute(f_n, f_qn[q])
            k += 1


class ZeroWidth(OccupationNumberCalculator):
    name = 'zero-width'
    extrapolate_factor = 0.0

    def todict(self):
        return {'width': 0.0}

    def __str__(self):
        return '# Zero width'

    def distribution(self, eig_n, fermi_level):
        f_n = np.zeros_like(eig_n)
        f_n[eig_n < fermi_level] = 1.0
        f_n[eig_n == fermi_level] = 0.5
        return f_n, np.zeros_like(eig_n), np.zeros_like(eig_n)

    def _calculate(self,
                   nelectrons,
                   eig_qn,
                   weight_q,
                   f_qn,
                   fermi_level_guess=nan,
                   fix_fermi_level=False):
        eig_kn, weight_k, nkpts_r = collect_eigelvalues(eig_qn, weight_q,
                                                        self.bd, self.kpt_comm)

        if eig_kn.size != 0:
            # Try to use integer weights (avoid round-off errors):
            N = int(round(1 / min(weight_k)))
            w_k = (weight_k * N).round().astype(int)
            if abs(w_k - N * weight_k).max() > 1e-10:
                # Did not work.  Use original fractional weights:
                w_k = weight_k
                N = 1

            f_kn = np.zeros_like(eig_kn)
            f_m = f_kn.ravel()
            w_kn = np.empty_like(eig_kn, dtype=w_k.dtype)
            w_kn[:] = w_k[:, np.newaxis]
            eig_m = eig_kn.ravel()
            w_m = w_kn.ravel()
            m_i = eig_m.argsort()
            if fix_fermi_level:
                fermi_level = fermi_level_guess
                f_m[eig_m < fermi_level] = 1.0
                f_m[eig_m == fermi_level] = 0.5
            else:
                w_i = w_m[m_i]
                sum_i = np.add.accumulate(w_i)
                filled_i = (sum_i <= nelectrons * N)
                i = sum(filled_i)
                f_m[m_i[:i]] = 1.0
                if i == len(m_i):
                    fermi_level = inf
                else:
                    extra = nelectrons * N - (sum_i[i - 1] if i > 0 else 0.0)
                    if extra > 0:
                        assert extra <= w_i[i]
                        f_m[m_i[i]] = extra / w_i[i]
                        fermi_level = eig_m[m_i[i]]
                    else:
                        fermi_level = (eig_m[m_i[i]] + eig_m[m_i[i - 1]]) / 2
        else:
            f_kn = None
            fermi_level = nan

        distribute_occupation_numbers(f_kn, f_qn, nkpts_r,
                                      self.bd, self.kpt_comm)

        if self.kpt_comm.rank == 0:
            fermi_level = broadcast_float(fermi_level, self.bd.comm)
        fermi_level = broadcast_float(fermi_level, self.kpt_comm)

        e_entropy = 0.0
        return fermi_level, e_entropy


class FixedOccupationNumbers(OccupationNumberCalculator):
    extrapolate_factor = 0.0

    def __init__(self, numbers, parallel_layout: ParallelLayout = None):
        """Fixed occupation numbers.

        f_sn: ndarray, shape=(nspins, nbands)
            Occupation numbers (in the range from 0 to 1)

        Example (excited state with 4 electrons)::

            occ = FixedOccupationNumbers([[1, 0, 1, 0], [1, 1, 0, 0]])

        """
        OccupationNumberCalculator.__init__(self, parallel_layout)
        self.f_sn = np.array(numbers)

    def _calculate(self,
                   nelectrons,
                   eig_qn,
                   weight_q,
                   f_qn,
                   fermi_level_guess=nan,
                   fix_fermi_level=False):

        calc_fixed(self.bd, self.f_sn, f_qn)

        return inf, 0.0

    def todict(self):
        return {'name': 'fixed', 'numbers': self.f_sn}


class FixedOccupationNumbersUniform(OccupationNumberCalculator):

    extrapolate_factor = 0.0
    name = 'fixed-uniform'

    def __init__(self, nelectrons, nspins, magmom, nkpts, nbands,
                 parallel_layout: ParallelLayout = None):
        """
        Uniform distribution of occupation numbers: each
        k-point per spin has the same number of occupied states
        Magnetic moment defines difference between two spins occ. numb.
        """
        OccupationNumberCalculator.__init__(self, parallel_layout)

        def get_f(nelectrons, magmom, nkpts, nbands, spin):
            """
            :param nelectrons:
            :param magmom:
            :param nkpts:
            :param nbands:
            :param spin: +1 or -1
            :return: occupation numbers per spin channel
            """

            f_qn = np.zeros(shape=(nkpts, nbands))
            nelecps = (nelectrons + spin * magmom) / 2
            assert int(nelecps) < nbands, 'need more bands!'

            f_qn[:, : int(nelecps)] = 1.0
            f_qn[:, int(nelecps)] = nelecps - int(nelecps)

            return f_qn

        if nspins == 2:
            f1_qn = get_f(nelectrons, magmom, nkpts, nbands, 1)
            f2_qn = get_f(nelectrons, magmom, nkpts, nbands, -1)
            f_qn = []
            for f1_n, f2_n in zip(f1_qn, f2_qn):
                f_qn += [f1_n, f2_n]
        else:
            f_qn = get_f(nelectrons, magmom, nkpts, nbands, 0)

        self.f_sn = np.array(f_qn)
        self.nspins = nspins
        self.magmom = magmom

    def _calculate(self,
                   nelectrons,
                   eig_qn,
                   weight_q,
                   f_qn,
                   fermi_level_guess=nan,
                   fix_fermi_level=False):
        assert not fix_fermi_level

        calc_fixed(self.bd, self.f_sn, f_qn)

        eig_kn, weight_k, nkpts_r = collect_eigelvalues(
            eig_qn, weight_q, self.bd, self.kpt_comm)

        def get_homo(eig_kn, nelectrons, deg, magmom, spin):
            nelecps = int((nelectrons * deg + spin * magmom) / 2)
            return np.max(eig_kn[:, np.maximum(nelecps - 1, 0)])

        def get_lumo(eig_kn, nelectrons, deg, magmom, spin):
            nelecps = int((nelectrons * deg + spin * magmom) / 2)
            return np.min(eig_kn[:, nelecps])

        deg = 3 - self.nspins
        mm = self.magmom
        if eig_kn.size != 0:
            if self.nspins == 2:
                hup = get_homo(eig_kn[::2], nelectrons, deg, mm, 1)
                hdown = get_homo(eig_kn[1::2], nelectrons, deg, mm, -1)
                homo = np.maximum(hup, hdown)

                lup = get_lumo(eig_kn[::2], nelectrons, deg, mm, 1)
                ldown = get_lumo(eig_kn[1::2], nelectrons, deg, mm, -1)
                lumo = np.maximum(lup, ldown)

            else:
                homo = get_homo(eig_kn, nelectrons, deg, mm, 0)
                lumo = get_lumo(eig_kn, nelectrons, deg, mm, 0)
            fermi_level = (homo + lumo) / 2
        else:
            fermi_level = nan

        if self.kpt_comm.rank == 0:
            fermi_level = broadcast_float(fermi_level, self.bd.comm)
        fermi_level = broadcast_float(fermi_level, self.kpt_comm)

        return fermi_level, 0.0

    def todict(self):
        return {'name': 'fixed-uniform'}

    def __str__(self):
        return '# Uniform distribution of occupation numbers'


def calc_fixed(bd, f_sn, f_qn):
    if bd.nbands == f_sn.shape[1]:
        for q, f_n in enumerate(f_qn):
            s = q % len(f_sn)
            bd.distribute(f_sn[s], f_n)
    else:
        # Non-collinear calculation:
        bd.distribute(f_sn.T.flatten().copy(), f_qn[0])


def FixedOccupations(f_sn):
    warnings.warn(
        "Please use occupations={'name': 'fixed', 'numbers': ...} instead.")
    if len(f_sn) == 1:
        f_sn = np.array(f_sn) / 2
    return {'name': 'fixed', 'numbers': f_sn}


class ThomasFermiOccupations(OccupationNumberCalculator):
    name = 'orbital-free'
    extrapolate_factor = 0.0

    def _calculate(self,
                   nelectrons,
                   eig_qn,
                   weight_q,
                   f_qn,
                   fermi_level_guess=nan,
                   fix_fermi_level=False):
        assert len(f_qn) == 1
        f_qn[0][:] = [nelectrons]
        return inf, 0.0


def create_occ_calc(dct: Dict[str, Any],
                    *,
                    parallel_layout: ParallelLayout = None,
                    fixed_magmom_value=None,
                    rcell=None,
                    monkhorst_pack_size=None,
                    bz2ibzmap=None,
                    nspins=None,
                    nelectrons=None,
                    nkpts=None,
                    nbands=None
                    ) -> OccupationNumberCalculator:
    """Surprise: Create occupation-number object.

    The unit of width is eV and name must be one of:

    * 'fermi-dirac'
    * 'marzari-vanderbilt'
    * 'methfessel-paxton'
    * 'fixed'
    * 'tetrahedron-method'
    * 'improved-tetrahedron-method'
    * 'orbital-free'

    >>> occ = create_occ_calc({'width': 0.0})
    >>> occ.calculate(nelectrons=3,
    ...               eigenvalues=[[0, 1, 2], [0, 2, 3]],
    ...               weights=[1, 1])
    (array([[1., 1., 0.],
           [1., 0., 0.]]), [1.5], 0.0)
    """
    kwargs = dct.copy()
    fix_the_magnetic_moment = kwargs.pop('fixmagmom', False)
    name = kwargs.pop('name', '')
    kwargs['parallel_layout'] = parallel_layout

    if name == 'unknown':
        return OccupationNumberCalculator(**kwargs)

    occ: OccupationNumberCalculator

    if kwargs.get('width') == 0.0:
        del kwargs['width']
        occ = ZeroWidth(**kwargs)
    elif name == 'methfessel-paxton':
        occ = MethfesselPaxtonCalculator(**kwargs)
    elif name == 'fermi-dirac':
        occ = FermiDiracCalculator(**kwargs)
    elif name == 'marzari-vanderbilt':
        occ = MarzariVanderbiltCalculator(**kwargs)
    elif name in {'tetrahedron-method', 'improved-tetrahedron-method'}:
        from gpaw.tetrahedron import TetrahedronMethod
        occ = TetrahedronMethod(rcell,
                                monkhorst_pack_size,
                                name == 'improved-tetrahedron-method',
                                bz2ibzmap,
                                **kwargs)
    elif name == 'orbital-free':
        return ThomasFermiOccupations(**kwargs)
    elif name == 'fixed':
        return FixedOccupationNumbers(**kwargs)
    elif name == 'fixed-uniform':
        return FixedOccupationNumbersUniform(
            nelectrons, nspins, fixed_magmom_value,
            nkpts, nbands, **kwargs)
    else:
        raise ValueError(f'Unknown occupation number object name: {name}')

    if fix_the_magnetic_moment:
        occ = FixMagneticMomentOccupationNumberCalculator(
            occ, fixed_magmom_value)

    return occ


def occupation_numbers(occ, eig_skn, weight_k, nelectrons):
    """Calculate occupation numbers from eigenvalues in eV (**deprecated**).

    occ: dict
        Example: {'name': 'fermi-dirac', 'width': 0.05} (width in eV).
    eps_skn: ndarray, shape=(nspins, nibzkpts, nbands)
        Eigenvalues.
    weight_k: ndarray, shape=(nibzkpts,)
        Weights of k-points in IBZ (must sum to 1).
    nelectrons: int or float
        Number of electrons.

    Returns a tuple containing:

    * f_skn (sums to nelectrons)
    * fermi-level [Hartree]
    * magnetic moment
    * entropy as -S*T [Hartree]
    """

    warnings.warn('Please use one of the OccupationNumbers implementations',
                  DeprecationWarning)
    occ = create_occ_calc(occ)
    f_kn, (fermi_level,), e_entropy = occ.calculate(
        nelectrons * len(eig_skn) / 2,
        [eig_n for eig_kn in eig_skn for eig_n in eig_kn],
        list(weight_k) * len(eig_skn))

    f_kn *= np.array(weight_k)[:, np.newaxis]

    if len(eig_skn) == 1:
        f_skn = np.array([f_kn]) * 2
        e_entropy *= 2
        magmom = 0.0
    else:
        f_skn = np.array([f_kn[::2], f_kn[1::2]])
        f1, f2 = f_skn.sum(axis=(1, 2))
        magmom = f1 - f2

    return f_skn, fermi_level * Ha, magmom, e_entropy * Ha


def FermiDirac(width, fixmagmom=False):
    return dict(name='fermi-dirac', width=width, fixmagmom=fixmagmom)


def MarzariVanderbilt(width, fixmagmom=False):
    return dict(name='marzari-vanderbilt', width=width, fixmagmom=fixmagmom)


def MethfesselPaxton(width, order=0, fixmagmom=False):
    return dict(name='methfessel-paxton', width=width, order=order,
                fixmagmom=fixmagmom)
