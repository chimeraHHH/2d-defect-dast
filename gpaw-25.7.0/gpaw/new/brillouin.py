"""Brillouin-zone sampling."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from ase.dft.kpoints import monkhorst_pack
from gpaw.mpi import MPIComm
from gpaw.typing import Array1D, ArrayLike2D
from gpaw.symmetry import reduce_kpts
if TYPE_CHECKING:
    from gpaw.new.symmetry import Symmetries


class BZPoints:
    def __init__(self, points: ArrayLike2D):
        self.kpt_Kc = np.array(points)
        assert self.kpt_Kc.ndim == 2
        assert self.kpt_Kc.shape[1] == 3
        self.gamma_only = len(self.kpt_Kc) == 1 and not self.kpt_Kc.any()

    def __len__(self):
        """Number of k-points in the BZ."""
        return len(self.kpt_Kc)

    def __repr__(self):
        if self.gamma_only:
            return 'BZPoints([<gamma only>])'
        return f'BZPoints([<{len(self)} points>])'

    def reduce(self,
               symmetries: Symmetries,
               *,
               comm: MPIComm = None,
               strict: bool = True,
               use_time_reversal=True,
               tolerance=1e-7) -> IBZ:
        """Find irreducible set of k-points."""
        if not use_time_reversal and len(symmetries) == 1:
            N = len(self)
            return IBZ(symmetries,
                       self,
                       ibz2bz=np.arange(N),
                       bz2ibz=np.arange(N),
                       weights=np.ones(N) / N,
                       bz2bz_Ks=np.arange(N).reshape((N, 1)),
                       s_K=np.zeros(N, int),
                       time_reversal_K=np.zeros(N, bool))

        if symmetries.has_inversion:
            use_time_reversal = False
        (_, weight_k, sym_K, time_reversal_K, bz2ibz_K, ibz2bz_k,
         bz2bz_Ks) = reduce_kpts(self.kpt_Kc,
                                 symmetries.rotation_scc,
                                 use_time_reversal,
                                 comm,
                                 tolerance)

        if strict and -1 in bz2bz_Ks:
            raise ValueError(
                'Your k-points are not as symmetric as your crystal!')

        return IBZ(symmetries, self, ibz2bz_k, bz2ibz_K, weight_k, bz2bz_Ks,
                   sym_K, time_reversal_K)


class BZBandPath(BZPoints):
    def __init__(self, band_path):
        self.band_path = band_path
        super().__init__(band_path.kpts)


class MonkhorstPackKPoints(BZPoints):
    def __init__(self, size, shift=(0, 0, 0)):
        self.size_c = size
        self.shift_c = np.array(shift)
        super().__init__(monkhorst_pack(size) + shift)

    def __repr__(self):
        return f'MonkhorstPackKPoints({self.size_c}, shift={self.shift_c})'

    def __str__(self):
        a, b, c = self.size_c
        l, m, n = self.shift_c
        return (f'monkhorst-pack size: [{a}, {b}, {c}]\n'
                f'monkhorst-pack shift: [{l}, {m}, {n}]\n')


class IBZ:
    def __init__(self,
                 symmetries: Symmetries,
                 bz: BZPoints,
                 ibz2bz, bz2ibz, weights,
                 bz2bz_Ks=None, s_K=None, time_reversal_K=None):
        self.symmetries = symmetries
        self.bz = bz
        self.weight_k = weights
        self.kpt_kc = bz.kpt_Kc[ibz2bz]
        self.ibz2bz_k = ibz2bz
        self.bz2ibz_K = bz2ibz
        self.bz2bz_Ks = bz2bz_Ks
        self.s_K = s_K
        self.time_reversal_K = time_reversal_K

    def __len__(self):
        """Number of k-points in the IBZ."""
        return len(self.kpt_kc)

    def __repr__(self):
        return (f'IBZ(<points: {len(self)}, '
                f'symmetries: {len(self.symmetries)}>)')

    def __str__(self):
        N = len(self)
        txt = ('bz sampling:\n'
               f'  number of bz points: {len(self.bz)}\n'
               f'  number of ibz points: {N}\n')

        if self.bz2bz_Ks is not None and -1 in self.bz2bz_Ks:
            txt += '  your k-points are not as symmetric as your crystal!\n'

        if isinstance(self.bz, MonkhorstPackKPoints):
            txt += '  ' + str(self.bz).replace('\n', '\n  ', 1)

        txt += '  points and weights: [\n'
        k = 0
        while k < N:
            if k == 10:
                if N > 10:
                    txt += '    # ...\n'
                k = N - 1
            a, b, c = self.kpt_kc[k]
            w = self.weight_k[k]
            t = ',' if k < N - 1 else ']'
            txt += (f'    [[{a:12.8f}, {b:12.8f}, {c:12.8f}], '
                    f'{w:.8f}]{t}  # {k}\n')
            k += 1
        return txt

    def ranks(self, comm: MPIComm) -> Array1D:
        """Distribute k-points over MPI-communicator."""
        return ranks(comm.size, len(self))


def ranks(N, K) -> Array1D:
    """Distribute k-points over MPI-communicator.

    >>> ranks(4, 6)
    array([0, 1, 2, 2, 3, 3])
    """
    n, x = divmod(K, N)
    rnks = np.empty(K, int)
    r = N - x
    for k in range(r * n):
        rnks[k] = k // n
    for k in range(r * n, K):
        rnks[k] = (k - r * n) // (n + 1) + r
    return rnks
