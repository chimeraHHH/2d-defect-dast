from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Literal, Generic, TypeVar

import numpy as np
from ase.geometry.cell import cellpar_to_cell

from gpaw.fftw import get_efficient_fft_size
from gpaw.mpi import MPIComm, serial_comm
from gpaw.typing import (Array2D, ArrayLike, ArrayLike1D, ArrayLike2D,
                         DTypeLike, Vector, Self)

if TYPE_CHECKING:
    from gpaw.core import UGDesc
    from gpaw.core.arrays import DistributedArrays


def normalize_cell(cell: ArrayLike) -> Array2D:
    """...

    >>> normalize_cell([1, 2, 3])
    array([[1., 0., 0.],
           [0., 2., 0.],
           [0., 0., 3.]])
    """
    cell = np.array(cell, float)
    if cell.ndim == 2:
        return cell
    if len(cell) == 3:
        return np.diag(cell)
    return cellpar_to_cell(cell)


XArray = TypeVar('XArray', bound='DistributedArrays')


class Domain(Generic[XArray]):
    itemsize: int

    def __init__(self,
                 cell: ArrayLike1D | ArrayLike2D,
                 pbc=(True, True, True),
                 kpt: Vector | None = None,
                 comm: MPIComm = serial_comm,
                 dtype: DTypeLike | None = None):
        """"""
        if isinstance(pbc, int):
            pbc = (pbc,) * 3
        self.cell_cv = normalize_cell(cell)
        self.pbc_c = np.array(pbc, bool)
        self.comm = comm

        self.volume = abs(np.linalg.det(self.cell_cv))
        self.orthogonal = not (self.cell_cv -
                               np.diag(self.cell_cv.diagonal())).any()

        assert np.dtype(dtype) in \
            [None, np.float32, np.float64, np.complex64, np.complex128], dtype

        # XXX: Gotta be careful about precision here:
        if kpt is not None:
            if dtype is None:
                dtype = complex
        else:
            if dtype is None:
                dtype = float
            kpt = (0.0, 0.0, 0.0)

        self.kpt_c = np.array(kpt, float)

        if self.kpt_c.any():
            if dtype == float:
                raise ValueError(f'dtype must be complex for kpt={kpt}')
            for p, k in zip(pbc, self.kpt_c):
                if not p and k != 0:
                    raise ValueError(f'Bad k-point {kpt} for pbc={pbc}')

        self.dtype = np.dtype(dtype)  # type: ignore

        self.myshape: tuple[int, ...]

    def new(self,
            *,
            kpt=None,
            dtype=None,
            comm: MPIComm | Literal['inherit'] | None = 'inherit'
            ) -> Self:
        raise NotImplementedError

    def __repr__(self):
        comm = self.comm
        if self.kpt_c.any():
            k = f', kpt={self.kpt_c.tolist()}'
        else:
            k = ''
        if (self.cell_cv == np.diag(self.cell_cv.diagonal())).all():
            cell = self.cell_cv.diagonal().tolist()
        else:
            cell = self.cell_cv.tolist()
        return (f'Domain(cell={cell}, '
                f'pbc={self.pbc_c.tolist()}, '
                f'comm={comm.rank}/{comm.size}, '
                f'dtype={self.dtype}{k})')

    def global_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def cell(self):
        return self.cell_cv.copy()

    @property
    def pbc(self):
        return self.pbc_c.copy()

    @property
    def kpt(self):
        return self.kpt_c.copy()

    def empty(self,
              shape: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm, xp=None) -> XArray:
        raise NotImplementedError

    def zeros(self,
              shape: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm, xp=None) -> XArray:
        array = self.empty(shape, comm, xp=xp)
        array.data[:] = 0.0
        return array

    @property
    def icell(self):
        """Inverse of unit cell.

        >>> d = Domain([1, 2, 4])
        >>> d.icell
        array([[1.  , 0.  , 0.  ],
               [0.  , 0.5 , 0.  ],
               [0.  , 0.  , 0.25]])
        >>> d.cell @ d.icell.T
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
        """
        return np.linalg.inv(self.cell).T

    def uniform_grid_with_grid_spacing(self,
                                       grid_spacing: float,
                                       n: int = 1,
                                       factors: Sequence[int] = (2, 3, 5, 7)
                                       ) -> UGDesc:
        from gpaw.core import UGDesc

        L_c = (np.linalg.inv(self.cell_cv)**2).sum(0)**-0.5
        size_c = np.maximum(n, (L_c / grid_spacing / n + 0.5).astype(int) * n)
        if factors:
            size_c = np.array([get_efficient_fft_size(N, n, factors)
                               for N in size_c])
        return UGDesc(size=size_c,
                      cell=self.cell_cv,
                      pbc=self.pbc_c,
                      kpt=self.kpt_c,
                      dtype=self.dtype,
                      comm=self.comm)
