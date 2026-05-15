from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar, Callable, Literal

import gpaw.fftw as fftw
import numpy as np
from ase.io.ulm import NDArrayReader
from gpaw.core.domain import Domain
from gpaw.core.matrix import Matrix
from gpaw.mpi import MPIComm
from gpaw.typing import Array1D, Self, ArrayND
from gpaw.gpu import XP
from gpaw.new import trace

if TYPE_CHECKING:
    from gpaw.core.uniform_grid import UGArray, UGDesc

from gpaw.new import prod

DomainType = TypeVar('DomainType', bound=Domain)


class XArrayWithNoData:
    def __init__(self,
                 comm,
                 dims,
                 desc,
                 xp):
        self.comm = comm
        self.dims = dims
        self.desc = desc
        self.xp = xp
        self.data = None

    def morph(self, desc):
        from gpaw.new.calculation import ReuseWaveFunctionsError
        raise ReuseWaveFunctionsError


class DistributedArrays(Generic[DomainType], XP):
    desc: DomainType

    def __init__(self,
                 dims: int | tuple[int, ...],
                 myshape: tuple[int, ...],
                 comm: MPIComm,
                 domain_comm: MPIComm,
                 data: np.ndarray | None,
                 dv: float,
                 dtype,
                 xp=None):
        self.myshape = myshape
        self.comm = comm
        self.domain_comm = domain_comm
        self.dv = dv

        # convert int to tuple:
        self.dims = dims if isinstance(dims, tuple) else (dims,)

        if self.dims:
            mydims0 = (self.dims[0] + comm.size - 1) // comm.size
            d1 = min(comm.rank * mydims0, self.dims[0])
            d2 = min((comm.rank + 1) * mydims0, self.dims[0])
            mydims0 = d2 - d1
            self.mydims = (mydims0,) + self.dims[1:]
        else:
            self.mydims = ()

        fullshape = self.mydims + self.myshape

        if data is not None:
            if data.shape != fullshape:
                raise ValueError(
                    f'Bad shape for data: {data.shape} != {fullshape}')
            if data.dtype != dtype:
                raise ValueError(
                    f'Bad dtype for data: {data.dtype} != {dtype}')
            if xp is not None:
                assert (xp is np) == isinstance(
                    data, (np.ndarray, NDArrayReader)), xp
        else:
            data = (xp or np).empty(fullshape, dtype)

        self.data = data
        if isinstance(data, (np.ndarray, NDArrayReader)):
            xp = np
        else:
            from gpaw.gpu import cupy as cp
            xp = cp
        XP.__init__(self, xp)
        self._matrix: Matrix | None = None

    def new(self, data=None, dims=None) -> DistributedArrays:
        raise NotImplementedError

    def create_work_buffer(self, data_buffer: np.ndarray):
        """Create new Distributed array object of same
        kind, to be used as a buffer array when doing
        sliced operations.

        Parameters
        ----------
        data_buffer:
            Array to use for storage.
        """
        assert isinstance(data_buffer, self.xp.ndarray)
        assert len(self.dims) >= 1
        data_buffer = data_buffer.view(self.data.dtype)
        datasize = data_buffer.size
        X = self.data.shape[1:]
        nX = int(np.prod(X))
        # Choose mybands, s.t. they fit into
        # data_buffer. Hence, datasize divided by nX
        # rounded down.
        mybands = min(datasize // nX,
                      self.data.shape[0])
        data = data_buffer[:mybands * nX].reshape(
            (mybands,) + X)
        totalbands = self.comm.sum_scalar(mybands)
        # Dims is (totalbands,) + self.dims[1:], where
        # self.dims[1:] is extra dimensions, such as spin.
        return self.new(data=data,
                        dims=(totalbands,) + self.dims[1:])

    def copy(self):
        return self.new(data=self.data.copy())

    def sanity_check(self) -> None:
        """Sanity check."""
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    def __bool__(self):
        raise ValueError

    def __len__(self):
        return self.dims[0]

    def __iter__(self):
        for index in range(self.dims[0]):
            yield self[index]

    def flat(self) -> Self:
        if self.dims == ():
            yield self
        else:
            for index in np.indices(self.dims).reshape((len(self.dims), -1)).T:
                yield self[tuple(index)]

    def to_xp(self, xp):
        if xp is self.xp:
            assert xp is np, 'cp -> cp should not be needed!'
            return self
        if xp is np:
            return self.new(data=self.xp.asnumpy(self.data))
        else:
            return self.new(data=xp.asarray(self.data))

    @property
    def matrix(self) -> Matrix:
        if self._matrix is not None:
            return self._matrix

        nx = prod(self.myshape)
        shape = (self.dims[0], prod(self.dims[1:]) * nx)
        myshape = (self.mydims[0], prod(self.mydims[1:]) * nx)
        dist = (self.comm, -1, 1)

        data = self.data.reshape(myshape)
        self._matrix = Matrix(*shape, data=data, dist=dist)

        return self._matrix

    @trace
    def matrix_elements(self,
                        other: Self,
                        *,
                        out: Matrix | None = None,
                        symmetric: bool | Literal['_default'] = '_default',
                        function=None,
                        domain_sum=True,
                        cc: bool = False) -> Matrix:
        if symmetric == '_default':
            symmetric = self is other

        comm = self.comm

        if out is None:
            out = Matrix(self.dims[0], other.dims[0],
                         dist=(comm, -1, 1),
                         dtype=self.desc.dtype,
                         xp=self.xp)

        if comm.size == 1:
            assert other.comm.size == 1
            if function:
                assert symmetric
                other = function(other)

            M1 = self.matrix
            M2 = other.matrix
            out = M1.multiply(M2, opb='C', alpha=self.dv,
                              symmetric=symmetric, out=out)

            # Plane-wave expansion of real-valued
            # functions needs a correction:
            self._matrix_elements_correction(M1, M2, out, symmetric)
        else:
            if symmetric:
                _parallel_me_sym(self, out, function)
            else:
                _parallel_me(self, other, out)

        if not cc:
            out.complex_conjugate()

        if domain_sum:
            self.domain_comm.sum(out.data)
        return out

    def _matrix_elements_correction(self,
                                    M1: Matrix,
                                    M2: Matrix,
                                    out: Matrix,
                                    symmetric: bool) -> None:
        """Hook for PlaneWaveExpansion."""
        pass

    def abs_square(self,
                   weights: Array1D,
                   out: UGArray) -> None:
        """Add weighted absolute square of data to output array.

        See also :xkcd:`849`.
        """
        raise NotImplementedError

    def add_ked(self,
                weights: Array1D,
                out: UGArray) -> None:
        """Add weighted absolute square of gradient of data to output array."""
        raise NotImplementedError

    def gather(self, out=None, broadcast=False):
        raise NotImplementedError

    def gathergather(self):
        a_xX = self.gather()  # gather X
        if a_xX is not None:
            m_xX = a_xX.matrix.gather()  # gather x
            if m_xX.dist.comm.rank == 0:
                data = m_xX.data
                if a_xX.data.dtype != data.dtype:
                    data = data.view(complex)
                return self.desc.new(comm=None).from_data(data)

    def scatter_from(self, data: ArrayND | None = None) -> None:
        raise NotImplementedError

    def redist(self,
               domain,
               comm1: MPIComm, comm2: MPIComm) -> DistributedArrays:
        result = domain.empty(self.dims)
        if comm1.rank == 0:
            a = self.gather()
        else:
            a = None
        if comm2.rank == 0:
            result.scatter_from(a)
        comm2.broadcast(result.data, 0)
        return result

    def interpolate(self,
                    plan1: fftw.FFTPlans | None = None,
                    plan2: fftw.FFTPlans | None = None,
                    grid: UGDesc | None = None,
                    out: UGArray | None = None) -> UGArray:
        raise NotImplementedError

    def integrate(self, other: Self | None = None) -> np.ndarray:
        raise NotImplementedError

    def norm2(self, kind: str = 'normal', skip_sum=False) -> np.ndarray:
        raise NotImplementedError


def _parallel_me(psit1_nX: DistributedArrays,
                 psit2_nX: DistributedArrays,
                 M_nn: Matrix) -> None:

    comm = psit2_nX.comm
    nbands = psit2_nX.dims[0]

    psit1_nX = psit1_nX[:]

    B = (nbands + comm.size - 1) // comm.size

    n_r = [min(r * B, nbands) for r in range(comm.size + 1)]

    xp = psit1_nX.xp
    buf1_nX = psit1_nX.desc.empty(B, xp=xp)
    buf2_nX = psit1_nX.desc.empty(B, xp=xp)
    psit_nX = psit2_nX

    for shift in range(comm.size):
        rrequest = None
        srequest = None

        if shift < comm.size - 1:
            srank = (comm.rank + shift + 1) % comm.size
            rrank = (comm.rank - shift - 1) % comm.size
            n1 = n_r[rrank]
            n2 = n_r[rrank + 1]
            mynb = n2 - n1
            if mynb > 0:
                rrequest = comm.receive(buf1_nX.data[:mynb], rrank, 11, False)
            if psit2_nX.data.size > 0:
                srequest = comm.send(psit2_nX.data, srank, 11, False)

        r2 = (comm.rank - shift) % comm.size
        n1 = n_r[r2]
        n2 = n_r[r2 + 1]
        m_nn = psit1_nX.matrix_elements(psit_nX[:n2 - n1],
                                        cc=True, domain_sum=False)

        M_nn.data[:, n1:n2] = m_nn.data

        if rrequest:
            comm.wait(rrequest)
        if srequest:
            comm.wait(srequest)

        psit_nX = buf1_nX
        buf1_nX, buf2_nX = buf2_nX, buf1_nX


def _parallel_me_sym(psit1_nX: DistributedArrays,
                     M_nn: Matrix,
                     operator: None | Callable[[DistributedArrays],
                                               DistributedArrays]
                     ) -> None:
    """..."""
    comm = psit1_nX.comm
    nbands = psit1_nX.dims[0]
    B = (nbands + comm.size - 1) // comm.size
    mynbands = psit1_nX.mydims[0]

    n_r = [min(r * B, nbands) for r in range(comm.size + 1)]
    mynbands_r = [n_r[r + 1] - n_r[r] for r in range(comm.size)]
    assert mynbands_r[comm.rank] == mynbands

    xp = psit1_nX.xp
    psit2_nX = psit1_nX
    buf1_nX = psit1_nX.desc.empty(B, xp=xp)
    buf2_nX = psit1_nX.desc.empty(B, xp=xp)
    half = comm.size // 2

    for shift in range(half + 1):
        rrequest = None
        srequest = None

        if shift < half:
            srank = (comm.rank + shift + 1) % comm.size
            rrank = (comm.rank - shift - 1) % comm.size
            skip = comm.size % 2 == 0 and shift == half - 1
            rmynb = mynbands_r[rrank]
            if not (skip and comm.rank < half) and rmynb > 0:
                rrequest = comm.receive(buf1_nX.data[:rmynb], rrank, 11, False)
            if not (skip and comm.rank >= half) and psit1_nX.data.size > 0:
                srequest = comm.send(psit1_nX.data, srank, 11, False)

        if shift == 0:
            if operator is not None:
                op_psit1_nX = operator(psit1_nX)
            else:
                op_psit1_nX = psit1_nX
            op_psit1_nX = op_psit1_nX[:]  # local view

        if not (comm.size % 2 == 0 and shift == half and comm.rank < half):
            r2 = (comm.rank - shift) % comm.size
            n1 = n_r[r2]
            n2 = n_r[r2 + 1]
            m_nn = op_psit1_nX.matrix_elements(psit2_nX[:n2 - n1],
                                               symmetric=(shift == 0),
                                               cc=True, domain_sum=False)
            M_nn.data[:, n1:n2] = m_nn.data

        if rrequest:
            comm.wait(rrequest)
        if srequest:
            comm.wait(srequest)

        psit2_nX = buf1_nX
        buf1_nX, buf2_nX = buf2_nX, buf1_nX

    requests = []
    blocks = []
    nrows = (comm.size - 1) // 2
    for row in range(nrows):
        for column in range(comm.size - nrows + row, comm.size):
            if comm.rank == row:
                n1 = n_r[column]
                n2 = n_r[column + 1]
                if mynbands > 0 and n2 > n1:
                    requests.append(
                        comm.send(M_nn.data[:, n1:n2].T.conj().copy(),
                                  column, 12, False))
            elif comm.rank == column:
                n1 = n_r[row]
                n2 = n_r[row + 1]
                if mynbands > 0 and n2 > n1:
                    block = xp.empty((mynbands, n2 - n1), M_nn.dtype)
                    blocks.append((n1, n2, block))
                    requests.append(comm.receive(block, row, 12, False))

    comm.waitall(requests)
    for n1, n2, block in blocks:
        M_nn.data[:, n1:n2] = block
