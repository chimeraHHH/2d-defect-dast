"""BLACS distributed matrix object."""
from __future__ import annotations

from types import ModuleType
from typing import Dict, Tuple
import gpaw.cgpaw as cgpaw
import numpy as np
import scipy.linalg as sla

import gpaw.utilities.blas as blas
from gpaw import debug, get_scipy_version
from gpaw.gpu import cupy as cp, cupy_eigh, XP, gpu_gemm
from gpaw.mpi import MPIComm, _Communicator, serial_comm
from gpaw.typing import Array1D, ArrayLike1D, ArrayLike2D, Array2D

_global_blacs_context_store: Dict[Tuple[_Communicator, int, int], int] = {}


def suggest_blocking(N: int, ncpus: int) -> tuple[int, int, int]:
    """Suggest blocking of ``NxN`` matrix.

    Returns rows, columns, blocksize tuple.

    >>> suggest_blocking(10, 6)
    (3, 2, 2)
    """

    nprow = ncpus
    npcol = 1

    # Make npcol and nprow as close to each other as possible
    npcol_try = npcol
    while npcol_try < nprow:
        if ncpus % npcol_try == 0:
            npcol = npcol_try
            nprow = ncpus // npcol
        npcol_try += 1

    assert npcol * nprow == ncpus

    # ScaLAPACK creates trouble if there aren't at least a few whole blocks.
    # Choose block size so that there will always be at least one whole block
    # and at least two blocks in total.
    blocksize = max((N - 2) // max(nprow, npcol), 1)
    # The next commented line would give more whole blocks.
    # blocksize = max(N // max(nprow, npcol) - 2, 1)

    # Use block size that is a power of 2 and at most 64
    blocksize = 2**int(np.log2(blocksize))
    blocksize = max(min(blocksize, 64), 1)

    return nprow, npcol, blocksize


class MatrixWithNoData:
    def __init__(self,
                 M: int,
                 N: int,
                 *,
                 dtype=None,
                 dist: MatrixDistribution | tuple | None = None):
        self.shape = (M, N)
        self.dtype = dtype
        self.data = np.empty((0, 0), dtype)
        dist = dist or ()
        if isinstance(dist, tuple):
            kwargs = {key: val for key, val in zip(['comm', 'r', 'c', 'b'],
                                                   dist)}
            dist = create_distribution(M, N, **kwargs)
        self.dist = dist

    def create(self) -> Matrix:
        return Matrix(*self.shape, dtype=self.dtype, dist=self.dist)


class Matrix(XP):
    def __init__(self,
                 M: int,
                 N: int,
                 *,
                 dtype=None,
                 data: ArrayLike2D | None = None,
                 dist: MatrixDistribution | tuple | None = None,
                 xp=None):
        """Matrix object.

        Parameters
        ----------
        M:
            Rows.
        N:
            Columns.
        dtype:
            Data type (float or complex).
        dist:
            BLACS distribution given as
            (communicator, rows, columns, blocksize)
            tuple.  Default is None meaning no distribution.
        data:
            Numpy ndarray to use for storage.  By default, a new ndarray
            will be allocated.
            """
        self.shape = (M, N)

        if data is None or isinstance(data, (np.ndarray, cp.ndarray)):
            pass
        else:
            data = np.asarray(data)

        if dtype is None:
            if data is None:
                dtype = float
            else:
                dtype = data.dtype
        self.dtype = np.dtype(dtype)
        assert np.dtype(self.dtype) in \
            [np.float32, np.float64, np.complex64, np.complex128], dtype

        self.xp: ModuleType
        if xp is None:
            if isinstance(dist, CuPyDistribution):
                xp = cp
            elif data is not None and not isinstance(data, np.ndarray):
                xp = cp
            else:
                xp = np
        XP.__init__(self, xp)

        dist = dist or ()
        if isinstance(dist, tuple):
            kwargs = {key: val for key, val in zip(['comm', 'r', 'c', 'b'],
                                                   dist)}
            dist = create_distribution(M, N, xp=self.xp, **kwargs)
        else:
            assert self.shape == dist.full_shape
        self.dist = dist

        self.data: Array2D
        if data is None:
            self.data = self.xp.empty(dist.shape, self.dtype)
        else:
            assert data.shape == dist.shape, (data.shape, dist.shape, dist)
            self.data = data

    def __repr__(self):
        dist = str(self.dist).split('(')[1]
        if self.xp is cp:
            dist = 'xp=cp, ' + dist
        return f'Matrix({self.dtype.name}: {dist}'

    def new(self, dist='inherit', data=None) -> Matrix:
        """Create new matrix of same shape and dtype.

        Default is to use same BLACS distribution.  Use dist to use another
        distribution.
        """
        return Matrix(*self.shape,
                      dtype=self.dtype,
                      dist=self.dist if dist == 'inherit' else dist,
                      data=data,
                      xp=self.xp)

    def copy(self) -> Matrix:
        """Create a copy."""
        M = self.new()
        M.data[:] = self.data
        return M

    def __setitem__(self, item, value):
        assert item == slice(None)
        assert isinstance(value, Matrix)
        self.data[:] = value.data

    def __iadd__(self, other):
        if isinstance(other, Matrix):
            other = other.data
        self.data += other
        return self

    def multiply(self,
                 other,
                 alpha=1.0,
                 opa='N',
                 opb='N',
                 out=None,
                 data_buffer=None,
                 beta=0.0,
                 symmetric=False) -> Matrix:
        """BLAS matrix-multiplication with other matrix."""
        if not isinstance(other, Matrix):
            other = other.matrix
        A = self
        B = other
        dist = self.dist
        if out is None:
            assert beta == 0.0
            M = A.shape[0] if opa == 'N' else A.shape[1]
            N = B.shape[1] if opb == 'N' else B.shape[0]
            out = Matrix(M, N, dtype=A.dtype, dist=dist.new(M, N))
        elif not isinstance(out, Matrix):
            out = out.matrix
        if out.data is other.data:
            # Repeatably call multiply using data_buffer
            assert opa == 'N', 'Not implemented'
            assert opb == 'N', 'Not implemented'
            assert not beta, 'Not implemented'
            assert other.shape[0] == self.shape[0]

            # Assert simple (only row distributed) distributions:
            assert self.shape[1] == self.data.shape[1]
            assert other.shape[1] == other.data.shape[1]
            assert out.shape[1] == out.data.shape[1]

            if data_buffer is None:
                raise ValueError('other is out, and data_buffer is None')

            assert isinstance(data_buffer, other.xp.ndarray)
            dtype = other.data.dtype
            data_buffer = data_buffer.view(dtype)
            if other.data.shape[0] > 0:
                # Obtain buffer size s.t. the maximum number of
                # columns in other.data fits into data_buffer
                buffer_size = min(
                    data_buffer.size // other.data.shape[0],
                    other.data.shape[1])
            else:
                # There is no data in other. Thus buffer_size
                # fits all.
                buffer_size = other.data.shape[1]
            buffer_size = dist.comm.min_scalar(buffer_size)
            max_B = other.data.shape[1]

            if buffer_size >= max_B:
                # No need for sliced multiply
                other_buffer = other.new(
                    data=data_buffer[:other.data.size].reshape(
                        other.data.shape))
                other_buffer.data[:] = other.data
                dist.multiply(alpha, A, opa, other_buffer, opb, beta, out,
                              symmetric=symmetric)
                return out

            # Sliced multiply
            for i in range(0, max_B, buffer_size):
                r_buffer_size = min(max(other.data.shape[1] - i, 0),
                                    buffer_size)
                l_buffer_size = r_buffer_size * other.data.shape[0]
                buffer = Matrix(
                    M=other.shape[0],
                    N=r_buffer_size,
                    data=data_buffer[
                        :l_buffer_size].reshape(
                        (other.data.shape[0], r_buffer_size)
                    ),
                    dist=dist.new(M=other.shape[0], N=r_buffer_size),
                    xp=other.xp)
                buffer.data[:] \
                    = other.data[:, i:i + buffer_size]
                out_view = buffer.new(
                    data=out.data[:, i:i + buffer_size])
                dist.multiply(alpha, A, opa, buffer,
                              opb, beta, out_view, symmetric=False)
            return out

        dist.multiply(alpha, A, opa, B, opb, beta, out, symmetric=symmetric)
        return out

    def redist(self, other: Matrix) -> None:
        """Redistribute to other BLACS layout."""
        if self is other:
            return
        d1 = self.dist
        d2 = other.dist
        n1 = d1.rows * d1.columns
        n2 = d2.rows * d2.columns
        if n1 == n2 == 1:
            other.data[:] = self.data
            return

        if n2 == 1 and d1.blocksize is None:
            assert d2.blocksize is None
            assert d1.columns == 1
            comm = d1.comm
            if comm.rank == 0:
                M = self.shape[0]
                m = (M + comm.size - 1) // comm.size
                other.data[:m] = self.data
                for r in range(1, comm.size):
                    m1 = min(r * m, M)
                    m2 = min(m1 + m, M)
                    comm.receive(other.data[m1:m2], r)
            else:
                comm.send(self.data, 0)
            return

        if n1 == 1 and d2.blocksize is None:
            assert d1.blocksize is None
            assert d1.columns == 1
            comm = d1.comm
            if comm.rank == 0:
                M = self.shape[0]
                m = (M + comm.size - 1) // comm.size
                other.data[:] = self.data[:m]
                for r in range(1, comm.size):
                    m1 = min(r * m, M)
                    m2 = min(m1 + m, M)
                    comm.send(self.data[m1:m2], r)
            else:
                comm.receive(other.data, 0)
            return

        c = d1.comm if d1.comm.size > d2.comm.size else d2.comm
        n = max(n1, n2)
        if n < c.size:
            c = c.new_communicator(np.arange(n))
        if c is not None:
            M, N = self.shape
            d1 = create_distribution(M, N, c,
                                     d1.rows, d1.columns, d1.blocksize)
            d2 = create_distribution(M, N, c,
                                     d2.rows, d2.columns, d2.blocksize)
            if n1 == n:
                ctx = d1.desc[1]
            else:
                ctx = d2.desc[1]
            redist(d1, self.data, d2, other.data, ctx)

    def gather(self, root: int = 0, broadcast=False) -> Matrix:
        """Gather the Matrix on the root rank.

        Returns a new Matrix distributed so that all data is on the root rank
        """
        assert root == 0
        if self.dist.comm.size > 1:
            S = self.new(dist=(self.dist.comm, 1, 1))
            self.redist(S)
            if broadcast:
                if self.dist.comm.rank > 0:
                    S = self.new(dist=None)
                self.dist.comm.broadcast(S.data, 0)
        else:
            S = self

        return S

    def inv(self, uplo='L'):
        """Inplace inversion."""
        assert uplo == 'L'
        M, N = self.shape
        assert M == N
        dist = self.dist
        if dist.comm.size == 1:
            self.tril2full()
            self.data[:] = sla.inv(self.data,
                                   overwrite_a=True,
                                   check_finite=debug)
            return
        bc, br = dist.desc[4:6]
        assert bc == br
        info = cgpaw.scalapack_inverse(self.data, dist.desc, 'U')
        if info != 0:
            raise ValueError(f'scalapack_inverse error: {info}')

    def invcholesky(self) -> None:
        """In-place inverse of Cholesky decomposition.

        Calculate a lower triangle matrix `L` where:::

             †
          LSL = 1,

        and `S` is self.  Only the lower part of `S` is used.

        >>> S = Matrix(2, 2, data=[[1.0, np.nan],
        ...                        [0.1, 1.0]])
        >>> S.invcholesky()
        >>> S.data
        array([[ 1.        , -0.        ],
               [-0.10050378,  1.00503782]])
        """
        S = self.gather()
        if self.dist.comm.rank == 0:
            if isinstance(S.data, np.ndarray):
                if debug:
                    S.data[np.triu_indices(S.shape[0], 1)] = 42.0
                L_nn = sla.cholesky(S.data,
                                    lower=True,
                                    overwrite_a=True,
                                    check_finite=debug)
                S.data[:] = sla.inv(L_nn,
                                    overwrite_a=True,
                                    check_finite=debug)
            else:
                S.tril2full()
                L_nn = cp.linalg.cholesky(S.data)
                S.data[:] = cp.linalg.inv(L_nn)

        if S is not self:
            S.redist(self)

    def eigh(self,
             S=None,
             *,
             cc=False,
             scalapack=(None, 1, 1, None),
             limit: int | None = None) -> Array1D:
        """Calculate eigenvectors and eigenvalues.

        Matrix must be symmetric/hermitian and stored in lower half.
        If ``S`` is given, solve a generalized eigenvalue problem.

        Parameters
        ----------
        cc: bool
            Complex conjugate matrix before finding eigenvalues.
        scalapack: tuple
            BLACS distribution for ScaLapack to use.  Default is to do serial
            diagonalization.
        limit:
            Number of eigenvector and values to find.  Defaults to all.
        """
        slcomm, rows, columns, blocksize = scalapack
        slcomm = slcomm or self.dist.comm
        dist = (slcomm, rows, columns, blocksize)

        redist = (rows != self.dist.rows or
                  columns != self.dist.columns or
                  blocksize != self.dist.blocksize)

        if redist:
            H = self.new(dist=dist)
            self.redist(H)
            if S is not None:
                S0 = S
                S = S0.new(dist=dist)
                S0.redist(S)
        else:
            assert self.dist.comm.size == slcomm.size
            H = self

        if limit == H.shape[0]:
            limit = None

        if limit:
            eps = self.xp.empty(limit)
        else:
            eps = self.xp.empty(H.shape[0])

        if rows * columns == 1:
            if self.dist.comm.rank == 0:
                if cc and np.issubdtype(H.dtype, np.complexfloating):
                    np.negative(H.data.imag, H.data.imag)
                if debug:
                    H.data[np.triu_indices(H.shape[0], 1)] = 42.0
                if S is None:
                    if self.xp is not np:
                        assert isinstance(H.data, cp.ndarray)
                        eps[:], H.data.T[:] = cupy_eigh(H.data, UPLO='L')
                    else:
                        eps[:], H.data.T[:] = sla.eigh(
                            H.data,
                            lower=True,
                            overwrite_a=True,
                            check_finite=debug,
                            driver='evx' if H.data.size == 1 else 'evd')
                else:
                    if self.xp is cp:
                        assert self.dist.comm.size == 1
                        S.invcholesky()
                        self.tril2full()
                        eigs = self.eighg(S)
                        self.data[:] = self.data.T.copy()
                        return eigs
                    if debug:
                        S.data[self.xp.triu_indices(H.shape[0], 1)] = 42.0
                    eps, evecs = sla.eigh(
                        H.data,
                        S.data,
                        lower=True,
                        overwrite_a=True,
                        overwrite_b=True,
                        check_finite=debug,
                        subset_by_index=(0, limit - 1) if limit else None)
                    limit = limit or len(eps)
                    H.data.T[:, :limit] = evecs
            self.dist.comm.broadcast(eps, 0)
        else:
            if slcomm.rank < rows * columns:
                assert cc
                assert S is None
                array = H.data.copy()
                info = cgpaw.scalapack_diagonalize_dc(array, H.dist.desc, 'U',
                                                      H.data, eps)
                assert info == 0, info

            # necessary to broadcast eps when some ranks are not used
            # in current scalapack parameter set
            # eg. (2, 1, 2) with 4 processes
            if rows * columns < slcomm.size:
                H.dist.comm.broadcast(eps, 0)

        if redist:
            H.redist(self)

        return eps

    def eighg(self, L: Matrix, comm2: MPIComm = serial_comm) -> Array1D:
        """Solve generalized eigenvalue problem.

        With `H` being self, we solve for the eigenvectors `C` and the
        eigenvalues `Λ` (a diagonal matrix):::

          HC = SCΛ,

        where `L` is a lower triangle matrix such that:::

             †
          LSL = 1.

        The solution has these three steps:::

           ~      †   ~~   ~         †~
           H = LHL ,  HC = CΛ,  C = L C.

        Note that `H` must be the full matrix not just half of it!

        """
        M, N = self.shape
        assert M == N
        comm = self.dist.comm

        if comm2.rank == 0:
            if comm.size == 1:
                H = self
                L0 = L
            else:
                # TODO: Use scalapack
                H = self.new(dist=(comm,))
                self.redist(H)
                L0 = self.new(dist=(comm,))
                L.redist(L0)
            if comm.rank == 0:
                if self.xp is not np:
                    return self.dist.eighg(self, L0)
                tmp_MM = np.empty_like(H.data)
                L_MM = L0.data
                blas.mmm(1.0, L_MM, 'N', H.data, 'N', 0.0, tmp_MM)
                blas.r2k(0.5, tmp_MM, L_MM, 0.0, H.data)
                # Ht_MM = L_MM @ H.data @ L_MM.conj().T
                if get_scipy_version() >= [1, 9]:
                    driver = 'evx' if M == 1 else 'evd'
                else:
                    driver = None
                eig_n, Ct_Mn = sla.eigh(
                    H.data,
                    overwrite_a=True,
                    check_finite=debug,
                    driver=driver)
                assert Ct_Mn.flags.f_contiguous
                blas.mmm(1.0, L_MM, 'C', Ct_Mn.T, 'T', 0.0, H.data)
                # H.data[:] = L_MM.T.conj() @ Ct_Mn
            else:
                eig_n = np.empty(M)

            if comm.size > 1:
                H.redist(self)
                comm.broadcast(eig_n, 0)

        if comm2.rank > 0:
            eig_n = np.empty(M)
        comm2.broadcast(eig_n, 0)
        comm2.broadcast(self.data, 0)

        return eig_n

    def complex_conjugate(self) -> None:
        """Inplace complex conjugation."""
        if np.issubdtype(self.dtype, np.complexfloating):
            self.xp.negative(self.data.imag, self.data.imag)

    def add_hermitian_conjugate(self, scale: float = 1.0) -> None:
        """Add hermitian conjugate to myself."""
        if self.dist.comm.size == 1:
            if scale != 1.0:
                self.data *= scale
            self.data += self.data.conj().T
            return
        tmp = self.copy()
        cgpaw.pblas_tran(*self.shape, scale, tmp.data, scale, self.data,
                         self.dist.desc, self.dist.desc, True)

    def tril2full(self) -> None:
        """Fill in upper triangle from lower triangle.

        For a real matrix::

          a ? ?    a b d
          b c ? -> b c e
          d e f    d e f

        For a complex matrix, the complex conjugate of the lower part will
        be inserted into the upper part.
        """
        M, N = self.shape
        assert M == N

        dist = self.dist

        if dist.comm.size == 1 or dist.rows == 1 and dist.columns == 1:
            if dist.comm.rank == 0:
                lower = self.xp.tri(M, k=-1, dtype=bool)
                self.data.T[lower] = self.data[lower].conj()
            return

        desc = dist.desc
        cgpaw.scalapack_set(self.data, desc, 0.0, 0.0, 'L', M - 1, M - 1, 2, 1)
        buf = self.data.copy()
        # Set diagonal to zero in the copy:
        cgpaw.scalapack_set(buf, desc, 0.0, 0.0, 'L', M, M, 1, 1)
        # Now transpose tmp_mm adding the result to the original matrix:
        cgpaw.pblas_tran(M, M, 1.0, buf, 1.0, self.data, desc, desc, True)

    def add_to_diagonal(self, d: ArrayLike1D | float) -> None:
        """Add list of numbers or single number to diagonal of matrix."""
        n1, n2 = self.dist.my_row_range()
        M, N = self.shape
        assert M == N
        self.data.ravel()[n1::N + 1] += d

    def to_cpu(self) -> Matrix:
        """Create new matrix object with values transfered from GPU to CPU."""
        return self.to_xp(np)

    def to_xp(self, xp) -> Matrix:
        """Create new matrix object with data on GPU or CPU."""
        if xp is self.xp:
            assert xp is np, 'cp -> cp should not be needed!'
            return self
        if xp is np:
            return self.dist.matrix(data=cp.asnumpy(self.data))
        return self.dist.matrix(data=cp.asarray(self.data))

    def to_dtype(self, dtype) -> Matrix:
        """Convert to new data type."""
        if dtype == self.dtype:
            return self
        return self.dist.matrix(data=self.data.astype(dtype))


def _matrix(M):
    """Dig out Matrix object from wrapper(s)."""
    if isinstance(M, Matrix):
        return M
    return _matrix(M.matrix)


def redist(dist1, M1, dist2, M2, context):
    cgpaw.scalapack_redist(dist1.desc, dist2.desc,
                           M1, M2,
                           dist1.desc[2], dist1.desc[3],
                           1, 1, 1, 1,  # 1-indexing
                           context, 'G')


def create_distribution(M: int,
                        N: int,
                        comm: MPIComm | None = None,
                        r: int = 1,
                        c: int = 1,
                        b: int | None = None,
                        xp=None) -> MatrixDistribution:
    if xp is cp:
        assert b is None
        if r == 1 and c == 1:
            pass  # comm = None
        comm = comm or serial_comm
        return CuPyDistribution(M, N, comm,
                                r if r != -1 else comm.size,
                                c if c != -1 else comm.size,
                                b)

    if comm is None or comm.size == 1:
        assert r == 1 and abs(c) == 1 or c == 1 and abs(r) == 1
        return NoDistribution(M, N)

    return BLACSDistribution(M, N, comm,
                             r if r != -1 else comm.size,
                             c if c != -1 else comm.size,
                             b)


class MatrixDistribution:
    comm: MPIComm
    rows: int
    columns: int
    blocksize: int | None  # None means everything on rank=0
    shape: tuple[int, int]
    full_shape: tuple[int, int]
    desc: Array1D

    def matrix(self, dtype=None, data=None):
        return Matrix(*self.full_shape, dtype=dtype, data=data, dist=self)

    def multiply(self, alpha, a, opa, b, opb, beta, c, symmetric):
        raise NotImplementedError

    def eighg(self, H, L):
        raise NotImplementedError

    def new(self, M, N):
        raise NotImplementedError

    def my_row_range(self) -> tuple[int, int]:
        """Return indices for range of my rows.

        >>> Matrix(2, 2).dist.my_row_range()
        (0, 2)
        """
        ok = (self.rows == self.comm.size and
              self.columns == 1 and
              self.blocksize is None)
        if not ok:
            raise ValueError(f'Can not create slice of distribution: {self}')
        M = self.full_shape[0]
        b = (M + self.rows - 1) // self.rows
        n1 = self.comm.rank * b
        n2 = min(n1 + b, M)
        return n1, n2


class NoDistribution(MatrixDistribution):
    comm = serial_comm
    rows = 1
    columns = 1
    blocksize = None

    def __init__(self, M, N):
        self.shape = (M, N)
        self.full_shape = (M, N)

    def __str__(self):
        return 'NoDistribution({}x{})'.format(*self.shape)

    def global_index(self, n):
        return n

    def new(self, M, N):
        return NoDistribution(M, N)

    def multiply(self, alpha, a, opa, b, opb, beta, c, symmetric):
        if symmetric:
            if opa == 'N':
                assert opb == 'C' or opb == 'T' and a.dtype == float
                if a is b:
                    blas.rk(alpha, a.data, beta, c.data)
                else:
                    if beta == 1.0 and a.shape[1] == 0:
                        return
                    blas.r2k(0.5 * alpha, a.data, b.data, beta, c.data)
            else:
                1 / 0
                assert opa == 'C' and opb == 'N'
                assert a is not b
                blas.r2k(0.5 * alpha, a.data, b.data, beta, c.data, 'n')

        else:
            blas.mmm(alpha, a.data, opa, b.data, opb, beta, c.data)


class BLACSDistribution(MatrixDistribution):
    serial = False

    def __init__(self, M, N, comm, r, c, b):
        self.comm = comm
        self.rows = r
        self.columns = c
        self.blocksize = b
        self.full_shape = (M, N)
        self.simple = False

        key = (comm, r, c)
        context = _global_blacs_context_store.get(key)
        if context is None:
            try:
                context = cgpaw.new_blacs_context(comm.get_c_object(),
                                                  c, r, 'R')
            except AttributeError:
                pass
            else:
                _global_blacs_context_store[key] = context

        if b is None:
            if c == 1:
                br = (M + r - 1) // r
                bc = max(1, N)
                self.simple = True
            elif r == 1:
                br = M
                bc = (N + c - 1) // c
            else:
                raise ValueError('Please specify block size!')
        else:
            br = bc = b

        if context is None:
            assert b is None
            assert c == 1
            n = N
            m = min((comm.rank + 1) * br, M) - min(comm.rank * br, M)
        else:
            n, m = cgpaw.get_blacs_local_shape(context, N, M, bc, br, 0, 0)
        if n < 0 or m < 0:
            n = m = 0
        self.shape = (m, n)
        lld = max(1, n)
        if context is not None:
            self.desc = np.array([1, context, N, M, bc, br, 0, 0, lld],
                                 np.intc)

    def __str__(self):
        return ('BLACSDistribution(global={}, local={}, blocksize={})'
                .format(*('{}x{}'.format(*shape)
                          for shape in [self.desc[3:1:-1],
                                        self.shape,
                                        self.desc[5:3:-1]])))

    def global_index(self, myi):
        return self.comm.rank * int(self.desc[5]) + myi

    def new(self, M, N):
        return BLACSDistribution(M, N,
                                 self.comm,
                                 self.rows, self.columns,
                                 self.blocksize)

    def multiply(self, alpha, a, opa, b, opb, beta, c, symmetric):
        if self.comm.size > 1:
            ok = a.dist.simple and b.dist.simple and c.dist.simple
            if ok:
                # Special cases that don't need scalapack - most likely also
                # faster:
                if opa == 'N' and opb == 'N':
                    return mmm_nn(a, b, c, alpha, beta, blas.mmm)
                if opa == 'N' and opb == 'C':
                    if symmetric:
                        if beta == 1.0:
                            return mmm_nc_sym(a, b, c, alpha, blas.mmm)
                    else:
                        return mmm_nc(a, b, c, alpha, beta, blas.mmm)

        if symmetric:
            assert opa == 'N'
            assert opb == 'C' or opb == 'T' and a.dtype == float
            N, K = a.shape
            if a is b:
                cgpaw.pblas_rk(N, K, alpha, a.data,
                               beta, c.data,
                               a.dist.desc, c.dist.desc,
                               'U')
            else:
                cgpaw.pblas_r2k(N, K, 0.5 * alpha, b.data, a.data,
                                beta, c.data,
                                b.dist.desc, a.dist.desc, c.dist.desc,
                                'U')
        else:
            Ka, M = a.shape
            N, Kb = b.shape
            if opa == 'N':
                Ka, M = M, Ka
            if opb == 'N':
                N, Kb = Kb, N
            cgpaw.pblas_gemm(N, M, Ka, alpha, b.data, a.data,
                             beta, c.data,
                             b.dist.desc, a.dist.desc, c.dist.desc,
                             opb, opa)
        return c


def cublas_mmm(alpha, a, opa, b, opb, beta, c):
    if c.size == 0:
        return
    if a.size == 0 and beta == 1.0:
        return
    gpu_gemm(opa.replace('C', 'H'), opb.replace('C', 'H'),
             a, b, c, alpha, beta)


class CuPyDistribution(MatrixDistribution):
    def __init__(self, M, N, comm, r, c, b):
        self.comm = comm
        self.rows = r
        self.columns = c
        self.blocksize = b
        self.full_shape = (M, N)
        # assert r == comm.size, (M, N, comm, r, c, b)
        assert c == 1
        br = (M + r - 1) // r
        m = min((comm.rank + 1) * br, M) - min(comm.rank * br, M)
        self.shape = (m, N)

    def __str__(self):
        M, N = self.full_shape
        m, N = self.shape
        return f'CuPyDistribution(global={M}x{N}, local={m}x{N})'

    def global_index(self, n):
        1 / 0
        return n

    def new(self, M, N):
        return CuPyDistribution(M, N,
                                self.comm,
                                self.rows, self.columns,
                                self.blocksize)

    def multiply(self, alpha, a, opa, b, opb, beta, c, *, symmetric=False):
        if self.comm.size > 1:
            if opa == 'N' and opb == 'N':
                return mmm_nn(a, b, c, alpha, beta, cublas_mmm)
            if opa == 'N' and opb == 'C':
                if symmetric:
                    if beta == 1.0:
                        return mmm_nc_sym(a, b, c, alpha, cublas_mmm)
                else:
                    return mmm_nc(a, b, c, alpha, beta, cublas_mmm)
            1 / 0

        if symmetric:
            if opa == 'N':
                assert opb == 'C' or opb == 'T' and a.dtype == float
                if a is b:
                    gpu_gemm('N', 'H',
                             a.data, a.data, c.data,
                             alpha, beta)
                    # cp.cublas.syrk('N', a.data, c.data, alpha, beta, True)
                else:
                    if beta == 1.0 and a.shape[1] == 0:
                        return
                    if c.data.size > 0:
                        assert beta in [0.0, 1.0]
                        # CuPy doesn't have dsyrk, so we roll our own:
                        gpu_gemm('N', 'H',
                                 a.data, b.data, c.data,
                                 0.5 * alpha, beta)
                        gpu_gemm('N', 'H',
                                 b.data, a.data, c.data,
                                 0.5 * alpha, 1.0)
            else:
                1 / 0
                assert opa == 'C' and opb == 'N'
                assert a is not b
                raise NotImplementedError
                blas.r2k(0.5 * alpha, a.data, b.data, beta, c.data, 'n')

        else:
            cublas_mmm(alpha, a.data, opa, b.data, opb, beta, c.data)

    def eighg(self, H, L):
        """
        :::

           ~      †   ~~   ~         †~
           H = LHL ,  HC = CΛ,  C = L C.
        """
        assert self.comm.size == 1
        tmp = H.new()
        self.multiply(1.0, L, 'N', H, 'N', 0.0, tmp)
        self.multiply(1.0, tmp, 'N', L, 'C', 0.0, H, symmetric=True)
        eig_M, Ct_MM = cupy_eigh(H.data, UPLO='L')
        assert Ct_MM.flags.f_contiguous
        Ct = H.new(data=Ct_MM.T)
        self.multiply(1.0, L, 'C', Ct, 'T', 0.0, H)
        # H.complex_conjugate()
        return eig_M


def mmm_nn(m1, m2, m3, alpha, beta, mmm):
    """Parallel matrix-matrix multiplication.

    :::

        m  <- αm m + βm
         3      1 2    3
    """
    comm = m1.dist.comm
    buf1 = m2.data
    xp = m1.xp

    N = m1.shape[1]
    assert N == m2.shape[0], f'{N}, {m2.shape[0]}'
    n = (N + comm.size - 1) // comm.size

    for r in range(comm.size):
        if r == 0:
            # Buffers...
            buf2 = xp.empty((n, buf1.shape[1]), dtype=buf1.dtype)

        rrequest = None
        srequest = None
        if r < comm.size - 1:
            rrank = (comm.rank + r + 1) % comm.size
            rn1 = min(rrank * n, N)
            rn2 = min(rn1 + n, N)
            if rn2 > rn1:
                rrequest = comm.receive(buf2[:rn2 - rn1], rrank, 21, False)
            srank = (comm.rank - r - 1) % comm.size
            if len(m2.data) > 0:
                srequest = comm.send(m2.data, srank, 21, False)

        r0 = (comm.rank + r) % comm.size
        n1 = min(r0 * n, N)
        n2 = min(n1 + n, N)
        # Contiguity...
        mmm(alpha, m1.data[:, n1:n2], 'N', buf1[:n2 - n1], 'N', beta, m3.data)

        beta = 1.0

        if r == 0:
            # Buffers...
            buf1 = xp.empty_like(buf2)

        buf1, buf2 = buf2, buf1

        if rrequest:
            comm.wait(rrequest)
        if srequest:
            comm.wait(srequest)

    return m3


def mmm_nc_sym(a, b, out, alpha, mmm):
    """Symmetric parallel matrix-matrix multiplication.

    :::

                †
        c <- αab + c

    This function utilizes the fact that c is symmetric, s.t.:
                       †     †
        c <- 0.5 * (αab + αba) + c
    Only lower half of c is updated.
    """
    comm = a.dist.comm
    M, N = b.shape
    m = (M + comm.size - 1) // comm.size
    mym = len(b.data)
    xp = a.xp

    # Buffers...
    buf1 = xp.empty((m, N), dtype=a.dtype)
    buf2 = xp.empty((m, N), dtype=a.dtype)
    half = comm.size // 2
    aa = a.data
    bb = b.data

    for r in range(half + 1):
        rrequest = None
        srequest = None

        if r < half:
            srank = (comm.rank + r + 1) % comm.size
            rrank = (comm.rank - r - 1) % comm.size
            skip = (comm.size % 2 == 0 and r == half - 1)
            m1 = min(rrank * m, M)
            m2 = min(m1 + m, M)
            if not (skip and comm.rank < half) and m2 > m1:
                rrequest = comm.receive(buf1[:m2 - m1], rrank, 11, False)
            if not (skip and comm.rank >= half) and mym > 0:
                srequest = comm.send(b.data, srank, 11, False)

        if not (comm.size % 2 == 0 and r == half and comm.rank < half):
            m1 = min(((comm.rank - r) % comm.size) * m, M)
            m2 = min(m1 + m, M)
            if r == 0:
                # symmmmmmmmmmmmmmmmmmmmmmetricccccccccccccccc
                # Contiguity...
                mmm(alpha, aa, 'N', bb, 'C', 1.0, out.data[:, m1:m2])
            else:
                beta = 1.0 if r <= comm.rank else 0.0
                mmm(alpha, aa, 'N', buf2[:m2 - m1], 'C',
                    beta, out.data[:, m1:m2])
            # out.data[:, m1:m2] = m12.data[:, :m2 - m1]

        if rrequest:
            comm.wait(rrequest)
        if srequest:
            comm.wait(srequest)

        bb = buf1
        buf1, buf2 = buf2, buf1

    requests = []
    blocks = []
    nrows = (comm.size - 1) // 2
    for row in range(nrows):
        for column in range(comm.size - nrows + row, comm.size):
            if comm.rank == row:
                m1 = min(column * m, M)
                m2 = min(m1 + m, M)
                if mym > 0 and m2 > m1:
                    requests.append(
                        comm.send(out.data[:, m1:m2].T.conj().copy(),
                                  column, 12, False))
            elif comm.rank == column:
                m1 = min(row * m, M)
                m2 = min(m1 + m, M)
                if mym > 0 and m2 > m1:
                    block = xp.empty((mym, m2 - m1), out.dtype)
                    blocks.append((m1, m2, block))
                    requests.append(comm.receive(block, row, 12, False))

    comm.waitall(requests)
    for m1, m2, block in blocks:
        out.data[:, m1:m2] += block

    return out


def mmm_nc(a, b, out, alpha, beta, mmm):
    """Parallel matrix-matrix multiplication.

    :::

                †
        c <- αab  + βc
    """
    comm = a.dist.comm
    M, N = b.shape
    m = (M + comm.size - 1) // comm.size
    mym = len(b.data)
    xp = a.xp

    # Nasty buffers
    buf1 = xp.empty((m, N), dtype=a.dtype)
    buf2 = xp.empty((m, N), dtype=a.dtype)
    aa = a.data
    bb = b.data

    for r in range(comm.size):
        rrequest = None
        srequest = None

        if r < comm.size - 1:
            srank = (comm.rank + r + 1) % comm.size
            rrank = (comm.rank - r - 1) % comm.size
            m1 = min(rrank * m, M)
            m2 = min(m1 + m, M)
            if m2 > m1:
                rrequest = comm.receive(buf1[:m2 - m1], rrank, 11, False)
            if mym > 0:
                srequest = comm.send(b.data, srank, 11, False)

        m1 = min(((comm.rank - r) % comm.size) * m, M)
        m2 = min(m1 + m, M)
        # symmmmmmmmmmmmmmmmmmmmmmetricccccccccccccccc ??
        mmm(alpha, aa, 'N', bb[:m2 - m1], 'C', beta, out.data[:, m1:m2])

        if rrequest:
            comm.wait(rrequest)
        if srequest:
            comm.wait(srequest)

        bb = buf1
        buf1, buf2 = buf2, buf1

    return out
