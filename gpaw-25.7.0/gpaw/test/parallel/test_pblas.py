"""Test of PBLAS Level 2 & 3 : rk, r2k, gemv, gemm.

The test generates random matrices A0, B0, X0, etc. on a
1-by-1 BLACS grid. They are redistributed to a mprocs-by-nprocs
BLACS grid, BLAS operations are performed in parallel, and
results are compared against BLAS.
"""

import pytest
import numpy as np

from gpaw.mpi import world, rank, broadcast_float
from gpaw.blacs import BlacsGrid, Redistributor
from gpaw.utilities import compiled_with_sl
from gpaw.utilities.blas import r2k, rk
from gpaw.utilities.scalapack import \
    pblas_simple_gemm, pblas_gemm, \
    pblas_simple_gemv, pblas_gemv, \
    pblas_simple_r2k, pblas_simple_rk, \
    pblas_simple_hemm, pblas_hemm, \
    pblas_simple_symm, pblas_symm
from gpaw.utilities.tools import tri2full

pytestmark = pytest.mark.skipif(not compiled_with_sl(),
                                reason='not compiled with scalapack')

# may need to be be increased if the mprocs-by-nprocs
# BLACS grid becomes larger
tol = 5.0e-13

mnprocs_i = [(1, 1)]
if world.size >= 2:
    mnprocs_i += [(1, 2), (2, 1)]
if world.size >= 4:
    mnprocs_i += [(2, 2)]
if world.size >= 8:
    mnprocs_i += [(2, 4), (4, 2)]


def initialize_random(seed, dtype):
    gen = np.random.Generator(np.random.PCG64(seed))
    if dtype == complex:
        def random(*args):
            return gen.random(*args) + 1.0j * gen.random(*args)
    else:
        def random(*args):
            return gen.random(*args)
    return random


def initialize_alpha_beta(simple, random):
    if simple:
        alpha = 1.0
        beta = 0.0
    else:
        alpha = random()
        beta = random()
    return alpha, beta


def initialize_matrix(grid, M, N, mb, nb, random):
    block_desc = grid.new_descriptor(M, N, mb, nb)
    local_desc = block_desc.as_serial()
    A0 = random(local_desc.shape)
    A0 = np.ascontiguousarray(A0)
    local_desc.checkassert(A0)
    A = local_desc.redistribute(block_desc, A0)
    block_desc.checkassert(A)
    return A0, A, block_desc


def calculate_error(ref_A0, A, block_desc):
    local_desc = block_desc.as_serial()
    A0 = block_desc.redistribute(local_desc, A)
    comm = block_desc.blacsgrid.comm
    if comm.rank == 0:
        err = np.abs(ref_A0 - A0).max()
    else:
        err = np.nan
    err = broadcast_float(err, comm)
    return err


@pytest.mark.parametrize('mprocs, nprocs', mnprocs_i)
@pytest.mark.parametrize('dtype', [float, complex])
def test_pblas_rk_r2k(dtype, mprocs, nprocs,
                      M=160, K=140, seed=42):
    gen = np.random.RandomState(seed)
    grid = BlacsGrid(world, mprocs, nprocs)

    if dtype == complex:
        epsilon = 1.0j
    else:
        epsilon = 0.0

    # Create descriptors for matrices on master:
    globA = grid.new_descriptor(M, K, M, K)
    globD = grid.new_descriptor(M, K, M, K)
    globS = grid.new_descriptor(M, M, M, M)
    globU = grid.new_descriptor(M, M, M, M)

    # print globA.asarray()
    # Populate matrices local to master:
    A0 = gen.rand(*globA.shape) + epsilon * gen.rand(*globA.shape)
    D0 = gen.rand(*globD.shape) + epsilon * gen.rand(*globD.shape)

    # Local result matrices
    S0 = globS.zeros(dtype=dtype)  # zeros needed for rank-updates
    U0 = globU.zeros(dtype=dtype)  # zeros needed for rank-updates

    # Local reference matrix product:
    if rank == 0:
        r2k(1.0, A0, D0, 0.0, S0)
        rk(1.0, A0, 0.0, U0)
    assert globA.check(A0)
    assert globD.check(D0) and globS.check(S0) and globU.check(U0)

    # Create distributed destriptors with various block sizes:
    distA = grid.new_descriptor(M, K, 2, 2)
    distD = grid.new_descriptor(M, K, 2, 3)
    distS = grid.new_descriptor(M, M, 2, 2)
    distU = grid.new_descriptor(M, M, 2, 2)

    # Distributed matrices:
    A = distA.empty(dtype=dtype)
    D = distD.empty(dtype=dtype)
    S = distS.zeros(dtype=dtype)  # zeros needed for rank-updates
    U = distU.zeros(dtype=dtype)  # zeros needed for rank-updates
    Redistributor(world, globA, distA).redistribute(A0, A)
    Redistributor(world, globD, distD).redistribute(D0, D)

    pblas_simple_r2k(distA, distD, distS, A, D, S)
    pblas_simple_rk(distA, distU, A, U)

    # Collect result back on master
    S1 = globS.zeros(dtype=dtype)  # zeros needed for rank-updates
    U1 = globU.zeros(dtype=dtype)  # zeros needed for rank-updates
    Redistributor(world, distS, globS).redistribute(S, S1)
    Redistributor(world, distU, globU).redistribute(U, U1)

    if rank == 0:
        r2k_err = abs(S1 - S0).max()
        rk_err = abs(U1 - U0).max()
        print('r2k err', r2k_err)
        print('rk_err', rk_err)
    else:
        r2k_err = 0.0
        rk_err = 0.0

    # We don't like exceptions on only one cpu
    r2k_err = world.sum_scalar(r2k_err)
    rk_err = world.sum_scalar(rk_err)

    assert r2k_err == pytest.approx(0, abs=tol)
    assert rk_err == pytest.approx(0, abs=tol)


@pytest.mark.parametrize('mprocs, nprocs', mnprocs_i)
@pytest.mark.parametrize('simple', [True, False])
@pytest.mark.parametrize('transa', ['N', 'T', 'C'])
@pytest.mark.parametrize('dtype', [float, complex])
def test_pblas_gemv(dtype, simple, transa, mprocs, nprocs,
                    M=160, N=120, seed=42):
    """Test pblas_simple_gemv, pblas_gemv

    The operation is
    * y <- alpha*A*x + beta*y

    Additional options
    * alpha=1 and beta=0       if simple == True
    """
    random = initialize_random(seed, dtype)
    grid = BlacsGrid(world, mprocs, nprocs)

    # Initialize matrices
    alpha, beta = initialize_alpha_beta(simple, random)
    shapeA = (M, N)
    shapeX = {'N': (N, 1), 'T': (M, 1), 'C': (M, 1)}[transa]
    shapeY = {'N': (M, 1), 'T': (N, 1), 'C': (N, 1)}[transa]
    A0, A, descA = initialize_matrix(grid, *shapeA, 2, 2, random)
    X0, X, descX = initialize_matrix(grid, *shapeX, 4, 1, random)
    Y0, Y, descY = initialize_matrix(grid, *shapeY, 3, 1, random)

    if grid.comm.rank == 0:
        print(A0)

        # Calculate reference with numpy
        op_t = {'N': lambda M: M,
                'T': lambda M: np.transpose(M),
                'C': lambda M: np.conjugate(np.transpose(M))}
        ref_Y0 = alpha * np.dot(op_t[transa](A0), X0) + beta * Y0
    else:
        ref_Y0 = None

    # Calculate with scalapack
    if simple:
        pblas_simple_gemv(descA, descX, descY,
                          A, X, Y,
                          transa=transa)
    else:
        pblas_gemv(alpha, A, X, beta, Y,
                   descA, descX, descY,
                   transa=transa)

    # Check error
    err = calculate_error(ref_Y0, Y, descY)
    assert err < tol


@pytest.mark.parametrize('mprocs, nprocs', mnprocs_i)
@pytest.mark.parametrize('transb', ['N', 'T', 'C'])
@pytest.mark.parametrize('transa', ['N', 'T', 'C'])
@pytest.mark.parametrize('simple', [True, False])
@pytest.mark.parametrize('dtype', [float, complex])
def test_pblas_gemm(dtype, simple, transa, transb, mprocs, nprocs,
                    M=160, N=120, K=140, seed=42):
    """Test pblas_simple_gemm, pblas_gemm

    The operation is
    * C <- alpha*A*B + beta*C

    Additional options
    * alpha=1 and beta=0       if simple == True
    """
    random = initialize_random(seed, dtype)
    grid = BlacsGrid(world, mprocs, nprocs)

    # Initialize matrices
    alpha, beta = initialize_alpha_beta(simple, random)
    shapeA = {'N': (M, K), 'T': (K, M), 'C': (K, M)}[transa]
    shapeB = {'N': (K, N), 'T': (N, K), 'C': (N, K)}[transb]
    shapeC = (M, N)
    A0, A, descA = initialize_matrix(grid, *shapeA, 2, 2, random)
    B0, B, descB = initialize_matrix(grid, *shapeB, 2, 4, random)
    C0, C, descC = initialize_matrix(grid, *shapeC, 3, 2, random)

    if grid.comm.rank == 0:
        print(A0)

        # Calculate reference with numpy
        op_t = {'N': lambda M: M,
                'T': lambda M: np.transpose(M),
                'C': lambda M: np.conjugate(np.transpose(M))}
        ref_C0 = alpha * np.dot(op_t[transa](A0), op_t[transb](B0)) + beta * C0
    else:
        ref_C0 = None

    # Calculate with scalapack
    if simple:
        pblas_simple_gemm(descA, descB, descC,
                          A, B, C,
                          transa=transa, transb=transb)
    else:
        pblas_gemm(alpha, A, B, beta, C,
                   descA, descB, descC,
                   transa=transa, transb=transb)

    # Check error
    err = calculate_error(ref_C0, C, descC)
    assert err < tol


@pytest.mark.parametrize('mprocs, nprocs', mnprocs_i)
@pytest.mark.parametrize('uplo', ['L', 'U'])
@pytest.mark.parametrize('side', ['L', 'R'])
@pytest.mark.parametrize('simple', [True, False])
@pytest.mark.parametrize('hemm', [True, False])
@pytest.mark.parametrize('dtype', [float, complex])
def test_pblas_hemm_symm(dtype, hemm, simple, side, uplo, mprocs, nprocs,
                         M=160, N=120, seed=42):
    """Test pblas_simple_hemm, pblas_simple_symm, pblas_hemm, pblas_symm

    The operation is
    * C <- alpha*A*B + beta*C  if side == 'L'
    * C <- alpha*B*A + beta*C  if side == 'R'

    The computations are done with
    * lower triangular of A    if uplo == 'L'
    * upper triangular of A    if uplo == 'U'

    Additional options
    * A is Hermitian           if hemm == True
    * A is symmetric           if hemm == False
    * alpha=1 and beta=0       if simple == True
    """
    random = initialize_random(seed, dtype)
    grid = BlacsGrid(world, mprocs, nprocs)

    def generate_A_matrix(shape):
        A0 = random(shape)
        if grid.comm.rank == 0:
            if hemm:
                # Hermitian matrix
                A0 = A0 + A0.T.conj()
            else:
                # Symmetric matrix
                A0 = A0 + A0.T

            # Only lower or upper triangular is used, so
            # fill the other triangular with NaN to detect errors
            if uplo == 'L':
                A0 += np.triu(A0 * np.nan, 1)
            else:
                A0 += np.tril(A0 * np.nan, -1)
            A0 = np.ascontiguousarray(A0)
        return A0

    # Initialize matrices
    alpha, beta = initialize_alpha_beta(simple, random)
    shapeA = {'L': (M, M), 'R': (N, N)}[side]
    shapeB = (M, N)
    shapeC = (M, N)
    A0, A, descA = initialize_matrix(grid, *shapeA, 2, 2, generate_A_matrix)
    B0, B, descB = initialize_matrix(grid, *shapeB, 2, 4, random)
    C0, C, descC = initialize_matrix(grid, *shapeC, 3, 2, random)

    if grid.comm.rank == 0:
        print(A0)

        # Calculate reference with numpy
        tri2full(A0, uplo, map=np.conj if hemm else np.positive)
        if side == 'L':
            ref_C0 = alpha * np.dot(A0, B0) + beta * C0
        else:
            ref_C0 = alpha * np.dot(B0, A0) + beta * C0
    else:
        ref_C0 = None

    # Calculate with scalapack
    if simple and hemm:
        pblas_simple_hemm(descA, descB, descC,
                          A, B, C,
                          uplo=uplo, side=side)
    elif hemm:
        pblas_hemm(alpha, A, B, beta, C,
                   descA, descB, descC,
                   uplo=uplo, side=side)
    elif simple:
        pblas_simple_symm(descA, descB, descC,
                          A, B, C,
                          uplo=uplo, side=side)
    else:
        pblas_symm(alpha, A, B, beta, C,
                   descA, descB, descC,
                   uplo=uplo, side=side)

    # Check error
    err = calculate_error(ref_C0, C, descC)
    assert err < tol
