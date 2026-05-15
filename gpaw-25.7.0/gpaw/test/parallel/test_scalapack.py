"""Test of ScaLAPACK diagonalize and inverse cholesky.

The test generates a random symmetric matrix H0 and
positive definite matrix S0 on a 1-by-1 BLACS grid. They
are redistributed to a mprocs-by-nprocs BLACS grid,
diagonalized in parallel, and eigenvalues are compared
against LAPACK. Eigenvectors are not compared.
"""

import pytest
import numpy as np
from scipy.linalg import eigh, inv, cholesky

from gpaw.mpi import world, rank
from gpaw.blacs import BlacsGrid, Redistributor
from gpaw.utilities.tools import tri2full
from gpaw.utilities import compiled_with_sl
from gpaw.utilities.blas import rk
from gpaw.utilities.scalapack import \
    scalapack_general_diagonalize_dc, \
    scalapack_diagonalize_dc, \
    scalapack_inverse_cholesky, \
    scalapack_inverse, \
    scalapack_solve

from .test_pblas import \
    initialize_random, initialize_matrix, \
    calculate_error, mnprocs_i


pytestmark = pytest.mark.skipif(not compiled_with_sl(),
                                reason='not compiled with scalapack')


tol = 1.0e-8


@pytest.mark.parametrize('mprocs, nprocs', mnprocs_i)
@pytest.mark.parametrize('dtype', [float, complex])
def test_scalapack_diagonalize_inverse(dtype, mprocs, nprocs,
                                       N=72, seed=42):
    gen = np.random.RandomState(seed)
    grid = BlacsGrid(world, mprocs, nprocs)

    if dtype == complex:
        epsilon = 1.0j
    else:
        epsilon = 0.0

    # Create descriptors for matrices on master:
    glob = grid.new_descriptor(N, N, N, N)

    # print globA.asarray()
    # Populate matrices local to master:
    H0 = glob.zeros(dtype=dtype) + gen.rand(*glob.shape)
    S0 = glob.zeros(dtype=dtype) + gen.rand(*glob.shape)
    C0 = glob.empty(dtype=dtype)
    if rank == 0:
        # Complex case must have real numbers on the diagonal.
        # We make a simple complex Hermitian matrix below.
        H0 = H0 + epsilon * (0.1 * np.tri(N, N, k=-N // nprocs) +
                             0.3 * np.tri(N, N, k=-1))
        S0 = S0 + epsilon * (0.2 * np.tri(N, N, k=-N // nprocs) +
                             0.4 * np.tri(N, N, k=-1))
        # Make matrices symmetric
        rk(1.0, H0.copy(), 0.0, H0)
        rk(1.0, S0.copy(), 0.0, S0)
        # Overlap matrix must be semi-positive definite
        S0 = S0 + 50.0 * np.eye(N, N, 0)
        # Hamiltonian is usually diagonally dominant
        H0 = H0 + 75.0 * np.eye(N, N, 0)
        C0 = S0.copy()
        S0_inv = S0.copy()

    # Local result matrices
    W0 = np.empty((N), dtype=float)
    W0_g = np.empty((N), dtype=float)

    # Calculate eigenvalues / other serial results
    print(rank, C0.shape, C0.dtype)
    if rank == 0:
        W0 = eigh(H0, eigvals_only=True)
        W0_g = eigh(H0, S0, eigvals_only=True)
        C0 = inv(cholesky(C0, lower=True)).copy()  # result in lower triangle
        tri2full(S0_inv, 'L')
        S0_inv = inv(S0_inv)
        # tri2full(C0) # symmetrize

    print(rank, C0.shape, C0.dtype)
    assert glob.check(H0)
    assert glob.check(S0)
    assert glob.check(C0)

    # Create distributed destriptors with various block sizes:
    dist = grid.new_descriptor(N, N, 8, 8)

    # Distributed matrices:
    # We can use empty here, but end up with garbage on
    # on the other half of the triangle when we redistribute.
    # This is fine because ScaLAPACK does not care.

    H = dist.empty(dtype=dtype)
    S = dist.empty(dtype=dtype)
    Sinv = dist.empty(dtype=dtype)
    Z = dist.empty(dtype=dtype)
    C = dist.empty(dtype=dtype)
    Sinv = dist.empty(dtype=dtype)

    # Eigenvalues are non-BLACS matrices
    # W = np.empty((N), dtype=float)
    W_dc = np.empty((N), dtype=float)
    # W_mr3 = np.empty((N), dtype=float)
    # W_g = np.empty((N), dtype=float)
    W_g_dc = np.empty((N), dtype=float)
    # W_g_mr3 = np.empty((N), dtype=float)

    Glob2dist = Redistributor(world, glob, dist)
    Glob2dist.redistribute(H0, H, uplo='L')
    Glob2dist.redistribute(S0, S, uplo='L')
    Glob2dist.redistribute(S0, C, uplo='L')  # C0 was previously overwritten
    Glob2dist.redistribute(S0, Sinv, uplo='L')

    # we don't test the expert drivers anymore since there
    # might be a buffer overflow error
    # scalapack_diagonalize_ex(dist, H.copy(), Z, W, 'L')
    scalapack_diagonalize_dc(dist, H.copy(), Z, W_dc, 'L')
    # scalapack_diagonalize_mr3(dist, H.copy(), Z, W_mr3, 'L')
    # scalapack_general_diagonalize_ex(dist, H.copy(), S.copy(), Z, W_g, 'L')
    scalapack_general_diagonalize_dc(dist, H.copy(), S.copy(), Z, W_g_dc, 'L')

    scalapack_inverse_cholesky(dist, C, 'L')

    if dtype == complex:  # Only supported for complex for now
        scalapack_inverse(dist, Sinv, 'L')
    # Undo redistribute
    C_test = glob.empty(dtype=dtype)
    Sinv_test = glob.empty(dtype=dtype)
    Dist2glob = Redistributor(world, dist, glob)
    Dist2glob.redistribute(C, C_test)
    Dist2glob.redistribute(Sinv, Sinv_test)

    if rank == 0:
        diag_dc_err = abs(W_dc - W0).max()
        # diag_mr3_err = abs(W_mr3 - W0).max()
        # general_diag_ex_err = abs(W_g - W0_g).max()
        general_diag_dc_err = abs(W_g_dc - W0_g).max()
        # general_diag_mr3_err = abs(W_g_mr3 - W0_g).max()
        print(C_test[:2, :2])
        print(C0[:2, :2])
        inverse_chol_err = abs(C_test - C0).max()

        tri2full(Sinv_test, 'L')
        inverse_err = abs(Sinv_test - S0_inv).max()
        # print 'diagonalize ex err', diag_ex_err
        print('diagonalize dc err', diag_dc_err)
        # print 'diagonalize mr3 err', diag_mr3_err
        # print 'general diagonalize ex err', general_diag_ex_err
        print('general diagonalize dc err', general_diag_dc_err)
        # print 'general diagonalize mr3 err', general_diag_mr3_err
        print('inverse chol err', inverse_chol_err)
        if dtype == complex:
            print('inverse err', inverse_err)
    else:
        # diag_ex_err = 0.0
        diag_dc_err = 0.0
        # diag_mr3_err = 0.0
        # general_diag_ex_err = 0.0
        general_diag_dc_err = 0.0
        # general_diag_mr3_err = 0.0
        inverse_chol_err = 0.0
        inverse_err = 0.0

    # We don't like exceptions on only one cpu
    # diag_ex_err = world.sum(diag_ex_err)
    diag_dc_err = world.sum_scalar(diag_dc_err)
    # diag_mr3_err = world.sum(diag_mr3_err)
    # general_diag_ex_err = world.sum(general_diag_ex_err)
    general_diag_dc_err = world.sum_scalar(general_diag_dc_err)
    # general_diag_mr3_err = world.sum(general_diag_mr3_err)
    inverse_chol_err = world.sum_scalar(inverse_chol_err)
    inverse_err = world.sum_scalar(inverse_err)
    # assert diag_ex_err < tol
    assert diag_dc_err < tol
    # assert diag_mr3_err < tol
    # assert general_diag_ex_err < tol
    assert general_diag_dc_err < tol
    # assert general_diag_mr3_err < tol
    assert inverse_chol_err < tol
    if dtype == complex:
        assert inverse_err < tol


@pytest.mark.parametrize('mprocs, nprocs', mnprocs_i)
@pytest.mark.parametrize('dtype', [float, complex])
def test_scalapack_solve(dtype, mprocs, nprocs,
                         M=160, K=120, seed=42):
    """Test scalapack_solve()"""
    random = initialize_random(seed, dtype)
    grid = BlacsGrid(world, mprocs, nprocs)

    # Initialize matrices
    shapeA = (M, M)
    shapeB = (K, M)
    A0, A, descA = initialize_matrix(grid, *shapeA, 2, 2, random)
    B0, B, descB = initialize_matrix(grid, *shapeB, 3, 2, random)

    if grid.comm.rank == 0:
        # Calculate reference with numpy
        ref_B0 = np.linalg.solve(A0.T, B0.T).T
    else:
        ref_B0 = None

    # Calculate with scalapack
    scalapack_solve(descA, descB, A, B)

    # Check error
    err = calculate_error(ref_B0, B, descB)
    tol = {float: 8e-12, complex: 2e-13}[dtype]
    assert err < tol
