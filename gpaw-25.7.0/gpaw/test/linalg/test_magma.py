"""Tests for MAGMA eigensolver wrappers"""
import numpy as np
import pytest
from gpaw.new.magma import eigh_magma_cpu, eigh_magma_gpu
from gpaw.cgpaw import have_magma
from gpaw.gpu import cupy as cp
from gpaw.gpu import cupy_is_fake


def fix_eigenvector_phase(inout_arr):
    """Helper function for comparing eigenvector output from different
    solvers. Rotates eigenvectors in the input matrix so that the first
    element of each vector is real and non-negative.
    Input is modified in-place.
    NB: eigenvectors are on columns.
    """
    assert inout_arr.ndim == 2
    # Works for cupy arrays too because the dtypes are compatible

    if np.issubdtype(inout_arr.dtype, np.complexfloating):
        # Complex matrices
        for i in range(inout_arr.shape[1]):
            phase = np.angle(inout_arr[0, i])
            if phase != 0:
                rotation = np.exp(phase * (-1j))
                inout_arr[:, i] *= rotation

    elif np.issubdtype(inout_arr.dtype, np.floating):
        # Real matrices
        for i in range(inout_arr.shape[1]):
            if inout_arr[0, i] < 0:
                inout_arr[:, i] *= -1

    return inout_arr


@pytest.fixture
def eigh_test_matrix():
    def _generate(n: int, type: str = 'symmetric',
                  backend: str = 'numpy', seed: int = 42):

        assert type in ['symmetric', 'hermitian']
        assert backend in ['numpy', 'cupy']

        if backend == 'cupy':
            xp = cp
        else:
            xp = np

        rng = xp.random.default_rng(seed)
        if type == 'symmetric':
            A = rng.random((n, n))
            return (A + A.T) / 2

        else:
            # Create Hermitian matrix
            A = rng.random((n, n)) + 1j * rng.random((n, n))
            return (A + A.T.conj()) / 2

    return _generate


@pytest.mark.skipif(not have_magma, reason="No MAGMA")
@pytest.mark.parametrize("matrix_size, matrix_type, uplo",
                         [(2, 'symmetric', 'L'), (4, 'hermitian', 'U')])
def test_eigh_magma_cpu(eigh_test_matrix: np.ndarray,
                        matrix_size: int,
                        matrix_type: str,
                        uplo: str) -> None:
    """Compare eigh output of Numpy and MAGMA"""

    arr = eigh_test_matrix(matrix_size, type=matrix_type, backend='numpy')
    eigvals, eigvects = eigh_magma_cpu(arr, uplo)

    eigvals_np, eigvects_np = np.linalg.eigh(arr, UPLO=uplo)

    fix_eigenvector_phase(eigvects)
    fix_eigenvector_phase(eigvects_np)

    np.testing.assert_allclose(eigvals, eigvals_np, atol=1e-12)
    np.testing.assert_allclose(eigvects, eigvects_np, atol=1e-12)


# MAGMA seems to do small matrices (N <= 128) on the CPU.
# So need a large matrix for honest GPU tests
@pytest.mark.skipif(not have_magma, reason="No MAGMA")
@pytest.mark.skipif(cupy_is_fake,
                    reason="MAGMA GPU tests disabled for fake cupy")
@pytest.mark.gpu
@pytest.mark.parametrize("matrix_size, matrix_type, uplo",
                         [(16, 'symmetric', 'L'),
                          (150, 'hermitian', 'L'),
                          (256, 'symmetric', 'U')])
def test_eigh_magma_gpu(eigh_test_matrix: cp.ndarray,
                        matrix_size: int,
                        matrix_type: str,
                        uplo: str):
    """Compare eigh output of CUPY and MAGMA"""

    arr = eigh_test_matrix(matrix_size, type=matrix_type, backend='cupy')
    eigvals, eigvects = eigh_magma_gpu(arr, uplo)

    eigvals_cp, eigvects_cp = cp.linalg.eigh(arr, UPLO=uplo)

    fix_eigenvector_phase(eigvects)
    fix_eigenvector_phase(eigvects_cp)

    cp.testing.assert_allclose(eigvals, eigvals_cp, atol=1e-12)
    cp.testing.assert_allclose(eigvects, eigvects_cp, atol=1e-12)
