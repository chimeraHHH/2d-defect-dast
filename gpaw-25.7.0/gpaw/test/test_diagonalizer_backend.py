import numpy as np
import pytest
from ase.parallel import world
from gpaw.utilities import compiled_with_sl
from gpaw.utilities.elpa import LibElpa
from gpaw.eigensolvers.diagonalizerbackend import (
    DistributedBlacsDiagonalizer,
    ScipyDiagonalizer,
    ScalapackDiagonalizer,
    ElpaDiagonalizer)


def prepare_eigensolver_matrices(size_of_matrices, dtype):

    matrix_dimensions = [size_of_matrices, size_of_matrices]
    rng = np.random.Generator(np.random.PCG64(24589246))
    A = rng.random(matrix_dimensions).astype(dtype)
    B = rng.random(matrix_dimensions).astype(dtype)

    if dtype == complex:
        A += 1j * rng.random(matrix_dimensions)
        B += 1j * rng.random(matrix_dimensions)
    A = A + A.T.conj()
    B = B + B.T.conj()
    # Make sure B is positive definite
    B += np.eye(size_of_matrices) * size_of_matrices

    return A, B


@pytest.fixture(params=['eigh', 'scalapack', 'elpa'])
def backend_problemsize_kwargs(request):
    name = request.param
    eigenproblem_size = world.size * 64
    if name == 'eigh':
        return ScipyDiagonalizer, eigenproblem_size, {'comm': world}
    elif name == 'scalapack' or name == 'elpa':
        if not compiled_with_sl():
            pytest.skip()

        nrows = 2 if world.size > 1 else 1
        ncols = world.size // 2 if nrows > 1 else 1

        scalapack_kwargs = {
            'arraysize': eigenproblem_size,
            'grid_nrows': nrows,
            'grid_ncols': ncols,
            'scalapack_communicator': world,
            'blocksize': 32 if world.size == 1 else 64}

        if name == 'scalapack':
            return ScalapackDiagonalizer, eigenproblem_size, scalapack_kwargs

        if name == 'elpa':
            if not LibElpa.have_elpa():
                pytest.skip()
            return ElpaDiagonalizer, eigenproblem_size, scalapack_kwargs


@pytest.mark.parametrize('dtype,', [float, complex])
def test_diagonalizer_eigenproblem_correctness(backend_problemsize_kwargs,
                                               dtype):
    is_master_rank = world.rank == 0
    (
        diagonalizer_class,
        eigenproblem_size,
        diagonalizer_kwargs) = backend_problemsize_kwargs

    if diagonalizer_class is ScipyDiagonalizer:
        diagonalizer = diagonalizer_class(**diagonalizer_kwargs)
    elif issubclass(diagonalizer_class, DistributedBlacsDiagonalizer):
        diagonalizer = diagonalizer_class(**diagonalizer_kwargs, dtype=dtype)

    a, b = prepare_eigensolver_matrices(eigenproblem_size, dtype=dtype)
    eps = np.zeros(eigenproblem_size)
    a_copy = a.copy()
    b_copy = b.copy()

    # a_copy contains eigenvectors after this.
    diagonalizer.diagonalize(
        a_copy,
        b_copy,
        eps,
        debug=False)

    if is_master_rank:
        assert np.allclose(a @ a_copy, b @ a_copy @ np.diag(eps))
