import numpy as np
import pytest
from gpaw.core.matrix import Matrix
from gpaw.gpu import cupy as cp, as_np, as_xp
from gpaw.mpi import world
from gpaw.new.c import GPU_AWARE_MPI
from gpaw.gpu.mpi import CuPyMPI


@pytest.mark.gpu
@pytest.mark.serial
def test_zyrk():
    a = np.array([[1, 1 + 2j, 2], [1, 0.5j, -1 - 0.5j]])
    m = Matrix(2, 3, data=a)
    b = m.multiply(m, opb='C', beta=0.0, symmetric=True)
    b.tril2full()
    a = cp.asarray(a)
    m = Matrix(2, 3, data=a)
    b2 = m.multiply(m, opb='C', beta=0.0, symmetric=True)
    b2.tril2full()
    c = b2.to_cpu()
    assert (c.data == b.data).all()


@pytest.mark.gpu
@pytest.mark.serial
def test_eigh():
    H1 = Matrix(2, 2, data=np.array([[2, 42.1 + 42.1j], [0.1 - 0.1j, 3]]))
    S1 = Matrix(2, 2, data=np.array([[1, 42.1 + 42.2j], [0.1 - 0.2j, 0.9]]))
    H2 = Matrix(2, 2, data=cp.asarray(H1.data))
    S2 = Matrix(2, 2, data=cp.asarray(S1.data))

    E1 = H1.eigh(S1)

    S0 = S1.copy()
    S0.tril2full()

    E2 = H2.eigh(S2)
    assert as_np(E2) == pytest.approx(E1)

    C1 = H1.data
    C2 = H2.to_cpu().data

    # Check that eigenvectors are parallel:
    X = C1.conj() @ S0.data @ C2.T
    assert abs(X) == pytest.approx(np.eye(2))


def op(a: np.ndarray, o: str) -> np.ndarray:
    if o == 'N':
        return a
    if o == 'C':
        return a.T.conj()
    1 / 0


@pytest.mark.gpu
@pytest.mark.parametrize(
    'shape1, shape2, op1, op2, sym, same',
    [((5, 9), (5, 9), 'N', 'C', 1, 1),
     ((2, 3), (2, 3), 'N', 'C', 1, 0),
     ((5, 9), (5, 9), 'N', 'C', 0, 0),
     ((5, 9), (5, 9), 'C', 'N', 0, 0),
     ((5, 9), (9, 5), 'C', 'C', 0, 0),
     ((5, 5), (5, 9), 'N', 'N', 0, 0)])
@pytest.mark.parametrize('beta', [0.0, 1.0])
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('xp', [np, cp])
def test_mul(shape1, shape2, op1, op2, beta, sym, same, dtype, xp, rng):
    if world.size > 1 and xp is cp:
        if op1 == 'C' or (op1 == 'N' and op2 == 'C' and sym and beta == 0.0):
            pytest.skip('Not implemented!')
    alpha = 1.234
    comm = world if GPU_AWARE_MPI else CuPyMPI(world)

    shape3 = (shape1[0] if op1 == 'N' else shape1[1],
              shape2[1] if op2 == 'N' else shape2[0])
    m1, m2, m3 = (Matrix(*shape, dtype=dtype, dist=(comm, 1, 1), xp=xp)
                  for shape in [shape1, shape2, shape3])

    if world.rank == 0:
        for m in [m1, m2, m3]:
            data = m.data.view(float)
            data[:] = as_xp(rng.random(data.shape), xp)
        if sym:
            m2.data[:] = m1.data
            m3.data += m3.data.T.conj()

        # Correct result:
        a1, a2, a3 = (as_np(m.data) for m in [m1, m2, m3])
        a3 = beta * a3 + alpha * op(a1, op1) @ op(a2, op2)

    M1, M2, M3 = (Matrix(*shape, dtype=dtype, dist=(comm, -1, 1), xp=xp)
                  for shape in [shape1, shape2, shape3])
    for m, M in zip([m1, m2, m3], [M1, M2, M3]):
        m.redist(M)

    if same:
        M2 = M1

    M1.multiply(M2, alpha=alpha, opa=op1, opb=op2, beta=beta,
                out=M3, symmetric=sym)

    m3 = M3.gather()
    if world.rank == 0:
        if sym:
            m3.tril2full()
        error = abs(a3 - as_np(m3.data)).max()
    else:
        error = 0.0
    error = world.sum_scalar(error)
    assert error < 1e-13


if __name__ == '__main__':
    test_mul((1, 1), (1, 19), 'N', 'N', 0.0, 0, 0,
             complex, cp, np.random.default_rng(42))
