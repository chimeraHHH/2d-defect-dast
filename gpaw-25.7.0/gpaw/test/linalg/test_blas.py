import numpy as np
from gpaw.utilities.blas import axpy, r2k, rk, gemmdot, mmm, mmmx
from gpaw.utilities.tools import tri2full


def test_gemm_size_zero():
    c = np.ones((3, 3))
    a = np.zeros((0, 3))
    b = np.zeros((3, 0))
    d = np.zeros((0, 0))
    e = np.zeros((0, 3))
    # gemm(1.0, a, b, 0.0, c, 'n')
    mmm(1.0, b, 'N', a, 'N', 0.0, c)
    assert (c == 0.0).all()
    mmm(1.0, d, 'N', a, 'N', 0.0, e)


def test_linalg_blas():
    a = np.arange(5 * 7).reshape(5, 7) + 4.
    a2 = np.arange(3 * 7).reshape(3, 7) + 3.
    b = np.arange(7) - 2.

    # Check gemmdot with floats
    assert np.all(np.dot(a, b) == gemmdot(a, b))
    assert np.all(np.dot(a, a2.T) == gemmdot(a, a2, trans='t'))
    assert np.all(np.dot(a, a2.T) == gemmdot(a, a2, trans='c'))
    assert np.dot(b, b) == gemmdot(b, b)

    # Check gemmdot with complex arrays
    a = a * (2 + 1.j)
    a2 = a2 * (-1 + 3.j)
    b = b * (3 - 2.j)
    assert np.all(np.dot(a, b) == gemmdot(a, b))
    assert np.all(np.dot(a, a2.T) == gemmdot(a, a2, trans='t'))
    assert np.all(np.dot(a, a2.T.conj()) == gemmdot(a, a2, trans='c'))
    assert np.dot(b, b) == gemmdot(b, b, trans='n')
    assert np.dot(b, b.conj()) == gemmdot(b, b, trans='c')

    # Check gemm for transa='n'
    a2 = np.arange(7 * 5 * 1 * 3).reshape(7, 5, 1, 3) * (-1. + 4.j) + 3.
    c = np.tensordot(a, a2, [1, 0])
    mmmx(1., a, 'N', a2, 'N', -1., c)
    assert not c.any()

    # Check gemm for transa='c'
    a = np.arange(4 * 5 * 1 * 3).reshape(4, 5, 1, 3) * (3. - 2.j) + 4.
    c = np.tensordot(a, a2.conj(), [[1, 2, 3], [1, 2, 3]])
    mmmx(1., a, 'N', a2, 'C', -1., c)
    assert not c.any()

    # Check axpy
    c = 5.j * a
    axpy(-5.j, a, c)
    assert not c.any()

    # Check rk
    c = np.tensordot(a, a.conj(), [[1, 2, 3], [1, 2, 3]])
    rk(1., a, -1., c)
    tri2full(c)
    assert not c.any()

    # Check gemmdot for transa='c'
    c = np.tensordot(a, a2.conj(), [-1, -1])
    gemmdot(a, a2, beta=-1., out=c, trans='c')
    assert not c.any()

    # Check gemmdot for transa='n'
    a2.shape = 3, 7, 5, 1
    c = np.tensordot(a, a2, [-1, 0])
    gemmdot(a, a2, beta=-1., out=c, trans='n')
    assert not c.any()

    # Check r2k
    a2 = 5. * a
    c = np.tensordot(a, a2.conj(), [[1, 2, 3], [1, 2, 3]])
    r2k(.5, a, a2, -1., c)
    tri2full(c)
    assert not c.any()
