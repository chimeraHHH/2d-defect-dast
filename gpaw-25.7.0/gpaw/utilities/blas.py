# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Python wrapper functions for the ``C`` package:
Basic Linear Algebra Subroutines (BLAS)

See also:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
and
https://www.netlib.org/lapack/lug/node145.html
"""
from typing import TypeVar

import gpaw.cgpaw as cgpaw
import numpy as np
import scipy.linalg.blas as blas
from gpaw import debug
from gpaw.new import prod
from gpaw.typing import Array2D, ArrayND
from gpaw.utilities import is_contiguous


def is_finite(array, tril=False):
    if isinstance(array, np.ndarray):
        xp = np
    else:
        from gpaw.gpu import cupy as xp
    if tril:
        array = xp.tril(array)
    return xp.isfinite(array).all()


__all__ = ['mmm']

T = TypeVar('T', float, complex)


def mmm(alpha: T,
        a: Array2D,
        opa: str,
        b: Array2D,
        opb: str,
        beta: T,
        c: Array2D) -> None:
    """Matrix-matrix multiplication using dgemm or zgemm.

    For opa='N' and opb='N', we have:::

        c <- αab + βc.

    Use 'T' to transpose matrices and 'C' to transpose and complex conjugate
    matrices.
    """

    assert opa in 'NTC'
    assert opb in 'NTC'

    if opa == 'N':
        a1, a2 = a.shape
    else:
        a2, a1 = a.shape
    if opb == 'N':
        b1, b2 = b.shape
    else:
        b2, b1 = b.shape
    assert a2 == b1
    assert c.shape == (a1, b2)

    assert a.dtype == b.dtype == c.dtype
    assert a.strides[1] == c.itemsize or a.size == 0
    assert b.strides[1] == c.itemsize or b.size == 0
    assert c.strides[1] == c.itemsize or c.size == 0
    if a.dtype == float:
        assert not isinstance(alpha, complex)
        assert not isinstance(beta, complex)
    else:
        assert a.dtype == complex

    cgpaw.mmm(alpha, a, opa, b, opb, beta, c)


def gpu_mmm(alpha, a, opa, b, opb, beta, c):
    """Launch CPU or GPU version of mmm()."""
    m = b.shape[1] if opb == 'N' else b.shape[0]
    n = a.shape[0] if opa == 'N' else a.shape[1]
    k = b.shape[0] if opb == 'N' else b.shape[1]
    lda = a.strides[0] // a.itemsize
    ldb = b.strides[0] // b.itemsize
    ldc = c.strides[0] // c.itemsize
    cgpaw.mmm_gpu(alpha, a.data.ptr, lda, opa,
                  b.data.ptr, ldb, opb, beta,
                  c.data.ptr, ldc, c.itemsize,
                  m, n, k)


def gpu_scal(alpha, x):
    """alpha x

    Performs the operation::

      x <- alpha * x

    """
    if debug:
        if isinstance(alpha, complex):
            assert is_contiguous(x, complex)
        else:
            assert isinstance(alpha, float)
            assert x.dtype in [float, complex]
            assert x.flags.c_contiguous
    cgpaw.scal_gpu(alpha, x.data.ptr, x.shape, x.dtype)


def to2d(array: ArrayND) -> Array2D:
    """2D view af ndarray.

    >>> to2d(np.zeros((2, 3, 4))).shape
    (2, 12)
    """
    shape = array.shape
    return array.reshape((shape[0], prod(shape[1:])))


def mmmx(alpha: T,
         a: ArrayND,
         opa: str,
         b: ArrayND,
         opb: str,
         beta: T,
         c: ArrayND) -> None:
    """Matrix-matrix multiplication using dgemm or zgemm.

    Arrays a, b and c are converted to 2D arrays before calling mmm().
    """
    mmm(alpha, to2d(a), opa, to2d(b), opb, beta, to2d(c))


def gpu_gemm(alpha, a, b, beta, c, transa='n'):
    """General Matrix Multiply.

    Performs the operation::

      c <- alpha * b.a + beta * c

    If transa is "n", ``b.a`` denotes the matrix multiplication defined by::

                      _
                     \
      (b.a)        =  ) b  * a
           ijkl...   /_  ip   pjkl...
                      p

    If transa is "t" or "c", ``b.a`` denotes the matrix multiplication
    defined by::

                      _
                     \
      (b.a)        =  ) b    *    a
           ij        /_  iklm...   jklm...
                     klm...

    where in case of "c" also complex conjugate of a is taken.
    """
    if debug:
        assert beta == 0.0 or is_finite(c)

        assert (a.dtype == float and b.dtype == float and c.dtype == float and
                isinstance(alpha, float) and isinstance(beta, float) or
                a.dtype == complex and b.dtype == complex and
                c.dtype == complex)
        assert a.flags.c_contiguous
        if transa == 'n':
            assert c.flags.c_contiguous or (c.ndim == 2
                                            and c.strides[1] == c.itemsize)
            assert b.ndim == 2
            assert b.strides[1] == b.itemsize
            assert a.shape[0] == b.shape[1]
            assert c.shape == b.shape[0:1] + a.shape[1:]
        else:
            assert b.size == 0 or b[0].flags.c_contiguous
            assert c.strides[1] == c.itemsize
            assert a.shape[1:] == b.shape[1:]
            assert c.shape == (b.shape[0], a.shape[0])

    cgpaw.gemm_gpu(alpha, a.data.ptr, a.shape,
                   b.data.ptr, b.shape, beta,
                   c.data.ptr, c.shape,
                   a.dtype, transa)


def gpu_gemv(alpha, a, x, beta, y, trans='t'):
    """General Matrix Vector product.

    Performs the operation::

      y <- alpha * a.x + beta * y

    ``a.x`` denotes matrix multiplication, where the product-sum is
    over the entire length of the vector x and
    the first dimension of a (for trans='n'), or
    the last dimension of a (for trans='t' or 'c').

    If trans='c', the complex conjugate of a is used. The default is
    trans='t', i.e. behaviour like np.dot with a 2D matrix and a vector.
    """
    if debug:
        assert (a.dtype == float and x.dtype == float and y.dtype == float and
                isinstance(alpha, float) and isinstance(beta, float) or
                a.dtype == complex and x.dtype == complex and
                y.dtype == complex)
        assert a.flags.c_contiguous
        assert y.flags.c_contiguous
        assert x.ndim == 1
        assert y.ndim == a.ndim - 1
        if trans == 'n':
            assert a.shape[0] == x.shape[0]
            assert a.shape[1:] == y.shape
        else:
            assert a.shape[-1] == x.shape[0]
            assert a.shape[:-1] == y.shape

    cgpaw.gemv_gpu(alpha, a.data.ptr, a.shape,
                   x.data.ptr, x.shape, beta,
                   y.data.ptr, a.dtype,
                   trans)


which_axpy = {
    np.float32: blas.saxpy,
    np.float64: blas.daxpy,
    np.complex64: blas.caxpy,
    np.complex128: blas.zaxpy
}


def axpy(alpha, x, y):
    """alpha x plus y.

    Performs the operation::

      y <- alpha * x + y

    """
    if x.size == 0:
        return
    assert x.flags.contiguous
    assert y.flags.contiguous
    x = x.ravel()
    y = y.ravel()
    z = which_axpy[np.dtype(x.dtype).type](x, y, a=alpha)
    assert z is y, (x, y, x.shape, y.shape)


def gpu_axpy(alpha, x, y):
    """alpha x plus y.

    Performs the operation::

      y <- alpha * x + y

    """
    if debug:
        if isinstance(alpha, complex):
            assert is_contiguous(x, complex) and is_contiguous(y, complex)
        else:
            assert isinstance(alpha, float)
            assert x.dtype in [float, complex]
            assert x.dtype == y.dtype
            assert x.flags.c_contiguous and y.flags.c_contiguous
        assert x.shape == y.shape

    cgpaw.axpy_gpu(alpha, x.data.ptr, x.shape,
                   y.data.ptr, y.shape,
                   x.dtype)


def rk(alpha, a, beta, c, trans='c'):
    """Rank-k update of a matrix.

    For ``trans='c'`` the following operation is performed:::

              †
      c <- αaa + βc,

    and for ``trans='t'`` we get:::

             †
      c <- αa a + βc

    If the ``a`` array has more than 2 dimensions then the 2., 3., ...
    axes are combined.

    Only the lower triangle of ``c`` will contain sensible numbers.
    """
    if debug:
        assert beta == 0.0 or is_finite(c, tril=True)

        assert (a.dtype == float and c.dtype == float or
                a.dtype == complex and c.dtype == complex)
        assert a.flags.c_contiguous, (a.shape, a.strides, a.dtype)
        assert a.ndim > 1
        if trans == 'n':
            assert c.shape == (a.shape[1], a.shape[1])
        else:
            assert c.shape == (a.shape[0], a.shape[0])
        assert c.strides[1] == c.itemsize or c.size == 0

    cgpaw.rk(alpha, a, beta, c, trans)


def gpu_rk(alpha, a, beta, c, trans='c'):
    """Launch CPU or GPU version of rk()."""
    cgpaw.rk_gpu(alpha, a.data.ptr, a.shape,
                 beta, c.data.ptr, c.shape,
                 a.dtype)


def r2k(alpha, a, b, beta, c, trans='c'):
    """Rank-2k update of a matrix.

    Performs the operation::

                        dag        cc       dag
      c <- alpha * a . b    + alpha  * b . a    + beta * c

    or if trans='n'::
                    dag           cc   dag
      c <- alpha * a   . b + alpha  * b   . a + beta * c

    where ``a.b`` denotes the matrix multiplication defined by::

                 _
                \
      (a.b)   =  ) a         * b
           ij   /_  ipklm...     pjklm...
               pklm...

    ``cc`` denotes complex conjugation.

    ``dag`` denotes the hermitian conjugate (complex conjugation plus a
    swap of axis 0 and 1).

    Only the lower triangle of ``c`` will contain sensible numbers.
    """
    if debug:
        assert beta == 0.0 or is_finite(c, tril=True)
        assert (a.dtype == float and b.dtype == float and c.dtype == float or
                a.dtype == complex and b.dtype == complex and
                c.dtype == complex)
        assert a.flags.c_contiguous and b.flags.c_contiguous
        assert a.ndim > 1
        assert a.shape == b.shape
        if trans == 'c':
            assert c.shape == (a.shape[0], a.shape[0])
        else:
            assert c.shape == (a.shape[1], a.shape[1])
        assert c.strides[1] == c.itemsize or c.size == 0

    cgpaw.r2k(alpha, a, b, beta, c, trans)


def gpu_r2k(alpha, a, b, beta, c, trans='c'):
    """Launch CPU or GPU version of r2k()."""
    cgpaw.r2k_gpu(alpha, a.data.ptr, a.shape,
                  b.data.ptr, b.shape, beta,
                  c.data.ptr, c.shape,
                  a.dtype)


def gpu_dotc(a, b):
    r"""Dot product, conjugating the first vector with complex arguments.

    Returns the value of the operation::

        _
       \   cc
        ) a       * b
       /_  ijk...    ijk...
       ijk...

    ``cc`` denotes complex conjugation.
    """
    if debug:
        assert ((is_contiguous(a, float) and is_contiguous(b, float)) or
                (is_contiguous(a, complex) and is_contiguous(b, complex)))
        assert a.shape == b.shape

    return cgpaw.dotc_gpu(a.data.ptr, a.shape,
                          b.data.ptr, a.dtype)


def gpu_dotu(a, b):
    """Dot product, NOT conjugating the first vector with complex arguments.

    Returns the value of the operation::

        _
       \
        ) a       * b
       /_  ijk...    ijk...
       ijk...


    """
    if debug:
        assert ((is_contiguous(a, float) and is_contiguous(b, float)) or
                (is_contiguous(a, complex) and is_contiguous(b, complex)))
        assert a.shape == b.shape

    return cgpaw.dotu_gpu(a.data.ptr, a.shape,
                          b.data.ptr, a.dtype)


def _gemmdot(a, b, alpha=1.0, beta=1.0, out=None, trans='n'):
    """Matrix multiplication using gemm.

    return reference to out, where::

      out <- alpha * a . b + beta * out

    If out is None, a suitably sized zero array will be created.

    ``a.b`` denotes matrix multiplication, where the product-sum is
    over the last dimension of a, and either
    the first dimension of b (for trans='n'), or
    the last dimension of b (for trans='t' or 'c').

    If trans='c', the complex conjugate of b is used.
    """
    # Store original shapes
    ashape = a.shape
    bshape = b.shape

    # Vector-vector multiplication is handled by dotu
    if a.ndim == 1 and b.ndim == 1:
        assert out is None
        if trans == 'c':
            return alpha * np.vdot(b, a)  # dotc conjugates *first* argument
        else:
            return alpha * a.dot(b)

    # Map all arrays to 2D arrays
    a = a.reshape(-1, a.shape[-1])
    if trans == 'n':
        b = b.reshape(b.shape[0], -1)
        outshape = a.shape[0], b.shape[1]
    else:  # 't' or 'c'
        b = b.reshape(-1, b.shape[-1])

    # Apply BLAS gemm routine
    outshape = a.shape[0], b.shape[trans == 'n']
    if out is None:
        # (ATLAS can't handle uninitialized output array)
        out = np.zeros(outshape, a.dtype)
    else:
        out = out.reshape(outshape)
    mmmx(alpha, a, 'N', b, trans.upper(), beta, out)

    # Determine actual shape of result array
    if trans == 'n':
        outshape = ashape[:-1] + bshape[1:]
    else:  # 't' or 'c'
        outshape = ashape[:-1] + bshape[:-1]
    return out.reshape(outshape)


if not hasattr(cgpaw, 'mmm'):
    # These are the functions used with noblas=True
    # TODO: move these functions elsewhere so that
    # they can be used for unit tests

    def op(o, m):
        if o.upper() == 'N':
            return m
        if o.upper() == 'T':
            return m.T
        if o.upper() == 'C':
            return m.conj().T
        raise ValueError(f'unknown op: {o}')

    def rk(alpha, a, beta, c, trans='c'):  # noqa
        if c.size == 0:
            return
        if beta == 0:
            c[:] = 0.0
        else:
            c *= beta
        if trans == 'n':
            c += alpha * a.conj().T.dot(a)
        else:
            a = a.reshape((len(a), -1))
            c += alpha * a.dot(a.conj().T)

    def r2k(alpha, a, b, beta, c, trans='c'):  # noqa
        if c.size == 0:
            return
        if beta == 0.0:
            c[:] = 0.0
        else:
            c *= beta
        if trans == 'c':
            c += (alpha * a.reshape((len(a), -1))
                  .dot(b.reshape((len(b), -1)).conj().T) +
                  alpha * b.reshape((len(b), -1))
                  .dot(a.reshape((len(a), -1)).conj().T))
        else:
            c += alpha * (a.conj().T @ b + b.conj().T @ a)

    def mmm(alpha: T, a: np.ndarray, opa: str,  # noqa
            b: np.ndarray, opb: str,
            beta: T, c: np.ndarray) -> None:
        if beta == 0.0:
            c[:] = 0.0
        else:
            c *= beta
        c += alpha * op(opa, a).dot(op(opb, b))

    gemmdot = _gemmdot

elif not debug:
    mmm = cgpaw.mmm  # noqa
    rk = cgpaw.rk  # noqa
    r2k = cgpaw.r2k  # noqa
    gemmdot = _gemmdot

else:
    def gemmdot(a, b, alpha=1.0, beta=1.0, out=None, trans='n'):
        assert a.flags.c_contiguous
        assert b.flags.c_contiguous
        assert a.dtype == b.dtype
        if trans == 'n':
            assert a.shape[-1] == b.shape[0]
        else:
            assert a.shape[-1] == b.shape[-1]
        if out is not None:
            assert out.flags.c_contiguous
            assert a.dtype == out.dtype
            assert a.ndim > 1 or b.ndim > 1
            if trans == 'n':
                assert out.shape == a.shape[:-1] + b.shape[1:]
            else:
                assert out.shape == a.shape[:-1] + b.shape[:-1]
        return _gemmdot(a, b, alpha, beta, out, trans)
