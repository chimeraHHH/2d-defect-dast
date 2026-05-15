# Copyright (C) 2003  CAMP
# Copyright (C) 2010  Argonne National Laboratory
# Please see the accompanying LICENSE file for further information.

"""
Python wrapper functions for the C and Fortran packages:
Basic Linear Algebra Communication Subprogramcs (BLACS)
ScaLAPACK

See also:
https://www.netlib.org/blacs
and
https://www.netlib.org/scalapack
"""
import numpy as np

import gpaw.cgpaw as cgpaw

switch_lu = {'U': 'L', 'L': 'U'}
switch_lr = {'L': 'R', 'R': 'L'}


def scalapack_tri2full(desc, array, conj=True):
    """Write lower triangular part into upper triangular part of matrix.

    If conj == True, the lower triangular part is the complex conjugate
    of the upper triangular part.

    This function is a frightful hack, but we can improve the
    implementation later."""

    # Zero upper triangle:
    scalapack_zero(desc, array, 'U')
    buf = array.copy()
    # Set diagonal to zero in the copy:
    scalapack_set(desc, buf, alpha=0.0, beta=0.0, uplo='U')
    # Now transpose tmp_mm adding the result to the original matrix:
    pblas_tran(alpha=1.0, a_MN=buf,
               beta=1.0, c_NM=array,
               desca=desc, descc=desc,
               conj=conj)


def scalapack_zero(desca, a, uplo, ia=1, ja=1):
    """Zero the upper or lower half of a square matrix."""
    assert desca.gshape[0] == desca.gshape[1]
    p = desca.gshape[0] - 1
    if uplo == 'L':
        ia = ia + 1
    else:
        ja = ja + 1
    scalapack_set(desca, a, 0.0, 0.0, uplo, p, p, ia, ja)


def scalapack_set(desca, a, alpha, beta, uplo, m=None, n=None, ia=1, ja=1):
    """Set the diagonal and upper/lower triangular part of a.

    Set the upper or lower triangular part of a to alpha, and the diagonal
    of a to beta, where alpha and beta are real or complex numbers."""
    desca.checkassert(a)
    assert uplo in ['L', 'U']
    if m is None:
        m = desca.gshape[0]
    if n is None:
        n = desca.gshape[1]
    if not desca.blacsgrid.is_active():
        return
    cgpaw.scalapack_set(a, desca.asarray(), alpha, beta,
                        switch_lu[uplo], n, m, ja, ia)


def scalapack_diagonalize_dc(desca, a, z, w, uplo):
    """Diagonalize symmetric matrix using the divide & conquer algorithm.
    Orthogonal eigenvectors not guaranteed; no warning is provided.

    Solve the eigenvalue equation::

      A_nn Z_nn = w_N Z_nn

    Diagonalizes A_nn and writes eigenvectors to Z_nn.  Both A_nn
    and Z_nn must be compatible with desca descriptor.  Values in
    A_nn will be overwritten.

    Eigenvalues are written to the global array w_N in ascending order.

    The `uplo` flag can be either 'L' or 'U', meaning that the
    matrices are taken to be upper or lower triangular respectively.
    """
    desca.checkassert(a)
    desca.checkassert(z)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    assert uplo in ['L', 'U']
    if not desca.blacsgrid.is_active():
        return
    assert desca.gshape[0] == len(w)
    info = cgpaw.scalapack_diagonalize_dc(a, desca.asarray(),
                                          switch_lu[uplo], z, w)
    if info != 0:
        raise RuntimeError('scalapack_diagonalize_dc error: %d' % info)


def scalapack_diagonalize_ex(desca, a, z, w, uplo, iu=None):
    """Diagonalize symmetric matrix using the bisection and inverse
    iteration algorithm. Re-orthogonalization of eigenvectors
    is an issue for tightly clustered eigenvalue problems; it
    requires substantial memory and is not scalable. See ScaLAPACK
    pdsyevx.f routine for more information.

    Solve the eigenvalue equation::

      A_nn Z_nn = w_N Z_nn

    Diagonalizes A_nn and writes eigenvectors to Z_nn.  Both A_nn
    and Z_nn must be compatible with desca descriptor.  Values in
    A_nn will be overwritten.

    Eigenvalues are written to the global array w_N in ascending order.

    The `uplo` flag can be either 'L' or 'U', meaning that the
    matrices are taken to be upper or lower triangular respectively.

    The `iu` specifies how many eigenvectors and eigenvalues to compute.
    """
    desca.checkassert(a)
    desca.checkassert(z)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    if iu is None:  # calculate all eigenvectors and eigenvalues
        iu = desca.gshape[0]
    assert 1 < iu <= desca.gshape[0]
    # still need assert for eigenvalues
    assert uplo in ['L', 'U']
    if not desca.blacsgrid.is_active():
        return
    assert desca.gshape[0] == len(w)
    info = cgpaw.scalapack_diagonalize_ex(a, desca.asarray(),
                                          switch_lu[uplo],
                                          iu, z, w)
    if info != 0:
        # 0 means you are OK
        raise RuntimeError('scalapack_diagonalize_ex error: %d' % info)


def scalapack_diagonalize_mr3(desca, a, z, w, uplo, iu=None):
    """Diagonalize symmetric matrix using the MRRR algorithm.

    Solve the eigenvalue equation::

      A_nn Z_nn = w_N Z_nn

    Diagonalizes A_nn and writes eigenvectors to Z_nn.  Both A_nn
    and Z_nn must be compatible with this desca descriptor.  Values in
    A_nn will be overwritten.

    Eigenvalues are written to the global array w_N in ascending order.

    The `uplo` flag can be either 'L' or 'U', meaning that the
    matrices are taken to be upper or lower triangular respectively.

    The `iu` specifies how many eigenvectors and eigenvalues to compute.
    """
    desca.checkassert(a)
    desca.checkassert(z)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    if iu is None:  # calculate all eigenvectors and eigenvalues
        iu = desca.gshape[0]
    assert 1 < iu <= desca.gshape[0]
    # stil need assert for eigenvalues
    assert uplo in ['L', 'U']
    if not desca.blacsgrid.is_active():
        return
    assert desca.gshape[0] == len(w)
    info = cgpaw.scalapack_diagonalize_mr3(a, desca.asarray(),
                                           switch_lu[uplo],
                                           iu, z, w)
    if info != 0:
        raise RuntimeError('scalapack_diagonalize_mr3 error: %d' % info)


def scalapack_general_diagonalize_dc(desca, a, b, z, w, uplo):
    """Diagonalize symmetric matrix using the divide & conquer algorithm.
    Orthogonal eigenvectors not guaranteed; no warning is provided.

    Solve the generalized eigenvalue equation::

      A_nn Z_nn = w_N B_nn Z_nn

    B_nn is assumed to be positivde definite. Eigenvectors written to Z_nn.
    Both A_nn, B_nn and Z_nn must be compatible with desca descriptor.
    Values in A_nn and B_nn will be overwritten.

    Eigenvalues are written to the global array w_N in ascending order.

    The `uplo` flag can be either 'L' or 'U', meaning that the
    matrices are taken to be upper or lower triangular respectively.
    """
    desca.checkassert(a)
    desca.checkassert(b)
    desca.checkassert(z)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    assert uplo in ['L', 'U']
    if not desca.blacsgrid.is_active():
        return
    assert desca.gshape[0] == len(w)
    info = cgpaw.scalapack_general_diagonalize_dc(a, desca.asarray(),
                                                  switch_lu[uplo], b, z, w)
    if info != 0:
        raise RuntimeError('scalapack_general_diagonalize_dc error: %d' % info)


def scalapack_general_diagonalize_ex(desca, a, b, z, w, uplo, iu=None):
    """Diagonalize symmetric matrix using the bisection and inverse
    iteration algorithm. Re-orthogonalization of eigenvectors
    is an issue for tightly clustered eigenvalue problems; it
    requires substantial memory and is not scalable. See ScaLAPACK
    pdsyevx.f routine for more information.

    Solves the eigenvalue equation::

      A_nn Z_nn = w_N B_nn Z_nn

    B_nn is assumed to be positivde definite. Eigenvectors written to Z_nn.
    Both A_nn, B_nn and Z_nn must be compatible with desca descriptor.
    Values in A_nn and B_nn will be overwritten.

    Eigenvalues are written to the global array w_N in ascending order.

    The `uplo` flag can be either 'L' or 'U', meaning that the
    matrices are taken to be upper or lower triangular respectively.

    The `iu` specifies how many eigenvectors and eigenvalues to compute.
    """
    desca.checkassert(a)
    desca.checkassert(b)
    desca.checkassert(z)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    if iu is None:  # calculate all eigenvectors and eigenvalues
        iu = desca.gshape[0]
    assert 1 < iu <= desca.gshape[0]
    # still need assert for eigenvalues
    assert uplo in ['L', 'U']
    if not desca.blacsgrid.is_active():
        return
    assert desca.gshape[0] == len(w)
    info = cgpaw.scalapack_general_diagonalize_ex(a, desca.asarray(),
                                                  switch_lu[uplo],
                                                  iu, b, z, w)
    if info != 0:
        # 0 means you are OK
        raise RuntimeError('scalapack_general_diagonalize_ex error: %d' % info)


def scalapack_general_diagonalize_mr3(desca, a, b, z, w, uplo, iu=None):
    """Diagonalize symmetric matrix using the MRRR algorithm.

    Solve the generalized eigenvalue equation::

      A_nn Z_nn = w_N B_nn Z_nn

    B_nn is assumed to be positivde definite. Eigenvectors written to Z_nn.
    Both A_nn, B_nn and Z_nn must be compatible with desca descriptor.
    Values in A_nn and B_nn will be overwritten.

    Eigenvalues are written to the global array w_N in ascending order.

    The `uplo` flag can be either 'L' or 'U', meaning that the
    matrices are taken to be upper or lower triangular respectively.

    The `iu` specifies how many eigenvectors and eigenvalues to compute.
    """
    desca.checkassert(a)
    desca.checkassert(b)
    desca.checkassert(z)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    if iu is None:  # calculate all eigenvectors and eigenvalues
        iu = desca.gshape[0]
    assert 1 < iu <= desca.gshape[0]
    # still need assert for eigenvalues
    assert uplo in ['L', 'U']
    if not desca.blacsgrid.is_active():
        return
    assert desca.gshape[0] == len(w)
    info = cgpaw.scalapack_general_diagonalize_mr3(a, desca.asarray(),
                                                   switch_lu[uplo],
                                                   iu, b, z, w)
    if info != 0:
        raise RuntimeError('scalapack_general_diagonalize_mr3 error: %d' %
                           info)


def have_mkl():
    return hasattr(cgpaw, 'mklscalapack_diagonalize_geev')


def mkl_scalapack_diagonalize_non_symmetric(desca, a, z, w, transpose=True):
    """ Diagonalize non symmetric matrix.

    Requires mkl scalapack to function.
    Transpose is true by default (in order to match Fortran array ordering)
    Disable this if you want more control and reduced overhead.
    """
    desca.checkassert(a)
    desca.checkassert(z)

    assert desca.gshape[0] == desca.gshape[1]
    assert all([bsize >= 6 for bsize in desca.bshape]), \
        'Block size must be >= 6'

    if not desca.blacsgrid.is_active():
        return

    if transpose:
        a2 = desca.empty(dtype=complex)
        pblas_tran(1, a, 0, a2, desca, desca, conj=False)
    info = cgpaw.mklscalapack_diagonalize_geev(a2, z, w, desca.asarray())
    if transpose:
        z2 = desca.empty(dtype=complex)
        pblas_tran(1, z, 0, z2, desca, desca, conj=False)
        z[:] = z2

    if info != 0:
        raise RuntimeError('mkl_non_symmetric_diagonalize_geevx error: %d'
                           % info)


def scalapack_inverse_cholesky(desca, a, uplo):
    """Perform Cholesky decomposin followed by an inversion
    of the resulting triangular matrix.

    Only the upper or lower half of the matrix a will be
    modified; the other half is zeroed out.

    The `uplo` flag can be either 'L' or 'U', meaning that the
    matrices are taken to be upper or lower triangular respectively.
    """
    desca.checkassert(a)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    assert uplo in ['L', 'U']
    if not desca.blacsgrid.is_active():
        return
    info = cgpaw.scalapack_inverse_cholesky(a, desca.asarray(),
                                            switch_lu[uplo])
    if info != 0:
        raise RuntimeError('scalapack_inverse_cholesky error: %d' % info)


def scalapack_inverse(desca, a, uplo):
    """Perform a hermitian matrix inversion.

    """
    desca.checkassert(a)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    assert uplo in ['L', 'U']
    if not desca.blacsgrid.is_active():
        return
    info = cgpaw.scalapack_inverse(a, desca.asarray(), switch_lu[uplo])
    if info != 0:
        raise RuntimeError('scalapack_inverse error: %d' % info)


def scalapack_solve(desca, descb, a, b):
    """General matrix solve.

    Solve X from A*X = B. The array b will be replaced with the result.

    This function works on the transposed form. The equivalent
    non-distributed operation is numpy.linalg.solve(a.T, b.T).T.

    This function executes the following scalapack routine:
    * pzgesv if matrices are complex
    * pdgesv if matrices are real
    """
    desca.checkassert(a)
    descb.checkassert(b)
    assert desca.gshape[0] == desca.gshape[1], 'A not a square matrix'
    assert desca.bshape[0] == desca.bshape[1], 'A not having square blocks'
    assert desca.gshape[1] == descb.gshape[1], 'B shape not compatible with A'
    assert desca.bshape[1] == descb.bshape[1], 'B blocks not compatible with A'

    if not desca.blacsgrid.is_active():
        return
    info = cgpaw.scalapack_solve(a, desca.asarray(), b, descb.asarray())
    if info != 0:
        raise RuntimeError('scalapack_solve error: %d' % info)


def pblas_tran(alpha, a_MN, beta, c_NM, desca, descc, conj=True):
    """Matrix transpose.

    C <- alpha*A.H + beta*C  if conj == True
    C <- alpha*A.T + beta*C  if conj == False

    This function executes the following PBLAS routine:
    * pztranc if matrices are complex and conj == True
    * pztranu if matrices are complex and conj == False
    * pdtran  if matrices are real
    """
    desca.checkassert(a_MN)
    descc.checkassert(c_NM)
    M, N = desca.gshape
    assert N, M == descc.gshape
    cgpaw.pblas_tran(N, M, alpha, a_MN, beta, c_NM,
                     desca.asarray(), descc.asarray(),
                     conj)


def _pblas_hemm_symm(alpha, a_MM, b_MN, beta, c_MN, desca, descb, descc,
                     side, uplo, hemm):
    """Hermitian or symmetric matrix-matrix product.

    Do not call this function directly but
    use :func:`pblas_hemm` or :func:`pblas_symm` instead.

    C <- alpha*A*B + beta*C  if side == 'L'
    C <- alpha*B*A + beta*C  if side == 'R'

    Only lower or upper diagonal of a_MM is used.

    This function executes the following PBLAS routine:
    * pzhemm if matrices are complex and hemm == True
    * pzsymm if matrices are complex and hemm == False
    * pdsymm if matrices are real
    """
    # Note: if side == 'R', then a_MM matrix is actually size of a_NN
    desca.checkassert(a_MM)
    descb.checkassert(b_MN)
    descc.checkassert(c_MN)
    assert side in ['L', 'R'] and uplo in ['L', 'U']
    Ma, Ma2 = desca.gshape
    assert Ma == Ma2, 'A not square matrix'
    Mb, Nb = descb.gshape
    if side == 'L':
        assert Mb == Ma
    else:
        assert Nb == Ma
    M, N = descc.gshape
    assert M == Mb
    assert N == Nb

    if not desca.blacsgrid.is_active():
        return
    cgpaw.pblas_hemm_symm(switch_lr[side], switch_lu[uplo],
                          N, M, alpha, a_MM, b_MN, beta, c_MN,
                          desca.asarray(), descb.asarray(), descc.asarray(),
                          hemm)


def pblas_hemm(alpha, a_MM, b_MN, beta, c_MN, desca, descb, descc,
               side='L', uplo='L'):
    """Hermitian matrix-matrix product.

    C <- alpha*A*B + beta*C  if side == 'L'
    C <- alpha*B*A + beta*C  if side == 'R'

    Only lower or upper diagonal of a_MM is used.

    This function executes the following PBLAS routine:
    * pzhemm if matrices are complex
    * pdsymm if matrices are real
    """
    return _pblas_hemm_symm(alpha, a_MM, b_MN, beta, c_MN,
                            desca, descb, descc,
                            side, uplo, hemm=True)


def pblas_symm(alpha, a_MM, b_MN, beta, c_MN, desca, descb, descc,
               side='L', uplo='L'):
    """Symmetric matrix-matrix product.

    C <- alpha*A*B + beta*C  if side == 'L'
    C <- alpha*B*A + beta*C  if side == 'R'

    Only lower or upper diagonal of a_MM is used.

    This function executes the following PBLAS routine:
    * pzsymm if matrices are complex
    * pdsymm if matrices are real
    """
    return _pblas_hemm_symm(alpha, a_MM, b_MN, beta, c_MN,
                            desca, descb, descc,
                            side, uplo, hemm=False)


def pblas_gemm(alpha, a_MK, b_KN, beta, c_MN, desca, descb, descc,
               transa='N', transb='N'):
    """General matrix-matrix product.

    C <- alpha*A*B + beta*C

    This function executes the following PBLAS routine:
    * pzgemm if matrices are complex
    * pdgemm if matrices are real
    """
    desca.checkassert(a_MK)
    descb.checkassert(b_KN)
    descc.checkassert(c_MN)
    assert transa in ['N', 'T', 'C'] and transb in ['N', 'T', 'C']
    M, Ka = desca.gshape
    Kb, N = descb.gshape

    if transa in ['T', 'C']:
        M, Ka = Ka, M
    if transb in ['T', 'C']:
        Kb, N = N, Kb
    Mc, Nc = descc.gshape

    assert Ka == Kb
    assert M == Mc
    assert N == Nc

    if not desca.blacsgrid.is_active():
        return
    cgpaw.pblas_gemm(N, M, Ka, alpha, b_KN, a_MK, beta, c_MN,
                     descb.asarray(), desca.asarray(), descc.asarray(),
                     transb, transa)


def pblas_simple_gemm(desca, descb, descc, a_MK, b_KN, c_MN,
                      transa='N', transb='N'):
    alpha = 1.0
    beta = 0.0
    pblas_gemm(alpha, a_MK, b_KN, beta, c_MN, desca, descb, descc,
               transa, transb)


def pblas_simple_hemm(desca, descb, descc, a_MM, b_MN, c_MN,
                      side='L', uplo='L'):
    alpha = 1.0
    beta = 0.0
    pblas_hemm(alpha, a_MM, b_MN, beta, c_MN, desca, descb, descc, side, uplo)


def pblas_simple_symm(desca, descb, descc, a_MM, b_MN, c_MN,
                      side='L', uplo='L'):
    alpha = 1.0
    beta = 0.0
    pblas_symm(alpha, a_MM, b_MN, beta, c_MN, desca, descb, descc, side, uplo)


def pblas_gemv(alpha, a_MN, x_N, beta, y_M, desca, descx, descy,
               transa='N'):
    """General matrix-vector product.

    y <- alpha*A*x + beta*y

    This function executes the following PBLAS routine:
    * pzgemv if matrices are complex
    * pdgemv if matrices are real
    """
    desca.checkassert(a_MN)
    descx.checkassert(x_N)
    descy.checkassert(y_M)
    assert transa in ['N', 'T', 'C']
    M, N = desca.gshape
    Nx, Ox = descx.gshape
    My, Oy = descy.gshape
    assert Ox == 1
    assert Oy == 1
    if transa == 'N':
        assert Nx == N
        assert My == M
    else:
        assert Nx == M
        assert My == N

    # Switch transposition and handle complex conjugation manually
    if transa == 'C':
        a_MN = np.ascontiguousarray(a_MN.conj())
    switch_ntc = {'N': 'T', 'T': 'N', 'C': 'N'}

    if not desca.blacsgrid.is_active():
        return
    cgpaw.pblas_gemv(N, M, alpha,
                     a_MN, x_N, beta, y_M,
                     desca.asarray(),
                     descx.asarray(),
                     descy.asarray(),
                     switch_ntc[transa])


def pblas_simple_gemv(desca, descx, descy, a, x, y, transa='N'):
    alpha = 1.0
    beta = 0.0
    pblas_gemv(alpha, a, x, beta, y, desca, descx, descy, transa)


def pblas_r2k(alpha, a_NK, b_NK, beta, c_NN, desca, descb, descc,
              uplo='U'):
    if not desca.blacsgrid.is_active():
        return
    desca.checkassert(a_NK)
    descb.checkassert(b_NK)
    descc.checkassert(c_NN)
    assert descc.gshape[0] == descc.gshape[1]  # symmetric matrix
    assert desca.gshape == descb.gshape  # same shape
    assert uplo in ['L', 'U']
    N = descc.gshape[0]  # order of C
    # K must take into account implicit tranpose due to C ordering
    K = desca.gshape[1]  # number of columns of A and B
    cgpaw.pblas_r2k(N, K, alpha, a_NK, b_NK, beta, c_NN,
                    desca.asarray(),
                    descb.asarray(),
                    descc.asarray(),
                    uplo)


def pblas_simple_r2k(desca, descb, descc, a, b, c, uplo='U'):
    alpha = 1.0
    beta = 0.0
    pblas_r2k(alpha, a, b, beta, c,
              desca, descb, descc, uplo)


def pblas_rk(alpha, a_NK, beta, c_NN, desca, descc,
             uplo='U'):
    if not desca.blacsgrid.is_active():
        return
    desca.checkassert(a_NK)
    descc.checkassert(c_NN)
    assert descc.gshape[0] == descc.gshape[1]  # symmetrix matrix
    assert uplo in ['L', 'U']
    N = descc.gshape[0]  # order of C
    # K must take into account implicit tranpose due to C ordering
    K = desca.gshape[1]  # number of columns of A
    cgpaw.pblas_rk(N, K, alpha, a_NK, beta, c_NN,
                   desca.asarray(),
                   descc.asarray(),
                   uplo)


def pblas_simple_rk(desca, descc, a, c):
    alpha = 1.0
    beta = 0.0
    pblas_rk(alpha, a, beta, c,
             desca, descc)
