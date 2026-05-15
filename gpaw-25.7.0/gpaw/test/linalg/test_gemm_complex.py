# for n > 4 (?)
# when run with gpaw-python built with MKL matrices b and c have ~zero elements
# running with python (uses cgpaw.so) gives right (non-zero) results!

import numpy as np
from gpaw.utilities.blas import mmm


def test_linalg_gemm_complex():
    n = 5  # works for n = 4
    a1 = np.eye(n) + 1.j
    a2 = a1 + 1.j

    b = np.zeros((n, n), dtype=complex)

    c = np.dot(a2, a1)

    mmm(1.0, a2, 'N', a1, 'N', 0.0, b)

    thresh = 1.0e-7
    ref_max_value = -9.0

    # print b
    # print c
    numpy_dot = np.max(b).real
    gpaw_gemm = np.max(c).real
    # print gpaw_gemm
    assert abs(gpaw_gemm - numpy_dot) < thresh, (gpaw_gemm, numpy_dot, thresh)
    assert abs(gpaw_gemm - ref_max_value) < thresh, (gpaw_gemm,
                                                     ref_max_value, thresh)
