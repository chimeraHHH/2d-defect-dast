from time import time

import numpy as np
import pytest

from gpaw.utilities.blas import mmm


@pytest.mark.ci
def test_linalg_gemm():
    n = 1000
    a1 = np.eye(n)
    a1 += 0.0001
    a2 = 2 * a1

    b = np.zeros((n, n))
    t0 = time()
    mmm(1.0, a1, 'N', a2, 'N', 0.0, b)
    tgpaw = time() - t0
    print('gpaw.gemm  ', tgpaw)

    t0 = time()
    np.dot(a1, a2)
    tnumpy = time() - t0
    print('numpy.dot  ', tnumpy)

    """
    SLID:
    gpaw.gemm   0.41842508316
    numpy.dot   11.26800704

    p019:
    gpaw.gemm   0.444674015045
    numpy.dot   11.8213479519

    m022:
    gpaw.gemm   0.446084022522
    numpy.dot   11.9757530689

    u091:
    gpaw.gemm   0.377645015717
    numpy.dot   0.389540910721

    CASIMIR:
    gpaw.gemm   2.19097900391
    numpy.dot   8.21617603302

    THUL:
    gpaw.gemm   0.520259857178
    numpy.dot   0.505489110947
    """
