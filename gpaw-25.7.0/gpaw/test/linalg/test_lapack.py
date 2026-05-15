import numpy as np

import pytest
from gpaw.lrtddft.apmb import sqrt_matrix

# check sqrt of a matrix


def test_linalg_lapack():
    A = [[20, 4], [4, 1]]
    a = [[4.4, 0.8], [0.8, 0.6]]
    A = np.array(A, float)
    print('A=', A)
    a = np.array(a)
    b = sqrt_matrix(A)
    print('sqrt(A)=', b)
    assert ((a - b)**2).sum() == pytest.approx(0, abs=1.0e-12)
