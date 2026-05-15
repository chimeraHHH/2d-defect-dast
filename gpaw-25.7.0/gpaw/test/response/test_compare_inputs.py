from gpaw.response.g0w0 import compare_inputs
import numpy as np


def test_compare_inputs():
    A = {'A': [[1.0 + 0.9e-14, 10], np.array([1.0 + 0.9e-14, 10])],
         'C': [1, 2, 3]}
    B = {'A': [[1.0, 10], np.array([1.0, 10])], 'C': [1, 2, 3]}
    assert compare_inputs(A, B)
