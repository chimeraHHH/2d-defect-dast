import numpy as np
from gpaw.core import PWDesc
from gpaw.gpu import cupy as cp, T
from gpaw.new.pw.hamiltonian import precondition
import pytest


@pytest.mark.gpu
@pytest.mark.parametrize('xp', [np, cp])
def test_prec(xp):
    a = 2
    pw = PWDesc(cell=[a, a, a], ecut=200 / 27, dtype=complex)
    n = 2
    psit_nG, residual_nG, out_nG = pw.zeros((3, n), xp=xp)
    psit_nG.data[:, :2] = 1.0
    residual_nG.data[:] = 1.0
    print(residual_nG)
    for _ in range(1):
        with T():
            precondition(psit_nG, residual_nG, out_nG)
