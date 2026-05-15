from gpaw.new.c import pw_amend_insert_realwf_gpu
from gpaw.gpu import cupy_is_fake
import pytest


def reference_pw_amend_insert_realwf_gpu(array_nQ, n, m):
    for array_Q in array_nQ:
        t = array_Q[:, :, 0]
        print(t.strides)
        t[0, -m:] = t[0, m:0:-1].conj()
        t[n:0:-1, -m:] = t[-n:, m:0:-1].conj()
        t[-n:, -m:] = t[n:0:-1, m:0:-1].conj()
        t[-n:, 0] = t[n:0:-1, 0].conj()


@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
def test_pw_amend_insert():
    import cupy as cp
    Band, Ax, Ay, Az = cp.indices((10, 6, 8, 10))
    print(Ax.shape)
    A = Ax * (1.0 + 0.0j)
    A = cp.random.rand(10, 6, 8, 10) + 1j * cp.random.rand(10, 6, 8, 10)
    B = A.copy()
    reference_pw_amend_insert_realwf_gpu(A, 2, 3)
    pw_amend_insert_realwf_gpu(B, 2, 3)

    assert cp.allclose(A, B)
