import pytest
from gpaw.fftw import rfftn_patch, irfftn_patch
from gpaw.gpu import is_hip, cupy_is_fake


@pytest.mark.gpu
def test_hip_patch_on_cuda():
    if is_hip:
        pytest.skip('Test runs only on CUDA (have hip)')
    if cupy_is_fake:
        pytest.skip('Run runs only on CUDA (have fake cupy)')

    import cupy
    import cupyx

    def matrices():
        for sx in [1, 5, 6, 32, 101, 102]:
            for sy in [1, 5, 6, 32, 101, 102]:
                for sz in [5, 6, 32, 101, 102]:
                    yield cupy.random.rand(sx, sy, sz)
    for tmp_R in matrices():
        print(tmp_R.shape)
        tmp2_G = cupyx.scipy.fft.rfftn(tmp_R)
        tmp_G = rfftn_patch(tmp_R)
        assert cupy.allclose(tmp_G, tmp2_G, rtol=1e-10, atol=1e-10)
        back_R = irfftn_patch(tmp_G, tmp_R.shape)
        assert cupy.allclose(tmp_R, back_R, rtol=1e-10, atol=1e-10)
        back2_R = cupyx.scipy.fft.irfftn(tmp_G, tmp_R.shape)
        assert cupy.allclose(back_R, back2_R, rtol=1e-10, atol=1e-10)
