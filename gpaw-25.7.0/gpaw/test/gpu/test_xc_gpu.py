import pytest
from gpaw.xc import XC
from gpaw.gpu import cupy as cp, cupy_is_fake


@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.parametrize('nspins', [1, 2])
def test_gpu_pbe(nspins):
    from gpaw.cgpaw import evaluate_pbe_gpu
    ng = 10000
    n_sg = cp.exp(cp.log(10) * 5 * (cp.random.rand(nspins, ng) - 0.5))
    sigma_xg = cp.exp(cp.log(10) * 5 * (
                      cp.random.rand(2 * nspins - 1, ng) - 0.5))
    dedsigma_xg = cp.zeros_like(sigma_xg)
    cp.cuda.runtime.deviceSynchronize()
    v_sg = cp.zeros_like(n_sg)
    e_g = cp.zeros((ng,))

    cpun_sg = cp.asnumpy(n_sg)
    cpuv_sg = cp.asnumpy(v_sg)
    cpue_g = cp.asnumpy(e_g)
    cpusigma_xg = cp.asnumpy(sigma_xg)
    cpudedsigma_xg = cp.asnumpy(dedsigma_xg)
    xc = XC('PBE')
    import time
    start = time.time()
    xc.kernel.calculate(cpue_g, cpun_sg, cpuv_sg, cpusigma_xg, cpudedsigma_xg)
    stop = time.time()
    cpu = stop - start
    print('CPU took', stop - start)

    cp.cuda.runtime.deviceSynchronize()
    start = time.time()
    evaluate_pbe_gpu(n_sg, v_sg, e_g, sigma_xg, dedsigma_xg)
    cp.cuda.runtime.deviceSynchronize()
    stop = time.time()
    print('GPU took', stop - start)
    gpu = stop - start
    print('speedup', cpu / gpu)
    print('max error v_g', cp.max(cp.abs(v_sg - cp.asarray(cpuv_sg)).ravel()))
    assert cp.allclose(v_sg, cpuv_sg, atol=1e-7, rtol=1e-13)
    print('max error dedsigmax_g',
          cp.max(cp.abs(dedsigma_xg - cp.asarray(cpudedsigma_xg)).ravel()))
    assert cp.allclose(dedsigma_xg, cpudedsigma_xg, atol=1e-7, rtol=1e-13)
    print('max error e_g', cp.max(cp.abs(e_g - cp.asarray(cpue_g)).ravel()))
    assert cp.allclose(e_g, cpue_g, atol=1e-12, rtol=1e-12)


@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.parametrize('nspins', [1, 2])
def test_gpu_lda(nspins):
    from gpaw.cgpaw import evaluate_lda_gpu
    ng = 10000
    n_sg = cp.exp(cp.log(10) * 5 * (cp.random.rand(nspins, ng) - 0.5))
    cp.cuda.runtime.deviceSynchronize()
    v_sg = cp.zeros_like(n_sg)
    e_g = cp.zeros((ng,))

    cpun_sg = cp.asnumpy(n_sg)
    cpuv_sg = cp.asnumpy(v_sg)
    cpue_g = cp.asnumpy(e_g)
    xc = XC('LDA')
    import time
    start = time.time()
    xc.calculate_impl(None, cpun_sg, cpuv_sg, cpue_g)
    stop = time.time()
    cpu = stop - start
    print('CPU took', stop - start)

    cp.cuda.runtime.deviceSynchronize()
    start = time.time()
    evaluate_lda_gpu(n_sg, v_sg, e_g)
    cp.cuda.runtime.deviceSynchronize()
    stop = time.time()
    print('GPU took', stop - start)
    gpu = stop - start
    print('speedup', cpu / gpu)

    assert cp.allclose(v_sg, cpuv_sg, atol=1e-13, rtol=1e-13)
    assert cp.allclose(e_g, cpue_g, atol=1e-14, rtol=1e-14)
