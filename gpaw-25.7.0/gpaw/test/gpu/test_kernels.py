import pytest
import numpy as np

from gpaw.utilities import as_real_dtype
from gpaw.gpu import cupy as cp, cupy_is_fake

seed = 42


@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.parametrize("dtype", [np.float32, np.float64,
                                   np.complex64, np.complex128])
def test_dH_aii_times_P_ani(dtype):
    from _gpaw import dH_aii_times_P_ani_gpu as kernel_call
    from gpaw.purepython import dH_aii_times_P_ani_gpu as cupy_call
    assert cupy_call is not kernel_call

    rng = cp.random.RandomState(seed)
    aN = 3
    nN = 5
    iN = 12
    IN = aN * iN

    rdtype = as_real_dtype(dtype)
    dH_aii = rng.randn(aN * iN * iN, dtype=as_real_dtype(dtype))
    P_nI = rng.randn(nN, IN, dtype=as_real_dtype(dtype))
    if np.issubdtype(dtype, np.complexfloating):
        P_nI = P_nI + 1j * rng.randn(nN, IN, dtype=rdtype)
    out_kernel_nI = cp.zeros((nN, IN), dtype=dtype)
    out_cupy_nI = cp.zeros((nN, IN), dtype=dtype)
    ni_a = cp.ones(aN, dtype=np.int32) * iN

    kernel_call(dH_aii, ni_a, P_nI, out_kernel_nI)
    cupy_call(dH_aii, ni_a, P_nI, out_cupy_nI)
    assert out_kernel_nI.get() == \
        pytest.approx(out_cupy_nI.get(), abs=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.parametrize("dtype", [np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize("cc", [True, False])
def test_pwlfc_expand(dtype, cc):
    from _gpaw import pwlfc_expand_gpu as kernel_call
    from gpaw.purepython import pwlfc_expand_gpu as cupy_call
    assert cupy_call is not kernel_call

    rng = cp.random.RandomState(seed)
    GN = 100
    aN = 3
    LN = 25
    sN = 4
    JN = 8
    IN = 26

    rdtype = as_real_dtype(dtype)
    f_Gs = rng.randn(GN, sN, dtype=rdtype)
    Gk_Gv = rng.randn(GN, 3, dtype=rdtype)
    pos_av = rng.randn(aN, 3, dtype=rdtype)
    eikR_a = rng.randn(aN, dtype=rdtype) \
        + 1j * rng.randn(aN, dtype=rdtype)
    Y_GL = rng.randn(GN, LN, dtype=rdtype)

    gN = GN if np.issubdtype(dtype, np.complexfloating) else 2 * GN
    f_kernel_GI = cp.zeros((gN, IN), dtype=dtype)
    f_cupy_GI = cp.zeros((gN, IN), dtype=dtype)
    l_s = cp.arange(sN, dtype=np.int32)
    a_J = cp.ones(JN, dtype=np.int32)
    a_J[0] = 0
    a_J[-1] = aN - 1
    s_J = cp.ones(JN, dtype=np.int32)
    s_J[0] = 0
    s_J[-1] = sN - 1
    I_J = cp.zeros(JN, dtype=np.int32)
    I1 = 0
    for J, (a, s) in enumerate(zip(a_J, s_J)):
        l = l_s[s]
        I2 = I1 + 2 * l + 1
        I_J[J] = I1
        I1 = I2
    assert I2 == IN

    kernel_call(f_Gs, Gk_Gv, pos_av,
                eikR_a, Y_GL,
                l_s, a_J, s_J,
                cc, f_kernel_GI, I_J)
    cupy_call(f_Gs, Gk_Gv, pos_av,
              eikR_a, Y_GL,
              l_s, a_J, s_J,
              cc, f_cupy_GI, I_J)

    assert f_kernel_GI.get() == pytest.approx(f_cupy_GI.get(), abs=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pw_amend_insert_realwf(dtype):
    from _gpaw import pw_amend_insert_realwf_gpu as kernel_call
    from gpaw.purepython import pw_amend_insert_realwf_gpu as cupy_call
    assert cupy_call is not kernel_call

    rng = cp.random.RandomState(seed)
    nN = 10
    QN = (10, 15, 20)
    n = 3
    m = 7

    array_nQ = rng.randn(nN, *QN, dtype=dtype) \
        + 1j * rng.randn(nN, *QN, dtype=dtype)

    array_kernel_nQ = array_nQ.copy()
    array_cupy_nQ = array_nQ.copy()

    kernel_call(array_kernel_nQ, n, m)
    cupy_call(array_cupy_nQ, n, m)

    assert array_kernel_nQ.get() != pytest.approx(array_nQ.get())
    assert array_kernel_nQ.get() == pytest.approx(array_cupy_nQ.get())


@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.parametrize("dtype", [np.float32, np.float64,
                                   np.complex64, np.complex128])
def test_calculate_residuals(dtype):
    from _gpaw import calculate_residuals_gpu as kernel_call
    from gpaw.purepython import calculate_residuals_gpu as cupy_call
    assert cupy_call is not kernel_call

    rng = cp.random.RandomState(seed)
    nN = 10
    GN = 100

    rdtype = as_real_dtype(dtype)
    wfs_nG = rng.randn(nN, GN, dtype=rdtype)
    if np.issubdtype(dtype, np.complexfloating):
        wfs_nG = wfs_nG + 1j * rng.randn(nN, GN, dtype=rdtype)
    eps_n = rng.randn(nN, dtype=rdtype)
    residual_kernel_nG = cp.zeros((nN, GN), dtype=dtype)
    residual_cupy_nG = cp.zeros((nN, GN), dtype=dtype)

    kernel_call(residual_kernel_nG, eps_n, wfs_nG)
    cupy_call(residual_cupy_nG, eps_n, wfs_nG)

    assert residual_kernel_nG.get() == pytest.approx(residual_cupy_nG.get())


@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.parametrize("dtype", [np.float32, np.float64,
                                   np.complex64, np.complex128])
def test_add_to_density(dtype):
    from _gpaw import add_to_density_gpu as kernel_call
    from gpaw.purepython import add_to_density_gpu as cupy_call
    assert cupy_call is not kernel_call

    rng = cp.random.RandomState(seed)
    nN = 10
    RN = (10, 15, 20)

    rdtype = as_real_dtype(dtype)
    psit_nR = rng.randn(nN, *RN, dtype=rdtype)
    if np.issubdtype(dtype, np.complexfloating):
        psit_nR = psit_nR + 1j * rng.randn(nN, *RN, dtype=rdtype)

    weight_n = rng.randn(nN, dtype=np.float64)
    nt_kernel_R = cp.zeros(RN, dtype=np.float64)
    nt_cupy_R = cp.zeros(RN, dtype=np.float64)

    kernel_call(weight_n, psit_nR, nt_kernel_R)
    cupy_call(weight_n, psit_nR, nt_cupy_R)

    assert nt_kernel_R.get() == \
        pytest.approx(nt_cupy_R.get(), abs=1e-5)  # Not exact?


@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_pw_norm(dtype):
    from _gpaw import pw_norm_gpu as kernel_call
    from gpaw.purepython import pw_norm_gpu as cupy_call
    assert cupy_call is not kernel_call

    rng = cp.random.RandomState(seed)
    rdtype = as_real_dtype(dtype)

    xN = 20
    GN = 420

    C_xG = rng.randn(xN, GN, dtype=rdtype)
    if np.issubdtype(dtype, np.complexfloating):
        C_xG = C_xG + 1j * rng.randn(xN, GN, dtype=rdtype)
    result_kernel_x = cp.empty(xN, dtype=rdtype)
    result_cupy_x = cp.empty(xN, dtype=rdtype)

    kernel_call(result_kernel_x, C_xG)
    cupy_call(result_cupy_x, C_xG)

    assert result_kernel_x.get() == pytest.approx(result_cupy_x.get())


@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_pw_norm_kinetic(dtype):
    from _gpaw import pw_norm_kinetic_gpu as kernel_call
    from gpaw.purepython import pw_norm_kinetic_gpu as cupy_call
    assert cupy_call is not kernel_call

    rng = cp.random.RandomState(seed)
    rdtype = as_real_dtype(dtype)

    xN = 20
    GN = 420

    C_xG = rng.randn(xN, GN, dtype=rdtype)
    kin_G = rng.randn(GN, dtype=rdtype)
    if np.issubdtype(dtype, np.complexfloating):
        C_xG = C_xG + 1j * rng.randn(xN, GN, dtype=rdtype)
    result_kernel_x = cp.zeros(xN, dtype=rdtype)
    result_cupy_x = cp.zeros(xN, dtype=rdtype)

    kernel_call(result_kernel_x, C_xG, kin_G)
    cupy_call(result_cupy_x, C_xG, kin_G)

    assert result_kernel_x.get() == \
        pytest.approx(result_cupy_x.get(), abs=1e-4)  # Not exact?


@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_pw_insert(dtype):
    from _gpaw import pw_insert_gpu as kernel_call
    from gpaw.purepython import pw_insert_gpu as cupy_call
    assert cupy_call is not kernel_call

    rng = cp.random.RandomState(seed)
    rdtype = as_real_dtype(dtype)

    bN = 20
    nN = bN
    GN = 240
    QN = 360
    scale = 1.0
    nx = 6
    ny = 6
    nz = 10

    Q_G = cp.arange(GN, dtype=np.int32)
    Q_G[-1] = QN - 1
    psit_nG = rng.randn(nN, GN, dtype=rdtype)
    if np.issubdtype(dtype, np.complexfloating):
        psit_nG = psit_nG + 1j * rng.randn(nN, GN, dtype=rdtype)
    psit_kernel_bQ = cp.zeros((bN, QN), dtype=dtype)
    psit_cupy_bQ = cp.zeros((bN, QN), dtype=dtype)

    kernel_call(psit_nG, Q_G, scale, psit_kernel_bQ, nx, ny, nz)
    cupy_call(psit_nG, Q_G, scale, psit_cupy_bQ, nx, ny, nz)

    assert psit_kernel_bQ.get() == pytest.approx(psit_cupy_bQ.get())
