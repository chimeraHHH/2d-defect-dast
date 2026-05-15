"""Pure Python implementation of the _gpaw C-extension module.

Used if GPAW_NO_C_EXTENSION=1.  See also the gpaw.cgpaw module.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from gpaw.gpu import cupy as cp, cupy_is_fake
from gpaw.typing import Array1D, ArrayND

have_openmp = False


def gpaw_gpu_init():
    pass


def get_num_threads():
    return 1


class Spline:
    def __init__(self, l, rmax, f_g):
        self.spline = CubicSpline(np.linspace(0, rmax, len(f_g)), f_g,
                                  bc_type='clamped')
        self.l = l
        self.rmax = rmax

    def __call__(self, r):
        return self.spline(r)

    def get_angular_momentum_number(self):
        return self.l

    def get_cutoff(self):
        return self.rmax

    def map(self, r_g, out_g):
        out_g[:] = self.spline(r_g)
        out_g[r_g >= self.rmax] = 0.0


def hartree(l: int,
            nrdr: np.ndarray,
            r: np.ndarray,
            vr: np.ndarray) -> None:
    p = 0.0
    q = 0.0
    for g in range(len(r) - 1, 0, -1):
        R = r[g]
        rl = R**l
        dp = nrdr[g] / rl
        rlp1 = rl * R
        dq = nrdr[g] * rlp1
        vr[g] = (p + 0.5 * dp) * rlp1 - (q + 0.5 * dq) / rl
        p += dp
        q += dq
    vr[0] = 0.0
    f = 4.0 * np.pi / (2 * l + 1)
    vr[1:] += q / r[1:]**l
    vr[1:] *= f


def unpack(M, M2):
    n = len(M2)
    p = 0
    for r in range(n):
        for c in range(r, n):
            d = M[p]
            M2[r, c] = d
            M2[c, r] = d
            p += 1


def pack(M2):
    n = len(M2)
    M = np.empty(n * (n + 1) // 2)
    p = 0
    for r in range(n):
        M[p] = M2[r, r]
        p += 1
        for c in range(r + 1, n):
            M[p] = M2[r, c] + M2[c, r]
            p += 1
    return M


def add_to_density(f: float,
                   psit_X: ArrayND,
                   nt_X: ArrayND) -> None:
    nt_X += f * abs(psit_X)**2


def pw_precond(G2_G: Array1D,
               r_G: Array1D,
               ekin: float,
               o_G: Array1D) -> None:
    x = 1 / ekin / 3 * G2_G
    a = 27.0 + x * (18.0 + x * (12.0 + x * 8.0))
    xx = x * x
    o_G[:] = -4.0 / 3 / ekin * a / (a + 16.0 * xx * xx) * r_G


def pw_insert(coef_G: Array1D,
              Q_G: Array1D,
              x: float,
              array_Q: Array1D) -> None:
    array_Q[:] = 0.0
    array_Q.ravel()[Q_G] = x * coef_G


def pw_insert_gpu(psit_nG,
                  Q_G,
                  scale,
                  psit_bQ,
                  nx, ny, nz):
    assert scale == 1.0
    psit_bQ[..., Q_G] = psit_nG
    if nx * ny * nz != psit_bQ.shape[-1]:
        n, m = nx // 2 - 1, ny // 2 - 1
        pw_amend_insert_realwf_gpu(psit_bQ.reshape((-1, nx, ny, nz // 2 + 1)),
                                   n, m)


def pwlfc_expand(f_Gs, Gk_Gv, pos_av, eikR_a,
                 Y_GL, l_s, a_J, s_J,
                 cc, f_GI, xp=np):
    emiGR_Ga = Gk_Gv @ pos_av.T
    emiGR_Ga = \
        (xp.cos(emiGR_Ga) - 1j * xp.sin(emiGR_Ga)) * eikR_a
    real = np.issubdtype(f_GI.dtype, np.floating)
    I1 = 0
    for J, (a, s) in enumerate(zip(a_J, s_J)):
        l = l_s[s]
        I2 = I1 + 2 * l + 1
        f_Gi = (f_Gs[:, s] *
                emiGR_Ga[:, a] *
                Y_GL[:, l**2:(l + 1)**2].T *
                (-1.0j)**l).T
        if cc:
            np.conjugate(f_Gi, f_Gi)
        if real:
            f_GI[::2, I1:I2] = f_Gi.real
            f_GI[1::2, I1:I2] = f_Gi.imag
        else:
            f_GI[:, I1:I2] = f_Gi
        I1 = I2


def pwlfc_expand_gpu(f_Gs, Gk_Gv, pos_av, eikR_a,
                     Y_GL, l_s, a_J, s_J,
                     cc, f_GI, I_J):
    pwlfc_expand(f_Gs, Gk_Gv, pos_av, eikR_a,
                 Y_GL, l_s, a_J, s_J,
                 cc, f_GI, xp=cp)


def dH_aii_times_P_ani_gpu(dH_aii, ni_a,
                           P_nI, out_nI):
    I1 = 0
    J1 = 0
    for ni in ni_a.get():
        I2 = I1 + ni
        J2 = J1 + ni**2
        dH_ii = dH_aii[J1:J2].reshape((ni, ni))
        out_nI[:, I1:I2] = P_nI[:, I1:I2] @ dH_ii
        I1 = I2
        J1 = J2


def pw_amend_insert_realwf_gpu(array_nQ, n, m):
    for array_Q in array_nQ:
        t = array_Q[:, :, 0]
        t[0, -m:] = t[0, m:0:-1].conj()
        t[n:0:-1, -m:] = t[-n:, m:0:-1].conj()
        t[-n:, -m:] = t[n:0:-1, m:0:-1].conj()
        t[-n:, 0] = t[n:0:-1, 0].conj()


def calculate_residuals_gpu(residual_nG, eps_n, wfs_nG):
    for residual_G, eps, wfs_G in zip(residual_nG, eps_n, wfs_nG):
        residual_G -= eps * wfs_G


def add_to_density_gpu(weight_n, psit_nR, nt_R):
    for weight, psit_R in zip(weight_n, psit_nR):
        nt_R += float(weight) * cp.abs(psit_R)**2


def symmetrize_ft(a_R, b_R, r_cc, t_c, offset_c):
    if (r_cc == np.eye(3, dtype=int)).all() and not t_c.any():
        b_R[:] = a_R
        return
    raise NotImplementedError


def evaluate_lda_gpu(nt_sr, vxct_sr, e_r) -> None:
    if cupy_is_fake:
        from gpaw.xc.kernel import XCKernel
        XCKernel('LDA').calculate(e_r._data, nt_sr._data, vxct_sr._data)
    else:
        from _gpaw import evaluate_lda_gpu as evalf  # type: ignore
        evalf(nt_sr, vxct_sr, e_r)


def evaluate_pbe_gpu(nt_sr, vxct_sr, e_r, sigma_xr, dedsigma_xr) -> None:
    if cupy_is_fake:
        from gpaw.xc.kernel import XCKernel
        XCKernel('PBE').calculate(e_r._data, nt_sr._data, vxct_sr._data,
                                  sigma_xr._data, dedsigma_xr._data)
    else:
        from _gpaw import evaluate_pbe_gpu as evalf  # type: ignore
        evalf(nt_sr, vxct_sr, e_r, sigma_xr, dedsigma_xr)


def pw_norm_gpu(result_x, C_xG):
    if cupy_is_fake:
        result_x._data[:] = np.sum(np.abs(C_xG._data)**2, axis=1)
    else:
        result_x[:] = cp.sum(cp.abs(C_xG)**2, axis=1)


def pw_norm_kinetic_gpu(result_x, a_xG, kin_G):
    if cupy_is_fake:
        result_x._data[:] = np.sum(
            np.abs(a_xG._data)**2 * kin_G._data[None, :],
            axis=1)
    else:
        result_x[:] = cp.sum(cp.abs(a_xG)**2 * kin_G[None, :], axis=1)
