from __future__ import annotations
import numpy as np
from numpy.linalg import eigvals
from typing import List, Tuple
from gpaw.typing import Array1D, Array2D


null_pole_thr = 1e-5
pole_resolution = 1e-5
epsilon = 1e-8

#  -------------------------------------------------------------
#  Old reference code
#  -------------------------------------------------------------

# 1 pole


def Xeval(Omega_GGp, residues_GGp, omega_w):
    X_GGpw = (
        residues_GGp[..., :, np.newaxis] * 2 * Omega_GGp[..., :, np.newaxis] /
        (omega_w[None, None, None, :]**2 - Omega_GGp[..., :, np.newaxis]**2)
    )

    return np.sum(X_GGpw, axis=2)


def mpa_cond1(z: tuple[complex, complex] | Array1D,
              E2: tuple[complex] | Array1D) -> \
        tuple[[complex], [float]] | Array2D:
    PPcond_rate = 0
    if abs(E2) < null_pole_thr:  # need to check also NAN(abs(E))
        PPcond_rate = 1
    elif np.real(E2) > 0.:
        pass
    else:
        PPcond_rate = 1

    # DALV: since MPA uses complex poles we need to guarantee the time ordering
    E2 = complex(abs(E2.real), -abs(E2.imag))
    E = np.emath.sqrt(E2)

    return E, PPcond_rate


def mpa_E_1p_solver(z, x):
    E2 = (x[0] * z[0]**2 - x[1] * z[1]**2) / (x[0] - x[1])
    E, PPcond_rate = mpa_cond1(z, E2)
    return E, PPcond_rate


def pole_is_out(i, wmax, thr, E):
    is_out = False

    if np.real(E[i]) > wmax:
        is_out = True

    j = 0
    while j < i and not is_out:
        if abs(np.real(E[i]) - np.real(E[j])) < thr:
            is_out = True
            if abs(np.real(E[j])) > max(abs(np.imag(E[j])),
               abs(np.imag(E[i]))):
                E[j] = np.mean([np.real(E[j]), np.real(E[i])]) - 1j * \
                    max(abs(np.imag(E[j])), abs(np.imag(E[i])))
            else:
                E[j] = np.mean([np.real(E[j]), np.real(E[i])]) - 1j * \
                    min(abs(np.imag(E[j])), abs(np.imag(E[i])))
        j = j + 1

    return is_out


def mpa_cond(npols: int, z: List[complex], E) ->\
        Tuple[int, List[bool], List[complex]]:
    PPcond = np.full(npols, False)
    npr = npols
    wmax = np.max(np.real(np.emath.sqrt(z))) * 1.5
    Eaux = np.emath.sqrt(E)

    i = 0
    while i < npr:
        Eaux[i] = max(abs(np.real(Eaux[i])), abs(np.imag(Eaux[i]))) - 1j * \
            min(abs(np.real(Eaux[i])), abs(np.imag(Eaux[i])))
        is_out = pole_is_out(i, wmax, pole_resolution, Eaux)

        if is_out:
            Eaux[i] = np.emath.sqrt(E[npr - 1])
            Eaux[i] = max(abs(np.real(Eaux[i])), abs(np.imag(Eaux[i]))) - 1j \
                * min(abs(np.real(Eaux[i])), abs(np.imag(Eaux[i])))
            PPcond[npr - 1] = True
            npr = npr - 1
        else:
            i = i + 1

    E[:npr] = Eaux[:npr]
    if npr < npols:
        E[npr:npols] = 2 * wmax - 0.01j
        PPcond[npr:npols] = True

    return E, npr, PPcond


def mpa_R_1p_fit(npols, npr, w, x, E):
    # Transforming the problem into a 2* larger least square with real numbers:
    A = np.zeros((4, 2), dtype='complex64')
    b = np.zeros((4), dtype='complex64')
    for k in range(2):
        b[2 * k] = np.real(x[k])
        b[2 * k + 1] = np.imag(x[k])
        A[2 * k][0] = 2. * np.real(E / (w[k]**2 - E**2))
        A[2 * k][1] = -2. * np.imag(E / (w[k]**2 - E**2))
        A[2 * k + 1][0] = 2. * np.imag(E / (w[k]**2 - E**2))
        A[2 * k + 1][1] = 2. * np.real(E / (w[k]**2 - E**2))

    Rri = np.linalg.lstsq(A, b, rcond=None)[0]
    R = Rri[0] + 1j * Rri[1]
    return R


def mpa_R_fit(npols, npr, w, x, E):
    # Transforming the problem into a 2* larger least square with real numbers:
    A = np.zeros((2 * npols * 2, npr * 2), dtype='complex64')
    b = np.zeros((2 * npols * 2), dtype='complex64')
    for k in range(2 * npols):
        b[2 * k] = np.real(x[k])
        b[2 * k + 1] = np.imag(x[k])
        for i in range(npr):
            A[2 * k][2 * i] = 2. * np.real(E[i] / (w[k]**2 - E[i]**2))
            A[2 * k][2 * i + 1] = -2. * np.imag(E[i] / (w[k]**2 - E[i]**2))
            A[2 * k + 1][2 * i] = 2. * np.imag(E[i] / (w[k]**2 - E[i]**2))
            A[2 * k + 1][2 * i + 1] = 2. * np.real(E[i] / (w[k]**2 - E[i]**2))

    Rri = np.linalg.lstsq(A, b, rcond=None)[0]
    R = np.zeros(npols, dtype='complex64')
    R[:npr] = Rri[::2] + 1j * Rri[1::2]
    return R


def mpa_E_solver_Pade(npols, z, x):
    b_m1 = b = np.zeros(npols + 1, dtype='complex64')
    b_m1[0] = b[0] = 1
    c = np.copy(x)

    for i in range(1, 2 * npols):

        c_m1 = np.copy(c)
        c[i:] = (c_m1[i - 1] - c_m1[i:]) / ((z[i:] - z[i - 1]) * c_m1[i:])

        b_m2 = np.copy(b_m1)
        b_m1 = np.copy(b)

        b = b_m1 - z[i - 1] * c[i] * b_m2
        b_m2[npols:0:-1] = c[i] * b_m2[npols - 1::-1]
        b[1:] = b[1:] + b_m2[1:]

    Companion = np.polynomial.polynomial.polycompanion(b[:npols + 1])
    # DALV: /b[npols] it is carried inside

    E = eigvals(Companion)

    # DALV: here we need to force real(E) to be positive.
    # This is because of the way the residue integral is performed, later.
    E, npr, PPcond = mpa_cond(npols, z, E)

    return E, npr, PPcond


def mpa_RE_solver(npols, w, x):
    if npols == 1:  # DALV: we could particularize the solution for 2-3 poles
        E, PPcond_rate = mpa_E_1p_solver(w, x)
        R = mpa_R_1p_fit(1, 1, w, x, E)
        # DALV: if PPcond_rate=0, R = x[1]*(z[1]**2-E**2)/(2*E)
        MPred = 0
    else:
        # DALV: Pade-Thiele solver (mpa_sol='PT')
        E, npr, PPcond = mpa_E_solver_Pade(npols, w**2, x)
        R = mpa_R_fit(npols, npr, w, x, E)

        PPcond_rate = 1
        MPred = 1

    # for later: MP_err = err_func_X(np, R, E, w, x)

    return R, E, MPred, PPcond_rate  # , MP_err
