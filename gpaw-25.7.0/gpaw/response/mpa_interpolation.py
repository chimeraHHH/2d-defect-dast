"""
This file contains several routines to do the non linear
interpolation of poles and residues of respose fuctions
corresponding to the Multipole Approximation (MPA)
developed in the Ref. [1].

The implemented solver is the one based on Pade-Thiele
formula (See App. A of Ref. [1]).

[1] DA. Leon et al, PRB 104, 115157 (2021)
"""
from __future__ import annotations
from typing import Tuple, no_type_check
from gpaw.typing import Array1D, Array2D, Array3D
import numpy as np
from numpy.linalg import eigvals


def fit_residue(
    npr_GG: Array2D, omega_w: Array1D, X_wGG: Array3D, E_pGG: Array3D
) -> Array3D:
    npols = len(E_pGG)
    nw = len(omega_w)
    A_GGwp = np.zeros((*E_pGG.shape[1:], nw, npols), dtype=np.complex128)
    b_GGw = np.zeros((*E_pGG.shape[1:], nw), dtype=np.complex128)
    for w in range(nw):
        A_GGwp[:, :, w, :] = (
            2 * E_pGG / (omega_w[w]**2 - E_pGG**2)).transpose((1, 2, 0))
        b_GGw[:, :, w] = X_wGG[w, :, :]

    for p in range(npols):
        for w in range(A_GGwp.shape[2]):
            A_GGwp[:, :, w, p][p >= npr_GG] = 0.0

    temp_GGp = np.einsum('GHwp,GHw->GHp',
                         A_GGwp.conj(), b_GGw)
    XTX_GGpp = np.einsum('GHwp,GHwo->GHpo',
                         A_GGwp.conj(), A_GGwp)

    if XTX_GGpp.shape[2] == 1:
        # 1D matrix, invert the number
        XTX_GGpp = 1 / XTX_GGpp
        R_GGp = np.einsum('GHpo,GHo->GHp',
                          XTX_GGpp, temp_GGp)
    else:
        try:
            # Note: Numpy 2.0 changed the broadcasting rules of
            # `solve()`;
            # temporarily pad the array shape with an extra dimension to
            # emulate the old behavior
            R_GGp = np.linalg.solve(
                XTX_GGpp, temp_GGp.reshape(temp_GGp.shape + (1,)))[..., 0]
        except np.linalg.LinAlgError:
            XTX_GGpp = np.linalg.pinv(XTX_GGpp)
            R_GGp = np.einsum('GHpo,GHo->GHp',
                              XTX_GGpp, temp_GGp)

    return R_GGp.transpose((2, 0, 1))


class Solver:
    """
    X(w) is approximated as a sum of poles
    Form of one pole: 2*E*R/(w**2-E**2)
    The input are two w and X(w) for each pole
    The output are E and R coefficients
    """
    def __init__(self, omega_w: Array1D, threshold=1e-5, epsilon=1e-8):
        """
        Parameters
        ----------
        omega_w : Array of complex frequencies set in mpa sampling.
                  The length corresponds to twice the number of poles
        threshold : Threshold for small and too close poles
        epsilon : Precision for positive zero imaginary part of the poles
        """
        assert len(omega_w) % 2 == 0
        self.omega_w = omega_w
        self.npoles = len(omega_w) // 2
        self.threshold = threshold
        self.epsilon = epsilon

    def solve(self, X_wGG):
        """
        X_wGG is any response function evaluated at omega_w
        it returns a tuple of poles and residues (E_pGG, R_pGG)
        where p is the pole index
        """
        raise NotImplementedError


class SinglePoleSolver(Solver):
    def __init__(self, omega_w: Array1D):
        Solver.__init__(self, omega_w=omega_w)

    def solve(self, X_wGG: Array3D) -> Tuple[Array2D, Array2D]:
        """
        This interpolates X_wGG using a single pole (E_GG, R_GG)
        """
        assert len(X_wGG) == 2

        omega_w = self.omega_w
        E_GG = ((X_wGG[0, :, :] * omega_w[0]**2 -
                 X_wGG[1, :, :] * omega_w[1]**2) /
                (X_wGG[0, :, :] - X_wGG[1, :, :])
                )  # analytical solution

        def branch_sqrt_inplace(E_GG: Array2D):
            E_GG.real = np.abs(E_GG.real)  # physical pole
            E_GG.imag = -np.abs(E_GG.imag)  # correct time ordering
            E_GG[:] = np.emath.sqrt(E_GG)

        branch_sqrt_inplace(E_GG)
        mask = E_GG < self.threshold  # null pole
        E_GG[mask] = self.threshold - 1j * self.epsilon

        R_GG = fit_residue(
            npr_GG=np.zeros_like(E_GG) + 1,
            omega_w=omega_w,
            X_wGG=X_wGG,
            E_pGG=E_GG.reshape((1, *E_GG.shape)))[0, :, :]

        return E_GG.reshape((1, *E_GG.shape)), R_GG.reshape((1, *R_GG.shape))


class MultipoleSolver(Solver):
    def __init__(self, omega_w: Array1D):
        Solver.__init__(self, omega_w=omega_w)

    def solve(self, X_wGG: Array3D) -> Tuple[Array3D, Array3D]:
        """
        This interpolates X_wGG using a sveral poles (E_pGG, R_pGG)
        """
        assert len(X_wGG) == 2 * self.npoles

        # First the poles are obtained (non linear part of the problem)
        E_GGp, npr_GG = pade_solve(X_wGG, self.omega_w**2)
        E_pGG = E_GGp.transpose((2, 0, 1))
        # The residues are obtained in a linear least square problem with
        # complex variables
        R_pGG = fit_residue(npr_GG, self.omega_w, X_wGG, E_pGG)
        return E_pGG, R_pGG


def RESolver(omega_w: Array1D):
    assert len(omega_w) % 2 == 0
    npoles = len(omega_w) / 2
    assert npoles > 0
    if npoles == 1:
        return SinglePoleSolver(omega_w)
    else:
        return MultipoleSolver(omega_w)


def mpa_cond_vectorized(
    npols: int, z_w: Array1D, E_GGp: Array3D, pole_resolution: float = 1e-5
) -> Tuple[Array3D, Array2D]:
    wmax = np.max(np.real(np.emath.sqrt(z_w))) * 1.5

    E_GGp = np.emath.sqrt(E_GGp)
    args = np.abs(E_GGp.real), np.abs(E_GGp.imag)
    E_GGp = np.maximum(*args) - 1j * np.minimum(*args)
    E_GGp.sort(axis=2)  # Sort according to real part

    for i in range(npols):
        out_poles_GG = E_GGp[:, :, i].real > wmax
        E_GGp[out_poles_GG, i] = 2 * wmax - 0.01j
        for j in range(i + 1, npols):
            diff = E_GGp[:, :, j].real - E_GGp[:, :, i].real
            equal_poles_GG = diff < pole_resolution
            if np.sum(equal_poles_GG.ravel()):
                break

            # if the poles are to close, move them to the end, set value
            # to sort to the end of array (e.g., 2x wmax)
            E_GGp[:, :, j] = np.where(
                equal_poles_GG,
                (E_GGp[:, :, j].real + E_GGp[:, :, i].real) / 2
                + 1j * np.maximum(E_GGp[:, :, i].imag, E_GGp[:, :, j].imag),
                E_GGp[:, :, j],
            )
            E_GGp[equal_poles_GG, i] = 2 * wmax - 0.01j

    E_GGp.sort(axis=2)  # Sort according to real part

    npr_GG = np.sum(E_GGp.real < wmax, axis=2)
    return E_GGp, npr_GG


@no_type_check
def pade_solve(X_wGG: Array3D, z_w: Array1D) -> Tuple[Array3D, Array2D]:
    nw, nG1, nG2 = X_wGG.shape
    npols = nw // 2
    nm = npols + 1
    b_GGm = np.zeros((nG1, nG2, nm), dtype=np.complex128)
    b_GGm[..., 0] = 1.0
    bm1_GGm = b_GGm
    c_GGw = X_wGG.transpose((1, 2, 0)).copy()

    for i in range(1, 2 * npols):
        cm1_GGw = np.copy(c_GGw)
        bm2_GGm = np.copy(bm1_GGm)
        bm1_GGm = np.copy(b_GGm)

        current_z = z_w[i - 1]

        c_GGw[..., i:] = (
            (cm1_GGw[..., i - 1, np.newaxis] - cm1_GGw[..., i:]) /
            ((z_w[i:] - current_z) * cm1_GGw[..., i:])
        )

        b_GGm -= current_z * c_GGw[..., i, np.newaxis] * bm2_GGm
        bm2_GGm[..., 1:] = c_GGw[..., i, np.newaxis] * bm2_GGm[..., :-1]
        b_GGm[..., 1:] += bm2_GGm[..., 1:]

    companion_GGpp = np.zeros((nG1, nG2, npols, npols),
                              dtype=np.complex128)

    # Create a poly companion matrix in vectorized form
    # Equal to following serial code
    # for i in range(nG):
    #     for j in range(nG):
    #         companion_GGpp[i, j] = poly.polycompanion(b_GGm[i, j])
    b_GGm /= b_GGm[:, :, -1][..., None]
    companion_GGpp.reshape((nG1, nG2, -1))[:, :, npols::npols + 1] = 1
    companion_GGpp[:, :, :, -1] = -b_GGm[:, :, :npols]

    E_GGp = eigvals(companion_GGpp)
    E_GGp, npr_GG = mpa_cond_vectorized(npols=npols, z_w=z_w, E_GGp=E_GGp)
    return E_GGp, npr_GG
