import numpy as np
from time import time

from gpaw.response.mpa_sampling import mpa_frequency_sampling
from gpaw.response.mpa_interpolation import RESolver, fit_residue

from .mpa_interpolation_scalar import mpa_RE_solver, mpa_R_fit, Xeval


def test_pole_reduction(in_tmp_dir):
    npols, npol_fit = 1, 2
    rng = np.random.default_rng(seed=42)
    Omega_p = rng.random(npols) * 0.05 + 5.5 - 0.01j
    residues_p = rng.random(npols)

    omega_w = mpa_frequency_sampling(
        npoles=npol_fit,
        wrange=[0, 0.8],
        varpi=1,
        eta0=0.1,
        eta_rest=0.1,
        parallel_lines=2,
    )

    X_w = Xeval(Omega_p, residues_p, omega_w)
    R_p, E_p, _, _ = mpa_RE_solver(npol_fit, omega_w, X_w[0, 0])
    E_pGG, R_pGG = RESolver(omega_w).solve(X_w.reshape((-1, 1, 1)))

    E_GGp = E_pGG.transpose(1, 2, 0)
    R_GGp = R_pGG.transpose(1, 2, 0)

    assert np.allclose(E_GGp[0, 0, :], E_p, rtol=1e-4, atol=1e-6)
    assert np.allclose(R_GGp[0, 0], R_p, rtol=1e-4, atol=1e-6)

    if 0:
        from matplotlib import pyplot as plt
        w_x = np.linspace(0, 1, 1000)
        X_x = Xeval(Omega_p, residues_p, w_x)
        fit_x = Xeval(E_p, R_p, w_x)
        fit_vec_x = Xeval(E_GGp[0, 0, :], R_GGp[0, 0, :], w_x)

        plt.plot(w_x, X_x[0, 0, :].real, 'r')
        plt.plot(w_x, X_x[0, 0, :].imag, 'g')

        plt.plot(w_x, fit_vec_x[0, 0, :].real, 'r--')
        plt.plot(w_x, fit_vec_x[0, 0, :].imag, 'g--')

        plt.plot(w_x, fit_x[0, 0, :].real, 'k:')
        plt.plot(w_x, fit_x[0, 0, :].imag, 'k:')
        plt.show()


def test_residue_fit_1pole(in_tmp_dir):
    npols, npr, w, x, E = (
        1,
        1,
        np.array([0.0 + 0.0j, 0.0 + 1.0j]),
        np.array([0.13034393 - 0.40439649j, 0.36642415 - 0.67018998j]),
        np.array([1.654376 - 0.25302243j]),
    )

    R = mpa_R_fit(npols, npr, w, x, E)
    R_vec = fit_residue(npr_GG=np.array([[npols]]),
                        omega_w=w,
                        X_wGG=x.reshape((-1, 1, 1)),
                        E_pGG=E.reshape((-1, 1, 1)))
    assert np.allclose(R, R_vec)


def test_residue_fit(in_tmp_dir):
    npols, npr, w, x, E = (
        2,
        2,
        np.array([0.0 + 0.0j, 0.0 + 1.0j, 2.33 + 0.0j, 2.33 + 1.0j]),
        np.array(
            [
                0.13034393 - 0.40439649j,
                0.36642415 - 0.67018998j,
                0.77096523 + 0.85578656j,
                -0.38002124 - 0.20843867j,
            ]
        ),
        np.array([1.654376 - 0.25302243j, 2.4306366 - 0.15733093j]),
    )

    R = mpa_R_fit(npols, npr, w, x, E)
    R_vec = fit_residue(npr_GG=np.array([[npols]]),
                        omega_w=w,
                        X_wGG=x.reshape((-1, 1, 1)),
                        E_pGG=E.reshape((-1, 1, 1)))
    assert np.allclose(R, R_vec[:, 0, 0])


def test_ppa(in_tmp_dir):
    nG = 120
    rng = np.random.default_rng(seed=42)
    X_wGG = (2 * (rng.random((2, nG, nG)) - 0.5) + 1j *
             (rng.random((2, nG, nG)) - 0.5) * 2)
    omega_w = np.array([0, 2j])

    start = time()
    E_pGG, R_pGG = RESolver(omega_w).solve(X_wGG)
    stop = time()
    fail = False
    for i in range(nG):
        for j in range(nG):
            R, E, MPres, PPcond_rate = mpa_RE_solver(
                npols=1, w=omega_w, x=X_wGG[:, i, j])
            assert np.allclose(E, E_pGG[0, i, j])
            if not np.allclose(R, R_pGG[0, i, j]):
                fail = True
    assert not fail
    vectorized_time = stop - start
    print('\nVectorized', vectorized_time)
    start = time()
    for i in range(nG):
        for j in range(nG):
            R, E, _, _ = mpa_RE_solver(
                npols=1, w=omega_w, x=X_wGG[:, i, j])
    stop = time()
    serial_time = stop - start
    print('Not vectorized', serial_time)
    print('speedup', serial_time / vectorized_time)


def test_mpa(in_tmp_dir):
    nG = 8
    omega_w = np.array([0, 1j, 2. + 0.01j, 2. + 1j])
    nw = len(omega_w)
    rng = np.random.default_rng(seed=42)
    X_wGG = (2 * (rng.random((nw, nG, nG)) - 0.5) + 1j *
             (rng.random((nw, nG, nG)) - 0.5) * 2)

    E_pGG, R_pGG = RESolver(omega_w).solve(X_wGG)
    E_GGp = E_pGG.transpose(1, 2, 0)
    R_GGp = R_pGG.transpose(1, 2, 0)

    for i in range(nG):
        for j in range(nG):
            R_p, E_p, _, _ = mpa_RE_solver(
                npols=nw // 2, w=omega_w, x=X_wGG[:, i, j])
            ind = np.argsort(E_p.real)
            E_p = E_p[ind]
            R_p = R_p[ind]
            assert np.allclose(E_GGp[i, j, :], E_p, rtol=1e-5, atol=1e-6)
            assert np.allclose(R_GGp[i, j, :], R_p, rtol=1e-5, atol=1e-6)
