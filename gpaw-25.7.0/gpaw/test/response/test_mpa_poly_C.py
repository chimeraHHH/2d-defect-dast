import pytest
import numpy as np
from ase.units import Hartree as Ha
from gpaw.cgpaw import evaluate_mpa_poly as mpa_C


def mpa_py(omega, f, omegat_nGG, W_nGG, eta, factor):
    x1_nGG = f / (omega + omegat_nGG - 1j * eta)
    x2_nGG = (1.0 - f) / (omega - omegat_nGG + 1j * eta)

    x_GG = 2 * factor * np.sum(W_nGG * (x1_nGG + x2_nGG),
                               axis=0)

    eps = 0.0001 / Ha
    xp_nGG = f / (omega + eps + omegat_nGG - 1j * eta)
    xp_nGG += (1.0 - f) / (omega + eps - omegat_nGG + 1j * eta)
    xm_nGG = f / (omega - eps + omegat_nGG - 1j * eta)
    xm_nGG += (1.0 - f) / (omega - eps - omegat_nGG + 1j * eta)
    dx_GG = 2 * factor * np.sum(W_nGG * (xp_nGG - xm_nGG) / (2 * eps),
                                axis=0)
    return x_GG, dx_GG


@pytest.mark.parametrize('f', [0, 0.4, 1.0])
def test_residues(f):
    factor = 2.0
    eta = 0.1 * Ha
    nG = 5
    npols = 10
    omega = 0.5

    rng = np.random.default_rng(seed=1)
    omegat_nGG = rng.random((npols, nG, nG)) * 0.05 + 5.5 - 0.01j
    W_nGG = np.array(rng.random((npols, nG, nG)), dtype=complex)

    x_GG_py, dx_GG_py = mpa_py(omega, f, omegat_nGG, W_nGG, eta, factor)

    x_GG_C = np.empty(omegat_nGG.shape[1:], dtype=complex)
    dx_GG_C = np.empty(omegat_nGG.shape[1:], dtype=complex)
    mpa_C(x_GG_C, dx_GG_C, omega, f, omegat_nGG, W_nGG, eta, factor)

    assert np.allclose(x_GG_py, x_GG_C, atol=1e-6)
