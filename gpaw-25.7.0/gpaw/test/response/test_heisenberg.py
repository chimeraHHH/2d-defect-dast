"""Test the Heisenberg model based methodology of the response code."""

import numpy as np
import pytest

from gpaw.response.heisenberg import (calculate_fm_magnon_energies,
                                      calculate_single_site_magnon_energies)


@pytest.mark.ci
def test_single_site_magnons():
    """Check the single site magnon dispersion functionality."""
    rng = get_rng()
    # ---------- Inputs ---------- #

    # Magnetic moment
    mm = 1.
    # q-point grid
    nq = 11
    q_qc = get_randomized_qpoints(nq, rng)

    # Random J_q, with J=0 at q=0
    J_q = rng.rand(q_qc.shape[0])
    J_q[list(q_qc[:, 2]).index(0.)] = 0.

    # Cosine J_qD with different spin wave stiffnesses D
    D_D = np.linspace(400., 800., 5)
    J_qD = np.outer(np.cos(q_qc[:, 2]), D_D)

    # ---------- Script ---------- #

    # Calculate magnon energies
    E_q = calculate_single_site_magnon_energies(J_q, q_qc, mm)
    E_qD = calculate_single_site_magnon_energies(J_qD, q_qc, mm)

    # Check dimensions of arrays
    assert E_q.shape == (q_qc.shape[0],)
    assert E_qD.shape == J_qD.shape

    # Check versus formulas
    assert np.allclose(E_q, -2. / mm * J_q)  # Remember: J(0) = 0
    assert np.allclose(E_qD, 2. / mm * D_D[np.newaxis, :]
                       * (1. - np.cos(q_qc[:, 2]))[:, np.newaxis])


@pytest.mark.ci
def test_single_site_magnons_consistency():
    """Check that the generalized magnon dispersion calculation is consistent
    for a single site system with the simple analytical formula valid in that
    case."""
    rng = get_rng()
    # ---------- Inputs ---------- #

    # Magnetic moment
    mm = 1.
    # q-point grid
    nq = 11
    q_qc = get_randomized_qpoints(nq, rng)

    # Random isotropic exchange constants
    nJsamples = 6  # sample some different random combinations
    J_qx = rng.rand(q_qc.shape[0], nJsamples)

    # ---------- Script ---------- #

    # Calculate assuming a single site
    E_qx = calculate_single_site_magnon_energies(J_qx, q_qc, mm)

    # Calcualte using generalized functionality
    E_qnx = calculate_fm_magnon_energies(J_qx[:, np.newaxis, np.newaxis, :],
                                         q_qc, mm * np.ones((1, nJsamples)))

    # Test self-consistency
    assert E_qnx.shape[0] == E_qx.shape[0]
    assert E_qnx.shape[1] == 1
    assert E_qnx.shape[-1] == E_qx.shape[-1]
    assert np.allclose(E_qnx[:, 0, :], E_qx, atol=1e-8)


@pytest.mark.ci
def test_fm_random_magnons():
    """Check that the functionality to calculate the magnon dispersion of a
    ferromagnetic system with multiple sites works for a randomized system with
    three sites."""
    rng = get_rng()
    # ---------- Inputs ---------- #

    # Magnetic moments
    nsites = 3
    mm_a = 5. * rng.rand(nsites)
    # q-point grid
    nq = 11
    q_qc = get_randomized_qpoints(nq, rng)

    # Random isotropic exchange constants
    J_qab = 1.j * rng.rand(q_qc.shape[0], nsites, nsites)
    J_qab += rng.rand(q_qc.shape[0], nsites, nsites)
    # Take the Hermitian part of random tensor
    J_qab = (J_qab + np.transpose(np.conjugate(J_qab), (0, 2, 1))) / 2.
    # The q=0 component should furthermore be real
    J_qab[list(q_qc[:, 2]).index(0.)].imag = 0.

    # ---------- Script ---------- #

    # Calculate magnon energies
    E_qn = calculate_fm_magnon_energies(J_qab, q_qc, mm_a)
    E_qn = np.sort(E_qn, axis=1)  # Make sure the eigenvalues are sorted

    # Calculate the magnon energies manually
    mm_inv_ab = 2. / np.sqrt(np.outer(mm_a, mm_a))
    J0_ab = np.diag(np.sum(J_qab[list(q_qc[:, 2]).index(0.)], axis=1))
    H_qab = mm_inv_ab[np.newaxis] * (J0_ab[np.newaxis] - J_qab)
    test_E_qn, _ = np.linalg.eig(H_qab)

    assert E_qn.shape == (q_qc.shape[0], nsites)
    assert np.allclose(test_E_qn.imag, 0.)
    assert np.allclose(E_qn, np.sort(test_E_qn.real, axis=1))


@pytest.mark.ci
def test_fm_vectorized_magnons():
    """Check that the functionality to calculate the magnon dispersion of a
    ferromagnetic system with multiple sites works when supplying multiple
    sets of parameters for the same two-site systems."""
    rng = get_rng()
    # ---------- Inputs ---------- #

    # Magnetic moments
    nsites = 2
    nmagmoms = 4  # Test the same J_qab, but with different site magnetizations
    mm_ax = 5. * rng.rand(nsites, nmagmoms)
    # q-point grid
    nq = 11
    q_qc = get_randomized_qpoints(nq, rng)

    # Use a fixed structure for J_qab with known eigenvalues
    cos_q = np.cos(q_qc[:, 2])
    sin_q = np.sin(q_qc[:, 2])
    J_qab = np.empty((nq, nsites, nsites), dtype=complex)
    J_qab[:, 0, 0] = cos_q
    J_qab[:, 0, 1] = 1. + 1.j * sin_q
    J_qab[:, 1, 0] = 1. - 1.j * sin_q
    J_qab[:, 1, 1] = 2. * cos_q

    # Test different energy scales for the exchange interactions
    nJscales = 6
    Jscale_y = 800. * rng.rand(nJscales)

    # Combine different magnetic moments and scale for the exchange
    J_qabxy = np.empty(J_qab.shape + (nmagmoms, nJscales,), dtype=complex)
    J_qabxy[:] = np.tensordot(J_qab, Jscale_y,
                              axes=((), ()))[..., np.newaxis, :]
    mm_axy = np.moveaxis(np.tile(mm_ax, (nJscales, 1, 1)), 0, -1)

    # ---------- Script ---------- #

    # Calculate magnon energies
    E_qnxy = calculate_fm_magnon_energies(J_qabxy, q_qc, mm_axy)
    E_qnxy = np.sort(E_qnxy, axis=1)  # Make sure the eigenvalues are sorted

    # Calculate magnon energies analytically
    H_diag1_qxy = np.sqrt(mm_axy[1][np.newaxis] / mm_axy[0][np.newaxis])\
        * (2. - cos_q[:, np.newaxis, np.newaxis])
    H_diag2_qxy = np.sqrt(mm_axy[0][np.newaxis] / mm_axy[1][np.newaxis])\
        * (3. - 2. * cos_q[:, np.newaxis, np.newaxis])
    H_diag_avg_qxy = (H_diag1_qxy + H_diag2_qxy) / 2.
    H_diag_diff_qxy = (H_diag1_qxy - H_diag2_qxy) / 2.
    pm_n = np.array([-1., 1.])
    E_test_qnxy = H_diag_avg_qxy[:, np.newaxis]\
        + pm_n[np.newaxis, :, np.newaxis, np.newaxis]\
        * np.sqrt(H_diag_diff_qxy[:, np.newaxis]**2.
                  + (1 + sin_q[:, np.newaxis, np.newaxis, np.newaxis]**2.))
    E_test_qnxy *= 2. / np.sqrt(np.prod(mm_axy, axis=0))
    E_test_qnxy *= Jscale_y

    assert np.allclose(E_qnxy, E_test_qnxy)


# ---------- Test functionality ---------- #


def get_randomized_qpoints(nq, rng):
    """Make a simple, but shuffled, q-point array."""
    q_qc = np.zeros((nq, 3), dtype=float)
    q_qc[:, 2] = np.linspace(0., np.pi, nq)
    rng.shuffle(q_qc[:, 2])

    return q_qc


def get_rng():
    """Choose a specific random seed to make the tests reproducible."""
    rng = np.random.RandomState(23)

    return rng
