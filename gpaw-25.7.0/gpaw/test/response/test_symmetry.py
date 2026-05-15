import numpy as np
import pytest

from gpaw.response import ResponseContext, ResponseGroundStateAdapter
from gpaw.response.symmetry import QSymmetryAnalyzer
from gpaw.test.gpwfile import response_band_cutoff


@pytest.mark.response
@pytest.mark.parametrize('identifier', list(response_band_cutoff))
def test_qsymmetries(gpw_files, identifier, gpaw_new):
    if gpaw_new and identifier == 'v2br4_pw':
        pytest.skip('New-GPAW does not support interpolate=3')
    # Set up basic response code objects
    gs = ResponseGroundStateAdapter.from_gpw_file(gpw_files[identifier])
    context = ResponseContext()
    qsymmetry = QSymmetryAnalyzer()

    # Count all symmetries:
    symmetry = gs.kd.symmetry
    ndirect = len(symmetry.op_scc)
    nindirect = ndirect * (1 - symmetry.has_inversion)
    nsymmetries = ndirect + nindirect

    # Test symmetry analysis
    rng = np.random.default_rng(42)
    if np.linalg.norm(gs.kd.ibzk_kc, axis=1).min() < 1e-10:
        # If the ground state is Γ-centered, all IBZ k-points are valid
        # q-points as well (autocommensurate) and we check that the q-point
        # symmetry analyzer reproduces the symmetries of the ground state.
        for k, k_c in enumerate(gs.kd.ibzk_kc):
            # Add a bit of numerical noise:
            q_c = k_c + (rng.random(3) - 0.5) * 1e-15
            qsymmetries, _ = qsymmetry.analyze(q_c, gs.kpoints, context)
            # The number of q -> G + q symmetries is reduced by the
            # multiplicity of the corresponding k-point
            bzk_K = np.where(gs.kd.bz2ibz_k == k)[0]
            assert nsymmetries % len(bzk_K) == 0
            assert len(qsymmetries) == nsymmetries // len(bzk_K)
    else:
        # If the ground state isn't Γ-centered, we simply check that a "noisy"
        # Γ-point q vector recovers all symmetries of the system
        q_c = (rng.random(3) - 0.5) * 1e-15  # "Noisy" Γ-point
        qsymmetries, _ = qsymmetry.analyze(q_c, gs.kpoints, context)
        assert qsymmetries.ndirect == ndirect, f'{q_c}'
        assert qsymmetries.nindirect == nindirect, f'{q_c}'
