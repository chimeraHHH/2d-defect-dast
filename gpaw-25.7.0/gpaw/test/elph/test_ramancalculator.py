import numpy as np
import pytest

from ase.utils.filecache import MultiFileJSONCache
from gpaw import GPAW
from gpaw.lcao.dipoletransition import get_momentum_transitions
from gpaw.elph import ResonantRamanCalculator
from gpaw.mpi import world


def get_random_g(nk, nb):
    g_sqklnn = np.zeros((1, 1, 4, 3, 4, 4), dtype=complex)
    rng = np.random.default_rng()
    tmp = rng.random((4, 4)) + 1j * rng.random((4, 4))
    # make hermitian
    for i in range(4):
        for j in range(i + 1, 4):
            tmp[i, j] = tmp[j, i].conj()
    g_sqklnn[0, 0, 0, 2] = tmp
    return g_sqklnn


@pytest.mark.old_gpaw_only  # calc.initialize_positions(atoms) not implemented!
@pytest.mark.serial
def test_ramancalculator(gpw_files, in_tmp_dir):
    """Test of ResonantRamanCalculator object"""
    calc = GPAW(gpw_files['bcc_li_lcao'])
    atoms = calc.atoms
    # Initialize calculator if necessary
    if not hasattr(calc.wfs, 'C_nM'):
        calc.initialize_positions(atoms)
    # need to fiddle with some occupation numnbers as this exampe is
    # not properly converged
    for kpt in calc.wfs.kpt_u:
        kpt.f_n[0] = kpt.weight

    # prepare some required data
    wph_w = np.array([0., 0., 0.1])
    get_momentum_transitions(calc.wfs)
    if world.rank == 0:
        g_sqklnn = get_random_g(4, 4)
        np.save("gsqklnn.npy", g_sqklnn)

    rrc = ResonantRamanCalculator(calc, wph_w)
    assert rrc.mom_skvnm == pytest.approx(np.transpose(rrc.mom_skvnm,
                                                       (0, 1, 2, 4, 3)).conj())
    # Force momentum matrix elements to be the same in all directions
    # else R^ab won't be R^{ba*}
    # This is a bit of a dirty hack I guess. Ideally we need a test systm with
    # equivalent axes but no degenerate bands... so yeah
    rrc.mom_skvnm[0, :, 1] = rrc.mom_skvnm[0, :, 0]
    rrc.mom_skvnm[0, :, 2] = rrc.mom_skvnm[0, :, 0]

    # check reading of file cache
    check_cache = MultiFileJSONCache("Rlab")
    assert check_cache["phonon_frequencies"] == pytest.approx(wph_w)
    assert check_cache["frequency_grid"] is None

    rrc.calculate_raman_tensor(1.0)
    for i in range(3):
        for j in range(3):
            R_l = check_cache[f"{'xyz'[i]}{'xyz'[j]}"]
            assert R_l is not None
            assert R_l[0] == pytest.approx(0.0 + 1j * 0.)
            assert R_l[1] == pytest.approx(0.0 + 1j * 0.)

        if j > i:
            # need to make sure momentum matrix is perfectly hermitian too
            Rother_l = check_cache[f"{'xyz'[j]}{'xyz'[i]}"]
            print(i, j, R_l[2], Rother_l[2])
            assert R_l[2].real == pytest.approx(Rother_l[2].real)
            assert R_l[2].imag == pytest.approx(Rother_l[2].imag)

    # check proper kpt dependence. If we half all the weights,
    # the total intensity should be half as well
    for kpt in calc.wfs.kpt_u:
        kpt.weight /= 2
        kpt.f_n /= 2  # because f_n = kpt.f_n / kpt.weight

    for i in range(3):
        for j in range(3):
            R_l = check_cache[f"{'xyz'[i]}{'xyz'[j]}"]
            R_l_half = rrc.calculate(1.0, i, j)
            assert 2. * R_l_half[2].real == pytest.approx(R_l[2].real)
            assert 2. * R_l_half[2].imag == pytest.approx(R_l[2].imag)
