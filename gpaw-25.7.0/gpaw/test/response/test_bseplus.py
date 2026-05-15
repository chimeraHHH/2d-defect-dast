import numpy as np
from gpaw.response.chi0 import Chi0Calculator, get_frequency_descriptor
import pytest
from gpaw.response.pair import get_gs_and_context
from gpaw.mpi import world
from gpaw.response.bse import BSE, BSEPlus
from gpaw.response.df import Chi0DysonEquations
from gpaw.response.coulomb_kernels import CoulombKernel


@pytest.mark.response
def test_BSEPlus(in_tmp_dir, gpw_files, monkeypatch):
    """
    This test makes a BSE plus calculation with the BSEPlus class and
    manually to test that the BSEPlus code is working. It tests that the
    assertion work.
    """
    monkeypatch.chdir(in_tmp_dir)
    calc = gpw_files['si_pw_nbands10_converged']
    gs, context = get_gs_and_context(
        calc, txt=None, world=world, timer=None)
    ecut = 20
    eshift = 0.2
    eta = 0.1
    q_c = [0.0, 0.0, 0.0]
    bse_valence_bands = range(0, 4)
    bse_conduction_bands = range(4, 8)
    bse_nbands = 10
    rpa_nbands = 10
    bse = BSE(calc, ecut=ecut,
              q_c=q_c,
              valence_bands=bse_valence_bands,
              conduction_bands=bse_conduction_bands,
              eshift=eshift,
              mode='BSE',
              nbands=bse_nbands)

    w_w = np.array([-3, 0, 6])
    wd = get_frequency_descriptor(w_w, gs=gs)

    chi0calc_small = Chi0Calculator(gs, context,
                                    wd=wd,
                                    nbands=slice(0, 8),
                                    intraband=False,
                                    hilbert=False,
                                    eta=eta,
                                    ecut=ecut,
                                    eshift=eshift)

    chi0calc_large = Chi0Calculator(gs, context,
                                    wd=wd,
                                    nbands=rpa_nbands,
                                    intraband=False,
                                    hilbert=False,
                                    eta=eta,
                                    ecut=ecut,
                                    eshift=eshift)

    bseplus = BSEPlus(bse_gpw=calc,
                      bse_valence_bands=bse_valence_bands,
                      bse_conduction_bands=bse_conduction_bands,
                      bse_nbands=bse_nbands,
                      rpa_gpw=calc,
                      rpa_nbands=rpa_nbands,
                      w_w=w_w,
                      eshift=eshift,
                      eta=eta,
                      q_c=q_c,
                      ecut=ecut)

    bseplus.calculate_chi_wGG(optical=True,
                              bsep_name='chi_BSEPlus_3bands',
                              save_chi_BSE='chi_BSE',
                              save_chi_RPA='chi_RPA')
    if world.rank == 0:
        chi_BSEPlus_WGG = np.load("chi_BSEPlus_3bands.npy")
        chi_BSE_WGG = np.load("chi_BSE.npy")
        chi_RPA_WGG = np.load("chi_RPA.npy")

    coulomb_kernel = CoulombKernel.from_gs(gs, truncation=None)

    chi_irr_BSE_WGG = bse.get_chi_wGG(eta=eta,
                                      optical=True,
                                      irreducible=True,
                                      w_w=w_w)

    chi_BSE_WGG_from_bse = bse.get_chi_wGG(eta=eta,
                                           optical=True,
                                           irreducible=False,
                                           w_w=w_w)

    chi0_data_small = chi0calc_small.calculate(q_c)
    dyson_eqs_small = Chi0DysonEquations(chi0_data_small, coulomb_kernel,
                                         xc_kernel=None, cd=gs.cd)
    chi0_small_wGG = dyson_eqs_small.get_chi0_wGG(direction='x')
    chi0_small_WGG = dyson_eqs_small.wblocks.all_gather(chi0_small_wGG)

    chi0_data_large = chi0calc_large.calculate(q_c)
    dyson_eqs_large = Chi0DysonEquations(chi0_data_large, coulomb_kernel,
                                         xc_kernel=None, cd=gs.cd)
    chi0_large_wGG = dyson_eqs_large.get_chi0_wGG(direction='x')
    chi0_large_WGG = dyson_eqs_large.wblocks.all_gather(chi0_large_wGG)

    v_G = coulomb_kernel.V(chi0_data_large.qpd)

    bare_df = dyson_eqs_large.bare_dielectric_function(direction='x')
    chi_RPA_wGG_from_df = bare_df.vchibar_symm_wGG
    chi_RPA_WGG_from_df = dyson_eqs_large.wblocks.all_gather(
        chi_RPA_wGG_from_df)
    if world.rank == 0:
        sqrtV_G = v_G**0.5
        chi_RPA_WGG_from_df /= sqrtV_G * sqrtV_G[:, np.newaxis]

        assert chi_BSE_WGG_from_bse == pytest.approx(chi_BSE_WGG,
                                                     rel=8e-2, abs=8e-2)

        assert chi_RPA_WGG_from_df == pytest.approx(chi_RPA_WGG,
                                                    rel=1e-3, abs=1e-4)

    v_G[0] = 0.0

    if world.rank == 0:
        chi_irr_BSEPlus_WGG = \
            chi_irr_BSE_WGG - chi0_small_WGG + chi0_large_WGG

        eye = np.eye(len(chi_irr_BSEPlus_WGG[1]))

        chi_BSEPlus_WGG_manual = \
            np.linalg.solve(eye - chi_irr_BSEPlus_WGG @ np.diag(v_G),
                            chi_irr_BSEPlus_WGG)

        assert chi_BSEPlus_WGG_manual == pytest.approx(chi_BSEPlus_WGG,
                                                       rel=1e-3, abs=1e-4)

        ref_BSEPlus = [(-0.033315642421628745 - 0.025836336756360122j),
                       (-1.9089679632375633e-06 - 0.014016992849787593j),
                       (0.00017364808582158713 - 0.0008411354658348707j)]

        ref_BSE = [(-0.033585336180110614 - 0.02637670921698286j),
                   (-6.417710913846958e-08 - 0.01685348600618172j),
                   (3.521402023928122e-05 - 0.00039258798618966857j)]

        ref_RPA = [(-0.036080842971265584 - 0.06601130969506512j),
                   (-1.9785235796656932e-06 - 0.012059352300284204j),
                   (0.00023134668887131937 - 0.00041661801604023766j)]

        for i in range(3):
            assert np.allclose(chi_BSEPlus_WGG[i, i, i + 1],
                               ref_BSEPlus[i], rtol=1e-2, atol=1e-3)
            assert np.allclose(chi_BSE_WGG[i, i, i + 1],
                               ref_BSE[i], rtol=1e-2, atol=1e-3)
            assert np.allclose(chi_RPA_WGG[i, i, i + 1],
                               ref_RPA[i], rtol=1e-2, atol=1e-3)

    # assertion error if more bands in the bse calculation
    with pytest.raises(AssertionError, match=r'Large chi0 calculation*'):
        rpa_nbands = 5
        BSEPlus(bse_gpw=calc,
                bse_valence_bands=bse_valence_bands,
                bse_conduction_bands=bse_conduction_bands,
                bse_nbands=bse_nbands,
                rpa_gpw=calc,
                rpa_nbands=rpa_nbands,
                w_w=w_w,
                eshift=eshift,
                eta=eta,
                q_c=q_c,
                ecut=ecut)

    # assertion error if truncation is not none or 2d
    with pytest.raises(AssertionError):
        rpa_nbands = 8
        BSEPlus(bse_gpw=calc,
                bse_valence_bands=bse_valence_bands,
                bse_conduction_bands=bse_conduction_bands,
                bse_nbands=bse_nbands,
                rpa_gpw=calc,
                rpa_nbands=rpa_nbands,
                w_w=w_w,
                truncation='3D',
                eshift=eshift,
                eta=eta,
                q_c=q_c,
                ecut=ecut)

    # assertion error if truncation is 2d but system has pbc_c > 2.
    with pytest.raises(AssertionError):
        bseplus = BSEPlus(bse_gpw=calc,
                          bse_valence_bands=bse_valence_bands,
                          bse_conduction_bands=bse_conduction_bands,
                          bse_nbands=bse_nbands,
                          rpa_gpw=calc,
                          rpa_nbands=rpa_nbands,
                          w_w=w_w,
                          truncation='2D',
                          eshift=eshift,
                          eta=eta,
                          q_c=q_c,
                          ecut=ecut)

        bseplus.calculate_chi_wGG(optical=True)
