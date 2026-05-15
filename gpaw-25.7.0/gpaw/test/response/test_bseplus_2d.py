import numpy as np
from gpaw.response.chi0 import Chi0Calculator, get_frequency_descriptor
import pytest
from gpaw.response.pair import get_gs_and_context
from gpaw.mpi import world
from gpaw.response.bse import BSE, BSEPlus
from gpaw.response.df import Chi0DysonEquations
from gpaw.response.coulomb_kernels import CoulombKernel
from ase.units import Bohr


@pytest.mark.response
def test_BSEPlus_2d(in_tmp_dir, gpw_files):
    """
    This test makes a BSEPlus calculation with the BSEPlus class and
    manually to test that the BSEPlus code is working. It tests that the
    assertions work.
    """
    gs, context = get_gs_and_context(
        gpw_files['mos2_5x5_pw'], txt=None, world=world, timer=None)
    ecut = 25
    eshift = 0.2
    eta = 0.1
    q_c = [0.0, 0.0, 0.0]
    bse_valence_bands = range(7, 9)
    bse_conduction_bands = range(9, 11)
    bse_nbands = 12
    rpa_nbands = 15

    bse = BSE(gpw_files['mos2_5x5_pw'], ecut=ecut,
              truncation='2D',
              q_c=q_c,
              valence_bands=bse_valence_bands,
              conduction_bands=bse_conduction_bands,
              eshift=eshift,
              mode='BSE',
              nbands=bse_nbands)

    w_w = np.linspace(0, 10, 5)
    wd = get_frequency_descriptor(w_w, gs=gs)

    chi0calc_small = Chi0Calculator(gs, context,
                                    wd=wd,
                                    nbands=slice(7, 11),
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

    bseplus = BSEPlus(bse_gpw=gpw_files['mos2_5x5_pw'],
                      bse_valence_bands=bse_valence_bands,
                      bse_conduction_bands=bse_conduction_bands,
                      bse_nbands=bse_nbands,
                      rpa_gpw=gpw_files['mos2_5x5_pw'],
                      rpa_nbands=rpa_nbands,
                      w_w=w_w,
                      eshift=eshift,
                      eta=eta,
                      q_c=q_c,
                      ecut=ecut,
                      truncation='2D')

    bseplus.calculate_chi_wGG(optical=True)

    if world.rank == 0:
        chi_BSEPlus_WGG = np.load("chi_BSEPlus.npy")

    coulomb_kernel = CoulombKernel.from_gs(gs, truncation='2D')

    chi_irr_BSE_wGG = bse.get_chi_wGG(eta=eta,
                                      optical=True,
                                      irreducible=True,
                                      w_w=w_w)

    if world.rank == 0:
        chi_irr_BSE_WGG = chi_irr_BSE_wGG
    else:
        chi_irr_BSE_WGG = None

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

    coulomb_kernel_bare = CoulombKernel.from_gs(gs, truncation=None)
    v_G_bare = coulomb_kernel_bare.V(chi0_data_large.qpd, q_v=None)
    v_G = v_G / v_G_bare
    v_G_bare[0] = 0.0

    if world.rank == 0:
        chi0_small_WGG = chi0_small_WGG * v_G_bare[np.newaxis, np.newaxis, :]
        chi0_large_WGG = chi0_large_WGG * v_G_bare[np.newaxis, np.newaxis, :]
        chi_irr_BSE_WGG = chi_irr_BSE_WGG * v_G_bare[np.newaxis, np.newaxis, :]

        chi_irr_BSEPlus_WGG = \
            chi_irr_BSE_WGG - chi0_small_WGG + chi0_large_WGG

        eye = np.eye(len(chi_irr_BSEPlus_WGG[1]))

        chi_BSEPlus_WGG_manual = \
            np.linalg.solve(eye - chi_irr_BSEPlus_WGG @ np.diag(v_G),
                            chi_irr_BSEPlus_WGG)
        cell_cv = gs.gd.cell_cv
        pbc_c = gs.pbc
        V = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))
        V *= Bohr

        chi_BSEPlus_WGG_manual *= V / (np.pi * 4)

        assert chi_BSEPlus_WGG_manual == pytest.approx(chi_BSEPlus_WGG,
                                                       rel=1e-3, abs=1e-4)

        ref = [(3.397365320873605e-08 - 0.0011565040523532147j),
               (0.0808651014210496 + 8.040955353014965e-05j),
               (0.08902208746183858 + 0.04489347809737628j),
               (0.04277497642659831 + 0.013775402447791992j),
               (-0.0221078136579067 + 0.00854914022562708j)]

        for i, r in enumerate(ref):
            assert np.allclose(chi_BSEPlus_WGG[i, i, i + 1], r)
