import pytest
import numpy as np
from gpaw.mpi import world
from gpaw.test import findpeak
from gpaw.response.bse import BSE
from gpaw.response.df import read_response_function


def create_bse(gpwfile, q_c=(0, 0, 0)):
    bse = BSE(gpwfile,
              q_c=q_c,
              soc_tol=0.01,
              add_soc=True,
              ecut=10,
              valence_bands=2,
              conduction_bands=2,
              eshift=0.8,
              nbands=15,
              mode='BSE',
              truncation='2D')
    return bse


@pytest.mark.response
def test_response_bse_MoS2_cut(in_tmp_dir, scalapack, gpw_files):
    gpwfile = gpw_files['mos2_5x5_pw']
    bse = create_bse(gpwfile)

    outw_w, outalpha_w = bse.get_polarizability(eta=0.02,
                                                w_w=np.linspace(0., 5., 5001))
    world.barrier()
    w_w, alphareal_w, alphaimag_w = read_response_function('pol_bse.csv')

    # Check consistency with written results
    assert np.allclose(outw_w, w_w, atol=1e-5, rtol=1e-4)
    assert np.allclose(outalpha_w.real, alphareal_w, atol=1e-5, rtol=1e-4)
    assert np.allclose(outalpha_w.imag, alphaimag_w, atol=1e-5, rtol=1e-4)

    w0, I0 = findpeak(w_w[:1100], alphaimag_w[:1100])
    w1, I1 = findpeak(w_w[1100:1300], alphaimag_w[1100:1300])
    w1 += 1.1

    assert w0 == pytest.approx(0.58, abs=0.01)
    assert I0 == pytest.approx(38.8, abs=0.35)
    assert w1 == pytest.approx(2.22, abs=0.01)
    assert I1 == pytest.approx(6.3, abs=0.35)

    #################################################################
    # Absorption and EELS spectra for 2D materials should be identical
    # for q=0.
    #################################################################

    bse = create_bse(gpwfile)
    outw_w, eels = bse.get_eels_spectrum(w_w=w_w)

    bse = create_bse(gpwfile)
    factor = bse.gs.cd.nonperiodic_hypervolume / (4 * np.pi)
    outw_w, pol = bse.get_polarizability(w_w=w_w)
    assert np.allclose(pol.imag / factor, eels)

    #####################################################################
    # Absorption and EELS spectra for 2D materials should NOT be identical
    # for finite q.
    #####################################################################
    bse = create_bse(gpwfile, q_c=[0.2, 0.2, 0.0])
    outw_w, eels = bse.get_eels_spectrum(w_w=w_w)
    bse = create_bse(gpwfile, q_c=[0.2, 0.2, 0.0])
    outw_w, pol = bse.get_polarizability(w_w)

    assert not np.allclose(pol.imag / factor, eels)
