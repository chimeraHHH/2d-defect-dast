import pytest
from gpaw.mpi import world
import numpy as np
from gpaw.response.g0w0 import G0W0
import pickle


@pytest.mark.response
def test_do_GW_too(in_tmp_dir, gpw_files, scalapack):
    gw0 = G0W0(gpw_files['c_pw'], 'gw0',
               bands=(3, 5),
               nblocks=1,
               ecut_extrapolation=True,
               ecut=40)
    results0 = gw0.calculate()
    gw = G0W0(gpw_files['c_pw'], 'gwtoo',
              bands=(3, 5),
              nblocks=1,
              xc='rALDA',
              ecut_extrapolation=True,
              ecut=40,
              fxc_mode='GWP',
              do_GW_too=True)

    gw.calculate()

    world.barrier()

    files = gw.savepckl()

    with open(files['GW'], 'rb') as handle:
        results_GW = pickle.load(handle)
    np.testing.assert_allclose(results0['qp'], results_GW['qp'], rtol=1e-03)
