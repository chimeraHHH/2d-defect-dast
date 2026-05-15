import pytest
import numpy as np
from gpaw.response.g0w0 import G0W0
from ase.units import Hartree as Ha
from gpaw.mpi import world


@pytest.mark.response
@pytest.mark.parametrize('wigner_seitz', [True, False])
def test_mpa_WS(in_tmp_dir, gpw_files, scalapack, wigner_seitz):

    ref_result = {True: np.array([[[11.37680608, 21.56391991],
                                   [5.40811023, 16.11600678],
                                   [8.83575046, 22.42880098]]]),
                  False: np.asarray([[[11.283458, 21.601906],
                                      [5.326717, 16.066114],
                                      [8.73869, 22.457025]]])}

    mpa_dict = {'npoles': 4, 'wrange': [0 * Ha, 2 * Ha],
                'varpi': Ha,
                'eta0': 0.01 * Ha,
                'eta_rest': 0.1 * Ha,
                'alpha': 1}

    gw = G0W0(gpw_files['bn_pw'],
              bands=(3, 5),
              nblocks=min(2, world.size),
              ecut=40 + 20 * wigner_seitz,
              nbands=9,
              integrate_gamma='WS' if wigner_seitz else 'sphere',
              mpa=mpa_dict)

    results = gw.calculate()
    np.testing.assert_allclose(results['qp'], ref_result[wigner_seitz],
                               rtol=1e-03)
