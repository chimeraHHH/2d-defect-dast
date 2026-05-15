import numpy as np
import pytest

import gpaw.mpi as mpi
from gpaw import GPAW
from gpaw.xas import XAS


@pytest.mark.old_gpaw_only
def test_corehole_h2o(in_tmp_dir, add_cwd_to_setup_paths, gpw_files):
    if mpi.size != 1:  # 1 core only for now
        return

    # Generate setup for oxygen with half a core-hole:
    calc = GPAW(gpw_files['h2o_xas'])

    xas = XAS(calc)
    x, y = xas.get_spectra()
    e1_kn = xas.eps_kn
    de1 = e1_kn[0, 1] - e1_kn[0, 0]

    if mpi.size == 1:
        # calc = GPAW('h2o-xas.gpw')
        # poissonsolver=FDPoissonSolver(use_charge_center=True))
        # calc.initialize()
        xas = XAS(calc)
        x, y = xas.get_spectra()
        e2_kn = xas.eps_kn
        w_n = np.sum(xas.sigma_cmkn[:, 0, 0, :].real**2, axis=0)
        de2 = e2_kn[0, 1] - e2_kn[0, 0]

    assert de2 == pytest.approx(2.0733, abs=0.005)
    assert w_n[1] / w_n[0] == pytest.approx(2.22, abs=0.01)

    assert de1 == de2

    xnl, ynl = xas.get_spectra(linbroad=[0.5, 540, 550])
    assert len(xnl) == len(x)

    if 0:
        import matplotlib.pyplot as plt
        plt.plot(x, y[0])
        plt.plot(x, sum(y))
        plt.show()
