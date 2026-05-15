import pytest

import numpy as np

from gpaw import GPAW
import gpaw.mpi as mpi
from gpaw.response.df import DielectricFunction, read_response_function

from gpaw.test import findpeak


@pytest.mark.dielectricfunction
@pytest.mark.response
def test_mos2_polarizability(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['mos2_pw'], communicator=mpi.serial_comm)

    # Calculate the 2D polarizability within the RPA and ALDA
    df = DielectricFunction(calc, truncation='2D',
                            frequencies={'type': 'nonlinear', 'domega0': 0.05},
                            integrationmode='tetrahedron integration',
                            nblocks='max')
    df.get_polarizability(xc='RPA', filename='rpa_pol.csv')
    mpi.world.barrier()  # give rank 0 some time to write the files

    # Test against reference values
    refs = [  # rpa
        # w0r, w0i, wr, wi
        [1.875, 2.745, 1.949, 2.856],
        # I0r, I0i, Ir, Ii
        [10.602, 11.145, 9.886, 10.866],
    ]
    omega_w, alpha0_w, alpha_w = read_response_function('rpa_pol.csv')
    # plot_pol(omega_w, alpha0_w, alpha_w)
    w0r, I0r, w0i, I0i = identify_maxima(omega_w, alpha0_w)
    wr, Ir, wi, Ii = identify_maxima(omega_w, alpha_w)
    assert np.array([w0r, w0i, wr, wi]) == pytest.approx(
        np.array(refs[0]), abs=0.01)
    assert np.array([I0r, I0i, Ir, Ii]) == pytest.approx(
        np.array(refs[1]), abs=0.05)


def identify_maxima(omega_w, a_w):
    wr, Ir = findpeak(omega_w, a_w.real)
    wi, Ii = findpeak(omega_w, a_w.imag)
    return wr, Ir, wi, Ii


def plot_pol(omega_w, a0_w, a_w):
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.plot(omega_w, a0_w.real)
    plt.plot(omega_w, a_w.real)
    plt.subplot(1, 2, 2)
    plt.plot(omega_w, a0_w.imag)
    plt.plot(omega_w, a_w.imag)
    if mpi.world.rank == 0:
        plt.show()
