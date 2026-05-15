import time

import pytest
import numpy as np
from ase.parallel import parprint

from gpaw.test import findpeak
from gpaw.response.df import DielectricFunction, read_response_function
from gpaw.mpi import size, world


@pytest.mark.dielectricfunction
@pytest.mark.response
@pytest.mark.libxc
def test_response_aluminum_EELS_ALDA(gpw_files, in_tmp_dir):
    assert size <= 4**3

    # Using bse_al fixture, since it was closest to the previous test
    calc = gpw_files['bse_al']

    t1 = time.time()

    # Excited state calculation
    q = np.array([1 / 4, 0, 0])
    w = np.linspace(0, 24, 241)

    df = DielectricFunction(calc=calc, frequencies=w, eta=0.2, ecut=50,
                            hilbert=False)
    df.get_eels_spectrum(xc='ALDA', filename='EELS_Al_ALDA.csv', q_c=q)

    t2 = time.time()

    parprint('For excited state calc, it took', (t2 - t1) / 60, 'minutes')

    world.barrier()
    omega_w, eels0_w, eels_w = read_response_function('EELS_Al_ALDA.csv')

    # New results are compared with test values
    wpeak1, Ipeak1 = findpeak(omega_w, eels0_w)
    wpeak2, Ipeak2 = findpeak(omega_w, eels_w)

    test_wpeak1 = 15.1034604723  # eV
    test_Ipeak1 = 27.3106588260
    test_wpeak2 = 14.9421103838
    test_Ipeak2 = 25.1284001349

    if abs(test_wpeak1 - wpeak1) > 0.02 or abs(test_wpeak2 - wpeak2) > 0.02:
        print(test_wpeak1 - wpeak1, test_wpeak2 - wpeak2)
        raise ValueError('Plasmon peak not correct ! ')

    if abs(test_Ipeak1 - Ipeak1) > 1 or abs(test_Ipeak2 - Ipeak2) > 1:
        print(Ipeak1 - test_Ipeak1, Ipeak2 - test_Ipeak2)
        raise ValueError('Please check spectrum strength ! ')
