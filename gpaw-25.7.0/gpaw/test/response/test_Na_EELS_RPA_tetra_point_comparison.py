import pytest
import numpy as np

from gpaw import GPAW
from gpaw.test import findpeak
from gpaw.bztools import find_high_symmetry_monkhorst_pack
from gpaw.response.df import DielectricFunction, read_response_function
from gpaw.mpi import world


@pytest.mark.dielectricfunction
@pytest.mark.tetrahedron
@pytest.mark.response
def test_response_Na_EELS_RPA_tetra_point_comparison(in_tmp_dir, gpw_files):
    gpwname = gpw_files['na_chain']

    # Generate grid compatible with tetrahedron integration
    kpts = find_high_symmetry_monkhorst_pack(gpwname, 6.0)

    # Calculate the wave functions on the new kpts grid
    calc = GPAW(gpwname).fixed_density(kpts=kpts, update_fermi_level=True)
    calc.write('Na', 'all')

    # Excited state calculation
    q0_c = np.array([0., 0., 0.])
    q1_c = np.array([1 / 4., 0., 0.])
    w_w = np.linspace(0, 24, 241)

    # Calculate the eels spectrum using point integration at both q-points
    df1 = DielectricFunction(calc='Na', frequencies=w_w, eta=0.2, ecut=50,
                             hilbert=False, rate=0.2)
    df1.get_eels_spectrum(xc='RPA', filename='EELS_Na-PI_q0', q_c=q0_c)
    df1.get_eels_spectrum(xc='RPA', filename='EELS_Na-PI_q1', q_c=q1_c)

    # Calculate the eels spectrum using tetrahedron integration at both q
    df2 = DielectricFunction(calc='Na', eta=0.2, ecut=50,
                             integrationmode='tetrahedron integration',
                             hilbert=True, rate=0.2)
    df2.get_eels_spectrum(xc='RPA', filename='EELS_Na-TI_q0', q_c=q0_c)
    df2.get_eels_spectrum(xc='RPA', filename='EELS_Na-TI_q1', q_c=q1_c)

    world.barrier()
    omegaP0_w, _, eelsP0_w = read_response_function('EELS_Na-PI_q0')
    omegaP1_w, _, eelsP1_w = read_response_function('EELS_Na-PI_q1')
    omegaT0_w, _, eelsT0_w = read_response_function('EELS_Na-TI_q0')
    omegaT1_w, _, eelsT1_w = read_response_function('EELS_Na-TI_q1')

    # calculate tetra & point int. peaks and intensities
    wpeakP0, IpeakP0 = findpeak(omegaP0_w, eelsP0_w)
    wpeakP1, IpeakP1 = findpeak(omegaP1_w, eelsP1_w)
    wpeakT0, IpeakT0 = findpeak(omegaT0_w, eelsT0_w)
    wpeakT1, IpeakT1 = findpeak(omegaT1_w, eelsT1_w)

    # import matplotlib.pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.plot(omegaP0_w, eelsP0_w)
    # plt.plot(omegaT0_w, eelsT0_w)
    # plt.subplot(1, 2, 2)
    # plt.plot(omegaP1_w, eelsP1_w)
    # plt.plot(omegaT1_w, eelsT1_w)
    # plt.show()

    # tetra and point integrators should produce similar results:
    # confirm this by comparing the 2 integration methods
    assert wpeakT0 == pytest.approx(wpeakP0, abs=0.08)
    assert wpeakT1 == pytest.approx(wpeakP1, abs=0.12)

    # ensure the wpeak for point & tetra integration do not change
    assert wpeakP0 == pytest.approx(3.4811, abs=0.02)
    assert wpeakP1 == pytest.approx(3.8076, abs=0.02)
    assert wpeakT0 == pytest.approx(3.54, abs=0.02)
    assert wpeakT1 == pytest.approx(3.79, abs=0.13)

    # ensure the Ipeak for point & tetra integration do not change
    assert IpeakP0 == pytest.approx(8.6311, abs=1.)
    assert IpeakP1 == pytest.approx(7.7766, abs=1.)
    assert IpeakT0 == pytest.approx(8.77, abs=1.)
    assert IpeakT1 == pytest.approx(7.51, abs=1.)
