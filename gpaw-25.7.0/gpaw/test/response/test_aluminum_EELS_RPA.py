import time
import pytest
import numpy as np

from ase.build import bulk
from ase.parallel import parprint

from gpaw import GPAW, PW
from gpaw.test import findpeak
from gpaw.bztools import find_high_symmetry_monkhorst_pack
from gpaw.response.df import DielectricFunction, read_response_function
from gpaw.mpi import size, world


# Affected by https://gitlab.com/gpaw/gpaw/-/issues/840
# We are disabling assertions below as necessary, should be reenabled
# after fixing 840.
@pytest.mark.dielectricfunction
@pytest.mark.tetrahedron
@pytest.mark.response
def test_response_aluminum_EELS_RPA(in_tmp_dir):
    assert size <= 4**3

    # Ground state calculation

    t1 = time.time()

    a = 4.043
    atoms = bulk('Al', 'fcc', a=a)
    atoms.center()
    calc = GPAW(mode=PW(200),
                nbands=4,
                kpts=(4, 4, 4),
                parallel={'band': 1},
                xc='LDA')

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('Al_gs.gpw')

    # Generate grid compatible with tetrahedron integration
    kpts = find_high_symmetry_monkhorst_pack('Al_gs.gpw', 2.0)

    # Calculate the wave functions on the new kpts grid
    calc = GPAW('Al_gs.gpw').fixed_density(kpts=kpts, update_fermi_level=True)
    calc.write('Al.gpw', 'all')

    t2 = time.time()

    # Excited state calculation
    q0_c = np.array([0., 0., 0.])
    q1_c = np.array([1 / 4., 0., 0.])
    w_w = np.linspace(0, 24, 241)

    # Calculate the eels spectrum using point integration at both q-points
    df1 = DielectricFunction(calc='Al.gpw', frequencies=w_w, eta=0.2, ecut=50,
                             integrationmode='point integration',
                             hilbert=False, rate=0.2)
    df1.get_eels_spectrum(xc='RPA', filename='EELS_Al-PI_q0', q_c=q0_c)
    df1.get_eels_spectrum(xc='RPA', filename='EELS_Al-PI_q1', q_c=q1_c)

    t3 = time.time()

    # Calculate the eels spectrum using tetrahedron integration at q=0
    # NB: We skip the finite q-point, because the underlying symmetry
    # exploration runs excruciatingly slowly at finite q...
    df2 = DielectricFunction(calc='Al.gpw', eta=0.2, ecut=50,
                             integrationmode='tetrahedron integration',
                             hilbert=True, rate=0.2)
    df2.get_eels_spectrum(xc='RPA', filename='EELS_Al-TI_q0', q_c=q0_c)

    t4 = time.time()

    parprint('')
    parprint('For ground  state calc, it took', (t2 - t1) / 60, 'minutes')
    parprint('For PI excited state calc, it took', (t3 - t2) / 60, 'minutes')
    parprint('For TI excited state calc, it took', (t4 - t3) / 60, 'minutes')

    world.barrier()
    omegaP0_w, eels0P0_w, eelsP0_w = read_response_function('EELS_Al-PI_q0')
    omegaP1_w, eels0P1_w, eelsP1_w = read_response_function('EELS_Al-PI_q1')
    omegaT0_w, eels0T0_w, eelsT0_w = read_response_function('EELS_Al-TI_q0')

    # calculate the 1 & 2 wpeak and ipeak values for tetra and point int.
    wpeak1P0, Ipeak1P0 = findpeak(omegaP0_w, eels0P0_w)
    wpeak2P0, Ipeak2P0 = findpeak(omegaP0_w, eelsP0_w)
    wpeak1P1, Ipeak1P1 = findpeak(omegaP1_w, eels0P1_w)
    wpeak2P1, Ipeak2P1 = findpeak(omegaP1_w, eelsP1_w)
    wpeak1T0, Ipeak1T0 = findpeak(omegaT0_w, eels0T0_w)
    wpeak2T0, Ipeak2T0 = findpeak(omegaT0_w, eelsT0_w)

    # import matplotlib.pyplot as plt
    # plt.plot(omegaP0_w, eelsP0_w)
    # plt.plot(omegaT0_w, eelsT0_w)
    # plt.show()

    # tetra and point integrators should produce similar results; however,
    # Al converges very slowly w.r.t. kpts so we just make sure the
    # values don't change and tests consistency elsewhere
    assert wpeak1P0 == pytest.approx(15.7111, abs=0.02)
    assert wpeak2P0 == pytest.approx(15.7096, abs=0.02)
    assert wpeak1P1 == pytest.approx(15.8402, abs=0.02)
    assert wpeak2P1 == pytest.approx(15.8645, abs=0.02)
    # assert wpeak1T0 == pytest.approx(20.2119, abs=0.02)  # XXX #840
    # assert wpeak2T0 == pytest.approx(20.2179, abs=0.02)  # XXX #840

    assert Ipeak1P0 == pytest.approx(29.40, abs=1.)
    assert Ipeak2P0 == pytest.approx(27.70, abs=1.)
    assert Ipeak1P1 == pytest.approx(28.39, abs=1.)
    assert Ipeak2P1 == pytest.approx(26.89, abs=1.)
    # assert Ipeak1T0 == pytest.approx(46.24, abs=1.)  # XXX #840
    # assert Ipeak2T0 == pytest.approx(44.27, abs=1.)  # XXX #840
