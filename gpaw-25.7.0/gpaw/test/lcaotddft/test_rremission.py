import pytest
import numpy as np

from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.qed import RRemission
from gpaw.tddft.spectrum import read_td_file_data
from gpaw.tddft.units import as_to_au


@pytest.mark.rttddft
def test_rremission(gpw_files, in_tmp_dir):
    dt = 40
    niter = 20

    env = {'environment': 'waveguide',
           'quantization_plane': 0.5,
           'cavity_polarization': [0, 0, 1]}
    td_calc = LCAOTDDFT(gpw_files['na2_tddft_dzp'],
                        rremission=RRemission(env),
                        propagator={'name': 'scpc', 'tolerance': 1e-0})
    DipoleMomentWriter(td_calc, 'dm.dat')
    td_calc.absorption_kick([0.0, 0.0, 1e-5])
    td_calc.propagate(dt, niter)

    times = np.arange(0, dt * (niter + 0.1), dt) * as_to_au
    times = np.concatenate(([0], times))  # One inital zero before the kick
    data = np.loadtxt('dm.dat')

    # Check times in data file
    np.testing.assert_allclose(data[:, 0], times)

    # Check norm and x, y components
    np.testing.assert_allclose(data[:, 1:4], 0, atol=1e-10)

    # Check z component
    ref_i = [
        -3.982847629258e-14,
        -3.914902027642e-14,
        3.419488131280e-05,
        6.328738771521e-05,
        8.608533242884e-05,
        1.032870345833e-04,
        1.156375187581e-04,
        1.238315688509e-04,
        1.285064383632e-04,
        1.302401811969e-04,
        1.295514557099e-04,
        1.269006285844e-04,
        1.226921048004e-04,
        1.172777277082e-04,
        1.109610266703e-04,
        1.040020450314e-04,
        9.662244521506e-05,
        8.901056071782e-05,
        8.132605402745e-05,
        7.370387298427e-05,
        6.625729128738e-05,
        5.907997327753e-05,
    ]

    np.testing.assert_allclose(data[:, 4], ref_i, atol=1e-10)

    """
    Restart check for rremission. Does restarting change the outcome?
    """

    env = {'environment': 'waveguide',
           'quantization_plane': 0.5,
           'cavity_polarization': [0, 0, 1]}
    td_calc = LCAOTDDFT(gpw_files['na2_tddft_dzp'],
                        rremission=RRemission(env),
                        propagator={'name': 'scpc', 'tolerance': 1e-0})
    DipoleMomentWriter(td_calc, 'dm_rrsplit.dat')
    td_calc.absorption_kick([0.0, 0.0, 1e-5])
    td_calc.propagate(40, 10)
    td_calc.write('td_rrsplit0.gpw', mode='all')

    td_calc_restart = LCAOTDDFT('td_rrsplit0.gpw')
    DipoleMomentWriter(td_calc_restart, 'dm_rrsplit.dat')
    td_calc_restart.propagate(40, 10)

    dipole_full = read_td_file_data('dm.dat')[1][-10:]
    dipole_restart = read_td_file_data('dm_rrsplit.dat',)[1][-10:]
    assert np.allclose(dipole_full, dipole_restart)


@pytest.mark.ci
def test_legacy_parameters():
    quantization_plane = 0.5
    cavity_polarization = [0, 0, 1]
    env = {'environment': 'waveguide',
           'quantization_plane': quantization_plane,
           'cavity_polarization': cavity_polarization}

    # The proper way of setting up the RRemission
    rr = RRemission(env)

    # The legacy way
    with pytest.warns(FutureWarning):
        rr_legacy = RRemission(quantization_plane, cavity_polarization)

    # The environment should be the same
    assert rr.environment.todict() == rr_legacy.environment.todict()
