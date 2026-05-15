import numpy as np
import pytest

from ase.units import Bohr, Hartree
from gpaw.external import ConstantElectricField
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.laser import create_laser, register_custom_laser, Laser
from gpaw.mpi import world
from gpaw.tddft.units import as_to_au

# Settings
dt = 20.0
N1 = 5
N = 5 + N1
kick_v = np.ones(3) * 1e-5


@pytest.mark.rttddft
@pytest.mark.parametrize('pulse', [
    {'name': 'GaussianPulse', 'strength': 1e-5, 'time0': 0, 'frequency': 8.6,
     'sigma': 0.5, 'sincos': 'sin'},
    {'name': 'SincPulse', 'strength': 1e-6, 'time0': 0, 'cutoff_freq': 15,
     'relative_t0': False}])
def test_laser(gpw_files, in_tmp_dir, pulse):
    # Time-propagation calculation
    td_calc = LCAOTDDFT(gpw_files['na2_tddft_dzp'], txt='td.out')
    DipoleMomentWriter(td_calc, 'dm.dat')
    td_calc.absorption_kick(kick_v)
    td_calc.propagate(dt, N)

    # Pulse
    direction = kick_v
    ext = ConstantElectricField(Hartree / Bohr, direction)

    # Time-propagation calculation with pulse
    pulse = create_laser(pulse)
    td_calc = LCAOTDDFT(gpw_files['na2_tddft_dzp'],
                        td_potential={'ext': ext, 'laser': pulse},
                        txt='tdpulse.out')
    DipoleMomentWriter(td_calc, 'dmpulse.dat')
    td_calc.propagate(dt, N1)
    td_calc.write('td.gpw', mode='all')
    # Restart
    td_calc = LCAOTDDFT('td.gpw', txt='tdpulse2.out')
    DipoleMomentWriter(td_calc, 'dmpulse.dat')
    td_calc.propagate(dt, N - N1)

    # Convoluted dipole moment
    world.barrier()
    time_t = np.arange(0, dt * (N + 0.1), dt) * as_to_au
    pulse_t = pulse.strength(time_t)
    np.savetxt('pulse.dat', np.stack((time_t, pulse_t)).T)
    dm_tv = np.delete(np.loadtxt('dm.dat')[:, 2:], 1, axis=0)
    dm_tv /= np.linalg.norm(kick_v)
    pulsedm_tv = np.delete(np.loadtxt('dmpulse.dat')[:, 2:], N1, axis=0)

    tol = 5e-6
    for v in range(3):
        pulsedmconv_t = np.convolve(
            dm_tv[:, v], pulse_t)[:(N + 1)] * dt * as_to_au
        np.savetxt('dmpulseconv%d.dat' % v, pulsedmconv_t)
        assert pulsedm_tv[:, v] == pytest.approx(pulsedmconv_t, abs=tol)


@pytest.mark.rttddft
def test_custom(gpw_files, in_tmp_dir):
    gpw_fname = gpw_files['na2_tddft_dzp']

    class RandomPulse(Laser):
        def __init__(self, strength):
            self.rng = np.random.default_rng(42)
            self.dict = dict(name='RandomPulse', strength=strength)
            self.s0 = strength

        def strength(self, t):
            return self.s0 * self.rng.uniform(size=np.shape(t))

        def todict(self):
            return self.dict

    # We should be able to run and restart using custom pulses
    register_custom_laser('RandomPulse', RandomPulse)

    # Pulse
    direction = kick_v
    ext = ConstantElectricField(Hartree / Bohr, direction)

    # Time-propagation calculation with pulse
    pulse = RandomPulse(1e-5)
    td_calc = LCAOTDDFT(gpw_fname, td_potential={'ext': ext, 'laser': pulse},
                        txt='tdpulse.out')
    td_calc.propagate(dt, 1)

    td_calc.write('td.gpw', mode='all')

    # Restart
    td_calc = LCAOTDDFT('td.gpw', txt='tdpulse2.out')

    restart_pulse = td_calc.td_hamiltonian.td_potential.laser_i[0]

    assert isinstance(restart_pulse, RandomPulse)
    np.testing.assert_equal(restart_pulse.todict(), pulse.todict())
