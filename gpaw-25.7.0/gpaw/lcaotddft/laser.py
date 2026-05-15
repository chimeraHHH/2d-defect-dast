from __future__ import annotations
import numpy as np

from typing import Type
from gpaw.mpi import world
from gpaw.tddft.units import as_to_au, eV_to_au


known_lasers: dict[str, Type[Laser]] = dict()


def create_laser(name, **kwargs):
    """ Create Laser from dict """

    if isinstance(name, Laser):
        return name
    elif isinstance(name, dict):
        kwargs.update(name)
        return create_laser(**kwargs)

    if not known_lasers:
        _register_known_lasers()

    cls = known_lasers.get(name, None)
    if cls is None:
        raise ValueError(f'Unknown laser: {name}')
    return cls(**kwargs)


def register_custom_laser(name: str,
                          cls: Type[Laser]):
    """ Register a custom laser object

    This function must be used when restarting TDDFT calculations using
    user defined laser classes

    Parameters
    ----------
    name
        Name of laser object. Must be consistent with the name in todict()
    cls
        Class of the laser object
    """
    if not known_lasers:
        _register_known_lasers()

    known_lasers[name] = cls


def _register_known_lasers():
    for name in ['GaussianPulse', 'SincPulse', 'SumLaser']:
        known_lasers[name] = globals()[name]


class Laser:
    def __init__(self):
        pass

    def strength(self, time):
        return 0.0

    def derivative(self, time):
        return 0.0

    def fourier(self, omega):
        return 0.0

    def write(self, fname, time_t):
        """
        Write the values of the pulse to a file.

        Parameters
        ----------
        fname
            filename
        time_t
            times in attoseconds
        """
        if world.rank != 0:
            return
        time_t = time_t * as_to_au
        strength_t = self.strength(time_t)
        derivative_t = self.derivative(time_t)
        fmt = '%12.6f %20.10e %20.10e'
        header = '{:^10} {:^20} {:^20}'.format('time', 'strength',
                                               'derivative')
        np.savetxt(fname, np.stack((time_t, strength_t, derivative_t)).T,
                   fmt=fmt, header=header)


class SumLaser(Laser):
    def __init__(self, *lasers):
        self.laser_i = []
        dict_i = []
        for laser in lasers:
            laser = create_laser(laser)
            self.laser_i.append(laser)
            dict_i.append(laser.todict())
        self.dict = dict(name='SumLaser',
                         lasers=dict_i)

    def strength(self, time):
        s = 0.0
        for laser in self.laser_i:
            s += laser.strength(time)
        return s

    def fourier(self, omega):
        s = 0.0
        for laser in self.laser_i:
            s += laser.fourier(omega)
        return s

    def todict(self):
        return self.dict


class GaussianPulse(Laser):
    r"""
    Laser pulse with Gaussian envelope:

    .. math::

        g(t) = s_0 \sin(\omega_0 (t - t_0)) \exp(-\sigma^2 (t - t_0)^2 / 2)


    Parameters
    ----------
    strength: float
        value of :math:`s_0` in atomic units
    time0: float
        value of :math:`t_0` in attoseconds
    frequency: float
        value of :math:`\omega_0` in eV
    sigma: float
        value of :math:`\sigma` in eV
    sincos: 'sin' or 'cos'
        use sin or cos function
    stoptime: float
        pulse is set to zero after this value (in attoseconds)
    """

    def __init__(self, strength, time0, frequency, sigma, sincos='sin',
                 stoptime=np.inf):
        self.dict = dict(name='GaussianPulse',
                         strength=strength,
                         time0=time0,
                         frequency=frequency,
                         sigma=sigma,
                         sincos=sincos)
        self.s0 = strength
        self.t0 = time0 * as_to_au
        self.omega0 = frequency * eV_to_au
        self.sigma = sigma * eV_to_au
        self.stoptime = stoptime * as_to_au
        assert sincos in ['sin', 'cos']
        self.sincos = sincos

    def strength(self, t):
        """
        Return the value of the pulse :math:`g(t)`.

        Parameters
        ----------
        t
            time in atomic units

        Returns
        -------
        The value of the pulse.
        """
        s = self.s0 * np.exp(-0.5 * self.sigma**2 * (t - self.t0)**2)
        if self.sincos == 'sin':
            s *= np.sin(self.omega0 * (t - self.t0))
        else:
            s *= np.cos(self.omega0 * (t - self.t0))
        flt = t < self.stoptime

        return s * flt

    def derivative(self, t):
        """
        Return the derivative of the pulse :math:`g'(t)`.

        Parameters
        ----------
        t
            time in atomic units

        Returns
        -------
        The derivative of the pulse.
        """
        dt = t - self.t0
        s = self.s0 * np.exp(-0.5 * self.sigma**2 * dt**2)
        if self.sincos == 'sin':
            s *= (-self.sigma**2 * dt * np.sin(self.omega0 * dt) +
                  self.omega0 * np.cos(self.omega0 * dt))
        else:
            s *= (-self.sigma**2 * dt * np.cos(self.omega0 * dt) +
                  -self.omega0 * np.sin(self.omega0 * dt))
        return s

    def fourier(self, omega):
        r"""
        Return Fourier transform of the pulse :math:`g(\omega)`.

        Parameters
        ----------
        omega
            frequency in atomic units

        Returns
        -------
        Fourier transform of the pulse.
        """
        s = (self.s0 * np.sqrt(np.pi / 2) / self.sigma *
             np.exp(-0.5 * (omega - self.omega0)**2 / self.sigma**2) *
             np.exp(1.0j * self.t0 * omega))
        if self.sincos == 'sin':
            s *= 1.0j
        return s

    def todict(self):
        return self.dict


class SincPulse(Laser):
    r"""
    Laser pulse with sinc envelope:

    .. math::

        g(t) = s_0 \frac{\sin(\pi \omega_{cut} (t - t_0))}
        {\pi \omega_{cut} (t - t_0)}


    Parameters
    ----------
    strength: float
        value of :math:`s_0` in atomic units
    time0: float
        value of :math:`t_0` in attoseconds, or in units of
        :math:`\omega_{cut} / 2` if relative_t0 is True
    cutoff_freq: float
        Cutoff frequency: value of :math:`\omega_{cut}` in eV
    relative_t0:
        Specify time0 in units relative to the cutoff frequency
    """

    def __init__(self,
                 strength: float,
                 time0: float,
                 cutoff_freq: float,
                 relative_t0: bool):
        self.dict = dict(name='SincPulse',
                         strength=strength,
                         time0=time0,
                         cutoff_freq=cutoff_freq,
                         relative_t0=relative_t0)
        self.s0 = strength
        self.omega_cut = (cutoff_freq / np.pi) * eV_to_au
        if relative_t0:
            self.t0 = 2 * time0 / self.omega_cut
        else:
            self.t0 = time0 * as_to_au

    def strength(self, t):
        """
        Return the value of the pulse :math:`g(t)`.

        Parameters
        ----------
        t
            time in atomic units

        Returns
        -------
        The value of the pulse.
        """
        s = self.s0 * np.sinc(self.omega_cut * (t - self.t0))

        return s

    def derivative(self, t):
        raise NotImplementedError

    def fourier(self, omega):
        raise NotImplementedError

    def todict(self):
        return self.dict
