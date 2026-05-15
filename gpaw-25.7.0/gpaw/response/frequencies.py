from __future__ import annotations

import numbers
from typing import Any

import numpy as np
from ase.units import Ha
from gpaw.typing import ArrayLike1D


class FrequencyDescriptor:
    def __init__(self, omega_w: ArrayLike1D):
        """Frequency grid descriptor.

        Parameters
        ----------
        omega_w:
            Frequency grid in Hartree units.
        """
        self.omega_w = np.asarray(omega_w).copy()

    def __len__(self):
        return len(self.omega_w)

    def __repr__(self):
        emin = self.omega_w[0] * Ha
        emax = self.omega_w[-1] * Ha
        return (f'{self.__class__.__name__}'
                f'(from {emin:.3f} to {emax:.3f} eV, {len(self)} points)')

    @staticmethod
    def from_array_or_dict(input: dict[str, Any] | ArrayLike1D
                           ) -> FrequencyDescriptor:
        """Create frequency-grid descriptor.

        In case *input* is a list on frequencies (in eV) a
        :class:`FrequencyGridDescriptor` instance is returned.
        Othervise a :class:`NonLinearFrequencyDescriptor` instance is
        returned.

        >>> from ase.units import Ha
        >>> params = dict(type='nonlinear',
        ...               domega0=0.1,
        ...               omega2=10,
        ...               omegamax=50)
        >>> wd = FrequencyDescriptor.from_array_or_dict(params)
        >>> wd.omega_w[0:2] * Ha
        array([0.        , 0.10041594])
        """
        if isinstance(input, dict):
            assert input['type'] == 'nonlinear'
            domega0 = input.get('domega0')
            omega2 = input.get('omega2')
            omegamax = input['omegamax']
            return NonLinearFrequencyDescriptor(
                (0.1 if domega0 is None else domega0) / Ha,
                (10.0 if omega2 is None else omega2) / Ha,
                omegamax / Ha)
        return FrequencyGridDescriptor(np.asarray(input) / Ha)


class ComplexFrequencyDescriptor:

    def __init__(self, hz_z: ArrayLike1D):
        """Construct the complex frequency descriptor.

        Parameters
        ----------
        hz_z:
            Array of complex frequencies (in units of Hartree)
        """
        # Use a copy of the input array
        hz_z = np.array(hz_z)
        assert hz_z.dtype == complex

        self.hz_z = hz_z

    def __len__(self):
        return len(self.hz_z)

    def almost_eq(self, zd):
        if len(zd) != len(self):
            return False
        return np.allclose(self.hz_z, zd.hz_z)

    @staticmethod
    def from_array(frequencies: ArrayLike1D):
        """Create a ComplexFrequencyDescriptor from frequencies in eV."""
        return ComplexFrequencyDescriptor(np.asarray(frequencies) / Ha)

    @property
    def upper_half_plane(self):
        """All frequencies reside in the upper half complex frequency plane?"""
        return np.all(self.hz_z.imag > 0.)

    @property
    def horizontal_contour(self):
        """Do all frequency point lie on a horizontal contour?"""
        return np.ptp(self.hz_z.imag) < 1.e-6

    @property
    def omega_w(self):
        """Real part of the frequencies."""
        assert self.horizontal_contour, \
            'It only makes sense to index the frequencies by their real part '\
            'if they reside on a horizontal contour.'
        return self.hz_z.real


class FrequencyGridDescriptor(FrequencyDescriptor):

    def get_index_range(self, lim1_m, lim2_m):
        """Get index range. """

        i0_m = np.zeros(len(lim1_m), int)
        i1_m = np.zeros(len(lim2_m), int)

        for m, (lim1, lim2) in enumerate(zip(lim1_m, lim2_m)):
            i_x = np.logical_and(lim1 <= self.omega_w,
                                 lim2 >= self.omega_w)
            if i_x.any():
                inds = np.argwhere(i_x)
                i0_m[m] = inds.min()
                i1_m[m] = inds.max() + 1

        return i0_m, i1_m


class NonLinearFrequencyDescriptor(FrequencyDescriptor):
    def __init__(self,
                 domega0: float,
                 omega2: float,
                 omegamax: float):
        """Non-linear frequency grid.

        Units are Hartree.  See :ref:`frequency grid`.

        Parameters
        ----------
        domega0:
            Frequency grid spacing for non-linear frequency grid at omega = 0.
        omega2:
            Frequency at which the non-linear frequency grid has doubled
            the spacing.
        omegamax:
            The upper frequency bound for the non-linear frequency grid.
        """
        beta = (2**0.5 - 1) * domega0 / omega2
        wmax = int(omegamax / (domega0 + beta * omegamax))
        w = np.arange(wmax + 2)  # + 2 is for buffer
        omega_w = w * domega0 / (1 - beta * w)

        super().__init__(omega_w)

        self.domega0 = domega0
        self.omega2 = omega2
        self.omegamax = omegamax
        self.omegamin = 0

        self.beta = beta
        self.wmax = wmax
        self.omega_w = omega_w
        self.wmax = wmax

    def get_floor_index(self, o_m, safe=True):
        """Get closest index rounding down."""
        beta = self.beta
        w_m = (o_m / (self.domega0 + beta * o_m)).astype(int)
        if safe:
            if isinstance(w_m, np.ndarray):
                w_m[w_m >= self.wmax] = self.wmax - 1
            elif isinstance(w_m, numbers.Integral):
                if w_m >= self.wmax:
                    w_m = self.wmax - 1
            else:
                raise TypeError
        return w_m

    def get_index_range(self, omega1_m, omega2_m):
        omega1_m = omega1_m.copy()
        omega2_m = omega2_m.copy()
        omega1_m[omega1_m < 0] = 0
        omega2_m[omega2_m < 0] = 0
        w1_m = self.get_floor_index(omega1_m)
        w2_m = self.get_floor_index(omega2_m)
        o1_m = self.omega_w[w1_m]
        o2_m = self.omega_w[w2_m]
        w1_m[o1_m < omega1_m] += 1
        w2_m[o2_m < omega2_m] += 1
        return w1_m, w2_m
