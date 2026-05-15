from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
from ase.units import alpha, Hartree, Bohr
from gpaw.external import ConstantElectricField
from gpaw.lcaotddft.hamiltonian import KickHamiltonian


def create_environment(environment='waveguide', **kwargs):
    if isinstance(environment, Environment):
        assert not kwargs, 'kwargs must not be given here'
        return environment

    if isinstance(environment, dict):
        assert not kwargs, 'please do not give kwargs together with dict'
        kwargs = dict(environment)
        environment = kwargs.pop('environment', 'waveguide')

    if environment == 'waveguide':
        return WaveguideEnvironment(**kwargs)

    raise ValueError(f'Unknown environment {environment}')


def forward_finite_difference(coefficients: list[int],
                              data_tx: np.ndarray):
    """ Calculate derivatives by forward difference.

    The data corresponding to the last time is repeated several times,
    in order to obtain a result of the same shape as the original array.

    Parameters
    ----------
    coefficients
        List of coefficients.
        Check e.g. https://en.wikipedia.org/wiki/Finite_difference_coefficient
    data_tx
        Array of data, the first axis being the time axis.

    """
    N = len(data_tx)

    padded_shape = (N + len(coefficients) - 1, ) + data_tx.shape[1:]
    padded_data_tx = np.zeros(padded_shape, dtype=data_tx.dtype)
    padded_data_tx[:len(coefficients)] = data_tx[:1]
    padded_data_tx[len(coefficients):] = data_tx[1:]

    result_tx = np.zeros_like(data_tx)

    for start, coefficient in enumerate(coefficients):
        result_tx[:] += coefficient * padded_data_tx[start:start + N]

    return result_tx


def calculate_first_derivative(timestep, data_tx):
    """ Calculate the first derivative by first order forward difference.

    The data corresponding to the last time is repeated several times,
    in order to obtain a result of the same shape as the original array.

    Parameters
    ----------
    timestep
        Time step
    data_tx
        Array of data, the first axis being the time axis.

    Returns
    -------
    First derivative, array same shape as :attr:`data_tx`.
    """
    if len(data_tx) == 1 and timestep is None:
        return np.zeros_like(data_tx)

    # Coefficients from oldest times to newest times
    coefficients = [-1, 1]
    return forward_finite_difference(coefficients, data_tx) / timestep


def calculate_third_derivative(timestep, data_tx):
    """ Calculate the third derivative by first order forward difference.

    The data corresponding to the last time is repeated several times,
    in order to obtain a result of the same shape as the original array.

    Parameters
    ----------
    timestep
        Time step
    data_tx
        Array of data, the first axis being the time axis.

    Returns
    -------
    Third derivative, array same shape as :attr:`data_tx`.
    """
    if len(data_tx) == 1 and timestep is None:
        return np.zeros_like(data_tx)

    # Coefficients from oldest times to newest times
    coefficients = [-1, 3, -3, 1]
    # For a second order accuracy, the coefficients would be
    # [-5/2, 9, -12, 7, -3/2]
    return forward_finite_difference(coefficients, data_tx) / timestep ** 3


class RRemission:
    r"""
    Radiation-reaction potential according to Schaefer et al.
    [https://doi.org/10.1103/PhysRevLett.128.156402] and
    Schaefer [https://doi.org/10.48550/arXiv.2204.01602].
    The potential accounts for the friction
    forces acting on the radiating system of oscillating charges
    emitting into a single dimension. A more elegant
    formulation would use the current instead of the dipole.
    Please contact christian.schaefer.physics@gmail.com if any problems
    should appear or you would like to consider more complex systems.
    Big thanks to Tuomas Rossi and Jakub Fojt for their help.

    Parameters
    ----------
    environment
        Environment, or dictionary with environment parameters.

    Notes
    -----
    A legacy form for constructing the RRemission object is supported
    for backwards compatibility. In this form, two parameters are given.

        quantization_plane: float
            value of :math:`quantization_plane` in units of Å^2
        cavity_polarization: array
            value of :math:`cavity_polarization`; dimensionless (directional)
    """

    def __init__(self, *args):
        if len(args) == 1:
            self.environment = create_environment(args[0])
        elif len(args) == 2:
            # Legacy syntax
            self.environment = create_environment(
                'waveguide',
                quantization_plane=args[0],
                cavity_polarization=args[1])
            warnings.warn(
                "Use RRemission({'environment': 'waveguide', "
                "'quantization_plane': quantization_plane, "
                "'cavity_polarization': cavity_polarization})) instead of "
                'RRemission(quantization_plane, cavity_polarization).',
                FutureWarning)
        else:
            raise TypeError('Please provide only one argument '
                            '(two for legacy syntax)')

        # Recorded dipole moment over time
        # The entries all correspond to unique times
        # If there is a kick, the dipole after the kick is recorded
        # but not the dipole just before the kick
        self.dipole_tv = np.zeros((0, 3))
        # Time of last record
        self.time = 0

        # Time step in TDDFT
        self.timestep = None

    def read(self, reader):
        self.dipole_tv = reader.recorded_dipole / Bohr
        self.timestep = reader.timestep

    def write(self, writer):
        writer.write(recorded_dipole=self.dipole_tv * Bohr)
        writer.write(timestep=self.timestep)
        self.environment.write(writer)

    @classmethod
    def from_reader(cls, reader):
        parameters = reader.parameters.asdict()

        rremission = cls(parameters)
        rremission.read(reader)

        return rremission

    def initialize(self, paw):
        self.density = paw.density
        self.wfs = paw.wfs
        self.hamiltonian = paw.hamiltonian
        self.time = paw.time
        dipole_v = paw.density.calculate_dipole_moment()
        if len(self.dipole_tv) == 0:
            # If we are not reading from a restart file, this will be empty
            self.record_dipole(dipole_v)

        # Set up three external fields summing to the polarization cavity
        strength_v = self.environment.cavity_polarization * Hartree / Bohr
        assert len(strength_v) == 3
        ext_x = ConstantElectricField(strength_v[0], [1, 0, 0])
        ext_y = ConstantElectricField(strength_v[1], [0, 1, 0])
        ext_z = ConstantElectricField(strength_v[2], [0, 0, 1])
        ext_i = [ext_x, ext_y, ext_z]

        # Set up the dipole matrix
        get_matrix = self.wfs.eigensolver.calculate_hamiltonian_matrix
        self.V_iuMM = []
        for ext in ext_i:
            V_uMM = []
            hamiltonian = KickHamiltonian(self.hamiltonian, self.density, ext)
            for kpt in self.wfs.kpt_u:
                V_MM = get_matrix(hamiltonian, self.wfs, kpt,
                                  add_kinetic=False, root=-1)
                V_uMM.append(V_MM)
            self.V_iuMM.append(V_uMM)

    def record_dipole(self, dipole_v):
        """ Record the dipole for the next time step.

        Parameters
        ----------
        dipole_v
            The new dipole moment.
        """
        self.dipole_tv = np.vstack((self.dipole_tv, dipole_v))

    def update_dipole(self, time, dipole_v):
        """ Record the new dipole

        If the time has changed, record the new dipole in a new entry.

        If not, then it either means that we are iteratively
        performing propagator steps, or that a kick has been done.
        Either way, we overwrite the last entry in the record.

        Parameters
        ----------
        time
            The new time
        dipole_v
            The new dipole moment.
        """
        if time > self.time or len(self.dipole_tv) == 0:
            self.record_dipole(dipole_v)
            timestep = time - self.time
            if (self.timestep is not None and
                not np.isclose(self.timestep, timestep)):
                raise NotImplementedError(
                    'Variable time step in TDDFT not supported'
                    f'{self.timestep} != {timestep}')
            self.timestep = timestep
            self.time = time
        else:
            self.dipole_tv[-1] = dipole_v

    def get_MM(self, field_v, kpt):
        """ Get potential matrix.

        Parameters
        ----------
        field_v
            The radiation reaction field.
        kpt
            kpoint

        Returns
        -------
        The radiation reaction potential matrix.
        """
        kpt_rank, q = self.wfs.kd.get_rank_and_index(kpt.k)
        u = q * self.wfs.nspins + kpt.s

        Ni = len(self.V_iuMM)
        Vrr_MM = -field_v[0] * self.V_iuMM[0][u]
        for i in range(1, Ni):
            Vrr_MM -= field_v[i] * self.V_iuMM[i][u]

        return Vrr_MM

    def vradiationreaction(self, kpt, time):
        self.update_dipole(time, self.density.calculate_dipole_moment())

        field_v = self.environment.radiation_reaction_field(
            self.timestep, self.dipole_tv)

        Vrr_MM = self.get_MM(field_v, kpt)
        return Vrr_MM


class Environment(ABC):

    def write(self, writer):
        writer.child('parameters').write(**self.todict())

    @abstractmethod
    def radiation_reaction_field(self, timestep, dipole_tv):
        raise NotImplementedError

    @abstractmethod
    def todict(self):
        raise NotImplementedError


class WaveguideEnvironment(Environment):

    r""" The radiation reaction potential in a 1D waveguide is

    .. math::

        -\frac{4 \pi \alpha}{A}
        \boldsymbol{\varepsilon_c} \cdot \partial_t \boldsymbol{R}(t)
        \boldsymbol{\varepsilon_c} \cdot \hat{\boldsymbol{r}}

    Parameters
    ----------
    quantization_plane
        Quantization plane :math:`A` in units of Å^2.
    cavity_polarization
        The polarization of the cavity :math:`\boldsymbol{\varepsilon_c}`.
        Direction vector.
    """

    def __init__(self, quantization_plane, cavity_polarization):
        self.quantization_plane = quantization_plane / Bohr**2
        self.cavity_polarization = np.array(cavity_polarization)

    def radiation_reaction_field(self, timestep, dipole_tv):
        d1_dipole_v = calculate_first_derivative(timestep, dipole_tv)[-1]

        field_v = (4.0 * np.pi * alpha / self.quantization_plane
                   * (d1_dipole_v @ self.cavity_polarization)
                   * self.cavity_polarization)

        return field_v

    def todict(self):
        return {'environment': 'waveguide',
                'quantization_plane': self.quantization_plane * Bohr**2,
                'cavity_polarization': list(self.cavity_polarization)}
