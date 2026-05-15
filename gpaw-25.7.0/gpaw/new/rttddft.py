from __future__ import annotations

from typing import Generator, NamedTuple

import numpy as np
from numpy.linalg import solve

from ase.units import Bohr, Hartree

from gpaw.external import ExternalPotential, ConstantElectricField
from gpaw.typing import Vector
from gpaw.mpi import world
from gpaw.new.ase_interface import ASECalculator
from gpaw.new.calculation import DFTState, DFTCalculation
from gpaw.new.lcao.hamiltonian import HamiltonianMatrixCalculator
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.new.gpw import read_gpw
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.tddft.units import asetime_to_autime, autime_to_asetime, au_to_eA
from gpaw.utilities.timing import nulltimer


class TDAlgorithm:

    def kick(self,
             state: DFTState,
             pot_calc: PotentialCalculator,
             dm_calc: HamiltonianMatrixCalculator):
        raise NotImplementedError()

    def propagate(self,
                  time_step: float,
                  state: DFTState,
                  pot_calc: PotentialCalculator,
                  ham_calc: HamiltonianMatrixCalculator):
        raise NotImplementedError()

    def get_description(self):
        return self.__class__.__name__


def propagate_wave_functions_numpy(source_C_nM: np.ndarray,
                                   target_C_nM: np.ndarray,
                                   S_MM: np.ndarray,
                                   H_MM: np.ndarray,
                                   dt: float):
    SjH_MM = S_MM + (0.5j * dt) * H_MM
    target_C_nM[:] = source_C_nM @ SjH_MM.conj().T
    target_C_nM[:] = solve(SjH_MM.T, target_C_nM.T).T


class ECNAlgorithm(TDAlgorithm):

    def kick(self,
             state: DFTState,
             pot_calc: PotentialCalculator,
             dm_calc: HamiltonianMatrixCalculator):
        """Propagate wavefunctions by delta-kick.

        ::

                                 0+
                     ^        -1 /                     ^        -1
          U(0+, 0) = T exp[-iS   | δ(τ) V   (r) dτ ] = T exp[-iS  V   (r)]
                                 /       ext                       ext
                                 0

        (1) Calculate propagator U(0+, 0)
        (2) Update wavefunctions ψ_n(0+) ← U(0+, 0) ψ_n(0)
        (3) Update density and hamiltonian H(0+)

        Parameters
        ----------
        state
            Current state of the wave functions, that is to be updated
        pot_calc
            The potential calculator
        dm_calc
            Dipole moment operator calculator, which contains the dipole moment
            operator
        """
        for wfs in state.ibzwfs:
            assert isinstance(wfs, LCAOWaveFunctions)
            V_MM = dm_calc.calculate_matrix(wfs)

            # Phi_n <- U(0+, 0) Phi_n
            nkicks = 10
            for i in range(nkicks):
                propagate_wave_functions_numpy(wfs.C_nM.data, wfs.C_nM.data,
                                               wfs.S_MM.data,
                                               V_MM.data, 1 / nkicks)
        # Update density
        state.density.update(state.ibzwfs)

        # Calculate Hamiltonian H(t+dt) = H[n[Phi_n]]
        state.potential, state.energies, _ = pot_calc.calculate(
            state.density, state.potential.vHt_x)

    def propagate(self,
                  time_step: float,
                  state: DFTState,
                  pot_calc: PotentialCalculator,
                  ham_calc: HamiltonianMatrixCalculator):
        """ One explicit Crank-Nicolson propagation step, i.e.

        (1) Calculate propagator U[H(t)]
        (2) Update wavefunctions ψ_n(t+dt) ← U[H(t)] ψ_n(t)
        (3) Update density and hamiltonian H(t+dt)
        """
        for wfs in state.ibzwfs:
            assert isinstance(wfs, LCAOWaveFunctions)
            H_MM = ham_calc.calculate_matrix(wfs)

            # Phi_n <- U[H(t)] Phi_n
            propagate_wave_functions_numpy(wfs.C_nM.data, wfs.C_nM.data,
                                           wfs.S_MM.data,
                                           H_MM.data, time_step)
        # Update density
        state.density.update(state.ibzwfs)

        # Calculate Hamiltonian H(t+dt) = H[n[Phi_n]]
        state.potential, state.energies, _ = pot_calc.calculate(
            state.density, state.potential.vHt_x)


class RTTDDFTHistory:

    kick_strength: Vector | None  # Kick strength in atomic units
    niter: int  # Number of propagation steps
    time: float  # Simulation time in atomic units

    def __init__(self):
        """Object that keeps track of the RT-TDDFT history, that is

        - Has a kick been performed?
        - The number of propagation states performed
        """
        self.kick_strength = None
        self.niter = 0
        self.time = 0.0

    def absorption_kick(self,
                        kick_strength: Vector):
        """ Store the kick strength in history

        At most one kick can be done, and it must happen before any
        propagation steps

        Parameters
        ----------
        kick_strength
            Strength of the kick in atomic units
        """
        assert self.niter == 0, 'Cannot kick if already propagated'
        assert self.kick_strength is None, 'Cannot kick if already kicked'
        self.kick_strength = np.array(kick_strength, dtype=float)

    def propagate(self,
                  time_step: float) -> float:
        """ Increment the number of propagation steps and simulation time
        in history

        Parameters
        ----------
        time_step
            Time step in atomic units

        Returns
        -------
        The new simulation time in atomic units
        """
        self.niter += 1
        self.time += time_step

        return self.time

    def todict(self):
        absorption_kick = self.absorption_kick
        if absorption_kick is not None:
            absorption_kick = absorption_kick.tolist()
        return {'niter': self.niter, 'time': self.time,
                'absorption_kick': absorption_kick}


class RTTDDFTResult(NamedTuple):

    """ Results are stored in atomic units, but displayed to the user in
    ASE units
    """

    time: float  # Time in atomic units
    dipolemoment: Vector  # Dipole moment in atomic units

    def __repr__(self):
        timestr = f'{self.time * autime_to_asetime:.3f} Å√(u/eV)'
        dmstr = ', '.join([f'{dm * au_to_eA:10.4g}'
                           for dm in self.dipolemoment])
        dmstr = f'[{dmstr}]'

        return (f'{self.__class__.__name__}: '
                f'(time: {timestr}, dipolemoment: {dmstr} eÅ)')


class RTTDDFT:
    def __init__(self,
                 state: DFTState,
                 pot_calc: PotentialCalculator,
                 hamiltonian,
                 history: RTTDDFTHistory,
                 propagator: TDAlgorithm | None = None):
        if propagator is None:
            propagator = ECNAlgorithm()

        self.state = state
        self.pot_calc = pot_calc
        self.propagator = propagator
        self.hamiltonian = hamiltonian
        self.history = history

        self.kick_ext: ExternalPotential | None = None

        # Dipole moment operators in each Cartesian direction
        self.dm_operator_c: list[HamiltonianMatrixCalculator] | None = None

        self.timer = nulltimer
        self.log = print

        self.ham_calc = hamiltonian.create_hamiltonian_matrix_calculator(state)

    @classmethod
    def from_dft_calculation(cls,
                             calc: ASECalculator | DFTCalculation,
                             propagator: TDAlgorithm | None = None):

        if isinstance(calc, DFTCalculation):
            dft = calc
        else:
            assert calc.dft is not None
            dft = calc.dft

        state = dft.get_state()
        pot_calc = dft.pot_calc
        hamiltonian = dft.scf_loop.hamiltonian
        history = RTTDDFTHistory()

        return cls(state, pot_calc, hamiltonian, propagator=propagator,
                   history=history)

    @classmethod
    def from_dft_file(cls,
                      filepath: str,
                      propagator: TDAlgorithm | None = None):
        _, dft, params, builder = read_gpw(filepath,
                                           log='-',
                                           comm=world,
                                           dtype=complex)

        state = dft.get_state()
        pot_calc = dft.pot_calc
        hamiltonian = builder.create_hamiltonian_operator()
        history = RTTDDFTHistory()

        return cls(state, pot_calc, hamiltonian, propagator=propagator,
                   history=history)

    def absorption_kick(self,
                        kick_strength: Vector):
        """Kick with a weak electric field.

        Parameters
        ----------
        kick_strength
            Strength of the kick in atomic units
        """
        with self.timer('Kick'):
            kick_strength = np.array(kick_strength, dtype=float)
            self.history.absorption_kick(kick_strength)

            magnitude = np.sqrt(np.sum(kick_strength**2))
            direction = kick_strength / magnitude
            dirstr = [f'{d:.4f}' for d in direction]

            self.log('----  Applying absorption kick')
            self.log(f'----  Magnitude: {magnitude:.8f} Hartree/Bohr')
            self.log(f'----  Direction: {dirstr}')

            # Create Hamiltonian object for absorption kick
            cef = ConstantElectricField(magnitude * Hartree / Bohr, direction)

            # Propagate kick
            self.kick(cef)

    def kick(self,
             ext: ExternalPotential):
        """Kick with any external potential.

        Note that unless this function is called by absorption_kick, the kick
        is not logged in history

        Parameters
        ----------
        ext
            External potential
        """
        with self.timer('Kick'):
            self.log('----  Applying kick')
            self.log(f'----  {ext}')

            dm_operator_calc = self.hamiltonian.create_kick_matrix_calculator(
                self.state, ext, self.pot_calc)

            self.kick_ext = ext

            # Propagate kick
            self.propagator.kick(state=self.state,
                                 pot_calc=self.pot_calc,
                                 dm_calc=dm_operator_calc)

    def ipropagate(self,
                   time_step: float = 10.0,
                   maxiter: int = 2000,
                   ) -> Generator[RTTDDFTResult, None, None]:
        """Propagate the electronic system.

        Parameters
        ----------
        time_step
            Time step in ASE time units Å√(u/eV)
        iterations
            Number of propagation steps
        """

        time_step = time_step * asetime_to_autime

        for iteration in range(maxiter):
            self.propagator.propagate(time_step,
                                      state=self.state,
                                      pot_calc=self.pot_calc,
                                      ham_calc=self.ham_calc)
            time = self.history.propagate(time_step)
            # TODO This seems to be broken
            # dipolemoment = self.state.density.calculate_dipole_moment(
            #     self.pot_calc.relpos_ac)
            dipolemoment_xv = [
                self.calculate_dipole_moment(wfs)  # type: ignore
                for wfs in self.state.ibzwfs]
            dipolemoment_v = np.sum(dipolemoment_xv, axis=0)
            result = RTTDDFTResult(time=time, dipolemoment=dipolemoment_v)
            yield result

    def calculate_dipole_moment(self,
                                wfs: LCAOWaveFunctions) -> np.ndarray:
        """ Calculates the dipole moment

        The dipole moment is calculated as the expectation value of the
        dipole moment operator, i.e. the trace of it times the density matrix::

          d = - Σ  ρ   d
                μν  μν  νμ

        """
        if self.dm_operator_c is None:
            self.dm_operator_c = []

            # Create external potentials in each direction
            ext_c = [ConstantElectricField(Hartree / Bohr, dir)
                     for dir in np.eye(3)]
            dm_operator_c = [self.hamiltonian.create_kick_matrix_calculator(
                self.state, ext, self.pot_calc) for ext in ext_c]
            self.dm_operator_c = dm_operator_c

        dm_v = np.zeros(3)
        for c, dm_operator in enumerate(self.dm_operator_c):
            rho_MM = wfs.calculate_density_matrix()
            dm_MM = dm_operator.calculate_matrix(wfs)
            dm = - np.einsum('MN,NM->', rho_MM, dm_MM.data)
            assert np.abs(dm.imag) < 1e-20
            dm_v[c] = dm.real

        return dm_v
