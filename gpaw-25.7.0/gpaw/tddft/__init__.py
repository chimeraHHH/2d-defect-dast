"""This module implements a class for (true) time-dependent density
functional theory calculations.

"""
import time
import warnings
from math import log

import numpy as np

from gpaw.calculator import GPAW
from gpaw.mixer import DummyMixer
from gpaw.preconditioner import Preconditioner
from gpaw.tddft.units import (attosec_to_autime, autime_to_attosec,
                              aufrequency_to_eV)
from gpaw.tddft.utils import MultiBlas
from gpaw.tddft.solvers import create_solver
from gpaw.tddft.propagators import \
    create_propagator, \
    AbsorptionKick
from gpaw.tddft.tdopers import \
    TimeDependentHamiltonian, \
    TimeDependentOverlap, \
    TimeDependentWaveFunctions, \
    TimeDependentDensity, \
    AbsorptionKickHamiltonian
from gpaw.wavefunctions.fd import FD

from gpaw.tddft.spectrum import photoabsorption_spectrum
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.magneticmomentwriter import MagneticMomentWriter
from gpaw.lcaotddft.restartfilewriter import RestartFileWriter


__all__ = ['TDDFT', 'photoabsorption_spectrum',
           'DipoleMomentWriter', 'MagneticMomentWriter',
           'RestartFileWriter']


# T^-1
# Bad preconditioner
class KineticEnergyPreconditioner:
    def __init__(self, gd, kin, dtype):
        self.preconditioner = Preconditioner(gd, kin, dtype)
        self.preconditioner.allocate()

    def apply(self, kpt, psi, psin):
        for i in range(len(psi)):
            psin[i][:] = self.preconditioner(psi[i], kpt.phase_cd, None, None)


# S^-1
class InverseOverlapPreconditioner:
    """Preconditioner for TDDFT."""
    def __init__(self, overlap):
        self.overlap = overlap

    def apply(self, kpt, psi, psin):
        self.overlap.apply_inverse(psi, psin, kpt)
# ^^^^^^^^^^


class FDTDDFTMode(FD):
    def __call__(self, *args, **kwargs):
        reuse_wfs_method = kwargs.pop('reuse_wfs_method', None)
        assert reuse_wfs_method is None
        return TimeDependentWaveFunctions(self.nn, *args, **kwargs)


class TDDFT(GPAW):
    """Time-dependent density functional theory calculation based on GPAW.

    This class is the core class of the time-dependent density functional
    theory implementation and is the only class which a user has to use.
    """

    def __init__(self, filename: str, *,
                 td_potential: object = None,
                 calculate_energy: bool = True,
                 propagator: dict = None,
                 solver: dict = None,
                 tolerance: float = None,  # deprecated
                 parallel: dict = None,
                 communicator: object = None,
                 txt: str = '-'):
        """
        Parameters
        ----------
        filename
            File containing ground state or time-dependent state to propagate.
        td_potential
            Function class for the time-dependent potential. Must have a method
            ``strength(time)`` which returns the strength of
            the linear potential to each direction as a vector of three floats.
        calculate_energy
            Whether to calculate energy during propagation.
        propagator
            Time propagator for the Kohn-Sham wavefunctions.
        solver
            The iterative linear equations solver for propagator.
        tolerance
            Deprecated. Do not use this, but use solver dictionary instead.
            Tolerance for the linear solver.
        parallel
            Parallelization options
        communicator
            MPI communicator
        txt
            Text output
        """
        # Default values
        if propagator is None:
            propagator = dict(name='SICN')
        if solver is None:
            solver = dict(name='CSCG', tolerance=1e-8)

        assert filename is not None

        # For communicating with observers
        self.action = ''

        self.time = 0.0
        self.kick_strength = np.array([0.0, 0.0, 0.0], dtype=float)
        self.kick_gauge = ''
        self.niter = 0
        self.dm_file = None  # XXX remove and use observer instead

        # Parallelization dictionary should default to strided bands
        # but it does not work XXX
        # self.default_parallel = GPAW.default_parallel.copy()
        # self.default_parallel['stridebands'] = True

        self.default_parameters = GPAW.default_parameters.copy()
        self.default_parameters['mixer'] = DummyMixer()
        self.default_parameters['experimental']['reuse_wfs_method'] = None

        # NB: TDDFT restart files contain additional information which
        #     will override the initial settings for time/kick/niter.
        GPAW.__init__(self, filename, parallel=parallel,
                      communicator=communicator, txt=txt)
        if len(self.symmetry.op_scc) > 1:
            raise ValueError('Symmetries are not allowed for TDDFT. '
                             'Run the ground state calculation with '
                             'symmetry={"point_group": False}.')

        assert isinstance(self.wfs, TimeDependentWaveFunctions)
        assert isinstance(self.wfs.overlap, TimeDependentOverlap)

        self.set_positions()

        # Don't be too strict
        self.density.charge_eps = 1e-5

        self.rank = self.wfs.world.rank

        self.calculate_energy = calculate_energy
        if self.hamiltonian.xc.name.startswith('GLLB'):
            self.log('GLLB model potential. Not updating energy.')
            self.calculate_energy = False

        # Time-dependent variables and operators
        self.td_hamiltonian = TimeDependentHamiltonian(self.wfs, self.spos_ac,
                                                       self.hamiltonian,
                                                       td_potential)
        self.td_density = TimeDependentDensity(self)

        # Solver for linear equations
        if isinstance(solver, str):
            solver = dict(name=solver)
        if tolerance is not None:
            warnings.warn(
                "Please specify the solver tolerance using dictionary "
                "solver={'name': name, 'tolerance': tolerance}. "
                "Confirm the used tolerance from the output file. "
                "The old syntax will throw an error in the future.",
                FutureWarning)
            solver.update(tolerance=tolerance)
        self.solver = create_solver(solver)

        # Preconditioner
        # No preconditioner as none good found
        self.preconditioner = None  # TODO! check out SSOR preconditioning
        # self.preconditioner = InverseOverlapPreconditioner(self.overlap)
        # self.preconditioner = KineticEnergyPreconditioner(
        #    self.wfs.gd, self.td_hamiltonian.hamiltonian.kin, complex)

        # Time propagator
        if isinstance(propagator, str):
            propagator = dict(name=propagator)
        self.propagator = create_propagator(propagator)

        self.hpsit = None
        self.eps_tmp = None
        self.mblas = MultiBlas(self.wfs.gd)

        self.tddft_initialized = False

    def tddft_init(self):
        if self.tddft_initialized:
            return

        self.log('')
        self.log('')
        self.log('------------------------------------------')
        self.log('  Time-propagation TDDFT                  ')
        self.log('------------------------------------------')
        self.log('')

        self.log('Charge epsilon:', self.density.charge_eps)

        # Density mixer
        self.td_density.density.set_mixer(DummyMixer())

        # Solver
        self.solver.initialize(self.wfs.gd, self.timer)
        self.log('Solver:', self.solver.todict())

        # Preconditioner
        self.log('Preconditioner:', self.preconditioner)

        # Propagator
        self.propagator.initialize(self.td_density, self.td_hamiltonian,
                                   self.wfs.overlap, self.solver,
                                   self.preconditioner,
                                   self.wfs.gd, self.timer)
        self.log('Propagator:', self.propagator.todict())

        # Parallelization
        wfs = self.wfs
        if self.rank == 0:
            if wfs.kd.comm.size > 1:
                if wfs.nspins == 2:
                    self.log('Parallelization Over Spin')

                if wfs.gd.comm.size > 1:
                    self.log('Using Domain Decomposition: %d x %d x %d' %
                             tuple(wfs.gd.parsize_c))

                if wfs.bd.comm.size > 1:
                    self.log('Parallelization Over bands on %d Processors' %
                             wfs.bd.comm.size)
            self.log('States per processor =', wfs.bd.mynbands)

        # Restarting an FDTD run generates hamiltonian.fdtd_poisson, which
        # now overwrites hamiltonian.poisson
        if hasattr(self.hamiltonian, 'fdtd_poisson'):
            self.hamiltonian.poisson = self.hamiltonian.fdtd_poisson
            self.hamiltonian.poisson.set_grid_descriptor(self.density.finegd)

        # For electrodynamics mode
        if self.hamiltonian.poisson.get_description() == 'FDTD+TDDFT':
            self.hamiltonian.poisson.set_density(self.density)
            self.hamiltonian.poisson.print_messages(self.log)
            self.log.flush()

        # Update density and Hamiltonian
        self.propagator.update_time_dependent_operators(self.time)

        # XXX remove dipole moment handling and use observer instead
        self._dm_args0 = (self.density.finegd.integrate(self.density.rhot_g),
                          self.calculate_dipole_moment())

        # Call observers
        self.action = 'init'
        self.call_observers(self.niter)

        self.tddft_initialized = True

    def create_wave_functions(self, mode, *args, **kwargs):
        mode = FDTDDFTMode(mode.nn, mode.interpolation, True)
        GPAW.create_wave_functions(self, mode, *args, **kwargs)

    def read(self, filename):
        reader = GPAW.read(self, filename)
        if 'tddft' in reader:
            r = reader.tddft
            self.time = r.time
            self.niter = r.niter
            self.kick_strength = r.kick_strength

    def _write(self, writer, mode):
        GPAW._write(self, writer, mode)
        w = writer.child('tddft')
        w.write(time=self.time,
                niter=self.niter,
                kick_strength=self.kick_strength)

    def propagate(self, time_step: float, iterations: int,
                  dipole_moment_file: str = None,  # deprecated
                  restart_file: str = None,  # deprecated
                  dump_interval: int = None,  # deprecated
                  ):
        """Propagates wavefunctions.

        Parameters
        ----------
        time_step
            Time step in attoseconds (10^-18 s).
        iterations
            Number of iterations.
        dipole_moment_file
            Deprecated. Do not use this.
            Name of the data file where to the time-dependent dipole
            moment is saved.
        restart_file
            Deprecated. Do not use this.
            Name of the restart file.
        dump_interval
            Deprecated. Do not use this.
            After how many iterations restart data is dumped.
        """
        self.tddft_init()

        if self.propagator.todict()['name'] in ['EFSICN', 'EFSICN_HGH']:
            msg = ("You are using propagator for Ehrenfest dynamics. "
                   "Please use regular propagator.")
            raise RuntimeError(msg)

        def warn_deprecated(parameter, observer):
            warnings.warn(
                f"The {parameter} parameter is deprecated. "
                f"Please use {observer} observer instead. "
                "The old syntax will throw an error in the future.",
                FutureWarning)

        if dipole_moment_file is not None:
            warn_deprecated('dipole_moment_file', 'DipoleMomentWriter')

        if restart_file is not None:
            warn_deprecated('restart_file', 'RestartFileWriter')

        if dump_interval is not None:
            warn_deprecated('dump_interval', 'RestartFileWriter')
        else:
            dump_interval = 100

        if self.rank == 0:
            self.log()
            self.log('Starting time: %7.2f as'
                     % (self.time * autime_to_attosec))
            self.log('Time step:     %7.2f as' % time_step)
            header = """\
                        Simulation     Total         log10     Iterations:
             Time          time        Energy (eV)   Norm      Propagator"""
            self.log()
            self.log(header)

        # Convert to atomic units
        time_step = time_step * attosec_to_autime

        if dipole_moment_file is not None:
            self.initialize_dipole_moment_file(dipole_moment_file)

        # Set these as class properties for use of observers
        self.time_step = time_step
        self.dump_interval = dump_interval  # XXX remove, deprecated
        self.restart_file = restart_file  # XXX remove, deprecated

        niterpropagator = 0
        self.maxiter = self.niter + iterations

        # FDTD requires extra care
        if self.hamiltonian.poisson.get_description() == 'FDTD+TDDFT':
            self.hamiltonian.poisson.set_time(self.time)
            self.hamiltonian.poisson.set_time_step(self.time_step)

            # The propagate calculation_mode causes classical part to evolve
            # in time when self.hamiltonian.poisson.solve(...) is called
            self.hamiltonian.poisson.set_calculation_mode('propagate')

            # During each time step, self.hamiltonian.poisson.solve may be
            # called several times (depending on the used propagator).
            # Using the attached observer one ensures that actual propagation
            # takes place only once. This is because the FDTDPoissonSolver
            # changes the calculation_mode from propagate to
            # something else when the propagation is finished.
            self.attach(self.hamiltonian.poisson.set_calculation_mode, 1,
                        'propagate')

        self.timer.start('Propagate')
        while self.niter < self.maxiter:
            norm = self.density.finegd.integrate(self.density.rhot_g)

            # Write dipole moment at every iteration
            if dipole_moment_file is not None:
                if self._dm_args0 is not None:
                    self.update_dipole_moment_file(*self._dm_args0)
                    self._dm_args0 = None
                else:
                    dm = self.calculate_dipole_moment()
                    self.update_dipole_moment_file(norm, dm)

            # Propagate the Kohn-Shame wavefunctions a single timestep
            niterpropagator = self.propagator.propagate(self.time, time_step)
            self.time += time_step
            self.niter += 1

            # print output (energy etc.) every 10th iteration
            if self.niter % 10 == 0:
                self.get_td_energy()

                T = time.localtime()
                if self.rank == 0:
                    iter_text = 'iter: %3d  %02d:%02d:%02d %11.2f' \
                                '   %13.6f %9.1f %10d'
                    self.log(iter_text %
                             (self.niter, T[3], T[4], T[5],
                              self.time * autime_to_attosec,
                              self.Etot * aufrequency_to_eV,
                              log(abs(norm) + 1e-16) / log(10),
                              niterpropagator))

                    self.log.flush()

            # Call registered callback functions
            self.action = 'propagate'
            self.call_observers(self.niter)

            # Write restart data
            if restart_file is not None and self.niter % dump_interval == 0:
                self.write(restart_file, 'all')
                if self.rank == 0:
                    print('Wrote restart file.')
                    print(self.niter, ' iterations done. Current time is ',
                          self.time * autime_to_attosec, ' as.')

        self.timer.stop('Propagate')

        # Write final results and close dipole moment file
        if dipole_moment_file is not None:
            # TODO final iteration is propagated, but nothing is updated
            # norm = self.density.finegd.integrate(self.density.rhot_g)
            # self.finalize_dipole_moment_file(norm)
            self.finalize_dipole_moment_file()

        # Finalize FDTDPoissonSolver
        if self.hamiltonian.poisson.get_description() == 'FDTD+TDDFT':
            self.hamiltonian.poisson.finalize_propagation()

        if restart_file is not None:
            self.write(restart_file, 'all')

    def initialize_dipole_moment_file(self, dipole_moment_file):
        if self.rank == 0:
            if self.dm_file is not None and not self.dm_file.closed:
                raise RuntimeError('Dipole moment file is already open')

            if self.time == 0.0:
                mode = 'w'
            else:
                # We probably continue from restart
                mode = 'a'

            self.dm_file = open(dipole_moment_file, mode)

            # If the dipole moment file is empty, add a header
            if self.dm_file.tell() == 0:
                header = '# Kick = [%22.12le, %22.12le, %22.12le]\n' \
                    % (self.kick_strength[0], self.kick_strength[1],
                       self.kick_strength[2])
                header += '# %15s %15s %22s %22s %22s\n' \
                    % ('time', 'norm', 'dmx', 'dmy', 'dmz')
                self.dm_file.write(header)
                self.dm_file.flush()

    def calculate_dipole_moment(self):
        dm = self.density.finegd.calculate_dipole_moment(self.density.rhot_g)
        if self.hamiltonian.poisson.get_description() == 'FDTD+TDDFT':
            dm += self.hamiltonian.poisson.get_classical_dipole_moment()
        return dm

    def update_dipole_moment_file(self, norm, dm):
        if self.rank == 0:
            line = '%20.8lf %20.8le %22.12le %22.12le %22.12le\n' \
                % (self.time, norm, dm[0], dm[1], dm[2])
            self.dm_file.write(line)
            self.dm_file.flush()

    def finalize_dipole_moment_file(self, norm=None):
        if norm is not None:
            dm = self.calculate_dipole_moment()
            self.update_dipole_moment_file(norm, dm)

        if self.rank == 0:
            self.dm_file.close()
            self.dm_file = None

    def update_eigenvalues(self):

        kpt_u = self.wfs.kpt_u
        if self.hpsit is None:
            self.hpsit = self.wfs.gd.zeros(len(kpt_u[0].psit_nG),
                                           dtype=complex)
        if self.eps_tmp is None:
            self.eps_tmp = np.zeros(len(kpt_u[0].eps_n),
                                    dtype=complex)

        # self.Eband = sum_i <psi_i|H|psi_j>
        for kpt in kpt_u:
            self.td_hamiltonian.apply(kpt, kpt.psit_nG, self.hpsit,
                                      calculate_P_ani=False)
            self.mblas.multi_zdotc(self.eps_tmp, kpt.psit_nG,
                                   self.hpsit, len(kpt_u[0].psit_nG))
            self.eps_tmp *= self.wfs.gd.dv
            kpt.eps_n[:] = self.eps_tmp.real

        H = self.td_hamiltonian.hamiltonian

        # Nonlocal
        self.Enlkin = H.xc.get_kinetic_energy_correction()

        # PAW
        e_band = self.wfs.calculate_band_energy()
        self.Ekin = H.e_kinetic0 + e_band + self.Enlkin
        self.e_coulomb = H.e_coulomb
        self.Eext = H.e_external
        self.Ebar = H.e_zero
        self.Exc = H.e_xc
        self.Etot = self.Ekin + self.e_coulomb + self.Ebar + self.Exc

    def get_td_energy(self):
        """Calculate the time-dependent total energy"""
        self.tddft_init()

        if not self.calculate_energy:
            self.Etot = 0.0
            return 0.0

        self.wfs.overlap.update(self.wfs)
        self.td_density.update()
        self.td_hamiltonian.update(self.td_density.get_density(),
                                   self.time)
        self.update_eigenvalues()

        return self.Etot

    def set_absorbing_boundary(self, absorbing_boundary):
        self.td_hamiltonian.set_absorbing_boundary(absorbing_boundary)

    # exp(ip.r) psi
    def absorption_kick(self, kick_strength):
        """Delta absorption kick for photoabsorption spectrum.

        Parameters
        ----------
        kick_strength
            Strength of the kick in atomic units
        """
        self.tddft_init()

        if self.rank == 0:
            self.log('Delta kick =', kick_strength)

        self.kick_strength = np.array(kick_strength)

        abs_kick_hamiltonian = AbsorptionKickHamiltonian(
            self.wfs, self.spos_ac,
            np.array(kick_strength, float))
        abs_kick = AbsorptionKick(self.wfs, abs_kick_hamiltonian,
                                  self.wfs.overlap, self.solver,
                                  self.preconditioner, self.wfs.gd, self.timer)
        abs_kick.kick()

        # Kick the classical part, if it is present
        if self.hamiltonian.poisson.get_description() == 'FDTD+TDDFT':
            self.hamiltonian.poisson.set_kick(kick=self.kick_strength)

        # Update density and Hamiltonian
        self.propagator.update_time_dependent_operators(self.time)

        # Call observers after kick
        self.action = 'kick'
        self.call_observers(self.niter)
