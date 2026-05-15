"""
The solvated jellium method is contained in this module.
This enables electronically grand-canonical calculations to be calculated,
typically for simulating electrochemical interfaces.

This version the the solvated jellium method has been modified
and includes the Fluctuation Dissipation Theorem.
"""

import copy
import os
import textwrap

import ase.io
import gpaw.mpi
import numpy as np
from ase.calculators.calculator import (InputError, Parameters,
                                        PropertyNotPresent, equal)
from ase.parallel import paropen
from ase.units import Bohr, Ha, _e, kB
from gpaw import GPAW_NEW, ConvergenceError
from gpaw.dipole_correction import DipoleCorrection
from gpaw.fd_operators import Gradient
from gpaw.hamiltonian import RealSpaceHamiltonian
from gpaw.io.logger import indent
from gpaw.jellium import Jellium, JelliumSlab
from gpaw.solvation.calculator import OldSolvationGPAW
from gpaw.solvation.cavity import Power12Potential, get_pbc_positions
from gpaw.solvation.hamiltonian import SolvationRealSpaceHamiltonian
from gpaw.solvation.poisson import WeightedFDPoissonSolver
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy.stats import linregress


def SJM(*args, **kwargs):
    """Backwards compatibility ..."""
    if GPAW_NEW:
        from gpaw.new.ase_interface import GPAW
        from gpaw.new.sjm import SJM
        environment = SJM(cavity=kwargs.pop('cavity'),
                          dielectric=kwargs.pop('dielectric'),
                          interactions=kwargs.pop('interactions', None),
                          **kwargs.pop('sj'))
        return GPAW(
            *args, **kwargs,
            environment=environment)
    return OldSJM(*args, **kwargs)


class OldSJM(OldSolvationGPAW):
    r"""Solvated Jellium method.
    (Implemented as a subclass of the SolvationGPAW class.)

    The method allows the simulation of an electrochemical environment, where
    the potential can be varied by changing the charging (that is, number of
    electrons) of the system. For this purpose, it allows the usagelof non-
    neutral periodic slab systems. Cell neutrality is achieved by adding a
    background charge in the solvent region above the slab

    Further details are given in https://doi.org/10.1021/acs.jpcc.8b02465
    If you use this method, we appreciate it if you cite that work.

    The method can be run in three modes:

        - Constant charge: The number of excess electrons in the simulation
          can be directly specified with the 'excess_electrons' keyword,
          leaving 'target_potential' set to None.
        - Constant potential: The target potential (expressed as a work
          function) can be specified with the 'target_potential' keyword.
          Optionally, the 'excess_electrons' keyword can be supplied to specify
          the initial guess of the number of electrons.
        - Constant inner potential: The target potential (expressed as a inner
          potential) can be specified with the method='CIP' and
          'target_potential' keywords. The CIP-DFT method is detailled in
          https://doi.org/10.1038/s41524-023-01184-4

    By default, this method writes the grand-potential energy to the output;
    that is, the energy that has been adjusted with `- \mu N` (in this case,
    `\mu` is the work function and `N` is the excess electrons). This is the
    energy that is compatible with the forces in constant-potential mode and
    thus will behave well with optimizers, NEBs, etc. It is also frequently
    used in subsequent free-energy calculations.

    Within this method, the potential is expressed as the top-side work
    function of the slab or the inner potential of the electrode.
    In both cases, a potential of 0 V_SHE corresponds to target_potential of
    roughly 4.4 eV. (That is, the user should specify
    target_potential as 4.4 in this case.) Because this method is
    attempting to bring either the work function or the inner potential to a
    target value, the work function and electrostatics need to be
    well-converged. For this reason, the 'work function' keyword is
    automatically added to the SCF convergence dictionary with a value of
    0.001. This can be overriden by the user.

    All methods requires a dipole correction, and this is turned on
    automatically, but can be overridden with the poissonsolver keyword.

    When using method='CIP', mixed Dirichlet/Neumann boundary conditions
    for the electrostatic potential are used by default.

    The SJM class takes a single argument, the sj dictionary. All other
    arguments are fed to the parent SolvationGPAW (and therefore GPAW)
    calculator.

    Parameters:

    sj: dict
        Dictionary of parameters for the solvated jellium method, whose
        possible keys are given below.

    Parameters in sj dictionary:

    excess_electrons: float
        Number of electrons added in the atomic system and (with opposite
        sign) in the background charge region. If the 'target_potential'
        keyword is also supplied, 'excess_electrons' is taken as an initial
        guess for the needed number of electrons.
    target_potential: float
        The potential that should be reached or kept in the course of the
        calculation. If set to 'None' (default) a constant-charge
        calculation based on the value of 'excess_electrons' is performed.
        Expressed as a work function, on the top side of the slab; see note
        above.
    tol: float
        Tolerance for the deviation of the target potential.
        If the potential is outside the defined range 'ne' will be
        changed in order to get inside again. Default: 0.01 V.
    max_iters: int
        In constant-potential mode, the maximum number of iterations to try
        to equilibrate the potential (by varying ne). Default: 10.
    jelliumregion: dict
        Parameters regarding the shape of the counter charge region.
        Implemented keys:

        'top': float or None
            Upper boundary of the counter charge region (z coordinate).
            A positive float is taken as the upper boundary coordiate.
            A negative float is taken as the distance below the uppper
            cell boundary.  Default: -1.
        'bottom': float or 'cavity_like' or None
            Lower boundary of the counter-charge region (z coordinate).
            A positive float is taken as the lower boundary coordinate.
            A negative float is interpreted as the distance above the highest
            z coordinate of any atom; this can be fixed by setting
            fix_bottom to True. If 'cavity_like' is given the counter
            charge will take the form of the cavity up to the 'top'.
            Default: -3.
        'thickness': float or None
            Thickness of the counter charge region in Angstroms.
            Can only be used if start is not 'cavity_like'. Default: None
        'fix_bottom': bool
            Do not allow the lower limit of the jellium region to adapt
            to a change in the atomic geometry during a relaxation.
            Omitted in a 'cavity_like' jelliumregion.
            Default: False
    grand_output: bool
        Write the grand-potential energy into output files such as
        trajectory files. Default: True
    always_adjust: bool
        Adjust ne again even when potential is within tolerance.
        This is useful to set to True along with a loose potential
        tolerance (tol) to allow the potential and structure to be
        simultaneously optimized in a geometry optimization, for example.
        Default: False.
    slope : float or None
        Initial guess of the slope, in volts per electron, of the relationship
        between the electrode potential and excess electrons, used to
        equilibrate the potential.
        This only applies to the first step, as the slope is calculated
        internally at subsequent steps. If None, will be guessed based on
        apparent capacitance of 10 uF/cm^2.
    max_step : float
        When equilibrating the potential, if a step results in the
        potential being further from the target and changing by more than
        'max_step' V, then the step size is cut in half and we try again. This
        is to avoid leaving the linear region. You can set to np.inf to turn
        this off. Default: 2 V.
    mixer : float
        Damping for the slope estimate. Because estimating slopes can
        sometimes be noisy (particularly for small changes, or when
        positions have also changed), some fraction of the previous slope
        estimate can be "mixed" with the current slope estimate before the
        new slope is established. E.g., new_slope = mixer * old_slope +
        (1. - mixer) * current_slope_estimate. Set to 0 for no damping.
        Default: 0.5.
    fdt : bool or dict
        Keyboard for switching on/off the computation of the charge using
        the fluctuation dissipation theorem
        Default: False.
        If set to True, use standard parameters for the FDT calculation.
        If dict, the following keys are implemented:

        'dt': float
            Time step for the FDT calculation in fs. Default: 0.5.
        'po_time': float
            Relaxation-time constant of potentiostat. Default: 100.
        'th_temp': float
            Thermal temperature for the FDT calculation in K. Default: 300.

    slope_regression_depth : int
        Number of previous attempts to use for the slope regression.
        Default: 4.
    pot_ref: 'wf' or 'CIP'
        potential reference scale
        wf: original SJM using workfunction for the absolute potential
        CIP: use inner potential as the abs. el. pot.
    cip: dict, parameters when using CIP-DFT
        inner_region: list of floats: [bottom, top]
             bottom is the starting point for computing the inner potential,
             likewise top is the ending point.
        filter: int
            number of points for smoothing the el.stat. pot.
        autoinner: dict with {'nlayers':int, 'threshold':0.001}
            if inner_region is given, autoinner is automatically disabled
            nlayers: number of layers
            threshold: Required threshold of peaks, the innerpotential
            difference at neighboring grid points
        mu_pzc: float
            Fermi level at potential of zero charge
            Sets the reference scale for the absolute potential level for CIP
        phi_pzc: float
          Use only with method='CIP', the value corresponds to the inner
          potential of the neutral electrode

    Special SJM methods (in addition to those of GPAW/SolvationGPAW):

    get_electrode_potential
        Returns the potential of the simulated electrode, in V, relative
        to the vacuum.

    write_sjm_traces
        Write traces of quantities like electrostatic potential or cavity
        to disk.

    """
    implemented_properties = ['energy', 'forces', 'stress', 'dipole',
                              'magmom', 'magmoms', 'excess_electrons',
                              'electrode_potential']
    _sj_default_parameters = Parameters(
        {'excess_electrons': 0.,
         'jelliumregion': {'top': -1.,
                           'bottom': -3.,
                           'thickness': None,
                           'fix_bottom': False},
         'target_potential': None,
         'pot_ref': 'wf',
         'tol': 0.01,
         'always_adjust': False,
         'grand_output': True,
         'max_iters': 10,
         'max_step': 2.,
         'slope': None,
         'mixer': 0.5,
         'fdt': False,
         'previous_electrons': [],
         'previous_potentials': [],
         'slope_regression_depth': 4,
         'cip': {'autoinner': {'nlayers': None,
                               'threshold': 0.0001},
                 'inner_region': None,
                 'mu_pzc': None,
                 'phi_pzc': None,
                 'filter': 10}})

    _sj_default_parameters.update({'dirichlet': False})
    default_parameters = copy.deepcopy(OldSolvationGPAW.default_parameters)
    default_parameters.update({'poissonsolver': {'dipolelayer': 'xy'}})
    default_parameters['convergence'].update({'work function': 0.001})
    default_parameters.update({'sj': _sj_default_parameters})

    def __init__(self, restart=None, **kwargs):

        deprecated_keys = ['ne', 'potential', 'write_grandcanonical_energy',
                           'potential_equilibration_mode', 'dpot',
                           'max_pot_deviation', 'doublelayer', 'verbose']
        msg = ('{:s} is no longer a supported keyword argument for the SJM '
               'class. All SJM arguments should be sent in via the "sj" '
               'dict.')
        for key in deprecated_keys:
            if key in kwargs:
                raise InputError(textwrap.fill(msg.format(key)))

        # Note the below line calls self.set().
        OldSolvationGPAW.__init__(self, restart, **kwargs)

    def set(self, **kwargs):
        """Change parameters for calculator.

        It differs from the standard `set` function in two ways:
        - Keywords in the `sj` dictionary are handled.
        - It does not reinitialize and delete `self.wfs` if only the
          background charge is changed.

        """
        p = self.parameters['sj']

        # We handle 'sj' and 'background_charge' internally; passing to GPAW's
        # set function will trigger a deletion of the density, etc.
        sj_changes = kwargs.pop('sj', {})
        try:
            sj_changes = {key: value for key, value in sj_changes.items()
                          if not equal(value, p[key])}
        except KeyError:
            raise InputError(
                'Unexpected key(s) provided to sj dict. '
                'Keys provided were "{}". '
                'Only keys allowed are "{}".'
                .format(', '.join(sj_changes),
                        ', '.join(self.default_parameters['sj'])))
        self.fill_cip_keywords(p.cip)

        if p.pot_ref == 'CIP':
            p['dirichlet'] = True

            if p.cip['mu_pzc'] is None or p.cip['phi_pzc'] is None:
                p.cip['mu_pzc'] = 0
                p.cip['phi_pzc'] = 0
                msg = ('Warning: a CIP calculation has been activated '
                       'but mu_pzc and/or phi_pzc was none. This is fine '
                       'for CIP calibration but meaningful references '
                       'must be provided for production calculations\n')

            if p.cip['inner_region'] is None and p.cip['autoinner'] is None:
                raise RuntimeError("The inner region cannot be none" +
                                   "when using inner potential as the" +
                                   "reference. Please, set up" +
                                   "either bottom/top values to define the" +
                                   "electrode bulk or set autoinner to True")

            if p.cip.get('inner_region') and p.cip.get('autoinner'):
                raise RuntimeError("Only inner_region or autoinner" +
                                   "can be set to define the inner potential")

            if p.cip['inner_region'] is not None:
                p.cip['inner_region'] = np.array(p.inner_region)
                p.cip['autoinner'] = None
            else:
                assert p.cip['autoinner']['nlayers'] is not None

        p.update(sj_changes)

        background_charge = kwargs.pop('background_charge', None)
        kwargs['_set_ok'] = True
        OldSolvationGPAW.set(self, **kwargs)

        # parent_changed checks if GPAW needs to be reinitialized
        # The following key do not need reinitialization
        parent_changed = False
        for key in kwargs:
            if key not in ['mixer', 'verbose', 'txt', 'hund', 'random',
                           'eigensolver', 'convergence', 'fixdensity',
                           'maxiter', '_set_ok']:
                parent_changed = True

        if len(sj_changes):
            if self.wfs is None:
                self.log('Non-default Solvated Jellium parameters:')
            else:
                self.log('Changed Solvated Jellium parameters:')
            self.log.print_dict({i: p[i] for i in sj_changes})
            self.log()

        if 'target_potential' in sj_changes and p.target_potential is not None:
            # If target potential is changed by the user and the slope is
            # known, a step towards the new potential is taken right away.
            try:
                true_potential = self.get_electrode_potential()
            # TypeError is needed for the case of starting from a gpw
            # file and changing the target potential at the start.
            except (AttributeError, TypeError):
                pass
            else:
                if self.atoms and p.slope:

                    p.excess_electrons += ((p.target_potential -
                                            true_potential) / p.slope)
                    self.log('Number of electrons changed to {:.4f} based '
                             'on slope of {:.4f} V/electron.'
                             .format(p.excess_electrons, p.slope))

        if (any(key in ['target_potential', 'excess_electrons',
            'jelliumregion'] for key in sj_changes) and
                not parent_changed):
            self.results = {}
            # SolvationGPAW will not reinitialize anymore if only
            # 'sj' keywords are set. The lines below will reinitialize and
            # apply the changed charges.
            if self.atoms:
                self.set(background_charge=self._create_jellium())

        if 'tol' in sj_changes:
            try:
                true_potential = self.get_electrode_potential()
            except (AttributeError, TypeError):
                pass
            else:
                msg = ('Potential tolerance changed to {:1.4f} V: '
                       .format(p.tol))
                if abs(true_potential - p.target_potential) > p.tol:
                    msg += 'new calculation required.'
                    self.results = {}
                    if self.atoms and p.slope is not None:
                        p.excess_electrons += ((p.target_potential -
                                               true_potential) / p.slope)
                        self.set(background_charge=self._create_jellium())
                        msg += ('\n Excess electrons changed to {:.4f} based '
                                'on slope of {:.4f} V/electron.'
                                .format(p.excess_electrons, p.slope))
                else:
                    msg += 'already within tolerance.'
                self.log(msg)

        if background_charge:
            # background_charge is a GPAW parameter that we handle internally,
            # as it contains the jellium countercharge. Note if a user tries to
            # specify an *additional* background charge this will probably
            # conflict, but we know of no such use cases.
            if self.wfs is None:
                kwargs.update({'background_charge': background_charge,
                               '_set_ok': True})
                OldSolvationGPAW.set(self, **kwargs)
            else:
                if parent_changed:
                    self.density = None
                else:
                    if self.density.background_charge:
                        self.density.background_charge = background_charge
                        self.density.background_charge.set_grid_descriptor(
                            self.density.finegd)
                    self._quick_reinitialization()

                self.wfs.nvalence = self.setups.nvalence + p.excess_electrons
                self.log('Number of valence electrons is now {:.5f}'
                         .format(self.wfs.nvalence))

        if self.parameters['sj']['fdt'] is True:
            # Default parameters if fdt is True
            self.parameters['sj']['fdt'] = {
                'dt': 0.5,
                'po_time': 100.0,
                'th_temp': 300.0}
        elif isinstance(self.parameters['sj']['fdt'], dict):
            # If fdt is a dict, ensure the dictionary is complete
            fdt_dict = self.parameters['sj']['fdt']
            self.parameters['sj']['fdt'] = {
                'dt': fdt_dict.get('dt', 0.5),
                'po_time': fdt_dict.get('po_time', 100.0),
                'th_temp': fdt_dict.get('th_temp', 300.0)}

    def _quick_reinitialization(self):
        """Minimal reinitialization of electronic-structure stuff when only
        background charge changes."""
        if self.density.nct_G is None:
            self.initialize_positions()

        self.density.reset()
        self._set_atoms(self.atoms)
        self.density.mixer.reset()
        self.wfs.initialize(self.density, self.hamiltonian,
                            self.spos_ac)
        self.wfs.eigensolver.reset()
        if self.scf:
            self.scf.reset()

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['cell']):
        """Perform an electronic structure calculation, with either a
        constant number of electrons or a target potential, as requested by
        the user in the 'sj' dict."""

        if atoms and not self.atoms:
            # Need to be set before ASE's Calculator.calculate gets to it.
            self.atoms = atoms.copy()

        if len(system_changes) == 0 and len(self.results) > 0:
            # Potential is already equilibrated.
            OldSolvationGPAW.calculate(self, atoms, properties, system_changes)
            return

        self.log('Solvated jellium method (SJM) calculation:')

        p = self.parameters['sj']

        if p.target_potential is None:
            self.log('Constant-charge calculation with {:.5f} excess '
                     'electrons'.format(p.excess_electrons))
            # Background charge is set here, not earlier, because atoms needed.
            self.set(background_charge=self._create_jellium())
            OldSolvationGPAW.calculate(self, atoms, ['energy'], system_changes)
            self.log('Potential found to be {:.5f} V (with {:+.5f} '
                     'electrons)'.format(self.get_electrode_potential(),
                                         p.excess_electrons))
        else:
            self.log(' Constant-potential mode.')
            self.log(' Target potential: {:.5f} +/- {:.5f}'
                     .format(p.target_potential, p.tol))
            self.log(' Initial guess of excess electrons: {:.5f}'
                     .format(p.excess_electrons))
            if 'workfunction' in self.parameters.convergence:
                if self.parameters.convergence['workfunction'] >= p.tol:
                    msg = ('Warning: it appears that your work function '
                           'convergence criterion ({:g}) is higher than your '
                           'desired potential tolerance ({:g}). This may lead '
                           'to issues with potential convergence.'
                           .format(self.parameters.convergence['workfunction'],
                                   p.tol))
                    self.log(textwrap.fill(msg))
            self._equilibrate_potential(atoms, system_changes)
        if properties != ['energy']:
            # The equilibration loop only calculated energy, to save
            # unnecessary computations (mostly of forces) in the loop.
            OldSolvationGPAW.calculate(self, atoms, properties, [])

        # Note that grand-potential energies were assembled in summary,
        # which in turn was called by GPAW.calculate.

        if p.grand_output:
            self.results['energy'] = self.omega_extrapolated * Ha
            self.results['free_energy'] = self.omega_free * Ha
            self.log('Grand-potential energy was written into results.\n')
        else:
            self.log('Canonical energy was written into results.\n')

        self.results['excess_electrons'] = p.excess_electrons
        self.results['electrode_potential'] = self.get_electrode_potential()
        self.log.fd.flush()

    def _equilibrate_potential(self, atoms, system_changes):
        """Adjusts the number of electrons until the potential reaches the
        desired value."""
        p = self.parameters['sj']
        iteration = 0

        rerun = False
        while iteration <= p.max_iters:
            self.log('Attempt {:d} to equilibrate potential to {:.3f} +/-'
                     ' {:.3f} V'
                     .format(iteration, p.target_potential, p.tol))
            self.log('Current guess of excess electrons: {:+.5f}\n'
                     .format(p.excess_electrons))
            if iteration == 1:
                self.timer.start('Potential equilibration loop')
                # We don't want SolvationGPAW to see any more system
                # changes, like positions, after attempt 0.
                system_changes = []

            if any([iteration, rerun, 'positions' in system_changes]):
                self.set(background_charge=self._create_jellium())

            # Do the calculation.
            OldSolvationGPAW.calculate(self, atoms, ['energy'], system_changes)
            true_potential = self.get_electrode_potential()
            self.log()
            msg = (f'Potential found to be {true_potential:.5f} V (with '
                   f'{p.excess_electrons:+.5f} excess electrons, attempt '
                   f'{iteration:d}/{p.max_iters:d}')
            msg += ' rerun).' if rerun else ').'
            self.log(msg, flush=True)

            # Check if we took too big of a step, that moved us further
            # from the target potential rather than closer. This can
            # happen when large geometric changes happened in the
            # last step, the slope is noisy or, at very low
            # workfunctions, where the electron density starts to spill out
            # into the vacuum and the slope is unreliable.
            # The rerun can happen multiple times if needed and the stepsize
            # will be reduced by factor of 2 every time.
            # The rerun is disabled if the FDT is used.

            if len(p.previous_potentials):

                stepsize = abs(true_potential - p.previous_potentials[-1])

                if (stepsize > p.max_step and
                   abs(p.previous_potentials[-1] - p.target_potential) <
                   abs(true_potential - p.target_potential)) and not p.fdt:
                    self.log('Step resulted in a potential change of '
                             f'{stepsize:.2f} V, larger than max_step '
                             f'({p.max_step:.2f} V) and\n surpassed the'
                             ' target potential by a dangerous amount.\n'
                             ' The step is rejected and the change in'
                             ' excess_electrons will be halved.')

                    if p.fdt:
                        rerun = False
                    else:
                        pe, ce = p.previous_electrons[-1], p.excess_electrons
                        if abs(pe - ce) < 1e-5:
                            msg = ('Step size is too small to be halved in '
                                   'rerun. To avoid this try to change your '
                                   'initial guess of excess electrons. '
                                   'Potential equilibration failed.')
                            raise PotentialConvergenceError(msg)

                        p.excess_electrons = (pe + ce) / 2.

                        rerun = True
                    continue  # back to while

            # Increase iteration count.
            iteration += 1
            rerun = False

            # Store attempt and calculate slope.
            p.previous_electrons.append(float(p.excess_electrons))
            p.previous_potentials.append(float(true_potential))

            # The following solves a bug, where the code would crash if the
            # user sets the right number of electrons to reach the target
            # potential in the first iteration and then changes the target
            # potential. The code would crash because the slope has not been
            # calculated yet and so no step is taken towards the new potential.
            # As two equal charges are added to p.previous_electrons, the
            # regression of the slope will fail.
            if len(p.previous_electrons) > 1:
                if not p.previous_electrons[-2] - p.previous_electrons[-1]:
                    del p.previous_electrons[-2], p.previous_potentials[-2]

            if len(p.previous_electrons) > 1:
                slope = _calculate_slope(p.previous_electrons,
                                         p.previous_potentials,
                                         p.slope_regression_depth)

                nreg = len(p.previous_electrons[-p.slope_regression_depth:])
                self.log(f'Slope regressed from last {nreg:d} attempts is '
                         f'{slope:.4f} V/electron,')
                area = np.linalg.det(atoms.cell[:2, :2])
                # get capacitance in muF/cm^2
                capacitance = - _e * 1e22 / (area * slope)
                self.log(f'or apparent capacitance of {capacitance:.4f} '
                         'muF/cm^2')

                if p.slope is not None:
                    p.slope = p.mixer * p.slope + (1. - p.mixer) * slope
                    self.log(f'After mixing with {p.mixer:.2f}, new slope is '
                             f'{p.slope:.4f} V/electron.')
                else:
                    p.slope = slope

                self.log.flush()

            # Check if we're equilibrated and exit if always_adjust is False.
            if abs(true_potential - p.target_potential) < p.tol and not p.fdt:
                self.log('Potential is within tolerance. Equilibrated.')
                if iteration >= 2:
                    self.timer.stop('Potential equilibration loop')
                if not p.always_adjust:
                    return

            # Guess slope if we don't have enough information yet.
            if p.slope is None or (p.slope > 0. and p.fdt):
                area = np.linalg.det(atoms.cell[:2, :2])
                p.slope = -1.6022e3 / (area * 10.)
                if p.fdt:
                    self.log('Positive slope! Guessing a slope of '
                             f'{p.slope:.4f} corresponding\nto an apparent '
                             'capacitance of 10 muF/cm^2.')
                else:
                    self.log('No slope provided, guessing a slope of '
                             f'{p.slope:.4f} corresponding\nto an apparent '
                             'capacitance of 10 muF/cm^2.')

            if p.fdt:
                fdt_dict = p['fdt']
                dt = fdt_dict['dt']
                po_time = fdt_dict['po_time']
                th_temp = fdt_dict['th_temp']

                rn = np.random.standard_normal(1)

                self.world.broadcast(rn, 0)
                # set capacitance again
                area = np.linalg.det(atoms.cell[:2, :2])
                if abs(p.slope) < 1e-10:
                    raise ValueError(
                        "Slope cannot be zero when calculating capacitance.")

                capacitance = - _e * 1e22 / (area * p.slope)

                p.excess_electrons += (
                    capacitance * (true_potential - p.target_potential)
                    * (1 - np.exp(-dt / po_time))
                    + rn[0] * np.sqrt(
                        kB * th_temp * capacitance
                            * (1 - np.exp(-2 * dt / po_time))
                    )
                )
                self.log(
                    f'Number of electrons is {p.excess_electrons:.4f} '
                    f'using the FDT, with slope of {p.slope:.4f} V/electron '
                    f'and capacitance of ({capacitance:.4f} muF/cm2).'
                )

            else:
                p.excess_electrons += ((p.target_potential - true_potential)
                                       / p.slope)
                self.log(
                    f'Number of electrons changed to {p.excess_electrons:.4f}'
                    f' based on slope of {p.slope:.4f} V/electron.')

            # Check if we're equilibrated and exit if always_adjust is True.
            if (abs(true_potential - p.target_potential) < p.tol
                    and p.always_adjust) or p.fdt:
                return

        msg = (f'Potential could not be reached after {iteration - 1:d} '
               'iterations. This may indicate your workfunction is noisier '
               'than your potential tol. You may try setting the '
               'convergence["workfunction"] keyword. The last values of '
               'excess_electrons and the potential are listed below; '
               'plotting them could give you insight into the problem.')
        msg = textwrap.fill(msg) + '\n'
        for n, p in zip(p.previous_electrons, p.previous_potentials):
            msg += f'{n:+.6f} {p:.6f}\n'
        self.log(msg, flush=True)
        raise PotentialConvergenceError(msg)

    def write_sjm_traces(self, path='sjm_traces', style='z',
                         props=('potential', 'cavity', 'background_charge')):
        """Write traces of quantities in `props` to file on disk; traces will
        be stored within specified path. Default is to save as vertical traces
        (style 'z'), but can also save as cube (specify `style='cube'`)."""
        grid = self.density.finegd
        data = {'cavity': self.hamiltonian.cavity.g_g,
                'background_charge': self.density.background_charge.mask_g,
                'potential': (self.hamiltonian.vHt_g * Ha -
                              self.get_fermi_level())}
        if not os.path.exists(path) and gpaw.mpi.world.rank == 0:
            os.makedirs(path)
        for prop in props:
            if style == 'z':
                _write_trace_in_z(grid, data[prop], prop + '.txt', path)
            elif style == 'cube':
                _write_property_on_grid(grid, data[prop], self.atoms,
                                        prop + '.cube', path)

    def summary(self):
        """Writes summary information to the log file.
        This varies from the implementation in gpaw.calculator.GPAW by the
        inclusion of grand potential quantities."""
        # Standard GPAW summary.
        self.hamiltonian.summary(self.wfs, self.log)
        # Add grand-canonical terms.
        p = self.parameters['sj']
        self.log()
        mu_N = -self.get_electrode_potential()
        mu_N *= p.excess_electrons / Ha
        self.omega_free = self.hamiltonian.e_total_free - mu_N
        self.omega_extrapolated = self.hamiltonian.e_total_extrapolated - mu_N
        self.log('Legendre-transformed energies (grand potential, '
                 'Omega = E - N mu)')
        self.log(' N (excess electrons):  {:+11.6f}'
                 .format(p.excess_electrons))
        if p['pot_ref'] == 'wf':
            self.log(' mu (-workfunction, eV): {:+11.6f}'
                     .format(-self.get_electrode_potential()))
        elif p['pot_ref'] == 'CIP':
            self.log('Electrode potential from inner potential: {:+11.6f} [eV]'
                     .format(self.get_electrode_potential()))
            self.log('The absolute inner potential is: {:+11.6f} [eV]'
                     .format(self.get_inner_potential(self.atoms,
                             p['cip']['inner_region'])))

        self.log(' (Grand) free energy:   {:+11.6f}'
                 .format(Ha * self.omega_free))
        self.log(' (Grand) extrapolated:  {:+11.6f}'
                 .format(Ha * self.omega_extrapolated))
        self.log()
        # Back to standard GPAW summary.
        self.density.summary(self.atoms, self.results.get('magmom', 0.0),
                             self.log)
        self.wfs.summary(self.log)
        self._print_gapinfo()
        self.log.fd.flush()

    def _create_jellium(self):
        """Creates the counter charge according to the user's specs."""
        atoms = self.atoms

        p = self.parameters['sj']
        jellium = p['jelliumregion']
        defaults = self.default_parameters['sj']['jelliumregion']

        # Populate missing keywords.
        missing = {'top': None,
                   'bottom': None,
                   'thickness': None,
                   'fix_bottom': False}
        for key in missing:
            if key not in jellium:
                jellium[key] = missing[key]

        # Catch incompatible specifications.
        if jellium.get('thickness') and jellium['bottom'] == 'cavity_like':
            raise InputError("With a cavity-like counter charge only the "
                             "keyword 'top' (not 'thickness') allowed.")

        # We need 2 of the 3 "specifieds" below.
        specifieds = [jellium['top'] is not None,
                      jellium['bottom'] is not None,
                      jellium['thickness'] is not None]

        if sum(specifieds) == 3:
            raise InputError('The jellium region has been overspecified.')
        if sum(specifieds) == 0:
            top = defaults['top']
            bottom = defaults['bottom']
            thickness = defaults['thickness']
        if sum(specifieds) == 2:
            top = jellium['top']
            bottom = jellium['bottom']
            thickness = jellium['thickness']
        if specifieds == [True, False, False]:
            top = jellium['top']
            bottom = defaults['bottom']
            thickness = None
        elif specifieds == [False, True, False]:
            top = defaults['top']
            bottom = jellium['bottom']
            thickness = None
        elif specifieds == [False, False, True]:
            top = None
            bottom = defaults['bottom']
            thickness = jellium['thickness']

        # Deal with negative specifications of upper and lower limits;
        # as well as fixed/free lower limit.
        if top is not None and top < 0.:
            top = atoms.cell[2][2] + top
        if bottom not in [None, 'cavity_like'] and bottom < 0.:
            bottom = (max(atoms.positions[:, 2]) - bottom)
            if jellium['fix_bottom'] is True:
                try:
                    bottom = self._fixed_lower_limit
                except AttributeError:
                    self._fixed_lower_limit = bottom

        # Use thickness if needed.
        if top is None:
            top = bottom + thickness
        if bottom is None:
            bottom = top - thickness

        # Catch unphysical limits.
        if top > atoms.cell[2][2]:
            raise InputError('The upper limit of the jellium region lies '
                             'outside of unit cell. If you did not set it '
                             'manually, increase your unit cell size or '
                             'translate the atomic system down along the '
                             'z-axis.')
        if bottom != 'cavity_like':
            if bottom > top:
                raise InputError('Your jellium region has a bottom at {:.3f}'
                                 ' AA, which is above the top at {:.3f} AA.'
                                 .format(bottom, top))

        # Finally, make the jellium.
        if bottom == 'cavity_like':
            if self.hamiltonian is None:
                self.initialize(atoms)
                self.set_positions(atoms)
                g_g = self.hamiltonian.cavity.g_g.copy()
                self.wfs = None
                self.density = None
                self.hamiltonian = None
                self.initialized = False
            else:
                self.set_positions(atoms)
                # XXX If you start with a fixed bottom, run a calc,
                # then hot-switch to cavity_like it crashes here. Not sure
                # that's common enough to worry about.
                g_g = self.hamiltonian.cavity.g_g
            self.log('Jellium counter-charge defined with:\n'
                     ' Bottom: cavity-like\n'
                     ' Top: {:7.3f} AA\n'
                     ' Charge: {:5.4f}\n'
                     .format(top, p.excess_electrons))
            return CavityShapedJellium(charge=p.excess_electrons,
                                       g_g=g_g,
                                       z2=top)

        self.log('Jellium counter-charge defined with:\n'
                 ' Bottom: {:7.3f} AA\n'
                 ' Top: {:7.3f} AA\n'
                 ' Charge: {:5.4f}\n'
                 .format(bottom, top, p.excess_electrons))
        return JelliumSlab(charge=p.excess_electrons,
                           z1=bottom,
                           z2=top)

    def get_electrode_potential(self, pot_ref=None,
                                return_referenced=True):
        """Returns the potential of the simulated electrode, in V, relative
        to the vacuum. This comes directly from the work function."""

        if pot_ref is None:
            pot_ref = self.parameters.sj['pot_ref']

        if pot_ref == 'wf':
            try:
                return Ha * self.hamiltonian.get_workfunctions(self.wfs)[1]
            except TypeError:
                # Error happens on freshly-opened *.gpw file.
                if 'electrode_potential' in self.results:
                    return self.results['electrode_potential']
                else:
                    msg = ('Electrode potential could not be read. Make sure a'
                           'DFT calculation has been performed before reading '
                           'the potential.')
                    raise PropertyNotPresent(textwrap.fill(msg))

        elif pot_ref == 'CIP':
            cip = self.parameters.sj.cip
            inner_potential = self.get_inner_potential(self.atoms)
            if return_referenced is True:
                inner_potential = - (cip['mu_pzc'] - cip['phi_pzc'] +
                                     inner_potential)
                return inner_potential

            return cip['phi_pzc'] - inner_potential

    def get_inner_potential(self, atoms, z=None):

        cip = self.parameters.sj.cip
        self.fill_cip_keywords(cip)

        el = self.hamiltonian.vHt_g * Ha
        gd = self.hamiltonian.finegd
        elstat = gd.collect(el, broadcast=True)
        el_stat_z = elstat.mean(0).mean(0)
        smooth_elstat_z = uniform_filter1d(el_stat_z, size=cip['filter'])

        if cip['inner_region'] is not None:
            z_top = (np.abs(gd.coords(2) - (z[1] / Bohr))).argmin()
            z_bottom = (np.abs(gd.coords(2) - (z[0] / Bohr))).argmin()
            el_av = np.average(smooth_elstat_z[z_bottom:z_top])
        else:
            # find minima of elstat potentials (position of nuclei)
            peaks, _ = find_peaks(-smooth_elstat_z,
                                  threshold=cip['autoinner']['threshold'])
            # choose maxima of elstatpot using the number of layers
            if (cip['autoinner']['nlayers'] % 2) == 0:
                # even number of peaks
                b = int(cip['autoinner']['nlayers'] / 2)
                bottom = peaks[b - 1]
                top = peaks[b]
                el_av = np.average(smooth_elstat_z[bottom:top])
            else:
                # odd number of peaks
                c = int((cip['autoinner']['nlayers'] - 1) / 2)
                bottom = peaks[c - 1]
                top = peaks[c + 1]
                el_av = np.average(smooth_elstat_z[bottom:top])

        return el_av

    def fill_cip_keywords(self, cip):

        defaults_auto = self.default_parameters['sj']['cip']['autoinner']
        missing = {'nlayers': None,
                   'threshold': None}
        for key in missing:
            if key not in cip['autoinner']:
                cip['autoinner'][key] = defaults_auto[key]

        defaults_cip = self.default_parameters['sj']['cip']
        missing = {'inner_region': None,
                   'mu_pzc': None,
                   'phi_pzc': None,
                   'filter': None}

        for key in missing:
            if key not in cip:
                cip[key] = defaults_cip[key]

    def initialize(self, atoms=None, reading=False):
        """Inexpensive initialization.

        This catches CavityShapedJellium, which GPAW's initialize does not
        know how to handle on restart. We delete and it will be recreated by
        the _create_jellium method when needed.
        """
        background_charge = self.parameters['background_charge']
        if isinstance(background_charge, dict):
            if 'z1' in background_charge:
                if background_charge['z1'] == 'cavity_like':
                    self.parameters['background_charge'] = None
        OldSolvationGPAW.initialize(self=self, atoms=atoms, reading=reading)

    def create_hamiltonian(self, realspace, mode, xc):
        """This differs from SolvationGPAW's create_hamiltonian method by the
        ability to use dipole corrections."""
        if not realspace:
            raise NotImplementedError(
                'SJM does not support calculations in reciprocal space yet'
                ' due to a lack of an implicit solvent module.')

        dens = self.density

        self.hamiltonian = SJM_RealSpaceHamiltonian(
            *self.stuff_for_hamiltonian,
            gd=dens.gd, finegd=dens.finegd,
            nspins=dens.nspins,
            collinear=dens.collinear,
            setups=dens.setups,
            timer=self.timer,
            xc=xc,
            world=self.world,
            redistributor=dens.redistributor,
            vext=self.parameters.external,
            psolver=self.parameters.poissonsolver,
            stencil=mode.interpolation,
            dirichlet=self.parameters['sj']['dirichlet'])

        xc.set_grid_descriptor(self.hamiltonian.finegd)


def _write_trace_in_z(grid, property, name, dir):
    """Writes out a property (like electrostatic potential, cavity, or
    background charge) as a function of the z coordinate only. `grid` is the
    grid descriptor, typically self.density.finegd. `property` is the property
    to be output, on the same grid."""
    property = grid.collect(property, broadcast=True)
    property_z = property.mean(0).mean(0)
    with paropen(os.path.join(dir, name), 'w') as f:
        for i, val in enumerate(property_z):
            f.write(f'{(i + 1) * grid.h_cv[2][2] * Bohr:f} {val:1.8f}\n')


def _write_property_on_grid(grid, property, atoms, name, dir):
    """Writes out a property (like electrostatic potential, cavity, or
    background charge) on the grid, as a cube file. `grid` is the
    grid descriptor, typically self.density.finegd. `property` is the property
    to be output, on the same grid."""
    property = grid.collect(property, broadcast=True)
    ase.io.write(os.path.join(dir, name), atoms, data=property)


def _calculate_slope(previous_electrons, previous_potentials, n_prev_pot):
    """Calculates the slope of potential versus number of electrons;
    regresses based on (up to) last four data points to smooth noise."""
    # debug

    ans = linregress(previous_electrons[-n_prev_pot:],
                     previous_potentials[-n_prev_pot:])
    return ans[0]


class SJMPower12Potential(Power12Potential):
    r"""Inverse power-law potential.
    Inverse power law potential for SJM, inherited from the
    Power12Potential of gpaw.solvation. This is a 1/r^{12} repulsive
    potential taking the value u0 at the atomic radius. In SJM one also has the
    option of removing the solvent from the electrode backside and adding
    ghost plane/atoms to remove the solvent from the electrode-water interface.

    Parameters:

    atomic_radii: function
        Callable mapping an ase.Atoms object to an iterable of atomic radii
        in Angstroms. If not provided, defaults to van der Waals radii.
    u0: float
        Strength of the potential at the atomic radius in eV.
        Defaults to 0.18 eV, the best-fit value for water from Held &
        Walter.
    pbc_cutoff: float
        Cutoff in eV for including neighbor cells in a calculation with
        periodic boundary conditions.
    H2O_layer: bool, int or str
        True: Exclude the implicit solvent from the interface region
        between electrode and water. Ghost atoms will be added below
        the water layer.
        False: The opposite of True. [default]
        int: Explicitly account for the given number of water molecules above
        electrode. This is handy if H2O is directly adsorbed and a water layer
        is present in the unit cell at the same time.
        'plane': Use a plane instead of ghost atoms for freeing the surface.
    unsolv_backside: bool
        Exclude implicit solvent from the region behind the electrode

    """

    depends_on_el_density = False
    depends_on_atomic_positions = True

    def __init__(self, atomic_radii=None, u0=0.180, pbc_cutoff=1e-6,
                 tiny=1e-10, H2O_layer=False,
                 unsolv_backside=True, communicator=gpaw.mpi.world):
        super().__init__(atomic_radii, u0, pbc_cutoff, tiny)
        self.H2O_layer = H2O_layer
        self.unsolv_backside = unsolv_backside
        self.communicator = communicator

    def todict(self):
        return {
            **super().todict(),
            'H2O_layer': self.H2O_layer,
            'unsolv_backside': self.unsolv_backside}

    def __str__(self):
        s = Power12Potential.__str__(self)
        s += indent(f'  H2O layer: {self.H2O_layer}\n')
        s += indent(f'  Only solvate front side: {self.unsolv_backside}\n')
        return s

    def write(self, writer):
        writer.write(
            name='SJMPower12Potential',
            u0=self.u0,
            atomic_radii=self.atomic_radii_output,
            H2O_layer=self.H2O_layer,
            unsolv_backside=self.unsolv_backside)

    def update(self, atoms, density):
        if atoms is None:
            return False
        self.r12_a = (self.atomic_radii_output / Bohr) ** 12
        r_cutoff = (self.r12_a.max() * self.u0 / self.pbc_cutoff) ** (1. / 12.)
        self.pos_aav = get_pbc_positions(atoms, r_cutoff)
        self.u_g.fill(.0)
        self.grad_u_vg.fill(.0)
        na = np.newaxis

        if self.unsolv_backside:
            # Removing solvent from electrode backside
            for z in range(self.u_g.shape[2]):
                if (self.r_vg[2, 0, 0, z] - atoms.positions[:, 2].min() /
                        Bohr < 0):
                    self.u_g[:, :, z] = np.inf
                    self.grad_u_vg[:, :, :, z] = 0

        if self.H2O_layer:
            # Add ghost coordinates and indices to pos_aav dictionary if
            # a water layer is present.

            all_oxygen_ind = [atom.index for atom in atoms
                              if atom.symbol == 'O']

            # Disregard oxygens that don't belong to the water layer
            allwater_oxygen_ind = []
            for ox in all_oxygen_ind:
                nH = 0

                for i, atm in enumerate(atoms):
                    for period_atm in self.pos_aav[i]:
                        dist = period_atm * Bohr - atoms[ox].position
                        if np.linalg.norm(dist) < 1.3 and atm.symbol == 'H':
                            nH += 1

                if nH >= 2:
                    allwater_oxygen_ind.append(ox)

            # If the number of waters in the water layer is given as an input
            # (H2O_layer=i) then only the uppermost i water molecules are
            # regarded for unsolvating the interface (this is relevant if
            # water is adsorbed on the surface)
            if not isinstance(self.H2O_layer, (bool, str)):
                if self.H2O_layer % 1 < self.tiny:
                    self.H2O_layer = int(self.H2O_layer)
                else:
                    raise InputError('Only an integer number of water '
                                     'molecules is possible in the water '
                                     'layer')

                allwaters = atoms[allwater_oxygen_ind]
                indizes_water_ox_ind = np.argsort(allwaters.positions[:, 2],
                                                  axis=0)

                water_oxygen_ind = []
                for i in range(self.H2O_layer):
                    water_oxygen_ind.append(
                        allwater_oxygen_ind[indizes_water_ox_ind[-1 - i]])

            else:
                water_oxygen_ind = allwater_oxygen_ind

            oxygen = self.pos_aav[water_oxygen_ind[0]] * Bohr
            if len(water_oxygen_ind) > 1:
                for windex in water_oxygen_ind[1:]:
                    oxygen = np.concatenate(
                        (oxygen, self.pos_aav[windex] * Bohr))

            O_layer = []
            if isinstance(self.H2O_layer, str):
                # Add a virtual plane
                if len(self.H2O_layer.split('-')) > 1:
                    plane_z = float(self.H2O_layer.split('-')[1]) - \
                        1.0 * self.atomic_radii_output[water_oxygen_ind[0]]
                else:
                    plane_rel_oxygen = -1.5 * self.atomic_radii_output[
                        water_oxygen_ind[0]]
                    plane_z = oxygen[:, 2].min() + plane_rel_oxygen

                r_diff_zg = self.r_vg[2, :, :, :] - plane_z / Bohr
                r_diff_zg[r_diff_zg < self.tiny] = self.tiny
                r_diff_zg2 = r_diff_zg ** 2
                u_g = self.r12_a[water_oxygen_ind[0]] / r_diff_zg2 ** 6
                self.u_g += u_g.copy()
                u_g /= r_diff_zg2
                r_diff_zg *= u_g.copy()
                self.grad_u_vg[2, :, :, :] += r_diff_zg

            else:
                # Ghost atoms are added below the explicit water layer
                cell = atoms.cell.copy() / Bohr
                cell[2][2] = 1.
                natoms_in_plane = [round(np.linalg.norm(cell[0]) * 1.5),
                                   round(np.linalg.norm(cell[1]) * 1.5)]

                plane_z = (oxygen[:, 2].min() - 1.75 *
                           self.atomic_radii_output[water_oxygen_ind[0]])
                nghatoms_z = int(round(oxygen[:, 2].min() -
                                 atoms.positions[:, 2].min()))

                for i in range(int(natoms_in_plane[0])):
                    for j in range(int(natoms_in_plane[1])):
                        for k in np.linspace(atoms.positions[:, 2].min(),
                                             plane_z, num=nghatoms_z):

                            O_layer.append(np.dot(np.array(
                                [(1.5 * i - natoms_in_plane[0] / 4) /
                                 natoms_in_plane[0],
                                 (1.5 * j - natoms_in_plane[1] / 4) /
                                 natoms_in_plane[1],
                                 k / Bohr]), cell))

                # Add additional ghost O-atoms below the actual water O atoms
                # of water which frees the interface in case of corrugated
                # water layers
                for ox in oxygen / Bohr:
                    O_layer.append([ox[0], ox[1], ox[2] - 1.0 *
                                    self.atomic_radii_output[
                                        water_oxygen_ind[0]] / Bohr])

                r12_add = []
                for i in range(len(O_layer)):
                    self.pos_aav[len(atoms) + i] = [O_layer[i]]
                    r12_add.append(self.r12_a[water_oxygen_ind[0]])
                r12_add = np.array(r12_add)
                # r12_a must have same dimensions as pos_aav items
                self.r12_a = np.concatenate((self.r12_a, r12_add))

        for index, pos_av in self.pos_aav.items():
            pos_av = np.array(pos_av)
            r12 = self.r12_a[index]
            for pos_v in pos_av:
                origin_vg = pos_v[:, na, na, na]
                r_diff_vg = self.r_vg - origin_vg
                r_diff2_g = (r_diff_vg ** 2).sum(0)
                r_diff2_g[r_diff2_g < self.tiny] = self.tiny
                u_g = r12 / r_diff2_g ** 6
                self.u_g += u_g
                u_g /= r_diff2_g
                r_diff_vg *= u_g[na, ...]
                self.grad_u_vg += r_diff_vg

        self.u_g *= self.u0 / Ha
        self.grad_u_vg *= -12. * self.u0 / Ha
        self.grad_u_vg[self.grad_u_vg < -1e20] = -1e20
        self.grad_u_vg[self.grad_u_vg > 1e20] = 1e20

        return True


class SJM_RealSpaceHamiltonian(SolvationRealSpaceHamiltonian):
    """Realspace Hamiltonian with continuum solvent model in the context of
    SJM.

    See also Section III of
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).

    In contrast to the standard implicit solvent model a dipole correction can
    also be applied; this is the only difference from its parent.
    """

    def __init__(self, cavity, dielectric, interactions, gd, finegd, nspins,
                 setups, timer, xc, world, redistributor, vext=None,
                 psolver=None, stencil=3, collinear=None, dirichlet=False):

        self.cavity = cavity
        self.dielectric = dielectric
        self.interactions = interactions
        cavity.set_grid_descriptor(finegd)
        dielectric.set_grid_descriptor(finegd)
        for ia in interactions:
            ia.set_grid_descriptor(finegd)

        if psolver is None:
            psolver = WeightedFDPoissonSolver()
            self.dipcorr = False
        elif isinstance(psolver, dict):
            # Sadly the 'dipolelayer' cannot be pop'd because
            # CavityShapedJellium calls this twice.
            poi_par = {a: b for a, b in psolver.items() if a != 'dipolelayer'}
            psolver = SJMDipoleCorrection(WeightedFDPoissonSolver(**poi_par),
                                          psolver['dipolelayer'],
                                          dirichlet=dirichlet)
            self.dipcorr = True

        if self.dipcorr:
            psolver.poissonsolver.set_dielectric(self.dielectric)
        else:
            psolver.set_dielectric(self.dielectric)

        self.gradient = None

        RealSpaceHamiltonian.__init__(
            self,
            gd, finegd, nspins, collinear, setups, timer, xc, world,
            vext=vext, psolver=psolver,
            stencil=stencil, redistributor=redistributor)

        for ia in interactions:
            setattr(self, 'e_' + ia.subscript, None)
        self.new_atoms = None
        self.vt_ia_g = None
        self.e_total_free = None
        self.e_total_extrapolated = None

    def initialize(self):
        if self.dipcorr:
            self.gradient = [Gradient(self.finegd, i, 1.0,
                             self.poisson.poissonsolver.nn)
                             for i in (0, 1, 2)]
        else:
            self.gradient = [Gradient(self.finegd, i, 1.0,
                             self.poisson.nn)
                             for i in (0, 1, 2)]

        self.vt_ia_g = self.finegd.zeros()
        self.cavity.allocate()
        self.dielectric.allocate()
        for ia in self.interactions:
            ia.allocate()
        RealSpaceHamiltonian.initialize(self)


class CavityShapedJellium(Jellium):
    """The jellium object, where the counter charge takes the form of the
    solvent cavity. It puts the jellium background charge where the solvent is
    present and z < z2.

    Parameters:
    ----------
    charge: float
        The total jellium background charge.
    g_g: array
        The g function from the implicit solvent model, representing the
        percentage of the actual dielectric constant on the grid.
    z2: float
        Position of upper surface in Angstrom units.
    """

    def __init__(self, charge, g_g, z2):

        Jellium.__init__(self, charge)
        self.g_g = g_g
        self.z2 = (z2 - 0.0001) / Bohr

    def todict(self):
        dct = Jellium.todict(self)
        dct.update(z2=self.z2 * Bohr + 0.0001,
                   z1='cavity_like')
        return dct

    def get_mask(self):
        r_gv = self.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        mask = np.logical_not(r_gv[:, :, :, 2] > self.z2).astype(float)
        mask *= self.g_g
        return mask


class SJMDipoleCorrection(DipoleCorrection):
    """Dipole-correcting wrapper around another PoissonSolver specific for SJM.

    Iterative dipole correction class as applied in SJM.

    Notes
    -----

    The modules can easily be incorporated in the trunk version of GPAW
    by just adding the `fd_solv_solve`  and adapting the `solve` modules
    in the `DipoleCorrection` class.

    This module is currently calculating the correcting dipole potential
    iteratively and we would be very grateful if anybody could
    provide an analytical solution.

    New Parameters
    ---------
    corrterm: float
        Correction factor for the added countering dipole. This is calculated
        iteratively.

    last_corrterm: float
        Corrterm in the last iteration for getting the change of slope with
        change in corrterm

    last_slope: float
        Same as for `last_corrterm`

    """
    def __init__(self, poissonsolver, direction, width=1.0, dirichlet=False):
        """Construct dipole correction object."""

        DipoleCorrection.__init__(self, poissonsolver, direction, width=1.0)
        self.corrterm = 1
        self.elcorr = None
        self.last_corrterm = None
        self.dirichlet = dirichlet

    def solve(self, pot, dens, **kwargs):
        if isinstance(dens, np.ndarray):
            # finite-diference Poisson solver:
            if hasattr(self.poissonsolver, 'dielectric'):
                return self.fd_solv_solve(pot, dens, **kwargs)
            else:
                return self.fdsolve(pot, dens, **kwargs)
        # Plane-wave solver:
        self.pwsolve(pot, dens)

    def fd_solv_solve(self, vHt_g, rhot_g, **kwargs):

        gd = self.poissonsolver.gd
        slope_lim = 1e-8
        slope = slope_lim * 10

        dipmom = gd.calculate_dipole_moment(rhot_g)[2]

        if self.elcorr is not None:
            vHt_g[:, :] -= self.elcorr

        iters2 = self.poissonsolver.solve(vHt_g, rhot_g, **kwargs)
        sawtooth_z = self.sjm_sawtooth(dirichlet=self.dirichlet)
        L = gd.cell_cv[2, 2]

        while abs(slope) > slope_lim:
            vHt_g2 = vHt_g.copy()
            self.correction = 2 * np.pi * dipmom * L / \
                gd.volume * self.corrterm
            elcorr = -2 * self.correction

            elcorr *= sawtooth_z
            elcorr2 = elcorr[gd.beg_c[2]:gd.end_c[2]]
            vHt_g2[:, :] += elcorr2

            VHt_g = gd.collect(vHt_g2, broadcast=True)
            VHt_z = VHt_g.mean(0).mean(0)
            slope = VHt_z[2] - VHt_z[10]

            if abs(slope) > slope_lim:
                if self.last_corrterm is not None:
                    ds = (slope - self.last_slope) / \
                        (self.corrterm - self.last_corrterm)
                    con = slope - (ds * self.corrterm)
                    self.last_corrterm = self.corrterm
                    self.corrterm = -con / ds
                else:
                    self.last_corrterm = self.corrterm
                    self.corrterm -= slope * 10.
                self.last_slope = slope
            else:
                vHt_g[:, :] += elcorr2
                self.elcorr = elcorr2

        return iters2

    def sjm_sawtooth(self, dirichlet):
        """Creates a linear function normalized between -0.5 and 0.5 whose
           slope is scaled based on the xy-averaged dielectric constant vs z"""
        gd = self.poissonsolver.gd
        c = self.c
        L = gd.cell_cv[c, c]
        step = gd.h_cv[c, c] / L

        eps_g = gd.collect(self.poissonsolver.dielectric.eps_gradeps[0],
                           broadcast=True)
        eps_z = eps_g.mean(0).mean(0)

        saw = np.zeros((int(L / gd.h_cv[c, c])))
        for i, eps in enumerate(eps_z):
            saw[i + 1] = saw[i] + step / eps
        saw /= saw[-1] + step / eps_z[-1] - saw[0]

        if dirichlet:
            saw -= saw[-1]
        else:
            saw -= (saw[0] + saw[-1] + step / eps_z[-1]) / 2.

        return saw


class PotentialConvergenceError(ConvergenceError):
    """Raised if potential did not equilibrate."""
