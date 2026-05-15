from __future__ import annotations

import itertools
import warnings
from math import inf
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from gpaw import KohnShamConvergenceError
from gpaw.convergence_criteria import (Criterion, check_convergence,
                                       dict2criterion)
from gpaw.new.energies import DFTEnergies
from gpaw.new.logger import indent
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.scf import write_iteration
from gpaw.typing import Array2D


class TooFewBandsError(KohnShamConvergenceError):
    """Not enough bands for CBM+x convergence criterion."""


class SCFLoop:
    def __init__(self,
                 hamiltonian,
                 occ_calc,
                 eigensolver,
                 mixer,
                 comm,
                 convergence,
                 maxiter):
        self.hamiltonian = hamiltonian
        self.eigensolver = eigensolver
        self.mixer = mixer
        self.occ_calc = occ_calc
        self.comm = comm
        self.convergence = create_convergence_criteria(convergence)
        self.maxiter = maxiter
        self.niter = 0
        self.update_density_and_potential = True
        self.fix_fermi_level = False

    def __repr__(self):
        return 'SCFLoop(...)'

    def __str__(self):
        return (f'eigensolver:\n{indent(self.eigensolver)}\n'
                f'{self.mixer}\n'
                f'occupation numbers:\n{indent(self.occ_calc)}\n')

    def iterate(self,
                ibzwfs: IBZWaveFunctions,
                density,
                potential,
                energies: DFTEnergies,
                pot_calc,
                *,
                maxiter=None,
                calculate_forces=None,
                log=None):
        cc = self.convergence
        maxiter = maxiter or self.maxiter

        if log:
            log('convergence criteria:')
            for criterion in cc.values():
                if criterion.description is not None:
                    log('- ' + criterion.description)
            log(f'maximum number of iterations: {self.maxiter}\n')

        self.mixer.reset()

        self.occ_calc.initialize_reference_orbitals()

        if self.update_density_and_potential:
            dens_error = self.mixer.mix(density)
        else:
            dens_error = 0.0

        for self.niter in itertools.count(start=1):
            eig_error, wfs_error, energies = self.eigensolver.iterate(
                ibzwfs, density, potential,
                self.hamiltonian, pot_calc, energies)
            nelectrons = (density.nvalence - density.charge +
                          pot_calc.environment.charge)
            e_band, e_entropy, e_extrapolation = ibzwfs.calculate_occs(
                self.occ_calc,
                nelectrons,
                fix_fermi_level=self.fix_fermi_level)

            energies.set(**pot_calc.xc.energies,
                         band=e_band,
                         entropy=e_entropy,
                         extrapolation=e_extrapolation)

            ctx = SCFContext(
                log, self.niter, energies,
                ibzwfs, density, potential,
                wfs_error, dens_error, eig_error,
                self.comm, calculate_forces,
                pot_calc, self.update_density_and_potential)

            yield ctx

            converged, converged_items, entries = check_convergence(cc, ctx)
            nconverged = self.comm.sum_scalar(int(converged))
            assert nconverged in [0, self.comm.size], converged_items

            if log:
                write_iteration(cc, converged_items, entries, ctx, log)

            if converged:
                converged = pot_calc.environment.post_scf_convergence(
                    ibzwfs, nelectrons, self.occ_calc, self.mixer, log)
                if converged:
                    break

            if self.niter == maxiter:
                if wfs_error < inf:
                    raise KohnShamConvergenceError
                raise TooFewBandsError

            if self.update_density_and_potential:
                density.update(ibzwfs, ked=pot_calc.xc.type == 'MGGA')
                dens_error = self.mixer.mix(density)
                potential, energies, _ = pot_calc.calculate(
                    density, ibzwfs, potential.vHt_x)

        self.eigensolver.postprocess(
            ibzwfs, density, potential, self.hamiltonian)


class SCFContext:
    def __init__(self,
                 log,
                 niter: int,
                 energies: DFTEnergies,
                 ibzwfs,
                 density,
                 potential,
                 wfs_error: float,
                 dens_error: float,
                 eig_error: float,
                 comm,
                 calculate_forces: Callable[[], Array2D],
                 pot_calc,
                 update_density_and_potential):
        self.log = log
        self.niter = niter
        self.energies = energies
        self.ibzwfs = ibzwfs
        self.density = density
        self.potential = potential
        energy = energies.total_extrapolated
        self.ham = SimpleNamespace(e_total_extrapolated=energy,
                                   get_workfunctions=self._get_workfunctions)
        self.wfs = SimpleNamespace(nvalence=density.nvalence +
                                   pot_calc.environment.charge,
                                   world=comm,
                                   eigensolver=SimpleNamespace(
                                       error=wfs_error),
                                   nspins=density.ndensities,
                                   collinear=density.collinear)
        self.dens = SimpleNamespace(
            calculate_magnetic_moments=density.calculate_magnetic_moments,
            fixed=not update_density_and_potential,
            error=dens_error)
        self.eig_error = eig_error
        self.calculate_forces = calculate_forces
        self.poisson_solver = pot_calc.poisson_solver

    def _get_workfunctions(self, _):
        vacuum_level = self.potential.get_vacuum_level()
        (fermi_level,) = self.ibzwfs.fermi_levels
        wf = vacuum_level - fermi_level
        delta = self.poisson_solver.dipole_layer_correction()
        return np.array([wf + delta, wf - delta])


def create_convergence_criteria(criteria: dict[str, Any]
                                ) -> dict[str, Criterion]:
    criteria = criteria.copy()
    for k, v in [('energy', 0.0005),        # eV / electron
                 ('density', 1.0e-4),       # electrons / electron
                 ('eigenstates', 4.0e-8)]:  # eV^2 / electron
        if k not in criteria:
            criteria[k] = v
    # Gather convergence criteria for SCF loop.
    custom = criteria.pop('custom', [])
    for name, criterion in criteria.items():
        if hasattr(criterion, 'todict'):
            # 'Copy' so no two calculators share an instance.
            criteria[name] = dict2criterion(criterion.todict())
        else:
            criteria[name] = dict2criterion({name: criterion})

    if not isinstance(custom, (list, tuple)):
        custom = [custom]
    for criterion in custom:
        if isinstance(criterion, dict):  # from .gpw file
            msg = ('Custom convergence criterion "{:s}" encountered, '
                   'which GPAW does not know how to load. This '
                   'criterion is NOT enabled; you may want to manually'
                   ' set it.'.format(criterion['name']))
            warnings.warn(msg)
            continue

        criteria[criterion.name] = criterion
        msg = ('Custom convergence criterion {:s} encountered. '
               'Please be sure that each calculator is fed a '
               'unique instance of this criterion. '
               'Note that if you save the calculator instance to '
               'a .gpw file you may not be able to re-open it. '
               .format(criterion.name))
        warnings.warn(msg)

    for criterion in criteria.values():
        criterion.reset()

    return criteria
