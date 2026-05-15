from __future__ import annotations

import warnings
from functools import cached_property
from typing import Any, TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.units import Bohr, Ha
from gpaw.core import UGArray, UGDesc
from gpaw.core.atom_arrays import AtomDistribution
from gpaw.densities import Densities
from gpaw.electrostatic_potential import ElectrostaticPotential
from gpaw.gpu import as_np
from gpaw.mpi import broadcast as bcast
from gpaw.mpi import broadcast_float, world
from gpaw.new import trace, zips
from gpaw.new.density import Density
from gpaw.new.energies import DFTEnergies
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.logger import Logger
from gpaw.new.potential import Potential
from gpaw.new.scf import SCFLoop
from gpaw.setup import Setups
from gpaw.typing import Array1D, Array2D
from gpaw.utilities import (check_atoms_too_close,
                            check_atoms_too_close_to_boundary)
from gpaw.utilities.partition import AtomPartition
if TYPE_CHECKING:
    from gpaw.dft import Parameters


class ReuseWaveFunctionsError(Exception):
    """Reusing the old wave functions after cell-change failed.

    Most likekly, the number of k-points changed.
    """


class NonsenseError(Exception):
    """Operation doesn't make sense."""


class CalculationModeError(Exception):
    """Calculation mode does not match what is expected from a given method.

    For example, if a method only works in collinear mode and receives a
    calculator in non-collinear mode, this exception should be raised.
    """


units = {'energy': Ha,
         'free_energy': Ha,
         'forces': Ha / Bohr,
         'stress': Ha / Bohr**3,
         'dipole': Bohr,
         'magmom': 1.0,
         'magmoms': 1.0,
         'non_collinear_magmom': 1.0,
         'non_collinear_magmoms': 1.0}


class DFTCalculation:
    def __init__(self,
                 atoms: Atoms,
                 ibzwfs: IBZWaveFunctions,
                 density: Density,
                 potential: Potential,
                 setups: Setups,
                 scf_loop: SCFLoop,
                 pot_calc,
                 log: Logger,
                 params: Parameters,
                 energies: DFTEnergies | None = None):
        self.atoms = atoms
        self.ibzwfs = ibzwfs
        self.density = density
        self.potential = potential
        self.setups = setups
        self.scf_loop = scf_loop
        self.pot_calc = pot_calc
        self.log = log
        self.comm = log.comm
        self.params = params

        self.results: dict[str, Any] = {}
        self.relpos_ac = self.pot_calc.relpos_ac
        self.energies = energies or DFTEnergies()
        self.forces_have_been_printed = False

    def __getattr__(self, name):
        matches = [ext
                   for ext in self.pot_calc.extensions
                   if ext.name == name]
        if len(matches) != 1:
            raise AttributeError
        return matches[0]

    @classmethod
    def from_parameters(cls,
                        atoms: Atoms,
                        params: Parameters,
                        comm=None,
                        log=None) -> DFTCalculation:
        """Create DFTCalculation object from parameters and atoms."""
        check_atoms_too_close(atoms)
        check_atoms_too_close_to_boundary(atoms)

        if not isinstance(log, Logger):
            log = Logger(log, comm or world)

        builder = params.dft_component_builder(atoms, log=log)

        basis_set = builder.create_basis_set()

        # The SCF-loop has a Hamiltonian that has an fft-plan that is
        # cached for later use, so best to create the SCF-loop first
        # FIX this!
        scf_loop = builder.create_scf_loop()

        pot_calc = builder.create_potential_calculator()

        density = builder.density_from_superposition(basis_set)
        if len(atoms) == 0:
            density.nt_sR.data[:] = 1.0
        density.normalize(pot_calc.environment.charge)

        potential, energies, _ = pot_calc.calculate_without_orbitals(
            density, kpt_band_comm=builder.communicators['D'])
        ibzwfs = builder.create_ibz_wave_functions(
            basis_set, potential)

        if ibzwfs.wfs_qs[0][0]._eig_n is not None:
            nelectrons = (density.nvalence - density.charge +
                          pot_calc.environment.charge)
            ibzwfs.calculate_occs(scf_loop.occ_calc, nelectrons)

        write_atoms(atoms, builder.initial_magmom_av, builder.grid, log)
        log(ibzwfs)
        log(density)
        log(potential)
        log(builder.setups)
        log(scf_loop)
        log(pot_calc)

        return cls(atoms, ibzwfs, density, potential,
                   builder.setups, scf_loop, pot_calc, log,
                   params=params, energies=energies)

    def ase_calculator(self):
        """Create ASE-compatible GPAW calculator.
        """
        from gpaw.new.ase_interface import ASECalculator
        return ASECalculator(params=self.params,
                             log=self.log,
                             dft=self,
                             atoms=self.atoms)

    def move_atoms(self, atoms) -> DFTCalculation:
        check_atoms_too_close(atoms)

        self.atoms = atoms
        self.relpos_ac = np.ascontiguousarray(atoms.get_scaled_positions())
        self.comm.broadcast(self.relpos_ac, 0)

        atomdist = self.density.D_asii.layout.atomdist
        grid = self.density.nt_sR.desc
        rank_a = grid.ranks_from_fractional_positions(self.relpos_ac)
        atomdist = AtomDistribution(rank_a, atomdist.comm)

        self.pot_calc.move(self.relpos_ac, atomdist)
        self.ibzwfs.move(self.relpos_ac, atomdist)
        self.density.move(self.relpos_ac, atomdist)
        if self.ibzwfs.has_wave_functions():
            self.density.update(self.ibzwfs)
        self.potential.move(atomdist)

        self.potential, self.energies, _ = self.pot_calc.calculate(
            self.density, self.ibzwfs, self.potential.vHt_x)

        mm_av = self.results['non_collinear_magmoms']
        write_atoms(atoms, mm_av, self.density.nt_sR.desc, self.log)

        self.results = {}
        self.forces_have_been_printed = False

        return self

    def iconverge(self, maxiter=None, calculate_forces=None):
        self.ibzwfs.make_sure_wfs_are_read_from_gpw_file()
        for ctx in self.scf_loop.iterate(self.ibzwfs,
                                         self.density,
                                         self.potential,
                                         self.energies,
                                         self.pot_calc,
                                         maxiter=maxiter,
                                         calculate_forces=calculate_forces,
                                         log=self.log):
            self.energies = ctx.energies
            self.potential = ctx.potential
            yield ctx

    @trace
    def converge(self,
                 maxiter=None,
                 steps=99999999999999999,
                 calculate_forces=None):
        """Converge to self-consistent solution of Kohn-Sham equation."""
        for step, _ in enumerate(self.iconverge(maxiter,
                                                calculate_forces),
                                 start=1):
            if step == steps:
                break
        else:  # no break
            self.log('SCF steps:', step)

    def energy(self):
        self.results['free_energy'] = broadcast_float(
            self.energies.total_free, self.comm)
        self.results['energy'] = broadcast_float(
            self.energies.total_extrapolated, self.comm)

        self.log('Energy contributions relative to reference atoms:',
                 f'(reference = {self.setups.Eref * Ha:.6f})\n')
        self.energies.summary(self.log)

    def dipole(self):
        if 'dipole' in self.results:
            return
        dipole_v = self.density.calculate_dipole_moment(self.relpos_ac)
        x, y, z = dipole_v * Bohr
        self.log(f'dipole moment: [{x:.6f}, {y:.6f}, {z:.6f}]  # |e|*Ang\n')
        self.results['dipole'] = dipole_v

    def magmoms(self) -> tuple[Array1D, Array2D]:
        mm_v, mm_av = self.density.calculate_magnetic_moments()
        self.results['magmom'] = mm_v[2]
        self.results['magmoms'] = mm_av[:, 2].copy()
        self.results['non_collinear_magmoms'] = mm_av
        self.results['non_collinear_magmom'] = mm_v

        if self.density.ncomponents > 1:
            x, y, z = mm_v
            self.log(f'total magnetic moment: [{x:.6f}, {y:.6f}, {z:.6f}]\n')
            self.log('local magnetic moments: [')
            for a, (setup, m_v) in enumerate(zips(self.setups, mm_av)):
                x, y, z = m_v
                c = ',' if a < len(mm_av) - 1 else ']'
                self.log(f'  [{x:9.6f}, {y:9.6f}, {z:9.6f}]{c}'
                         f'  # {setup.symbol:2} {a}')
            self.log()
        return mm_v, mm_av

    def forces(self):
        """Calculate atomic forces."""
        if 'forces' not in self.results:
            self._calculate_forces()

        if self.forces_have_been_printed:
            return

        self.forces_have_been_printed = True
        self.log('\nForces: [  # eV/Ang')
        F_av = self.results['forces'] * (Ha / Bohr)
        for a, setup in enumerate(self.setups):
            x, y, z = F_av[a]
            c = ',' if a < len(F_av) - 1 else ']'
            self.log(f'  [{x:10.4f}, {y:10.4f}, {z:10.4f}]{c}'
                     f'  # {setup.symbol:2} {a}')

    def _calculate_forces(self):
        xc = self.pot_calc.xc
        assert not hasattr(xc.xc, 'setup_force_corrections')

        # Force from projector functions (and basis set):
        F_av = self.ibzwfs.forces(self.potential)

        pot_calc = self.pot_calc
        Q_aL = self.density.calculate_compensation_charge_coefficients()
        Fcc_av, Fnct_av, Ftauct_av, Fvbar_av, Fext_av = \
            pot_calc.force_contributions(Q_aL,
                                         self.density,
                                         self.potential)

        # Force from compensation charges:
        F_av += Fcc_av

        # Force from smooth core charge:
        for a, dF_v in Fnct_av.items():
            F_av[a] += dF_v[:, 0]

        if Ftauct_av is not None:
            # Force from smooth core ked:
            for a, dF_v in Ftauct_av.items():
                F_av[a] += dF_v[:, 0]

        # Force from zero potential:
        for a, dF_v in Fvbar_av.items():
            F_av[a] += dF_v[:, 0]

        F_av = as_np(F_av)

        domain_comm = Q_aL.layout.atomdist.comm
        domain_comm.sum(F_av)

        # Force from extensions (only from rank 0)
        F_av += Fext_av

        F_av = self.ibzwfs.ibz.symmetries.symmetrize_forces(F_av)
        self.comm.broadcast(F_av, 0)
        self.results['forces'] = F_av

    def stress(self) -> None:
        if 'stress' in self.results:
            return
        stress_vv = self.pot_calc.stress(
            self.ibzwfs, self.density, self.potential)
        self.log('\nstress tensor: [  # eV/Ang^3')
        for (x, y, z), c in zips(stress_vv * (Ha / Bohr**3), ',,]'):
            self.log(f'  [{x:13.6f}, {y:13.6f}, {z:13.6f}]{c}')
        self.results['stress'] = stress_vv.flat[[0, 4, 8, 5, 2, 1]]

    def write_converged(self) -> None:
        self.ibzwfs.write_summary(self.log)
        vacuum_level = self.potential.get_vacuum_level()
        if not np.isnan(vacuum_level):
            self.log(f'vacuum-level: {vacuum_level:.3f}  # V')
            try:
                wf1, wf2 = self.workfunctions(_vacuum_level=vacuum_level)
            except NonsenseError:
                pass
            else:
                self.log(
                    f'Workfunctions: {wf1:.3f}, {wf2:.3f} eV (top, bottom)')
        self.log.fd.flush()

    def workfunctions(self,
                      *, _vacuum_level=None) -> tuple[float, float]:
        if _vacuum_level is None:
            _vacuum_level = self.potential.get_vacuum_level()
        if np.isnan(_vacuum_level):
            raise NonsenseError('No vacuum')
        try:
            correction = self.pot_calc.poisson_solver.dipole_layer_correction()
        except NotImplementedError:
            raise NonsenseError('No dipole layer')
        correction *= Ha
        fermi_level = self.ibzwfs.fermi_level * Ha
        wf = _vacuum_level - fermi_level
        return wf - correction, wf + correction

    def vacuum_level(self) -> float:
        return self.potential.get_vacuum_level()

    def electrostatic_potential(self) -> ElectrostaticPotential:
        return ElectrostaticPotential.from_calculation(self)

    def densities(self) -> Densities:
        return Densities.from_calculation(self)

    def wave_function(self, band: int, kpt=0, spin=None,
                      periodic=False,
                      broadcast=True) -> UGArray:
        psit_nR = self.wave_functions(n1=band, n2=band + 1, kpt=kpt, spin=spin,
                                      periodic=periodic, broadcast=broadcast)
        if psit_nR is not None:
            return psit_nR[0]

    def wave_functions(self, n1=0, n2=None, kpt=0, spin=None,
                       periodic=False,
                       broadcast=True,
                       _pad=True) -> UGArray:
        collinear = self.ibzwfs.collinear
        if collinear:
            if spin is None:
                spin = 0
        else:
            assert spin is None or spin == 0
        wfs = self.ibzwfs.get_wfs(spin=spin if collinear else 0,
                                  kpt=kpt,
                                  n1=n1, n2=n2)
        if wfs is not None:
            basis = getattr(self.scf_loop.hamiltonian, 'basis', None)
            grid = self.density.nt_sR.desc.new(comm=None)
            if collinear:
                wfs = wfs.to_uniform_grid_wave_functions(grid, basis)
                psit_nR = wfs.psit_nX
            else:
                psit_nsG = wfs.psit_nX
                grid = grid.new(kpt=psit_nsG.desc.kpt_c,
                                dtype=psit_nsG.desc.dtype)
                psit_nR = psit_nsG.ifft(grid=grid)
            if not psit_nR.desc.pbc.all() and _pad:
                psit_nR = psit_nR.to_pbc_grid()
            if periodic:
                psit_nR.multiply_by_eikr(-psit_nR.desc.kpt_c)
        else:
            psit_nR = None
        if broadcast:
            psit_nR = bcast(psit_nR, 0, self.comm)
        return psit_nR.scaled(cell=Bohr, values=Bohr**-1.5)

    @cached_property
    def _atom_partition(self):
        # Backwards compatibility helper
        atomdist = self.density.D_asii.layout.atomdist
        return AtomPartition(atomdist.comm, atomdist.rank_a)

    def new(self,
            atoms: Atoms,
            params: Parameters,
            log=None) -> DFTCalculation:
        """Create new DFTCalculation object."""

        if params.mode.name != 'pw':
            raise ReuseWaveFunctionsError

        ibzwfs = self.ibzwfs
        # if ibzwfs.domain_comm.size != 1:
        #     raise ReuseWaveFunctionsError

        check_atoms_too_close(atoms)
        check_atoms_too_close_to_boundary(atoms)

        builder = params.dft_component_builder(atoms, log=log)

        kpt_kc = builder.ibz.kpt_kc
        old_kpt_kc = ibzwfs.ibz.kpt_kc
        if len(kpt_kc) != len(old_kpt_kc):
            raise ReuseWaveFunctionsError
        if abs(kpt_kc - old_kpt_kc).max() > 1e-9:
            raise ReuseWaveFunctionsError

        log('Interpolating wave functions to new cell')

        scf_loop = builder.create_scf_loop()
        pot_calc = builder.create_potential_calculator()

        density = self.density.new(builder.grid,
                                   builder.interpolation_desc,
                                   builder.relpos_ac,
                                   builder.atomdist)
        density.normalize(pot_calc.environment.charge)

        # Make sure all have exactly the same density.
        # Not quite sure it is needed???
        # At the moment we skip it on GPU's because it doesn't
        # work!
        if density.nt_sR.xp is np:
            ibzwfs.kpt_band_comm.broadcast(density.nt_sR.data, 0)

        potential, energies, _ = pot_calc.calculate(density)

        old_ibzwfs = ibzwfs

        def create_wfs(spin, q, k, kpt_c, weight):
            wfs = old_ibzwfs.wfs_qs[q][spin]
            return wfs.morph(
                builder.wf_desc,
                builder.relpos_ac,
                builder.atomdist)

        ibzwfs = ibzwfs.create(
            ibz=builder.ibz,
            ncomponents=old_ibzwfs.ncomponents,
            create_wfs_func=create_wfs,
            kpt_comm=old_ibzwfs.kpt_comm,
            kpt_band_comm=old_ibzwfs.kpt_band_comm,
            comm=self.comm)

        write_atoms(atoms, builder.initial_magmom_av, builder.grid, log)
        log(ibzwfs)
        log(density)
        log(potential)
        log(builder.setups)
        log(scf_loop)
        log(pot_calc)

        return DFTCalculation(
            atoms, ibzwfs, density, potential,
            builder.setups, scf_loop, pot_calc, log,
            params=params, energies=energies)

    def get_state(self):
        return DFTState(self.ibzwfs, self.density, self.potential)

    @property
    def state(self):
        warnings.warn('Use of deprecated DFTCalculation.state attribute. '
                      'Use ibzwfs, density and potential attributes instead.')
        return self.get_state()


def write_atoms(atoms: Atoms,
                magmom_av: Array2D,
                grid: UGDesc,
                log) -> None:
    from gpaw.output import print_cell, print_positions
    print_positions(atoms, log, magmom_av)
    print_cell(grid._gd, grid.pbc, log)


class DFTState:
    def __init__(self,
                 ibzwfs: IBZWaveFunctions,
                 density: Density,
                 potential: Potential,
                 energies):
        """State of a Kohn-Sham calculation."""
        self.ibzwfs = ibzwfs
        self.density = density
        self.potential = potential
        self.energies = energies
