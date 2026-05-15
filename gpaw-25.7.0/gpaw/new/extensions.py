from ase.units import Hartree, Bohr
from ase.calculators.calculator import PropertyNotImplementedError
import numpy as np
from gpaw.mpi import serial_comm, broadcast_exception, broadcast_float
import uuid
from pathlib import Path
import os
from gpaw.dft import Extension as ExtensionParameter


class Extension:
    name = 'unnamed extension'

    def get_energy_contributions(self) -> dict[str, float]:
        return {}

    def force_contribution(self):
        raise NotImplementedError

    def move_atoms(self, relpos_ac) -> None:
        raise NotImplementedError

    def update_non_local_hamiltonian(self,
                                     D_sii,
                                     setup,
                                     atom_index,
                                     dH_sii) -> float:
        return 0.0

    def build(self, atoms, comms, log):
        return self


class D3(ExtensionParameter):
    name = 'd3'

    def __init__(self, *, xc, **kwargs):
        self.xc = xc
        self.kwargs = kwargs

    def todict(self) -> dict:
        return {'xc': self.xc, **self.kwargs}

    def build(self, atoms, communicators, log):
        atoms = atoms.copy()
        world = communicators['w']
        from ase.calculators.dftd3 import PureDFTD3

        # Since DFTD3 is filesystem based, and GPAW has no such requirements
        # we need to be absolutely sure that there are no race-conditions
        # in files. label cannot be used, because dftd3 executable still
        # writes gradients to fixed files, thus a unique folder needs to be
        # created.

        class D3Extension(Extension):
            name = 'd3'

            def __init__(self):
                super().__init__()
                self.stress_vv = np.zeros((3, 3)) * np.nan
                self.F_av = np.zeros_like(atoms.positions) * np.nan
                self.E = np.nan
                self._calculate(atoms)

            def _calculate(_self, atoms):
                # Circumvent a DFTD3 bug for an isolated atom ASE #1672
                if len(atoms) == 1 and not atoms.pbc.any():
                    _self.stress_vv = np.zeros((3, 3)) * np.nan
                    _self.F_av = np.zeros_like(atoms.positions)
                    _self.E = 0.0
                    return

                cwd = Path.cwd()
                assert atoms.calc is None
                # Call DFTD3 only with single core due to #1671
                with broadcast_exception(world):
                    if world.rank == 0:
                        try:
                            _self.calculate_single_core()
                        finally:
                            os.chdir(cwd)
                _self.E = broadcast_float(_self.E, world)
                world.broadcast(_self.F_av, 0)
                world.broadcast(_self.stress_vv, 0)

            def calculate_single_core(_self):
                """Single core method to calculate D3 forces and stresses"""

                label = uuid.uuid4().hex[:8]
                directory = Path('dftd3-ext-' + label).absolute()
                directory.mkdir()

                # Due to ase #1673, relative folders are not supported
                # neither are absolute folders due to 80 character limit.
                # The only way out, is to chdir to a temporary folder here.
                os.chdir(directory)
                log('Evaluating D3 corrections at temporary'
                    f' folder {directory}')
                atoms.calc = PureDFTD3(xc=self.xc,
                                       directory='.',
                                       comm=serial_comm,
                                       **self.kwargs)

                # XXX params.xc should be taken directly from the calculator.
                # XXX What if this is changed via set?
                _self.F_av = atoms.get_forces() / Hartree * Bohr

                try:
                    # Copy needed because array is not c-contigous
                    _self.stress_vv = atoms.get_stress(voigt=False).copy() \
                        / Hartree * Bohr**3
                except PropertyNotImplementedError:
                    _self.stress_vv = np.zeros((3, 3)) * np.nan

                _self.E = atoms.get_potential_energy() / Hartree
                try:
                    os.unlink('ase_dftd3.out')
                    os.unlink('ase_dftd3.POSCAR')
                    os.unlink('dftd3_cellgradient')
                    os.unlink('dftd3_gradient')
                    os.rmdir(directory.absolute())
                except OSError as e:
                    log('Unable to remove files and folder', e)
                atoms.calc = None

            def get_energy_contributions(_self) -> dict[str, float]:
                """Returns the energy contributions from D3 in Hartree"""
                return {f'D3 (xc={self.xc})': _self.E}

            def get_energy(self) -> float:
                """Returns the energy contribution from D3 in eV"""
                return self.E * Hartree

            def force_contribution(self):
                return self.F_av

            def stress_contribution(self):
                if np.isnan(self.stress_vv).all():
                    raise PropertyNotImplementedError
                return self.stress_vv

            def move_atoms(self, relpos_ac) -> None:
                atoms.set_scaled_positions(relpos_ac)
                self._calculate(atoms)

        return D3Extension()
