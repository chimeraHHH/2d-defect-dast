r"""Finite difference calculation of changes in potential.

The implementation is based on finite-difference calculations of the the atomic
gradients of the effective potential expressed on a real-space grid. The el-ph
couplings are obtained from LCAO representations of the atomic gradients of the
effective potential and the electronic states.

In PAW the matrix elements of the derivative of the effective potential is
given by the sum of the following contributions::

                  d                  d
            < i | -- V | j > = < i | -- V | j>
                  du  eff            du

                               _
                              \        ~a     d   .       ~a
                            +  ) < i | p  >   -- /_\H   < p | j >
                              /_        i     du     ij    j
                              a,ij

                               _
                              \        d  ~a     .        ~a
                            +  ) < i | -- p  >  /_\H    < p | j >
                              /_       du  i        ij     j
                              a,ij

                               _
                              \        ~a     .        d  ~a
                            +  ) < i | p  >  /_\H    < -- p  | j >
                              /_        i        ij    du  j
                              a,ij

where the first term is the derivative of the potential (Hartree + XC) and the
last three terms originate from the PAW (pseudopotential) part of the effective
DFT Hamiltonian.

Note: It is a bit difficult to find good references for how spin-polarisation
is supposed to be handled. Here we just handle the spin channels separately.
Use with care.

"""
from __future__ import annotations
import numpy as np

from ase import Atoms
from ase.phonons import Displacement

from gpaw.calculator import GPAW as OldGPAW
from gpaw.new.ase_interface import ASECalculator
from gpaw.utilities import pack_hermitian

dr_version = 1
# v1: saves natom, supercell, delta


class DisplacementRunner(Displacement):
    """Class for calculating the changes in effective potential.

    The derivative of the effective potential wrt atomic displacements is
    obtained from a finite difference approximation to the derivative by doing
    a self-consistent calculation for atomic displacements in the +/-
    directions. These calculations are carried out in the ``run`` member
    function.
    """

    def __init__(self,
                 atoms: Atoms,
                 calc: OldGPAW | ASECalculator,
                 supercell: tuple = (1, 1, 1),
                 name: str = 'elph',
                 delta: float = 0.01,
                 calculate_forces: bool = True) -> None:
        """Initialize with base class args and kwargs.

        Parameters
        ----------
        atoms: Atoms
            The atoms to work on. Primitive cell.
        calc: GPAW
            Calculator for the supercell finite displacement calculation.
        supercell: tuple, list
            Size of supercell given by the number of repetitions (l, m, n) of
            the small unit cell in each direction.
        name: str
            Name to use for files (default: 'elph').
        delta: float
            Magnitude of displacements. (default: 0.01 A)
        calculate_forces: bool
            If true, also calculate and store the dynamical matrix.
        """

        # Init base class and make the center cell in the supercell the
        # reference cell
        Displacement.__init__(self, atoms, calc=calc, supercell=supercell,
                              name=name, delta=delta, center_refcell=True)
        self.calculate_forces = calculate_forces

    def calculate(self, atoms_N: Atoms, disp):
        return self(atoms_N)

    def __call__(self, atoms_N: Atoms) -> dict:
        """Extract effective potential and projector coefficients."""

        # Do calculation
        atoms_N.get_potential_energy()

        # Calculate forces if desired
        if self.calculate_forces:
            forces = atoms_N.get_forces()
        else:
            forces = None

        # Get calculator
        calc = atoms_N.calc
        if not isinstance(calc, (OldGPAW, ASECalculator)):
            calc = calc.dft  # unwrap DFTD3 wrapper

        # Effective potential (in Hartree) and projector coefficients
        # Note: Need to use coarse grid, because we project onto basis later
        if isinstance(calc, ASECalculator):
            potential = calc.dft.potential
            Vt_sG = potential.vt_sR.gather(broadcast=True).data
            dH_all_asp = {a: pack_hermitian(dH_ii)
                          for a, dH_ii
                          in potential.dH_asii.gather(broadcast=True).items()}
        else:
            Vt_sG = calc.hamiltonian.vt_sG
            Vt_sG = calc.wfs.gd.collect(Vt_sG, broadcast=True)
            dH_asp = calc.hamiltonian.dH_asp

            setups = calc.wfs.setups
            nspins = calc.wfs.nspins

            dH_all_asp = {}
            for a, setup in enumerate(setups):
                ni = setup.ni
                nii = ni * (ni + 1) // 2
                dH_tmp_sp = np.zeros((nspins, nii))
                if a in dH_asp:
                    dH_tmp_sp[:] = dH_asp[a]
                calc.wfs.gd.comm.sum(dH_tmp_sp)
                dH_all_asp[a] = dH_tmp_sp

        output = {'Vt_sG': Vt_sG, 'dH_all_asp': dH_all_asp}
        if forces is not None:
            output['forces'] = forces
        return output

    def save_info(self) -> None:
        with self.cache.lock('info') as handle:
            if handle is not None:
                info = {'natom': len(self.atoms), 'supercell': self.supercell,
                        'delta': self.delta, 'dr_version': dr_version}
                handle.save(info)

    def run(self) -> None:
        """Run the calculations for the required displacements."""
        # Save some information about this run
        self.save_info()
        Displacement.run(self)
