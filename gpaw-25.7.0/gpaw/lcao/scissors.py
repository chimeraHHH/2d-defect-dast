"""Scissors operator for LCAO."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from ase.units import Ha

from gpaw.lcao.eigensolver import DirectLCAO
from gpaw.new.calculation import DFTCalculation
from gpaw.new.lcao.eigensolver import LCAOEigensolver
from gpaw.new.symmetry import Symmetries
from gpaw.core.matrix import Matrix


def non_self_consistent_scissors_shift(
        shifts: Sequence[tuple[float, float, int]],
        dft: DFTCalculation) -> np.ndarray:
    """Apply non self-consistent scissors shift.

    Return eigenvalues as a::

      (nspins, nibzkpts, nbands)

    shaped ndarray in eV units.

    The *shifts* are given as a sequence of tuples
    (energy shifts in eV)::

        [(<shift for occupied states>,
          <shift for unoccupied states>,
          <number of atoms>),
         ...]

    Here we open a gap for states on atoms with indices 3, 4 and 5::

      eig_skM = non_self_consistent_scissors_shift(
          [(0.0, 0.0, 3),
           (-0.5, 0.5, 3)],
          dft)
    """
    ibzwfs = dft.ibzwfs
    check_symmetries(ibzwfs.ibz.symmetries, shifts)
    shifts = [(homo / Ha, lumo / Ha, natoms)
              for homo, lumo, natoms in shifts]
    matcalc = dft.scf_loop.hamiltonian.create_hamiltonian_matrix_calculator(
        dft.potential)
    matcalc = MyMatCalc(matcalc, shifts)
    eig_skn = np.zeros((ibzwfs.nspins, len(ibzwfs.ibz), ibzwfs.nbands))
    for wfs in ibzwfs:
        H_MM = matcalc.calculate_matrix(wfs)
        eig_M = H_MM.eighg(wfs.L_MM, wfs.domain_comm)
        eig_skn[wfs.spin, wfs.k] = eig_M[:ibzwfs.nbands]
    ibzwfs.kpt_comm.sum(eig_skn)
    return eig_skn * Ha


def check_symmetries(symmetries: Symmetries,
                     shifts: Sequence[tuple[float, float, int]]) -> None:
    """Make sure shifts don't break any symmetries.

    >>> from gpaw.new.symmetry import create_symmetries_object
    >>> from ase import Atoms
    >>> atoms = Atoms('HH', [(0, 0, 1), (0, 0, -1)], cell=[3, 3, 3])
    >>> sym = create_symmetries_object(atoms)
    >>> check_symmetries(sym, [(1.0, 1.0, 1)])
    Traceback (most recent call last):
        ...
    ValueError: A symmetry maps atom 0 onto atom 1,
    but those atoms have different scissors shifts
    """
    b_sa = symmetries.atommap_sa
    shift_a = []
    for ho, lu, natoms in shifts:
        shift_a += [(ho, lu)] * natoms
    shift_a += [(0.0, 0.0)] * (b_sa.shape[1] - len(shift_a))
    for b_a in b_sa:
        for a, b in enumerate(b_a):
            if shift_a[a] != shift_a[b]:
                raise ValueError(f'A symmetry maps atom {a} onto atom {b},\n'
                                 'but those atoms have different '
                                 'scissors shifts')


class ScissorsLCAOEigensolver(LCAOEigensolver):
    def __init__(self,
                 basis,
                 shifts: Sequence[tuple[float, float, int]],
                 symmetries: Symmetries):
        """Scissors-operator eigensolver."""
        check_symmetries(symmetries, shifts)
        super().__init__(basis)
        self.shifts = []
        for homo, lumo, natoms in shifts:
            self.shifts.append((homo / Ha, lumo / Ha, natoms))

    def iterate(self,
                ibzwfs,
                density,
                potential,
                hamiltonian,
                pot_calc=None,
                energies=None):  # -> tuple[float, DFTEnergies]:
        eps_error, _, energies = \
            super().iterate(ibzwfs, density, potential,
                            hamiltonian, pot_calc, energies)
        if ibzwfs.wfs_qs[0][0]._occ_n is None:
            wfs_error = np.nan
        else:
            wfs_error = 0.0
        return eps_error, wfs_error, energies

    def iterate1(self,
                 wfs,
                 weight_n,
                 matrix_calculator):
        super().iterate1(wfs, weight_n,
                         MyMatCalc(matrix_calculator, self.shifts))

    def __repr__(self):
        txt = DirectLCAO.__repr__(self)
        txt += '\n    Scissors operators:\n'
        a1 = 0
        for homo, lumo, natoms in self.shifts:
            a2 = a1 + natoms
            txt += (f'      Atoms {a1}-{a2 - 1}: '
                    f'VB: {homo * Ha:+.3f} eV, '
                    f'CB: {lumo * Ha:+.3f} eV\n')
            a1 = a2
        return txt


class MyMatCalc:
    def __init__(self, matcalc, shifts):
        self.matcalc = matcalc
        self.shifts = shifts

    def calculate_matrix(self, wfs):
        H_MM = self.matcalc.calculate_matrix(wfs)

        try:
            nocc = int(round(wfs.occ_n.sum()))
        except ValueError:
            return H_MM

        self.add_scissors(wfs, H_MM, nocc)
        return H_MM

    def add_scissors(self, wfs, H_MM, nocc):
        ''' Serial implementation for readability:
        C_nM = wfs.C_nM.data
        S_MM = wfs.S_MM.data

        # Find Z=S^(1/2):
        e_N, U_MN = np.linalg.eigh(S_MM)
        # We now have: S_MM @ U_MN = U_MN @ diag(e_N)
        Z_MM = U_MN @ (e_N[np.newaxis]**0.5 * U_MN).T.conj()

        # Density matrix:
        A_nM = C_nM[:nocc].conj() @ Z_MM
        R_MM = A_nM.conj().T @ A_nM

        M1 = 0
        a1 = 0
        for homo, lumo, natoms in self.shifts:
            a2 = a1 + natoms
            M2 = M1 + sum(setup.nao for setup in wfs.setups[a1:a2])
            H_MM.data += Z_MM[:, M1:M2] @ \
                ((homo - lumo) * R_MM[M1:M2, M1:M2] + np.eye(M2 - M1) * lumo) \
                @ Z_MM.conj().T[M1:M2, :]
            a1 = a2
            M1 = M2

        return H_MM
        '''

        # Parallel implementation:
        U_NM = wfs.S_MM.copy()

        C_nM = wfs.C_nM
        comm = wfs.C_nM.dist.comm
        dist = (comm, comm.size, 1)

        M = C_nM.shape[1]
        C0_nM = C_nM.gather()
        C1_nM = Matrix(nocc, M, dtype=C_nM.dtype, dist=(comm, 1, 1))
        if comm.rank == 0:
            C1_nM.data[:] = C0_nM.data[:nocc, :]
        C_nM = C1_nM.new(dist=dist)
        C1_nM.redist(C_nM)

        # Find Z=S^(1/2):
        e_N = U_NM.eigh()
        e_NM = U_NM.copy()
        # We now have: S_MM @ U_MN = U_MN @ diag(e_N)

        # Next: Z_MM = U_MN @ (e_N[np.newaxis]**0.5 * U_MN).T.conj()
        n1, n2 = U_NM.dist.my_row_range()
        e_NM.data *= e_N[n1:n2, None]**0.5
        e_NM.complex_conjugate()
        Z_MM = U_NM.multiply(e_NM, opa='T')

        # Density matrix:
        C_nM.complex_conjugate()
        Q_nM = C_nM.multiply(Z_MM, opb='C')

        n = Q_nM.shape[0]

        M1 = 0
        a1 = 0
        for homo, lumo, natoms in self.shifts:
            a2 = a1 + natoms
            M2 = M1 + sum(setup.nao for setup in wfs.setups[a1:a2])
            A_Mm = Matrix(M, M2 - M1, dtype=Z_MM.dtype, dist=dist)
            A_Mm.data[:] = Z_MM.data[:, M1:M2]
            Q_nm = Matrix(n, M2 - M1, dtype=Q_nM.dtype, dist=dist)
            Q_nm.data[:] = Q_nM.data[:, M1:M2]

            Q2_nm = Q_nm.copy()
            Q2_nm.complex_conjugate()

            R_mm = Q2_nm.multiply(Q_nm, opa='T')
            R_mm.data *= (homo - lumo)
            R_mm.add_to_diagonal(lumo)
            B_mM = R_mm.multiply(A_Mm, opb='C')
            A_Mm.multiply(B_mM, beta=1.0, out=H_MM)

            a1 = a2
            M1 = M2

        return H_MM
