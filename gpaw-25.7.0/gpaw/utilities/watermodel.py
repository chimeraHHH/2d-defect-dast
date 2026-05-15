from ase.constraints import FixBondLengths
from ase.calculators.tip3p import TIP3P
from ase.calculators.tip3p import qH, sigma0, epsilon0

from gpaw.cgpaw import adjust_positions, adjust_momenta, calculate_forces_H2O
from ase.calculators.calculator import Calculator, all_changes

import numpy as np

A = 4 * epsilon0 * sigma0**12
B = -4 * epsilon0 * sigma0**6


class TIP3PWaterModel(TIP3P):
    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):

        Calculator.calculate(self, atoms, properties, system_changes)

        R = self.atoms.positions.reshape((-1, 3, 3))
        Z = self.atoms.numbers
        pbc = self.atoms.pbc
        diagcell = self.atoms.cell.diagonal()
        nh2o = len(R)

        assert (self.atoms.cell == np.diag(diagcell)).all(), 'not orthorhombic'
        assert ((diagcell >= 2 * self.rc) | ~pbc).all(), 'cutoff too large'
        if Z[0] == 8:
            o = 0
        else:
            o = 2

        assert o == 0
        assert (Z[o::3] == 8).all()
        assert (Z[(o + 1) % 3::3] == 1).all()
        assert (Z[(o + 2) % 3::3] == 1).all()

        charges = np.array([qH, qH, qH])
        charges[o] *= -2

        energy = 0.0
        forces = np.zeros((3 * nh2o, 3))

        cellmat = np.zeros((3, 3))
        np.fill_diagonal(cellmat, diagcell)   # c code wants 3x3 cell ...

        energy += calculate_forces_H2O(np.array(atoms.pbc, dtype=np.uint8),
                                       cellmat, A, B, self.rc, self.width,
                                       charges, self.atoms.get_positions(),
                                       forces)

        if self.pcpot:
            e, f = self.pcpot.calculate(np.tile(charges, nh2o),
                                        self.atoms.positions)
            energy += e
            forces += f

        self.results['energy'] = energy
        self.results['forces'] = forces


class FixBondLengthsWaterModel(FixBondLengths):

    def __init__(self, pairs, tolerance=1e-13, bondlengths=None,
                 iterations=None):
        FixBondLengths.__init__(self, pairs, tolerance=tolerance,
                                bondlengths=bondlengths,
                                iterations=iterations)

    def initialize_bond_lengths(self, atoms):
        bondlengths = FixBondLengths.initialize_bond_lengths(self, atoms)
        # Make sure that the constraints are compatible with the C-code
        assert len(self.pairs) % 3 == 0
        masses = atoms.get_masses()

        self.start = self.pairs[0][0]
        self.end = self.pairs[-3][0]
        self.NW = (self.end - self.start) // 3 + 1
        assert (self.end - self.start) % 3 == 0
        for i in range(self.NW):
            for j in range(3):
                assert self.pairs[i * 3 + j][0] == self.start + i * 3 + j
                assert self.pairs[i * 3 + j][1] == (self.start +
                                                    i * 3 + (j + 1) % 3)
                assert abs(bondlengths[i * 3 + j] - bondlengths[j]) < 1e-6
                assert masses[i * 3 + j + self.start] == masses[j + self.start]
        return bondlengths

    def select_indices(self, r):
        return r[self.start:self.start + self.NW * 3, :]

    def adjust_positions(self, atoms, new):
        masses = atoms.get_masses()

        if self.bondlengths is None:
            self.bondlengths = self.initialize_bond_lengths(atoms)

        return adjust_positions(self.bondlengths[:3],
                                masses[self.start:self.start + 3],
                                self.select_indices(atoms.get_positions()),
                                self.select_indices(new))

    def adjust_momenta(self, atoms, p):
        masses = atoms.get_masses()

        if self.bondlengths is None:
            self.bondlengths = self.initialize_bond_lengths(atoms)

        return adjust_momenta(masses[self.start:self.start + 3],
                              self.select_indices(atoms.get_positions()),
                              self.select_indices(p))

    def index_shuffle(self, atoms, ind):
        raise NotImplementedError
