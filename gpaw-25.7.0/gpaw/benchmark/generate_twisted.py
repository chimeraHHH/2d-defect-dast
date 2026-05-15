import numpy as np
from ase.build import make_supercell, niggli_reduce


def make_heterostructure(atoms_a, atoms_b,
                         transa_cc, transb_cc,
                         straina_vv, interlayer_dist,
                         vacuum=5):
    Satoms_a = make_supercell(atoms_a, transa_cc, wrap=True)
    Satoms_b = make_supercell(atoms_b, transb_cc, wrap=True)
    Satoms_a.set_cell(Satoms_a.cell @ straina_vv, scale_atoms=True)
    deformb_vv = np.linalg.solve(Satoms_b.cell, Satoms_a.cell)
    Satoms_b.set_cell(Satoms_b.cell @ deformb_vv, scale_atoms=True)
    Satoms_a.set_tags(0)
    Satoms_b.set_tags(1)
    Satoms_b.positions[:, 2] += interlayer_dist
    final_atoms = Satoms_a + Satoms_b
    niggli_reduce(final_atoms)
    final_atoms.center(axis=2, vacuum=vacuum)
    return final_atoms
