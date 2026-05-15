import numpy as np

from ase import Atoms

from gpaw.utilities import h2gpts
from gpaw.grid_descriptor import GridDescriptor


def adjust_cell(atoms: Atoms,
                border: float,
                h: float = 0.2,
                idiv: int = 4) -> None:
    """Adjust the cell such that
    1. The vacuum around all atoms is at least border
       in non-periodic directions
    2. The grid spacing chosen by GPAW will be as similar
       as possible to h in all directions
    """
    n_pbc = atoms.pbc.sum()
    if n_pbc == 3:
        return

    pos_ac = atoms.get_positions()
    lowest_c = np.minimum.reduce(pos_ac)
    largest_c = np.maximum.reduce(pos_ac)

    for i, v_c, in enumerate(atoms.cell):
        if (v_c == 0).all():
            assert not atoms.pbc[i]  # pbc with zero cell size make no sense
            atoms.cell[i, i] = 1

    if n_pbc:
        N_c = h2gpts(h, atoms.cell, idiv)
        gd = GridDescriptor(N_c, atoms.cell, atoms.pbc)
        h_c = gd.get_grid_spacings()
        h = 0
        for pbc, h1 in zip(atoms.pbc, h_c):
            if pbc:
                h += h1 / n_pbc

    # the optimal h to be set to non-periodic directions
    h_c = np.array([h, h, h])

    shift_c = np.zeros(3)

    # adjust each cell direction
    for i in range(3):
        if atoms.pbc[i]:
            continue

        # cell direction
        u_c = atoms.cell[i] / np.linalg.norm(atoms.cell[i])

        extension = (largest_c - lowest_c) * u_c
        min_size = extension + 2 * border

        h = h_c[i]
        N = min_size / h
        N = -(N // -idiv) * idiv  # ceil div
        size = N * h

        atoms.cell[i] = size * u_c

        # shift structure to the center
        shift_c += (size - extension) / 2 * u_c
        shift_c -= lowest_c * u_c

    atoms.translate(shift_c)
