import pytest

from ase import Atoms
from ase.build import fcc111, graphene_nanoribbon

from gpaw.utilities.adjust_cell import adjust_cell
from gpaw.utilities import h2gpts
from gpaw.grid_descriptor import GridDescriptor


def test_non_periodic():
    R = 2.0
    b = 4.0
    h = 0.2

    CO = Atoms(['C', 'O'], [(1, 0, 0), (1, 0, R)])

    adjust_cell(CO, b, h)
    cc = CO.get_cell()

    for c in range(3):
        width = 2 * b
        if c == 2:
            width += R + 2 * h
        assert cc[c, c] == pytest.approx(width, abs=1e-10)


def test_non_orthogonal_unitcell():
    a = 3.912
    box = 3.
    h = 0.2

    for atoms in [
            fcc111('Pt', (1, 1, 1), a=a),
            fcc111('Pt', (5, 6, 2), a=a, orthogonal=True)]:
        old_cell = atoms.cell.copy()

        adjust_cell(atoms, box, h)

        # check that the box ajusts for h in only non periodic directions
        assert atoms.cell[:2, :2] == pytest.approx(old_cell[:2, :2])
        # check that the atom is shifted in non periodic direction
        assert (atoms.positions[:, 2] >= box).all()

        N_c = h2gpts(h, atoms.cell)
        gd = GridDescriptor(N_c, atoms.cell, atoms.pbc)
        h_c = gd.get_grid_spacings()

        assert h_c[2] == pytest.approx(h_c[:2].sum() / 2)


def test_rotated_unitcell():
    h = 0.2
    box = 3

    gnt = graphene_nanoribbon(2, 2, sheet=True, vacuum=3)

    gnt.rotate('y', 'z', rotate_cell=True)

    adjust_cell(gnt, box, h)

    # check if the atoms are inside the unitcell
    gnt2 = gnt.copy()
    gnt2.pbc = False
    spos = gnt2.get_scaled_positions()

    assert ((spos) <= 1).all() and (spos >= 0).all()
    # check that the vacuum is equal on both sides in non periodic direction
    cell_z = gnt.cell[1, 2]
    assert (gnt.positions[:, 2] == cell_z / 2).all()

    N_c = h2gpts(h, gnt.cell)
    gd = GridDescriptor(N_c, gnt.cell, gnt.pbc)
    h_c = gd.get_grid_spacings()

    assert h_c[1] == pytest.approx((h_c[0] + h_c[2]) / 2)
