from math import sqrt
import numpy as np

from gpaw.new.symmetry import Symmetries
from gpaw.new.brillouin import MonkhorstPackKPoints


def test_si():
    """Primitive diamond lattice, with Si lattice parameter."""
    a = 5.475
    cell_cv = .5 * a * np.array([(1, 1, 0), (1, 0, 1), (0, 1, 1)])
    spos_ac = np.array([(.00, .00, .00),
                        (.25, .25, .25)])
    id_a = [1, 1]  # two identical atoms
    pbc_c = np.ones(3, bool)
    mp = MonkhorstPackKPoints((4, 4, 4))

    # Do check
    symm = Symmetries.from_cell(cell_cv, pbc=pbc_c)
    symm = symm.analyze_positions(spos_ac, id_a)
    assert len(symm) == 24
    ibz = mp.reduce(symm, strict=False)
    assert len(ibz) == 10
    a = 3 / 32
    b = 1 / 32
    c = 6 / 32
    assert np.all(ibz.weight_k == [a, b, a, c, c, a, a, a, a, b])
    assert not symm.rotation_scc.sum(0).any()

    # Rotate unit cell and check again:
    cell_cv = a / sqrt(2) * np.array([(1, 0, 0),
                                      (0.5, sqrt(3) / 2, 0),
                                      (0.5, sqrt(3) / 6, sqrt(2.0 / 3))])
    symm = Symmetries.from_cell(cell_cv, pbc=pbc_c)
    symm = symm.analyze_positions(spos_ac, id_a)
    assert len(symm) == 24
    ibz2 = mp.reduce(symm, strict=False)
    assert len(ibz) == 10
    assert abs(ibz.weight_k - ibz2.weight_k).sum() < 1e-14
    assert abs(ibz.kpt_kc - ibz2.kpt_kc).sum() < 1e-14
    assert not symm.rotation_scc.sum(0).any()

    mp = MonkhorstPackKPoints((3, 3, 3))
    ibz = mp.reduce(symm)
    assert len(ibz) == 4
    assert abs(ibz.weight_k * 27 - (1, 12, 6, 8)).sum() < 1e-14


def test_h4():
    # Linear chain of four atoms, with H lattice parameter
    cell_cv = np.diag((8., 5., 5.))
    spos_ac = np.array([[0.125, 0.5, 0.5],
                        [0.375, 0.5, 0.5],
                        [0.625, 0.5, 0.5],
                        [0.875, 0.5, 0.5]])
    id_a = [1, 1, 1, 1]  # four identical atoms
    pbc_c = np.array([1, 0, 0], bool)

    # Do check
    symm = Symmetries.from_cell(cell_cv, pbc=pbc_c)
    symm = symm.analyze_positions(spos_ac, id_a)
    assert len(symm) == 16
    mp = MonkhorstPackKPoints((3, 1, 1))
    ibz = mp.reduce(symm)
    assert len(ibz) == 2
    assert np.all(ibz.weight_k == [1 / 3., 2 / 3.])


def test_2():
    # Rocksalt Ni2O2
    a = 7.92
    x = 2. * np.sqrt(1 / 3)
    y = np.sqrt(1 / 8)
    z1 = np.sqrt(1 / 24)
    z2 = np.sqrt(1 / 6)
    cell_cv = a * np.array([(x, y, -z1), (x, -y, -z1), (x, 0., z2)])
    spos_ac = np.array([[0., 0., 0.],
                        [1. / 2., 1. / 2., 1. / 2.],
                        [1. / 4., 1. / 4., 1. / 4.],
                        [3. / 4., 3. / 4., 3. / 4.]])
    id_a = [1, 2, 3, 3]
    pbc_c = np.array([1, 1, 1], bool)

    # Do check
    symm = Symmetries.from_cell(cell_cv, pbc=pbc_c)
    symm = symm.analyze_positions(spos_ac, id_a)
    assert len(symm) == 12
    mp = MonkhorstPackKPoints((2, 2, 2))
    ibz = mp.reduce(symm)
    assert len(ibz) == 2
    assert np.all(ibz.weight_k == [3 / 4, 1 / 4])


def test_new():
    sym = Symmetries.from_cell([1, 2, 3])
    assert sym.has_inversion
    assert len(sym) == 8
    sym2 = sym.analyze_positions([[0, 0, 0], [0, 0, 0.5]],
                                 ids=[1, 2])
    assert len(sym2) == 8
