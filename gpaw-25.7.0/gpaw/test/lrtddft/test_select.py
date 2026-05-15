import pytest
from ase import Atoms
from gpaw import GPAW, FermiDirac
from gpaw.lrtddft import LrTDDFT


@pytest.fixture
def oxygen():
    atoms = Atoms('O')
    atoms.cell = [3, 4, 5]
    atoms.center()

    atoms.calc = GPAW(mode='fd',
                      occupations=FermiDirac(width=0.1),
                      nbands=5)
    atoms.get_potential_energy()
    return atoms


@pytest.mark.lrtddft
def test_digonalize(oxygen):
    """Test selection at diagonalization stage"""
    lr = LrTDDFT(oxygen.calc)

    # all
    lr.diagonalize()
    assert len(lr) == 10

    lr.diagonalize(restrict={'istart': 3})
    assert len(lr) == 1

    lr.diagonalize(restrict={'jend': 1})
    assert len(lr) == 1

    lr.diagonalize(restrict={'eps': 0.75})
    assert len(lr) == 2

    lr.diagonalize(restrict={'energy_range': 1})
    assert len(lr) == 3

    lr.diagonalize(restrict={'from': [0], 'to': [3, 4]})
    assert len(lr) == 2


@pytest.mark.lrtddft
def test_window(oxygen):
    """Test window selection at calculation step"""
    froml = [0, 1]
    tol = [4]
    lr = LrTDDFT(oxygen.calc, restrict={'from': froml, 'to': tol})
    assert len(lr) == len(froml) * len(tol)
    for ks in lr.kss:
        assert ks.i in froml
        assert ks.j in tol
