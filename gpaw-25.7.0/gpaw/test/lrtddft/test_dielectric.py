from pathlib import Path
import pytest
from ase import Atom, Atoms
from gpaw import GPAW
from gpaw.lrtddft import LrTDDFT
from gpaw.lrtddft.dielectric import get_dielectric, dielectric


@pytest.fixture
def H2():
    R = 0.7  # approx. experimental bond length
    a = 3.0
    c = 4.0
    H2 = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
                Atom('H', (a / 2, a / 2, (c + R) / 2))],
               cell=(a, a, c))

    H2.calc = GPAW(mode='fd', h=0.25, nbands=3, spinpol=False, txt=None)
    H2.get_potential_energy()
    return H2


@pytest.fixture
def lrtddft(H2):
    exlst = LrTDDFT(txt=None)
    exlst.calculate(H2)
    return exlst


@pytest.mark.lrtddft
def test_get(H2, lrtddft):
    energies, eps1, eps2, N, K, R = get_dielectric(lrtddft, H2.get_volume())
    for res in [eps1, eps2, N, K, R]:
        assert energies.shape == res.shape
    assert (energies >= 0).all()


@pytest.mark.lrtddft
def test_write(H2, lrtddft, in_tmp_dir):
    fname = 'dielectric.dat'
    dielectric(lrtddft, H2.get_volume(), fname)
    assert Path(fname).is_file()
