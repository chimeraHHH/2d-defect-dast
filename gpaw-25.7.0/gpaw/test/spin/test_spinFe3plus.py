from ase import Atoms

from gpaw import GPAW, FermiDirac
import pytest


def test_spin_spinFe3plus():
    h = 0.4
    q = 3

    s = Atoms('Fe')
    s.center(vacuum=2.5)
    convergence = {'eigenstates': 0.01, 'density': 0.1, 'energy': 0.1}

    # use Hunds rules

    c = GPAW(mode='fd', charge=q, h=h, nbands=5,
             hund=True,
             eigensolver='rmm-diis',
             occupations=FermiDirac(width=0.1),
             convergence=convergence)
    s.calc = c
    assert s.get_magnetic_moment() == pytest.approx(5, abs=0.1)

    # set magnetic moment

    mm = [5]
    s.set_initial_magnetic_moments(mm)
    c = GPAW(mode='fd', charge=q, h=h, nbands=5,
             occupations=FermiDirac(width=0.1, fixmagmom=True),
             convergence=convergence)
    s.calc = c
    assert s.get_magnetic_moment() == pytest.approx(5, abs=0.1)
