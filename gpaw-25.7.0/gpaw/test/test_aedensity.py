import pytest
from ase import Atom, Atoms

from gpaw import GPAW


def test_aedensity():
    h = 0.21  # gridspacing
    a = [6.5, 6.5, 7.7]  # unit cell
    d = 2.3608  # experimental bond length

    NaCl = Atoms([Atom('Na', [0, 0, 0]),
                  Atom('Cl', [0, 0, d])],
                 pbc=False, cell=a)
    NaCl.center()
    calc = GPAW(mode='fd',
                h=h,
                xc='LDA',
                nbands=5,
                setups={'Na': '1'},
                convergence={'eigenstates': 1e-6},
                spinpol=1)

    NaCl.calc = calc
    e = NaCl.get_potential_energy()

    assert e == pytest.approx(-4.907, abs=0.002)

    dv = NaCl.get_volume() / calc.get_number_of_grid_points().prod()
    nt1 = calc.get_pseudo_density(gridrefinement=1)
    Zt1 = nt1.sum() * dv
    nt2 = calc.get_pseudo_density(gridrefinement=2)
    Zt2 = nt2.sum() * dv / 8
    print('Integral of pseudo density:', Zt1, Zt2)
    assert Zt1 == pytest.approx(Zt2, abs=1e-12)

    for gridrefinement in [1, 2, 4]:
        n = calc.get_all_electron_density(gridrefinement=gridrefinement)
        Z = n.sum() * dv / gridrefinement**3
        print('Integral of all-electron density:', Z)
        assert Z == pytest.approx(28, abs=1e-5)
