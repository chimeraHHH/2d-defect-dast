import pytest
from ase import Atom, Atoms
from gpaw import GPAW, Davidson, Mixer
from gpaw.xc.hybrid import HybridXC


@pytest.mark.mgga
@pytest.mark.libxc
def test_xc_atomize(in_tmp_dir, gpaw_new):
    def xc(name):
        return {'name': name, 'stencil': 1}

    a = 6.  # Size of unit cell (Angstrom)
    c = a / 2
    # Hydrogen atom:
    atom = Atoms([Atom('H', (c, c, c), magmom=1)],
                 cell=(a, a, a), pbc=False)

    # gpaw calculator:
    calc = GPAW(mode='fd',
                gpts=(32, 32, 32),
                nbands=1,
                xc=xc('PBE'),
                txt='H.txt',
                eigensolver=Davidson(12),
                mixer=Mixer(0.5, 5),
                parallel=dict(kpt=1),
                convergence=dict(eigenstates=3.3e-8))
    atom.calc = calc

    e1 = atom.get_potential_energy()
    de1t = calc.get_xc_difference(xc('TPSS'))
    de1m = calc.get_xc_difference(xc('M06-L'))
    if not gpaw_new:
        de1x = calc.get_xc_difference(
            HybridXC('EXX', stencil=1, finegrid=True))
        de1xb = calc.get_xc_difference(
            HybridXC('EXX', stencil=1, finegrid=False))

    # Hydrogen molecule:
    d = 0.74  # Experimental bond length
    molecule = Atoms([Atom('H', (c - d / 2, c, c)),
                      Atom('H', (c + d / 2, c, c))],
                     cell=(a, a, a), pbc=False)

    molecule.calc = calc.new(txt='H2.txt')
    e2 = molecule.get_potential_energy()
    de2t = molecule.calc.get_xc_difference(xc('TPSS'))
    de2m = molecule.calc.get_xc_difference(xc('M06-L'))

    print('hydrogen atom energy:     %5.2f eV' % e1)
    print('hydrogen molecule energy: %5.2f eV' % e2)
    print('atomization energy:       %5.2f eV' % (2 * e1 - e2))
    print('atomization energy  TPSS: %5.2f eV' %
          (2 * (e1 + de1t) - (e2 + de2t)))
    print('atomization energy  M06-L: %5.2f eV' %
          (2 * (e1 + de1m) - (e2 + de2m)))
    PBETPSSdifference = (2 * e1 - e2) - (2 * (e1 + de1t) - (e2 + de2t))
    PBEM06Ldifference = (2 * e1 - e2) - (2 * (e1 + de1m) - (e2 + de2m))
    print(PBETPSSdifference)
    print(PBEM06Ldifference)
    # TPSS value is from JCP 120 (15) 6898, 2004
    # e.g. Table VII: DE(PBE - TPSS) = (104.6-112.9)*kcal/mol
    # EXX value is from PRL 77, 3865 (1996)
    assert PBETPSSdifference == pytest.approx(-0.3599, abs=0.04)
    assert PBEM06Ldifference == pytest.approx(-0.169, abs=0.01)

    energy_tolerance = 0.002
    assert e1 == pytest.approx(-1.081638, abs=energy_tolerance)
    assert e2 == pytest.approx(-6.726356, abs=energy_tolerance)

    if not gpaw_new:
        de2x = molecule.calc.get_xc_difference(
            HybridXC('EXX', stencil=1, finegrid=True))
        de2xb = molecule.calc.get_xc_difference(
            HybridXC('EXX', stencil=1, finegrid=False))

        print('atomization energy   EXX: %5.2f eV' %
              (2 * (e1 + de1x) - (e2 + de2x)))
        print('atomization energy   EXX: %5.2f eV' %
              (2 * (e1 + de1xb) - (e2 + de2xb)))
        PBEEXXdifference = (2 * e1 - e2) - (2 * (e1 + de1x) - (e2 + de2x))
        PBEEXXbdifference = (2 * e1 - e2) - (2 * (e1 + de1xb) - (e2 + de2xb))
        print(PBEEXXdifference)
        print(PBEEXXbdifference)
        assert PBEEXXdifference == pytest.approx(0.91, abs=0.005)
        assert PBEEXXbdifference == pytest.approx(0.91, abs=0.005)
