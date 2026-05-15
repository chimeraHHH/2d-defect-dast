import pytest
from ase import Atoms
from gpaw import GPAW, Mixer, Davidson

# ??? g = Generator('H', 'TPSS', scalarrel=True, nofiles=True)


@pytest.mark.mgga
def test_mgga_nsc_MGGA(in_tmp_dir):
    def xc(name):
        return dict(name=name, stencil=1)

    atoms = Atoms('H', magmoms=[1], pbc=True)
    atoms.center(vacuum=3)

    def getkwargs():
        return dict(mode='fd',
                    eigensolver=Davidson(3),
                    mixer=Mixer(0.7, 5, 50.0),
                    parallel=dict(augment_grids=True),
                    gpts=(32, 32, 32), nbands=1, xc=xc('oldPBE'))

    calc = GPAW(txt='Hnsc.txt', **getkwargs())
    atoms.calc = calc
    e1 = atoms.get_potential_energy()
    _ = calc.get_reference_energy()
    de12t = calc.get_xc_difference(xc('TPSS'))
    de12m = calc.get_xc_difference(xc('M06-L'))
    de12m2 = calc.get_xc_difference(xc('M06-L'))
    de12r = calc.get_xc_difference(xc('revTPSS'))

    print('================')
    print('e1 = ', e1)
    print('de12t = ', de12t)
    print('de12m = ', de12m)
    print('de12r = ', de12r)
    print('tpss = ', e1 + de12t)
    print('m06l = ', e1 + de12m)
    print('m06l = ', e1 + de12m2)
    print('revtpss = ', e1 + de12r)
    print('================')

    assert e1 + de12t == pytest.approx(-1.11723235592, abs=0.005)
    assert e1 + de12m == pytest.approx(-1.18207312133, abs=0.005)
    assert e1 + de12r == pytest.approx(-1.10093196353, abs=0.005)

    # ??? g = Generator('He', 'TPSS', scalarrel=True, nofiles=True)

    atomsHe = Atoms('He', pbc=True)
    atomsHe.center(vacuum=3)
    calc = GPAW(txt='Hensc.txt', **getkwargs())
    atomsHe.calc = calc
    e1He = atomsHe.get_potential_energy()
    _ = calc.get_reference_energy()
    de12tHe = calc.get_xc_difference(xc('TPSS'))
    de12mHe = calc.get_xc_difference(xc('M06-L'))
    de12rHe = calc.get_xc_difference(xc('revTPSS'))

    print('================')
    print('e1He = ', e1He)
    print('de12tHe = ', de12tHe)
    print('de12mHe = ', de12mHe)
    print('de12rHe = ', de12rHe)
    print('tpss = ', e1He + de12tHe)
    print('m06l = ', e1He + de12mHe)
    print('revtpss = ', e1He + de12rHe)
    print('================')

    assert e1He + de12tHe == pytest.approx(-0.409972893501, abs=0.005)
    assert e1He + de12mHe == pytest.approx(-0.487249688866, abs=0.005)
    assert e1He + de12rHe == pytest.approx(-0.447232286813, abs=0.005)

    energy_tolerance = 0.001
    assert e1 == pytest.approx(-1.124, abs=energy_tolerance)
    assert e1He == pytest.approx(0.0100192, abs=energy_tolerance)
