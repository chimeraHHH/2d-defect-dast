import pytest
from ase import Atoms

from gpaw import GPAW
from gpaw.calculator import DeprecatedParameterWarning


@pytest.mark.ci
@pytest.mark.mgga
def test_fixdensity(in_tmp_dir, gpaw_new):
    a = 2.5
    slab = Atoms('Li', cell=(a, a, 2 * a), pbc=1)
    slab.calc = GPAW(mode='fd',
                     xc='revTPSS',
                     h=0.15,
                     kpts=(3, 3, 1), txt='li-1.txt',
                     parallel=dict(kpt=1))
    slab.get_potential_energy()
    slab.calc.write('li.gpw', mode='all')
    slab.calc.write('li_nowf.gpw')

    # Gamma point:
    e1 = slab.calc.get_eigenvalues(kpt=0)[0]
    f1 = slab.calc.get_fermi_level()

    kpts = [(0, 0, 0)]

    # Fix density and continue:
    calc = slab.calc.fixed_density(
        txt='li-2.txt',
        nbands=5,
        kpts=kpts)
    e2 = calc.get_eigenvalues(kpt=0)[0]
    f2 = calc.get_fermi_level()

    # Start from gpw-file:
    calc = GPAW('li.gpw', txt=None)
    calc = calc.fixed_density(
        txt='li-3.txt',
        nbands=5,
        kpts=kpts)
    e3 = calc.get_eigenvalues(kpt=0)[0]
    f3 = calc.get_fermi_level()

    assert f2 == pytest.approx(f1, abs=1e-10)
    assert f3 == pytest.approx(f1, abs=1e-10)
    assert e2 == pytest.approx(e1, abs=3e-5)
    assert e3 == pytest.approx(e1, abs=3e-5)

    if not gpaw_new:
        calc = GPAW('li.gpw',
                    txt='li-4.txt',
                    fixdensity=True,
                    nbands=5,
                    kpts=kpts,
                    symmetry='off')
        try:
            with pytest.warns(DeprecatedParameterWarning):
                calc.get_potential_energy()
        except ValueError:
            pass
        else:
            assert False

    calc = GPAW('li_nowf.gpw')
    if not gpaw_new:
        with pytest.raises(ValueError):
            calc = calc.fixed_density(txt='li-3.txt', nbands=5, kpts=kpts)
    else:
        calc = calc.fixed_density(txt='li-3.txt', nbands=5, kpts=kpts)
        e4 = calc.get_eigenvalues(kpt=0)[0]
        f4 = calc.get_fermi_level()
        assert f4 == pytest.approx(f1, abs=1e-10)
        assert e4 == pytest.approx(e1, abs=3e-5)
