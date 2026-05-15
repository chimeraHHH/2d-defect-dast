import pytest
from gpaw.mpi import world
from ase import Atoms
from gpaw import GPAW

from gpaw.utilities.kspot import AllElectronPotential
pytestmark = pytest.mark.skipif(world.size > 1,
                                reason='world.size > 1')


@pytest.mark.old_gpaw_only
def test_muffintinpot(in_tmp_dir):
    if 1:
        be = Atoms(symbols='Be', positions=[(0, 0, 0)])
        be.center(vacuum=5)
        calc = GPAW(mode='fd',
                    gpts=(64, 64, 64),
                    xc='LDA',
                    nbands=1)  # 0.1 required for accuracy
        be.calc = calc
        e = be.get_potential_energy()

        energy_tolerance = 0.001
        assert e == pytest.approx(0.00246471, abs=energy_tolerance)

    # be, calc = restart("be.gpw")
    AllElectronPotential(calc).write_spherical_ks_potentials('bepot.txt')
    f = open('bepot.txt')
    lines = f.readlines()
    f.close()
    mmax = 0
    for l in lines:
        mmax = max(abs(eval(l.split(' ')[3])), mmax)

    print("Max error: ", mmax)
    assert mmax < 0.009
