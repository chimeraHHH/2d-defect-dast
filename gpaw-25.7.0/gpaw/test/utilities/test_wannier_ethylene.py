"""GPAW wannier example for ethylene corresponding to the ASE Wannier
tutorial.
"""
import pytest
import numpy as np
from ase import Atoms

from gpaw import GPAW
from gpaw.mpi import size
from gpaw.wannier.overlaps import calculate_overlaps
from gpaw.wannier.edmiston_ruedenberg import localize

pytestmark = pytest.mark.ci


@pytest.fixture(scope='module')
def ethylene():
    a = 6.0  # Size of unit cell (Angstrom)

    mol = Atoms('H2C2H2',
                [(-1.235, -0.936, 0),
                 (-1.235, 0.936, 0),
                 (-0.660, 0.000, 0),
                 (0.660, 0.000, 0),
                 (1.235, -0.936, 0),
                 (1.235, 0.936, 0)],
                cell=(a, a, a),
                pbc=True)
    mol.center()

    mol.calc = GPAW(mode='fd',
                    txt=None,
                    nbands=8,
                    gpts=(32, 32, 32),
                    convergence={'eigenstates': 3.3e-5})
    mol.get_potential_energy()
    return mol


@pytest.mark.skipif(size > 1, reason='Not parallelized')
def test_ethylene_energy(ethylene):
    e = ethylene.get_potential_energy()
    assert e == pytest.approx(-33.328, abs=0.002)


def check(calc):
    wannier = localize(calculate_overlaps(calc, n1=0, n2=6, nwannier=6))

    centers = wannier.centers
    print(centers)
    expected = [[1.950, 2.376, 3.000],
                [1.950, 3.624, 3.000],
                [3.000, 3.000, 2.671],
                [3.000, 3.000, 3.329],
                [4.050, 2.376, 3.000],
                [4.050, 3.624, 3.000]]
    assert wannier.value == pytest.approx(13.7995, abs=0.016)
    for center in centers:
        i = 0
        while np.sum((expected[i] - center)**2) > 0.01:
            i += 1
            if i == len(expected):
                raise RuntimeError('Correct center not found')
        expected.pop(i)


@pytest.mark.skipif(size > 1, reason='Not parallelized')
def test_wannier_centers(ethylene):
    check(ethylene.calc)


@pytest.mark.skipif(size > 1, reason='Not parallelized')
def test_wannier_centers_gpw(ethylene, in_tmp_dir):
    ethylene.calc.write('ethylene.gpw', 'all')
    check(GPAW('ethylene.gpw', txt=None))
