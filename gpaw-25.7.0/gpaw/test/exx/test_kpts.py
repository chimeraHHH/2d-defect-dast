"""Test case where q=k1-k2 has component outside 0<=q<1 range."""
from typing import Tuple

import pytest
import numpy as np
from ase import Atoms

from gpaw import GPAW, PW
from gpaw.hybrids.eigenvalues import non_self_consistent_eigenvalues
from gpaw.mpi import size
from gpaw.new.pw.nschse import NonSelfConsistentHSE06
from gpaw.new.ase_interface import GPAW as NewGPAW


@pytest.fixture(scope='module')
def atoms() -> Atoms:
    n = 7
    a = Atoms('HH',
              cell=[2, 2, 2.5, 90, 90, 60],
              pbc=1,
              positions=[[0, 0, 0], [0, 0, 0.75]])
    parallel = dict(zip(['domain', 'kpt', 'band'],
                        {1: [1, 1, 1],
                         2: [2, 1, 1],
                         4: [2, 2, 1],
                         8: [2, 2, 2]}[size]))
    a.calc = GPAW(mode=PW(200),
                  kpts=(n, n, 1),
                  xc='PBE',
                  parallel=parallel)
    a.get_potential_energy()
    return a


def bandgap(eps: np.ndarray) -> Tuple[int, int, float]:
    """Find band-gap."""
    k1 = eps[0, :, 0].argmax()
    k2 = eps[0, :, 1].argmin()
    return k1, k2, eps[0, k2, 1] - eps[0, k1, 0]


gaps = {'EXX': 21.45,
        'PBE0': 13.93,
        'HSE06': 14.44,
        'PBE': 11.63}


@pytest.mark.libxc
@pytest.mark.hybrids
@pytest.mark.parametrize('xc', ['EXX', 'PBE0', 'HSE06'])
def test_kpts(xc: str, atoms: Atoms) -> None:
    c = atoms.calc
    e0, v0, v = non_self_consistent_eigenvalues(c, xc)
    e = e0 - v0 + v
    k1, k2, gap = bandgap(e)
    assert k1 == 4 and k2 == 7
    assert gap == pytest.approx(gaps[xc], abs=0.01)
    k1, k2, gap = bandgap(e0)
    assert k1 == 4 and k2 == 7
    assert gap == pytest.approx(gaps['PBE'], abs=0.01)


def test_2d_non_self_consistent():
    a = Atoms('Li',
              [[0.0, 0.0, 1.0]],
              cell=[1.5, 1.5, 2.0, 90, 90, 120],
              pbc=(1, 1, 0))

    n = 2
    a.calc = NewGPAW(
        mode=PW(200),
        convergence={'density': 1e-5},
        kpts=(n, n, 1),
        txt=None)
    a.get_potential_energy()

    eref_kn = np.array(
        [[-6.0938938, 31.82737621, 36.83364518, 53.28369147],
         [13.0202785, 28.45570036, 38.86882486, 43.44290272]])

    if a.calc.dft.comm.size == 1:
        e0, v0, v = non_self_consistent_eigenvalues(a.calc, 'HSE06')
        e_skn = e0 - v0 + v
        assert e_skn[0] == pytest.approx(eref_kn, rel=1e-5)

    hse = NonSelfConsistentHSE06.from_dft_calculation(a.calc.dft)
    _, e_skn = hse.calculate(a.calc.dft.ibzwfs)
    assert e_skn[0] == pytest.approx(eref_kn, rel=1e-5)
    _, e_skn = hse.calculate(a.calc.dft.ibzwfs, na=0, nb=1)
    assert e_skn[0, :, 0] == pytest.approx(eref_kn[:, 0], rel=1e-5)


if __name__ == '__main__':
    test_2d_non_self_consistent()
