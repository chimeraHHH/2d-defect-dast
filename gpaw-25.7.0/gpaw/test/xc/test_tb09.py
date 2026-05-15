"""Test Tran Blaha potential."""
import warnings

import pytest
from ase.build import bulk
from ase.dft.bandgap import bandgap

from gpaw import GPAW, PW, Mixer


@pytest.mark.old_gpaw_only
@pytest.mark.libxc
@pytest.mark.mgga
def test_xc_tb09(in_tmp_dir):
    def xc(name):
        return {'name': name, 'stencil': 1}

    k = 8
    atoms = bulk('Si')
    atoms.calc = GPAW(mode=PW(300),
                      mixer=Mixer(0.8, 10, 50.0),
                      kpts={'size': (k, k, k), 'gamma': True},
                      xc=xc('TB09'),
                      convergence={'bands': -3},
                      txt='si.txt')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', module='gpaw.xc.libxc')
        atoms.get_potential_energy()
    gap, (sv, kv, nv), (sc, kc, nc) = bandgap(atoms.calc)
    c = atoms.calc.hamiltonian.xc.c
    print(gap, kv, kc)
    print('c:', c)
    assert abs(gap - 1.246) < 0.01
    assert kv == 0 and kc == 12
    assert abs(c - 1.135) < 0.01
