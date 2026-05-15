# Test that LCAO wavefunctions are available and equal after restarts
# in normal as well as 'all' mode

import pytest
from ase.build import molecule
from gpaw import GPAW
from gpaw.mpi import world


def test_restart(in_tmp_dir, gpaw_new):
    if gpaw_new and world.size > 1:
        pytest.skip('LCAO get_ps_w_f() not parallelized')
    system = molecule('H2')
    system.center(vacuum=2.5)

    calc = GPAW(mode='lcao', basis='sz(dzp)', h=0.3, nbands=1, txt=None)
    system.calc = calc
    system.get_potential_energy()
    wf = calc.get_pseudo_wave_function(0)

    for mode in ['all', 'normal']:
        fname = 'lcao-restart.%s.gpw' % mode
        calc.write(fname, mode=dict(normal='', all='all')[mode])

        calc2 = GPAW(fname, txt=None)
        if mode == 'normal':
            continue
        wf2 = calc2.get_pseudo_wave_function(0)
        assert wf2 == pytest.approx(wf, abs=1e-14)
