import numpy as np
import pytest

from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.mpi import world


@pytest.mark.rttddft
def test_lcaotddft_fxc_is_xc(gpw_files, in_tmp_dir):
    # Time-propagation calculation without fxc
    td_calc = LCAOTDDFT(gpw_files['na2_tddft_dzp'], txt='td.out')
    DipoleMomentWriter(td_calc, 'dm.dat')
    td_calc.absorption_kick(np.ones(3) * 1e-5)
    td_calc.propagate(20, 4)
    world.barrier()

    # Check dipole moment file
    ref = np.loadtxt('dm.dat').ravel()

    # Time-propagation calculation with fxc=xc
    td_calc = LCAOTDDFT(gpw_files['na2_tddft_dzp'], fxc='LDA',
                        txt='td_fxc.out')
    DipoleMomentWriter(td_calc, 'dm_fxc.dat')
    td_calc.absorption_kick(np.ones(3) * 1e-5)
    td_calc.propagate(20, 1)
    td_calc.write('td_fxc.gpw', mode='all')

    # Restart from the restart point
    td_calc = LCAOTDDFT('td_fxc.gpw', txt='td_fxc2.out')
    DipoleMomentWriter(td_calc, 'dm_fxc.dat')
    td_calc.propagate(20, 3)

    # Check dipole moment file
    data = np.loadtxt('dm_fxc.dat')[[0, 1, 2, 4, 5, 6]].ravel()

    tol = 1e-9
    assert data == pytest.approx(ref, abs=tol)
