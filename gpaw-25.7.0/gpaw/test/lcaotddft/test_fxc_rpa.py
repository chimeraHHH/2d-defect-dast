import numpy as np
import pytest

from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.mpi import world


@pytest.mark.rttddft
def test_lcaotddft_fxc_rpa(gpw_files, in_tmp_dir):
    # Time-propagation calculation with fxc
    td_calc = LCAOTDDFT(gpw_files['na2_tddft_poisson'], fxc='RPA',
                        txt='td.out')
    DipoleMomentWriter(td_calc, 'dm.dat')
    td_calc.absorption_kick(np.ones(3) * 1e-5)
    td_calc.propagate(20, 3)
    world.barrier()

    # Check dipole moment file
    data = np.loadtxt('dm.dat')[:, 2:].ravel()
    if 0:
        from gpaw.test import print_reference
        print_reference(data, 'ref', '%.12le')

    ref = [-9.383700894739e-16,
           -9.338586948130e-16,
           2.131582675483e-14,
           8.679923327633e-15,
           7.529517689096e-15,
           2.074867751820e-14,
           1.960742702185e-05,
           1.960742702128e-05,
           1.804030746475e-05,
           3.761997205571e-05,
           3.761997205562e-05,
           3.596681520068e-05,
           5.257367158160e-05,
           5.257367158104e-05,
           5.366663490365e-05]

    tol = 1e-8
    assert data == pytest.approx(ref, abs=tol)
