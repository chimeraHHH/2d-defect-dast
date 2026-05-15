import pytest
import numpy as np

from gpaw.tddft import TDDFT, DipoleMomentWriter
from gpaw.mpi import world


@pytest.mark.gllb
@pytest.mark.libxc
def test_tddft_fxc_linearize(in_tmp_dir, gpw_files):

    fxc = 'LDA'
    # Time-propagation calculation with linearize_to_fxc()
    td_calc = TDDFT(gpw_files['sih4_xc_gllbsc_fd'], txt='td.out')
    td_calc.linearize_to_xc(fxc)
    DipoleMomentWriter(td_calc, 'dm.dat')
    td_calc.absorption_kick(np.ones(3) * 1e-5)
    td_calc.propagate(20, 4)
    world.barrier()

    # Test the absolute values
    data = np.loadtxt('dm.dat').ravel()
    if 0:
        from gpaw.test import print_reference
        print_reference(data, 'ref', '%.12le')

    ref = [0.000000000000e+00,
           -1.578005120000e-15,
           -7.044387182078e-15,
           -1.863704251170e-14,
           5.661102599682e-15,
           0.000000000000e+00,
           5.675065320000e-16,
           1.468753200587e-10,
           1.579510728549e-10,
           1.480745858862e-10,
           8.268274700000e-01,
           9.001411310000e-16,
           5.997264413918e-05,
           5.996562448975e-05,
           5.996489353702e-05,
           1.653654930000e+00,
           -3.560336050000e-15,
           1.059270665342e-04,
           1.059358921818e-04,
           1.059367170602e-04,
           2.480482400000e+00,
           3.100362540000e-15,
           1.338025167034e-04,
           1.338146631769e-04,
           1.338165200102e-04,
           3.307309870000e+00,
           3.279248870000e-15,
           1.423796525306e-04,
           1.423871979410e-04,
           1.423890659304e-04]

    tol = 1e-7
    assert data == pytest.approx(ref, abs=tol)
