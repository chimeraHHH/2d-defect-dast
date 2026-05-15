import pytest
import numpy as np

from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.mpi import world


@pytest.mark.gllb
@pytest.mark.libxc
@pytest.mark.rttddft
def test_lcaotddft_fxc_vs_linearize(gpw_files, in_tmp_dir):
    fxc = 'LDA'
    # Time-propagation calculation with fxc
    td_calc = LCAOTDDFT(gpw_files['sih4_xc_gllbsc_lcao'],
                        fxc=fxc, txt='td_fxc.out')
    DipoleMomentWriter(td_calc, 'dm_fxc.dat')
    td_calc.absorption_kick(np.ones(3) * 1e-5)
    td_calc.propagate(20, 4)

    # Time-propagation calculation with linearize_to_fxc()
    td_calc = LCAOTDDFT(gpw_files['sih4_xc_gllbsc_lcao'], txt='td_lin.out')
    td_calc.linearize_to_xc(fxc)
    DipoleMomentWriter(td_calc, 'dm_lin.dat')
    td_calc.absorption_kick(np.ones(3) * 1e-5)
    td_calc.propagate(20, 4)

    # Test the equivalence
    world.barrier()
    ref = np.loadtxt('dm_fxc.dat').ravel()
    data = np.loadtxt('dm_lin.dat').ravel()

    tol = 1e-9
    assert data == pytest.approx(ref, abs=tol)

    # Test the absolute values
    if 0:
        from gpaw.test import print_reference
        print_reference(data, 'ref', '%.12le')

    ref = [0.00000000e+00, 1.62932507e-15,
           -7.87693913e-15, -1.14696427e-14,
           -5.80989675e-15, 0.00000000e+00,
           3.62636797e-15, -3.29113009e-14,
           -3.51289578e-14, -3.67931628e-14,
           8.26827470e-01, -1.81172385e-15,
           6.17633458e-05, 6.17633458e-05,
           6.17633458e-05, 1.65365493e+00,
           -8.01473100e-16, 9.88572381e-05,
           9.88572381e-05, 9.88572381e-05,
           2.48048240e+00, -2.63296003e-15,
           1.04139540e-04, 1.04139540e-04,
           1.04139540e-04, 3.30730987e+00,
           2.29033034e-15, 8.81733367e-05,
           8.81733367e-05, 8.81733367e-05]

    print('result')
    print(data.tolist())

    tol = 1e-12
    assert data == pytest.approx(ref, abs=tol)
