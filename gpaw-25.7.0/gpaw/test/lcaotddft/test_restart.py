import pytest
import numpy as np
import ase.io.ulm as ulm

from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.frequencydensitymatrix import FrequencyDensityMatrix
from gpaw.mpi import world
from gpaw.tddft.folding import frequencies


@pytest.mark.gllb
@pytest.mark.libxc
@pytest.mark.rttddft
def test_lcaotddft_restart(gpw_files, in_tmp_dir):
    # Time-propagation calculation
    td_calc = LCAOTDDFT(gpw_files['sih4_xc_gllbsc_lcao'], txt='td.out')
    DipoleMomentWriter(td_calc, 'dm.dat')
    dmat = DensityMatrix(td_calc)
    freqs = frequencies(np.arange(0.05, 6.01, 1.0), None, None)
    fdm = FrequencyDensityMatrix(td_calc, dmat, frequencies=freqs)
    td_calc.absorption_kick(np.ones(3) * 1e-5)
    td_calc.propagate(20, 3)

    # Write a restart point
    td_calc.write('td.gpw', mode='all')
    fdm.write('fdm_restart_point.ulm')

    # Keep propagating
    td_calc.propagate(20, 3)
    fdm.write('fdm_final_ref.ulm')

    # Restart from the restart point
    td_calc = LCAOTDDFT('td.gpw', txt='td2.out')
    DipoleMomentWriter(td_calc, 'dm.dat')
    dmat = DensityMatrix(td_calc)
    fdm = FrequencyDensityMatrix(td_calc, dmat,
                                 filename='fdm_restart_point.ulm')
    td_calc.propagate(20, 3)
    fdm.write('fdm_final_check.ulm')
    world.barrier()

    # Check dipole moment file
    data_tj = np.loadtxt('dm.dat')
    # Original run
    ref_i = data_tj[4:8].ravel()
    # Restarted steps
    data_i = data_tj[8:].ravel()

    tol = 1e-10
    assert data_i == pytest.approx(ref_i, abs=tol)

    tol = 1e-8
    # Check frequency density matrix file
    with ulm.open('fdm_final_ref.ulm') as fdm_ref:
        with ulm.open('fdm_final_check.ulm') as fdm_check:
            for key in ['FReDrho_wuMM', 'FImDrho_wuMM', 'rho0_uMM', 'time']:
                assert fdm_ref.get(key) == pytest.approx(fdm_check.get(key),
                                                         abs=tol)

    # Test the absolute values
    data = np.loadtxt('dm.dat')[:8].ravel()
    if 0:
        from gpaw.test import print_reference
        print_reference(data, 'ref', '%.12le')

    ref = [0.000000000000e+00,
           1.440746980000e-15,
           -5.150207058975e-14,
           -2.111502433286e-14,
           -7.898943127163e-15,
           0.000000000000e+00,
           2.611197480000e-15,
           -8.396549188150e-14,
           -2.905622138206e-14,
           -3.511635310469e-14,
           8.268274700000e-01,
           7.301260440000e-17,
           6.205222369795e-05,
           6.205222375484e-05,
           6.205222374486e-05,
           1.653654930000e+00,
           -2.352185940000e-15,
           1.001902218476e-04,
           1.001902218874e-04,
           1.001902218743e-04,
           2.480482400000e+00,
           1.199595930000e-15,
           1.069904191357e-04,
           1.069904191591e-04,
           1.069904191494e-04,
           3.307309870000e+00,
           1.993145110000e-16,
           9.190706194380e-05,
           9.190706194210e-05,
           9.190706193849e-05,
           4.134137330000e+00,
           1.527615730000e-15,
           6.808273775138e-05,
           6.808273772079e-05,
           6.808273772212e-05,
           4.960964800000e+00,
           1.490415960000e-15,
           4.135613094036e-05,
           4.135613089495e-05,
           4.135613089985e-05]

    print('result')
    print(data.tolist())

    tol = 1e-9
    assert data == pytest.approx(ref, abs=tol)
