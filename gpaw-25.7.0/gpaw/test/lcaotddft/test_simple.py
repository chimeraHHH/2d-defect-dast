import numpy as np
import pytest

from ase.units import Bohr, Hartree
from gpaw.external import ConstantElectricField
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.energywriter import EnergyWriter
from gpaw.lcaotddft.laser import create_laser
from gpaw.mpi import world
from gpaw.tddft.spectrum import photoabsorption_spectrum


@pytest.mark.rttddft
def test_lcaotddft_simple(gpw_files, in_tmp_dir):
    # Time-propagation calculation
    td_calc = LCAOTDDFT(gpw_files['na2_tddft_poisson'], txt='td.out')
    dmat = DensityMatrix(td_calc)
    DipoleMomentWriter(td_calc, 'dm.dat')
    EnergyWriter(td_calc, dmat, 'energy.dat')
    td_calc.absorption_kick(np.ones(3) * 1e-5)
    td_calc.propagate(20, 3)
    photoabsorption_spectrum('dm.dat', 'spec.dat', delta_e=5)
    world.barrier()

    # Test dipole moment
    data_i = np.loadtxt('dm.dat')[:, 2:].ravel()
    if 0:
        from gpaw.test import print_reference
        print_reference(data_i, 'ref_i', '%.12le')

    ref_i = [-9.383700894739e-16,
             -9.338586948130e-16,
             2.131582675483e-14,
             8.679923327633e-15,
             7.529517689096e-15,
             2.074867751820e-14,
             1.967175558125e-05,
             1.967175557952e-05,
             1.805004256446e-05,
             3.799528978877e-05,
             3.799528978943e-05,
             3.602506734201e-05,
             5.371491974467e-05,
             5.371491974534e-05,
             5.385046706407e-05]

    tol = 1e-8
    assert data_i == pytest.approx(ref_i, abs=tol)

    # Test spectrum
    data_i = np.loadtxt('spec.dat').ravel()
    if 0:
        from gpaw.test import print_reference
        print_reference(data_i, 'ref_i', '%.12le')

    ref_i = [0.000000000000e+00,
             0.000000000000e+00,
             0.000000000000e+00,
             0.000000000000e+00,
             5.000000000000e+00,
             4.500226856200e-03,
             4.500226856200e-03,
             4.408379542600e-03,
             1.000000000000e+01,
             1.659426124300e-02,
             1.659426124300e-02,
             1.623812256800e-02,
             1.500000000000e+01,
             3.244686838800e-02,
             3.244686838800e-02,
             3.168682490900e-02,
             2.000000000000e+01,
             4.684883744600e-02,
             4.684883744600e-02,
             4.559689861200e-02,
             2.500000000000e+01,
             5.466781222200e-02,
             5.466781222200e-02,
             5.290171209500e-02,
             3.000000000000e+01,
             5.231586230700e-02,
             5.231586230700e-02,
             5.008661764300e-02]

    tol = 1e-5
    assert data_i == pytest.approx(ref_i, abs=tol)

    # Test energy - almost no energy should be absorbed due to the delta kick
    data_i = np.loadtxt('energy.dat')[:, 1:].ravel()

    tol = 1e-8
    assert data_i == pytest.approx(0, abs=tol)


@pytest.mark.rttddft
def test_lcaotddft_laser(gpw_files, in_tmp_dir):
    # Simple test with laser instead of delta-kick, so
    # that system absorbs a meaningful amount of energy
    pulse = {'name': 'GaussianPulse', 'strength': 1e-3, 'time0': 0,
             'frequency': 8.6, 'sigma': 1, 'sincos': 'sin'}
    pulse = create_laser(pulse)

    ext = ConstantElectricField(Hartree / Bohr, [1, 1, 1])
    # Time-propagation calculation
    td_calc = LCAOTDDFT(gpw_files['na2_tddft_poisson'],
                        td_potential={'ext': ext, 'laser': pulse},
                        txt='tdout.out')
    dmat = DensityMatrix(td_calc)
    EnergyWriter(td_calc, dmat, 'energy.dat')
    td_calc.propagate(20, 5)
    world.barrier()

    # Test energy
    data_i = np.loadtxt('energy.dat')[:, 1:].ravel()
    if 0:
        from gpaw.test import print_reference
        print_reference(data_i, 'ref_i', '%.12le')

    ref_i = [0.000000000000e+00,
             0.000000000000e+00,
             0.000000000000e+00,
             0.000000000000e+00,
             0.000000000000e+00,
             0.000000000000e+00,
             3.722528463257e-08,
             -3.245893376302e-08,
             4.013500642941e-10,
             0.000000000000e+00,
             -5.280626269588e-09,
             1.342903965806e-08,
             5.637101546108e-07,
             -4.922112830652e-07,
             6.093572091359e-09,
             0.000000000000e+00,
             -8.051342392790e-08,
             2.051779581791e-07,
             2.611917989781e-06,
             -2.279857594711e-06,
             2.837519125221e-08,
             0.000000000000e+00,
             -3.781277957415e-07,
             9.681304530273e-07,
             7.288861718524e-06,
             -6.358019188912e-06,
             7.976064814175e-08,
             0.000000000000e+00,
             -1.075880055268e-06,
             2.773092058317e-06,
             1.513288133970e-05,
             -1.318711657139e-05,
             1.672415533838e-07,
             0.000000000000e+00,
             -2.291283559219e-06,
             5.959395212557e-06]

    tol = 1e-8
    assert data_i == pytest.approx(ref_i, abs=tol)


@pytest.mark.rttddft
def test_lcaotddft_fail_with_symmetry(gpw_files, in_tmp_dir):
    # Time-propagation calculation
    # should not be allowed with symmetries
    with pytest.raises(ValueError):
        LCAOTDDFT(gpw_files['na2_tddft_poisson_sym'], txt='td.out')
