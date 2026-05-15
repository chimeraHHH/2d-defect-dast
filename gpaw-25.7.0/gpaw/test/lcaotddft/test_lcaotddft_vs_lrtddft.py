import pytest
import numpy as np

from ase.units import Hartree
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.tddft.spectrum import photoabsorption_spectrum as spec_td
from gpaw.lrtddft import LrTDDFT
from gpaw.lrtddft import photoabsorption_spectrum as spec_lr
from gpaw.lrtddft2 import LrTDDFT2
from gpaw.mpi import world


pytestmark = [pytest.mark.usefixtures('module_tmp_path')]


@pytest.fixture(scope='module')
def time_propagation_calculation(gpw_files):
    td_calc = LCAOTDDFT(gpw_files['na2_tddft_sz'],
                        txt='na2_tddft_sz_td.out')
    DipoleMomentWriter(td_calc, 'dm.dat')
    td_calc.absorption_kick([0, 0, 1e-5])
    td_calc.propagate(30, 150)
    spec_td('dm.dat', 'spec_td.dat',
            e_min=0, e_max=10, width=0.5, delta_e=0.1)
    world.barrier()

    # Scale energy out due to \omega vs \omega_I difference in
    # broadened spectra in RT-TDDFT and LR-TDDFT
    data_ej = np.loadtxt('spec_td.dat')
    spec_e = data_ej[:, 3]
    spec_e[1:] /= data_ej[1:, 0]
    return spec_e


@pytest.fixture(scope='module')
def lrtddft_calculation(gpw_files):
    calc = GPAW(gpw_files['na2_tddft_sz'], txt=None)
    lr = LrTDDFT(calc, xc='LDA', txt='lr.out')
    lr.diagonalize()
    spec_lr(lr, 'spec_lr.dat',
            e_min=0, e_max=10, width=0.5, delta_e=0.1)
    world.barrier()

    # Scale energy out due to \omega vs \omega_I difference in
    # broadened spectra in RT-TDDFT and LR-TDDFT
    data_ej = np.loadtxt('spec_lr.dat')
    spec_e = data_ej[:, 4]
    spec_e[1:] /= lr[0].get_energy() * Hartree
    return spec_e


@pytest.fixture(scope='module')
def lrtddft2_calculation(gpw_files):
    calc = GPAW(gpw_files['na2_tddft_sz'], txt='lr2.out')
    lr2 = LrTDDFT2('lr2', calc, fxc='LDA')
    lr2.calculate()
    lr2.get_spectrum('spec_lr2.dat', 0, 10.1, 0.1, width=0.5)
    world.barrier()

    # Scale energy out due to \omega vs \omega_I difference in
    # broadened spectra in RT-TDDFT and LR-TDDFT
    data_ej = np.loadtxt('spec_lr2.dat')
    spec_e = data_ej[:, 5]
    spec_e[1:] /= lr2.lr_transitions.get_transitions()[0][0]
    return spec_e


@pytest.mark.rttddft
def test_lcaotddft_vs_lrtddft(time_propagation_calculation,
                              lrtddft_calculation):
    # One can decrease the tolerance by decreasing the time step
    # and other parameters
    assert (time_propagation_calculation
            == pytest.approx(lrtddft_calculation, abs=1e-2))


@pytest.mark.rttddft
def test_lcaotddft_vs_lrtddft2(time_propagation_calculation,
                               lrtddft2_calculation):
    # One can decrease the tolerance by decreasing the time step
    # and other parameters
    assert (time_propagation_calculation
            == pytest.approx(lrtddft2_calculation, abs=1e-2))


@pytest.mark.rttddft
def test_lrtddft_vs_lrtddft2(lrtddft_calculation,
                             lrtddft2_calculation):
    assert (lrtddft_calculation
            == pytest.approx(lrtddft2_calculation, abs=1e-3))
