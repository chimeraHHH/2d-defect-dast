import numpy as np
import pytest
from ase.build import molecule
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lrtddft2 import LrTDDFT2
from gpaw.mpi import world
from gpaw.tddft.spectrum import photoabsorption_spectrum


@pytest.mark.rttddft
def test_lcaotddft_lcaotddft_vs_lrtddft2_rpa(in_tmp_dir):
    atoms = molecule('Na2')
    atoms.center(vacuum=4.0)

    # Ground-state calculation
    calc = GPAW(nbands=2, h=0.4, setups=dict(Na='1'),
                basis='sz(dzp)', mode='lcao', xc='oldLDA',
                convergence={'density': 1e-8},
                symmetry={'point_group': False},
                txt='gs.out')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', mode='all')

    # Time-propagation calculation
    td_calc = LCAOTDDFT('gs.gpw', fxc='RPA', txt='td.out')
    DipoleMomentWriter(td_calc, 'dm.dat')
    td_calc.absorption_kick([0, 0, 1e-5])
    td_calc.propagate(30, 150)
    photoabsorption_spectrum('dm.dat', 'spec.dat',
                             e_max=10, width=0.5, delta_e=0.1)

    # LrTDDFT2 calculation
    calc = GPAW('gs.gpw', txt='lr.out')
    # It doesn't matter which fxc is here
    lr = LrTDDFT2('lr2', calc, fxc='LDA')
    lr.K_matrix.fxc_pre = 0.0  # Ignore fxc part
    lr.calculate()
    lr.get_spectrum('lr_spec.dat', 0, 10.1, 0.1, width=0.5)
    world.barrier()

    # Scale spectra due to slightly different definitions
    spec1_ej = np.loadtxt('spec.dat')
    data1_e = spec1_ej[:, 3]
    data1_e[1:] /= spec1_ej[1:, 0]
    spec2_ej = np.loadtxt('lr_spec.dat')
    E = lr.lr_transitions.get_transitions()[0][0]
    data2_e = spec2_ej[:, 5] / E

    # One can decrease the tolerance by decreasing the time step
    # and other parameters
    assert data1_e == pytest.approx(data2_e, abs=0.01)
