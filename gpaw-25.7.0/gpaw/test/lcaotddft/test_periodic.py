import numpy as np
import pytest

from ase.build import fcc111

from gpaw import GPAW
from gpaw.mpi import world, serial_comm
from gpaw.lcaotddft.wfwriter import WaveFunctionReader

from gpaw.test import only_on_master
from . import (parallel_options, calculate_error, calculate_time_propagation,
               check_wfs)

pytestmark = pytest.mark.usefixtures('module_tmp_path')

parallel_i = parallel_options(include_kpt=True)


@pytest.fixture(scope='module')
@only_on_master(world)
def initialize_system():
    comm = serial_comm

    # Ground-state calculation
    atoms = fcc111('Al', size=(1, 1, 2), vacuum=4.0)
    atoms.symbols[0] = 'Li'
    calc = GPAW(nbands=4,
                h=0.4,
                kpts={'size': (3, 3, 1)},
                basis='sz(dzp)',
                mode='lcao',
                convergence={'density': 1e-8},
                symmetry={'point_group': False},
                communicator=comm,
                txt='gs.out')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', mode='all')

    # Time-propagation calculation
    calculate_time_propagation('gs.gpw',
                               kick=[0, 0, 1e-5],
                               communicator=comm)


@pytest.mark.rttddft
def test_propagated_wave_function(initialize_system, module_tmp_path):
    wfr = WaveFunctionReader(module_tmp_path / 'wf.ulm')
    coeff = wfr[-1].wave_functions.coefficients
    coeff = coeff[np.ix_([0], [0, 1], [1, 3], [0, 1, 2])]
    # Normalize the wave function sign
    coeff = np.sign(coeff.real[..., 0, np.newaxis]) * coeff
    ref = [[[[5.4119034398864430e-01 + 4.6958807325576735e-01j,
              -5.8836045927143954e-01 - 5.1047688429408378e-01j,
              -6.5609314466400698e-06 - 5.8109609173527947e-06j],
             [1.6425837099429430e-06 - 1.4779657236004961e-06j,
              -8.7230715222772428e-07 + 8.9374679369814926e-07j,
              3.1300283337601806e+00 - 2.7306795126551076e+00j]],
            [[1.9820345503468246e+00 + 1.0562314330323577e+00j,
              -1.5008623926242098e-01 + 4.5817475674967340e-01j,
              -4.8385783015916195e-01 - 5.3676335879786385e-01j],
             [2.4227856141643818e+00 + 3.7767002050641824e-01j,
              -2.6174901880264838e+00 + 1.9885717875694848e+00j,
              7.2641847473298660e-01 + 1.6020733667409095e+00j]]]]
    err = calculate_error(coeff, ref)
    assert err < 7e-9


@pytest.mark.rttddft
@pytest.mark.parametrize('parallel', parallel_i)
def test_propagation(initialize_system, module_tmp_path, parallel, in_tmp_dir):
    calculate_time_propagation(module_tmp_path / 'gs.gpw',
                               kick=[0, 0, 1e-5],
                               parallel=parallel)
    check_wfs(module_tmp_path / 'wf.ulm', 'wf.ulm', atol=1e-12)
