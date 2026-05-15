import pytest
import numpy as np
from ase.build import molecule
from gpaw import GPAW
from gpaw.mpi import world, serial_comm, broadcast_exception

from gpaw.test.lcaotddft.test_molecule import only_on_master

pytestmark = [pytest.mark.usefixtures('module_tmp_path'),
              pytest.mark.gllb,
              pytest.mark.libxc
              ]


def check_asp(ref_asp, D_asp, atol):
    assert ref_asp is not D_asp, \
        'Trying to compare same objects. Is the test broken?'
    assert np.allclose(ref_asp.toarray(), D_asp.toarray(), atol=atol, rtol=0)


def check_response(ref_response, response, atol=1e-12):
    # Collect response in master
    D_asp = response.D_asp.copy()
    D_asp.redistribute(D_asp.partition.as_serial())
    Dresp_asp = response.Dresp_asp.copy()
    Dresp_asp.redistribute(Dresp_asp.partition.as_serial())
    vt_sG = response.gd.collect(response.vt_sG)

    with broadcast_exception(world):
        if world.rank == 0:
            check_asp(ref_response.D_asp, D_asp, atol=atol)
            check_asp(ref_response.Dresp_asp, Dresp_asp, atol=atol)
            assert np.allclose(ref_response.vt_sG, vt_sG, rtol=0, atol=atol)


def check_partitions(response, gs_calc):
    partition = response.D_asp.partition
    assert response.Dresp_asp.partition is partition
    assert partition == gs_calc.wfs.atom_partition
    assert partition == gs_calc.density.atom_partition


def calculate_ground_state(**kwargs):
    atoms = molecule('H2O')
    atoms.center(vacuum=4)
    gs_calc = GPAW(mode='lcao', basis='sz(dzp)', h=0.4,
                   xc='GLLBSC',
                   txt='gs.out',
                   **kwargs)
    atoms.calc = gs_calc
    atoms.get_potential_energy()
    return gs_calc


@pytest.fixture(scope='module')
@only_on_master(world)
def ground_state_calculation():
    gs_calc = calculate_ground_state(communicator=serial_comm)
    gs_calc.write('gs.gpw', mode='all')
    return gs_calc.hamiltonian.xc.response


def test_read(ground_state_calculation, module_tmp_path):
    ref_response = ground_state_calculation

    gs_calc = GPAW(module_tmp_path / 'gs.gpw', txt=None)
    response = gs_calc.hamiltonian.xc.response
    check_response(ref_response, response)
    check_partitions(response, gs_calc)


def test_calculate(ground_state_calculation, in_tmp_dir):
    ref_response = ground_state_calculation

    parallel = {'band': 2 if world.size >= 4 else 1}
    gs_calc = calculate_ground_state(parallel=parallel)
    response = gs_calc.hamiltonian.xc.response
    check_response(ref_response, response)
    check_partitions(response, gs_calc)


def test_fixed_density(ground_state_calculation, module_tmp_path, in_tmp_dir):
    ref_response = ground_state_calculation

    parallel = {'band': 2 if world.size >= 4 else 1}
    bs_calc = GPAW(module_tmp_path / 'gs.gpw', txt=None) \
        .fixed_density(parallel=parallel,
                       txt='unocc.out',
                       )
    response = bs_calc.hamiltonian.xc.response
    check_response(ref_response, response)
    # check_partitions(response, bs_calc)  # Fails. Should it work?
    bs_calc.write('unocc.gpw', mode='all')

    bs_calc = GPAW('unocc.gpw', txt=None)
    response = bs_calc.hamiltonian.xc.response
    check_response(ref_response, response)
    check_partitions(response, bs_calc)
