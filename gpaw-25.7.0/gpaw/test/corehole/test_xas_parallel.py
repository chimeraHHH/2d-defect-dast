import pytest

from gpaw import GPAW
from gpaw.xas import XAS
import gpaw.mpi as mpi


dks = 20


@pytest.fixture
def xas_sym_nosp(
        in_tmp_dir, add_cwd_to_setup_paths, gpw_files):
    comm = mpi.world.new_communicator([mpi.world.rank])
    calc1 = GPAW(gpw_files['si_corehole_sym_pw'], communicator=comm)
    xas1 = XAS(calc1)
    x1, y1 = xas1.get_oscillator_strength(dks=dks)
    return x1, y1


@pytest.mark.skipif(mpi.size % 4 != 0,
                    reason='works only for multiples of 4 cores')
def test_xas_paralell_kpts_and_domian(
        in_tmp_dir, add_cwd_to_setup_paths, gpw_files, xas_sym_nosp):

    parallel = {'kpt': 2,
                'band': 1}
    calc2 = GPAW(gpw_files['si_corehole_sym_pw'], parallel=parallel)
    xas2 = XAS(calc2)
    x2, y2 = xas2.get_oscillator_strength(dks=dks)

    x1, y1 = xas_sym_nosp

    assert x2 == pytest.approx(x1)
    assert y2 == pytest.approx(y1)


@pytest.mark.skipif(mpi.size % 2 != 0,
                    reason='works only for multiples of 2 cores')
def test_xas_paralell_multiple_kpt_pr_rank(
        in_tmp_dir, add_cwd_to_setup_paths, gpw_files):

    comm = mpi.world.new_communicator([mpi.world.rank])

    parallel = {'kpt': 2}
    calc2 = GPAW(gpw_files['si_corehole_nosym_pw'],
                 parallel=parallel)

    xas2 = XAS(calc2)
    x2, y2 = xas2.get_oscillator_strength(dks=dks)

    calc1 = GPAW(gpw_files['si_corehole_nosym_pw'],
                 communicator=comm)
    xas1 = XAS(calc1)

    x1, y1 = xas1.get_oscillator_strength(dks=dks)

    assert x2 == pytest.approx(x1)
    assert y2 == pytest.approx(y1)


@pytest.mark.skipif(mpi.size % 4 != 0,
                    reason='works only for multiples of 4 cores')
def test_xas_band_and_kpts_parallel(
        in_tmp_dir, add_cwd_to_setup_paths, gpw_files, xas_sym_nosp):

    parallel = {'band': 2,
                'kpt': 2}
    calc2 = GPAW(gpw_files['si_corehole_sym_pw'],
                 parallel=parallel)
    xas2 = XAS(calc2)
    x2, y2 = xas2.get_oscillator_strength(dks=dks)

    x1, y1 = xas_sym_nosp

    assert x2 == pytest.approx(x1)
    assert y2 == pytest.approx(y1)


@pytest.mark.skipif(mpi.size % 4 != 0,
                    reason='works only for multiples of 4 cores')
@pytest.mark.old_gpaw_only
def test_xas_kpts_domian_parallel_spinpol(
        in_tmp_dir, add_cwd_to_setup_paths, gpw_files):

    parallel = {'kpt': 2,
                'band': 1}

    calc2 = GPAW(gpw_files['si_corehole_sym_pw'],
                 spinpol=True, parallel=parallel)
    calc2.get_potential_energy()
    xas2 = XAS(calc2, spin=0)

    x2, y2 = xas2.get_oscillator_strength(dks=dks)

    comm = mpi.world.new_communicator([mpi.world.rank])

    calc1 = GPAW(gpw_files['si_corehole_sym_pw'],
                 communicator=comm, spinpol=True)

    calc1.get_potential_energy()

    xas1 = XAS(calc1, spin=0)
    x1, y1 = xas1.get_oscillator_strength(dks=dks)

    assert x2 == pytest.approx(x1, 1.1e-1)
    assert y2 == pytest.approx(y1, abs=1.3e-5)


@pytest.mark.skipif(mpi.size % 4 != 0,
                    reason='works only for multiples of 4 cores')
def test_xes_kpts_and_domain_parallel(
        in_tmp_dir, add_cwd_to_setup_paths, gpw_files):

    parallel = {'kpt': 2,
                'band': 1}

    calc2 = GPAW(gpw_files['si_corehole_sym_pw'],
                 parallel=parallel)

    xes2 = XAS(calc2, 'xes')
    x2, y2 = xes2.get_oscillator_strength(dks=dks)

    comm = mpi.world.new_communicator([mpi.world.rank])

    calc1 = GPAW(gpw_files['si_corehole_sym_pw'],
                 communicator=comm)

    xes1 = XAS(calc1, 'xes')

    x1, y1 = xes1.get_oscillator_strength(dks=dks)

    assert x2 == pytest.approx(x1)
    assert y2 == pytest.approx(y1)


@pytest.mark.skipif(mpi.size % 8 != 0,
                    reason='works only for multiples of 8 cores')
def test_all_band_and_kpts_parallel(
        in_tmp_dir, add_cwd_to_setup_paths, gpw_files):

    parallel = {'band': 4,
                'kpt': 2}

    calc2 = GPAW(gpw_files['si_corehole_sym_pw'],
                 parallel=parallel)

    xas2 = XAS(calc2, 'all')
    x2, y2 = xas2.get_oscillator_strength(dks=dks)

    comm = mpi.world.new_communicator([mpi.world.rank])

    calc1 = GPAW(gpw_files['si_corehole_sym_pw'],
                 communicator=comm)

    xas1 = XAS(calc1, 'all')

    x1, y1 = xas1.get_oscillator_strength(dks=dks)

    assert x2 == pytest.approx(x1)
    assert y2 == pytest.approx(y1)
