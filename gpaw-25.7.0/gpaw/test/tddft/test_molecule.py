import pytest

from ase.build import molecule

from gpaw import GPAW
from gpaw.tddft import TDDFT, DipoleMomentWriter
from gpaw.mpi import world, serial_comm
from gpaw.utilities import compiled_with_sl

from ..lcaotddft.test_molecule import only_on_master


pytestmark = pytest.mark.usefixtures('module_tmp_path')


def calculate_time_propagation(gpw_fpath, *,
                               iterations=3,
                               kick=[1e-5, 1e-5, 1e-5],
                               propagator='SICN',
                               communicator=world,
                               write_and_continue=False,
                               force_new_dm_file=False,
                               parallel={}):
    td_calc = TDDFT(gpw_fpath,
                    propagator=propagator,
                    communicator=communicator,
                    parallel=parallel,
                    txt='td.out')
    DipoleMomentWriter(td_calc, 'dm.dat',
                       force_new_file=force_new_dm_file)
    if kick is not None:
        td_calc.absorption_kick(kick)
    td_calc.propagate(20, iterations)
    if write_and_continue:
        td_calc.write('td.gpw', mode='all')
        # Switch dipole moment writer and output
        td_calc.observers.pop()
        dm = DipoleMomentWriter(td_calc, 'dm2.dat', force_new_file=True)
        dm._update(td_calc)
        td_calc.propagate(20, iterations)
    communicator.barrier()


def check_dm(ref_fpath, fpath, rtol=1e-8, atol=1e-12):
    from gpaw.tddft.spectrum import read_dipole_moment_file

    world.barrier()
    _, time_ref_t, _, dm_ref_tv = read_dipole_moment_file(ref_fpath)
    _, time_t, _, dm_tv = read_dipole_moment_file(fpath)
    assert time_t == pytest.approx(time_ref_t, abs=0)
    assert dm_tv == pytest.approx(dm_ref_tv, rel=rtol, abs=atol)


# Generate different parallelization options
parallel_i = [{}]
if world.size > 1:
    parallel_i.append({'band': 2})
if compiled_with_sl():
    parallel_i.append({'sl_auto': True})
    if world.size > 1:
        parallel_i.append({'sl_auto': True, 'band': 2})


@pytest.fixture(scope='module')
@only_on_master(world)
def ground_state():
    atoms = molecule('SiH4')
    atoms.center(vacuum=4.0)

    calc = GPAW(mode='fd', nbands=6, h=0.4,
                convergence={'density': 1e-8},
                communicator=serial_comm,
                xc='LDA',
                symmetry={'point_group': False},
                txt='gs.out')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', mode='all')


@pytest.fixture(scope='module')
@only_on_master(world)
def time_propagation_reference(ground_state):
    calculate_time_propagation('gs.gpw',
                               communicator=serial_comm,
                               write_and_continue=True)


def test_dipole_moment_values(time_propagation_reference,
                              module_tmp_path, in_tmp_dir):
    with open('dm.dat', 'w') as fd:
        fd.write('''
# DipoleMomentWriter[version=1](center=False, density='comp')
#            time            norm                    dmx                    dmy                    dmz
# Start; Time = 0.00000000
          0.00000000      -8.62679509e-16     8.856042552837e-09     5.230011358635e-11     1.624559936066e-10
# Kick = [    1.000000000000e-05,     1.000000000000e-05,     1.000000000000e-05]; Time = 0.00000000
          0.00000000       9.59295128e-16     8.826542661185e-09    -1.968118480737e-10    -1.260104338852e-10
          0.82682747       1.64702342e-15     6.016062457419e-05     6.015263997632e-05     6.015070820074e-05
          1.65365493       1.36035859e-15     1.075409609786e-04     1.075366083805e-04     1.075339737111e-04
          2.48048240       1.53109134e-15     1.388608139179e-04     1.388701618472e-04     1.388666380740e-04
'''.strip())  # noqa: E501

    with open('dm2.dat', 'w') as fd:
        fd.write('''
# DipoleMomentWriter[version=1](center=False, density='comp')
#            time            norm                    dmx                    dmy                    dmz
          2.48048240       1.53109134e-15     1.388608139179e-04     1.388701618472e-04     1.388666380740e-04
          3.30730987       1.36214053e-15     1.528275514998e-04     1.528424797241e-04     1.528388409079e-04
          4.13413733      -5.46885441e-16     1.498039918400e-04     1.498178744836e-04     1.498147055362e-04
          4.96096480      -3.62630566e-16     1.324275745486e-04     1.324479404917e-04     1.324450415780e-04
'''.strip())  # noqa: E501

    rtol = 5e-4
    atol = 1e-7
    check_dm('dm.dat', module_tmp_path / 'dm.dat', rtol=rtol, atol=atol)
    check_dm('dm2.dat', module_tmp_path / 'dm2.dat', rtol=rtol, atol=atol)


@pytest.mark.parametrize('parallel', parallel_i)
@pytest.mark.parametrize('propagator', [
    'SICN', 'ECN', 'ETRSCN', 'SIKE'])
def test_propagation(time_propagation_reference,
                     parallel, propagator,
                     module_tmp_path, in_tmp_dir):
    calculate_time_propagation(module_tmp_path / 'gs.gpw',
                               propagator=propagator,
                               parallel=parallel)
    atol = 1e-12
    if propagator == 'SICN':
        # This is the same propagator as the reference;
        # error comes only from parallelization
        rtol = 1e-8
        if 'band' in parallel:
            # XXX band parallelization is inaccurate!
            rtol = 7e-4
            atol = 5e-8
    else:
        # Other propagators match qualitatively
        rtol = 5e-2
        if 'band' in parallel:
            # XXX band parallelization is inaccurate!
            atol = 5e-8
    check_dm(module_tmp_path / 'dm.dat', 'dm.dat', rtol=rtol, atol=atol)


@pytest.mark.parametrize('parallel', parallel_i)
def test_restart(time_propagation_reference,
                 parallel,
                 module_tmp_path, in_tmp_dir):
    calculate_time_propagation(module_tmp_path / 'td.gpw',
                               kick=None,
                               force_new_dm_file=True,
                               parallel=parallel)
    rtol = 1e-8
    if 'band' in parallel:
        rtol = 5e-4
    check_dm(module_tmp_path / 'dm2.dat', 'dm.dat', rtol=rtol)
