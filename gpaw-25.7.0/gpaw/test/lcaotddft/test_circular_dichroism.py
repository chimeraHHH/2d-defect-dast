import numpy as np
import pytest

from ase import Atoms

from gpaw import GPAW
from gpaw.mpi import world, serial_comm
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.magneticmomentwriter import MagneticMomentWriter
from gpaw.lcaotddft.magneticmomentwriter import parse_header
from gpaw.tddft.spectrum import rotatory_strength_spectrum
from gpaw.tddft.units import as_to_au, eV_to_au, au_to_eV, rot_au_to_cgs

from gpaw.test import only_on_master
from . import parallel_options, check_txt_data, copy_and_cut_file

pytestmark = pytest.mark.usefixtures('module_tmp_path')

parallel_i = parallel_options()


# Parameterize over spin polarized and unpolarized calculations
@pytest.fixture(scope='module', params=[True, False])
@only_on_master(world)
def initialize_system(request):
    comm = serial_comm

    atoms = Atoms('LiNaNaNa',
                  positions=[[0.0, 0.0, 0.0],
                             [2.0, 1.0, 0.0],
                             [4.0, 0.0, 1.0],
                             [6.0, -1.0, 0.0]])
    atoms.center(vacuum=4.0)

    if request.param:
        atoms.set_initial_magnetic_moments([0.01] * len(atoms))

    calc = GPAW(nbands=2,
                h=0.4,
                setups={'Na': '1'},
                basis='sz(dzp)',
                mode='lcao',
                convergence={'density': 1e-12},
                communicator=comm,
                symmetry={'point_group': False},
                txt='gs.out')
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('gs.gpw', mode='all')

    for gauge in ['length', 'velocity']:
        if gauge == 'velocity':
            add = '_' + gauge
        else:
            add = ''
        td_calc = LCAOTDDFT('gs.gpw', communicator=comm,
                            txt='td' + add + '.out')
        dmat = DensityMatrix(td_calc)
        MagneticMomentWriter(td_calc, 'mm' + add + '.dat', dmat=dmat)
        MagneticMomentWriter(td_calc, 'mm_grid' + add + '.dat',
                             calculate_on_grid=True)
        MagneticMomentWriter(td_calc, 'mm_origin' + add + '.dat',
                             origin='zero', origin_shift=[1.0, 2.0, 3.0])
        td_calc.absorption_kick([1e-5, 0., 0.], gauge=gauge)
        td_calc.propagate(100, 3)
        td_calc.write('td' + add + '.gpw', mode='all')
        td_calc.propagate(100, 2)


@pytest.mark.rttddft
def test_magnetic_moment_velocity_gauge(initialize_system, module_tmp_path,
                                        in_tmp_dir):
    with open('mm_ref.dat', 'w', encoding='utf-8') as f:
        f.write('''
# MagneticMomentWriter[version=5](**{"origin": "COM", "origin_shift": [0.0, 0.0, 0.0], "calculate_on_grid": false, "only_pseudo": false})
# origin_v = [7.634300, 5.000000, 4.302858] Å
#            time               mmx                    mmy                    mmz
# Start; Time = 0.00000000
          0.00000000     0.000000000000e+00     0.000000000000e+00     0.000000000000e+00
# Kick = [    1.000000000000e-05,     0.000000000000e+00,     0.000000000000e+00]; Time = 0.00000000
          0.00000000     0.000000000000e+00     0.000000000000e+00     0.000000000000e+00
          4.13413733    -3.063974663768e-07    -3.574171353385e-07     1.245812807572e-06
          8.26827467    -1.197667430552e-06    -1.414585723558e-06     4.779431710530e-06
         12.40241200    -2.606870392812e-06    -3.132296132758e-06     1.010969895228e-05
         16.53654934    -4.424269197168e-06    -5.428714567811e-06     1.652651689986e-05
         20.67068667    -6.501209589454e-06    -8.162871785725e-06     2.323149911180e-05
'''.strip())  # noqa: E501
    check_txt_data(module_tmp_path / 'mm_velocity.dat',
                   'mm_ref.dat', atol=2e-14)

    with open('mm_grid_ref.dat', 'w', encoding='utf-8') as f:
        f.write('''
# MagneticMomentWriter[version=5](**{"origin": "COM", "origin_shift": [0.0, 0.0, 0.0], "calculate_on_grid": true, "only_pseudo": false})
# origin_v = [7.634300, 5.000000, 4.302858] Å
#            time               mmx                    mmy                    mmz
# Start; Time = 0.00000000
          0.00000000    -0.000000000000e+00    -0.000000000000e+00    -0.000000000000e+00
# Kick = [    1.000000000000e-05,     0.000000000000e+00,     0.000000000000e+00]; Time = 0.00000000
          0.00000000    -0.000000000000e+00    -0.000000000000e+00    -0.000000000000e+00
          4.13413733    -3.063224380042e-07    -3.573133338861e-07     1.245412048950e-06
          8.26827467    -1.197361494602e-06    -1.414167218866e-06     4.777891626962e-06
         12.40241200    -2.606162974554e-06    -3.131346591800e-06     1.010643346002e-05
         16.53654934    -4.422978770321e-06    -5.427020998550e-06     1.652116134041e-05
         20.67068667    -6.499162446085e-06    -8.160246795614e-06     2.322393973892e-05
'''.strip())  # noqa: E501

    check_txt_data(module_tmp_path / 'mm_grid_velocity.dat',
                   'mm_grid_ref.dat', atol=2e-14)

    with open('mm_origin_ref.dat', 'w', encoding='utf-8') as f:
        f.write('''
# MagneticMomentWriter[version=5](**{"origin": "zero", "origin_shift": [1.0, 2.0, 3.0], "calculate_on_grid": false, "only_pseudo": false})
# origin_v = [1.000000, 2.000000, 3.000000] Å
#            time               mmx                    mmy                    mmz
# Start; Time = 0.00000000
          0.00000000     0.000000000000e+00     0.000000000000e+00     0.000000000000e+00
# Kick = [    1.000000000000e-05,     0.000000000000e+00,     0.000000000000e+00]; Time = 0.00000000
          0.00000000     0.000000000000e+00     0.000000000000e+00     0.000000000000e+00
          4.13413733     3.055448269266e-07    -2.985624975165e-06     4.181522917320e-06
          8.26827467     1.055413708309e-06    -1.144251258824e-05     1.639709003121e-05
         12.40241200     1.830150711621e-06    -2.413141874059e-05     3.586908883647e-05
         16.53654934     2.098630938582e-06    -3.928697753788e-05     6.127421238813e-05
         20.67068667     1.409938523718e-06    -5.497359580260e-05     9.073479845925e-05
'''.strip())  # noqa: E501

    check_txt_data(module_tmp_path / 'mm_origin_velocity.dat',
                   'mm_origin_ref.dat', atol=1e-13)


@pytest.mark.rttddft
def test_magnetic_moment_values(initialize_system, module_tmp_path,
                                in_tmp_dir):
    with open('mm_ref.dat', 'w', encoding='utf-8') as f:
        f.write('''
# MagneticMomentWriter[version=4](origin='COM')
# origin_v = [7.634300, 5.000000, 4.302858] Å
#            time               mmx                    mmy                    mmz
          0.00000000     0.000000000000e+00     0.000000000000e+00     0.000000000000e+00
# Kick = [    1.000000000000e-05,     0.000000000000e+00,     0.000000000000e+00]; Time = 0.00000000
          0.00000000     8.192189793082e-06     1.038446327373e-05    -2.730498071751e-05
          4.13413733     7.838837723234e-06     1.000765310013e-05    -2.573300722038e-05
          8.26827467     6.809084660174e-06     8.879683492897e-06    -2.128890950807e-05
         12.40241200     5.175350632237e-06     7.009694921954e-06    -1.462938416394e-05
         16.53654934     3.058296873929e-06     4.443905967036e-06    -6.697375210691e-06
         20.67068667     6.247451722277e-07     1.298788405738e-06     1.460017881082e-06
'''.strip())  # noqa: E501

    check_txt_data(module_tmp_path / 'mm.dat', 'mm_ref.dat', atol=1e-13)


@pytest.mark.rttddft
def test_magnetic_moment_grid_evaluation(initialize_system, module_tmp_path):
    dpath = module_tmp_path
    check_txt_data(dpath / 'mm.dat', dpath / 'mm_grid.dat', atol=2e-8)


@pytest.mark.rttddft
@pytest.mark.parametrize('parallel', parallel_i)
@pytest.mark.parametrize('gauge', ['velocity', 'length'])
def test_magnetic_moment_parallel(initialize_system, module_tmp_path, parallel,
                                  in_tmp_dir, gauge):

    td_calc = LCAOTDDFT(module_tmp_path / 'gs.gpw',
                        parallel=parallel,
                        txt='td.out')
    print(parallel)
    add = '_velocity' if gauge == 'velocity' else ''
    MagneticMomentWriter(td_calc, f'mm{add}.dat')
    MagneticMomentWriter(td_calc, f'mm_grid{add}.dat', calculate_on_grid=True)
    MagneticMomentWriter(td_calc, f'mm_origin{add}.dat',
                         origin='zero', origin_shift=[1.0, 2.0, 3.0])
    td_calc.absorption_kick([1e-5, 0., 0.], gauge=gauge)
    td_calc.propagate(100, 5)

    for fname in [f'mm{add}.dat', f'mm_grid{add}.dat', f'mm_origin{add}.dat']:
        check_txt_data(module_tmp_path / fname, fname, atol=7e-14)


@pytest.mark.rttddft
@pytest.mark.parametrize('parallel', parallel_i)
def test_magnetic_moment_restart(initialize_system, module_tmp_path, parallel,
                                 in_tmp_dir):
    td_calc = LCAOTDDFT(module_tmp_path / 'td.gpw',
                        parallel=parallel,
                        txt='td.out')
    for fname in ['mm.dat', 'mm_grid.dat', 'mm_origin.dat']:
        if world.rank == 0:
            copy_and_cut_file(module_tmp_path / fname, fname, cut_lines=3)
        world.barrier()
        MagneticMomentWriter(td_calc, fname)
    td_calc.propagate(100, 2)

    for fname in ['mm.dat', 'mm_grid.dat', 'mm_origin.dat']:
        check_txt_data(module_tmp_path / fname, fname, atol=7e-14)


@only_on_master(world)
def test_spectrum(in_tmp_dir, rng):
    from gpaw.utilities.folder import Folder

    # Parameters for test data
    kick_strength = 1e-5
    frequency_v = np.array([1.0, 2.0, 3.0]) * eV_to_au
    strength_v = np.array([1.0, 2.0, 3.0])

    # Create dummy magnetic moment files
    for v, kick in enumerate('xyz'):
        kick_v = [0.0, 0.0, 0.0]
        kick_v[v] = kick_strength
        time_t = np.arange(0, 31e3, 10.0) * as_to_au
        data_tv = np.zeros((len(time_t), 4))
        data_tv[:, 0] = time_t
        # Fill unused columns with random values
        data_tv[:, 1:] = rng.random((len(time_t), 3))
        # Diagonal column has the data used for spectrum
        data_tv[:, v + 1] = (kick_strength * strength_v[v]
                             * np.cos(frequency_v[v] * time_t))
        with open(f'mm-{kick}.dat', 'w', encoding='utf-8') as f:
            f.write(f'''
# MagneticMomentWriter[version=4](origin='COM')
#            time               mmx                    mmy                    mmz
          0.00000000     0.000000000000e+00     0.000000000000e+00     0.000000000000e+00
# Kick = {kick_v}; Time = 0.00000000
''')  # noqa: E501
            np.savetxt(f, data_tv)

    # Calculate spectrum
    folding_kwargs = dict(folding='Gauss', width=0.2)
    rotatory_strength_spectrum(['mm-x.dat', 'mm-y.dat', 'mm-z.dat'],
                               'spec.dat',
                               **folding_kwargs,
                               e_min=0.0, e_max=5.0, delta_e=0.01)

    # Reference spectrum
    energy_e = np.arange(0, 5.0 + 1e-8, 0.01)
    f = Folder(**folding_kwargs).fold_values
    spec_e = (f(frequency_v * au_to_eV, strength_v, energy_e)[1]
              + f(-frequency_v * au_to_eV, strength_v, energy_e)[1])
    spec_e *= 0.5
    spec_e *= rot_au_to_cgs * 1e40

    # Compare spectrum
    data_ei = np.loadtxt('spec.dat')
    assert np.allclose(data_ei[:, 0], energy_e)
    assert np.allclose(data_ei[:, 1], spec_e)

    # Test failure
    with pytest.raises(RuntimeError):
        rotatory_strength_spectrum(['mm-x.dat', 'mm-x.dat', 'mm-z.dat'],
                                   'spec.dat')
    with pytest.raises(RuntimeError):
        rotatory_strength_spectrum(['mm-y.dat', 'mm-x.dat', 'mm-z.dat'],
                                   'spec.dat')
    with pytest.raises(RuntimeError):
        rotatory_strength_spectrum(['mm-x.dat', 'mm-y.dat', 'mm-x.dat'],
                                   'spec.dat')


def test_parse_header():
    line = 'SomeWriter[version=42](**{"str": "value", "vector": [1.2, 3.4], "boolean": true})'  # noqa: E501
    name, version, kwargs = parse_header(line)
    assert name == 'SomeWriter'
    assert version == 42
    assert kwargs['str'] == 'value'
    assert kwargs['vector'] == [1.2, 3.4]
    assert kwargs['boolean']

    # Test failure
    with pytest.raises(ValueError):
        parse_header('wrong line')
    with pytest.raises(ValueError):
        parse_header('A[version=1](**{wrong json})')
