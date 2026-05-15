import collections.abc
from copy import deepcopy
from pprint import pp
import numpy as np
from gpaw.mpi import world
from time import time
from json import dumps, loads
from pathlib import Path
from collections import defaultdict

from gpaw.benchmark.systems import parse_system
from gpaw.utilities.memory import maxrss

pw_default_parameters = {'mode': {'name': 'pw', 'ecut': 400}}

pw_parameter_subsets = {'high': {'mode': {'ecut': 800}},
                        'low': {'mode': {'ecut': 400}},
                        'float32': {'mode': {'dtype': np.float32}}}

lcao_default_parameters = {'mode': {'name': 'lcao'}}

lcao_parameter_subsets = {'sz': {'basis': 'sz(dzp)'},
                          'dzp': {'basis': 'dzp'}}

kpts_parameter_subsets = {'gamma': {'kpts': (1, 1, 1)},
                          'density6': {'kpts': {'density': 6}},
                          'density10': {'kpts': {'density': 10}},
                          '411': ({'kpts': (4, 1, 1)})}

xc_parameter_subsets = {'PBE': {'xc': 'PBE'},
                        'LDA': {'xc': 'LDA'}}

eigensolver_parameter_subsets = {'RMMDIIS':
                                 {'eigensolver':
                                  {'name': 'rmm-diis',
                                   'trial_step': 0.1}},
                                 'DAV3':
                                 {'eigensolver':
                                  {'name': 'davidson',
                                   'niter': 3}}}

benchmarks_list = [
    ('C60_pw',
     'C60-pw.high:kpts.gamma',
     '1-56:4G:1GPU'),
    ('C60_lcao',
     'C60-lcao.dzp',
     '1-56:4G'),
    ('C60_lowpw',
     'C60-pw.low:kpts.gamma',
     '1-56:4G:1GPU'),
    ('C60_lowpw_float',
     'C60-pw.low.float32:kpts.gamma',
     '0:4G:1GPU'),
    ('MoS2_tube',
     'MoS2_tube-pw.high:kpts.411:xc.PBE:parallel.scalapack',
     '56-192:100G:4-16GPU'),
    ('676_graphene',
     'C676-pw:kpts.gamma:xc.PBE:parallel.scalapack',
     '56-192:100G:4-16GPU'),
    ('pw_C6000',
     'C6000-pw.low:kpts.gamma:parallel.domainband.scalapack',
     '576-:5000G:12-GPU'),
    ('pw_C2188',
     'C2188-pw.low:kpts.gamma:parallel.domainband.scalapack',
     '192-:1200G:4-GPU'),
    ('pw_C676',
     'C676-pw.high:kpts.gamma:parallel.scalapack:xc.PBE',
     '56-:500G:4-GPU'),
    ('pw_magbulk',
     'magbulk-pw.high:kpts.density6',
     '1-56:4G:1-4GPU'),
    ('pw_C60_DIIS32',
     'C60-pw.high.float32:kpts.gamma:xc.PBE:eigensolver.RMMDIIS',
     '0:4G:1GPU'),
    ('pw_C676_DIIS32',
     'C676-pw.low.float32:kpts.gamma:xc.PBE:eigensolver.RMMDIIS',
     '0:100G:1-4GPU'),
    ('pw_slab',
     'metalslab-pw.high:kpts.density10:xc.PBE:eigensolver.DAV3',
     '56-:100G:2-GPU')]


def get_domainband(size=None):
    """Divide a world size to domain and bands (as square as possible)

    If size is None then use the mpi.world.size.
    """
    if size is None:
        size = world.size

    mid = int(np.sqrt(size))
    while size % mid:
        mid -= 1
        assert mid > 0
    return {'band': size // mid,
            'domain': mid}


parallel_parameter_subsets = {'scalapack': {'parallel': {'sl_auto': True}},
                              'domainband': {'parallel': get_domainband()}}

# A parameter set is a 2-tuple with dictionary for gpaw-parameters,
# and additional dictionary with named sub parameter sets
gpaw_parameter_sets = {'pw': (pw_default_parameters, pw_parameter_subsets),
                       'lcao': (lcao_default_parameters,
                                lcao_parameter_subsets),
                       'eigensolver': ({}, eigensolver_parameter_subsets),
                       'kpts': ({}, kpts_parameter_subsets),
                       'xc': ({}, xc_parameter_subsets),
                       'parallel': ({}, parallel_parameter_subsets)}


def parse_range(s):
    """Parse a CPU or GPU range string.

    Returns a two tuple of lower and upper bound.
    Examples:
        0-1
        5
        -4GPU
    """
    s = s.replace('GPU', '')
    if '-' not in s:
        return int(s), int(s)
    min_str, max_str = s.split('-')
    if min_str:
        a = int(min_str)
    else:
        a = 0
    if max_str:
        b = int(max_str)
    else:
        b = np.inf
    return a, b


def parse_mem(memstr):
    """Memory string contains G, M or K as the last character.
    """
    mul = {'G': 1024**3,
           'M': 1024**2,
           'K': 1024**1}[memstr[-1]]
    return float(memstr[:-1]) * mul


def parse_requirement(req):
    """Parses the requirement string, which is a colon separated list of
    core_range:memory[:gpu_range]
    """
    syntax = req.split(':')
    min_cores, max_cores = parse_range(syntax[0])
    min_mem = parse_mem(syntax[1])
    if len(syntax) == 3:
        min_gpus, max_gpus = parse_range(syntax[2])
    else:
        min_gpus, max_gpus = (0, 0)
    return {'mincores': min_cores,
            'maxcores': max_cores,
            'minmem': min_mem,
            'mingpus': min_gpus,
            'maxgpus': max_gpus}


# Parse the benchmark definitions from the benchmarks_str into
# the benchmarks dictionary.
benchmarks = {}
benchmarks_reqs = {}
for nickname, definition, req in benchmarks_list:
    benchmarks[nickname] = definition
    benchmarks_reqs[nickname] = parse_requirement(req)


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def parse_parameters(parameter_sets):
    """Parses parameter_sets descriptor into a dictionary

    First, individual parameter sets are separated by :
    And a single parameter set contains first the main paramter descriptor,
    which can be further refined by . for parameters subsets.

    For example valid paramter strings are:
         pw.high:gamma
         pw.high:gamma:parallel.gpu
         lcao.dzp:kpt.density4:noscalapack
    """

    kwargs = {}
    parameter_sets = parameter_sets.split(':')
    for parameter_set in parameter_sets:
        firstsplit = parameter_set.split('.', 1)
        if len(firstsplit) == 1:
            firstsplit.append(None)
        print(firstsplit)
        set_name, parameter_subsets = firstsplit
        default_parameter_set, subsets = gpaw_parameter_sets[set_name]
        recursive_update(kwargs, deepcopy(default_parameter_set))
        if parameter_subsets is None:
            continue
        for subsetname in parameter_subsets.split('.'):
            recursive_update(kwargs, deepcopy(subsets[subsetname]))
    return kwargs


def list_benchmarks():
    lst = ''
    header = '{:20s} | {:35s}\n'.format('name', 'system-parameter sets')
    lst += header + '-' * len(header) + '\n'

    for benchmark, system_and_parameter_set in benchmarks.items():
        lst += f'{benchmark:20s} | {system_and_parameter_set:35s}\n'

    return lst


def benchmarks_error(name):
    err = f'Cannot find benckmark with name {name}\n\n'
    err += 'Available benchmarks\n'
    err += list_benchmarks()
    return err


def shell_command(cmd, cwd=None):
    import subprocess
    try:
        output = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                check=True,
                                shell=True,
                                cwd=cwd).stdout
    except subprocess.CalledProcessError as e:
        output = f'{e.output} {e.stderr}'

    return output


def gather_system_information():
    import gpaw
    return {'processor': shell_command('lscpu'),
            'memory': shell_command('lsmem'),
            'mpi-ranks': world.size,
            'date': shell_command('date'),
            'nvidia-smi': shell_command('nvidia-smi'),
            'rocm-smi': shell_command('rocm-smi'),
            'git-hash': shell_command('git rev-parse --verify HEAD',
                                      cwd=Path(gpaw.__file__).parent),
            'git-status': shell_command('git status',
                                        cwd=Path(gpaw.__file__).parent),
            'hostname': shell_command('hostname')}


def parse_name(name):
    """Parse (either long or nickname input) possibly containing calc info

    The name can be
        C60_pw
        C60_pw#new
        C60-pw.high:kpts.density6
        C60-pw.high:kpts.density6#new
    """
    names = name.split('#')
    if len(names) > 1:
        calc_info = names[1]
        assert calc_info in {'new', 'old'}
        assert len(names) == 2
    else:
        # Default behaviour is new calculation
        calc_info = 'new'

    name = names[0]
    # Replace nickname with long name
    if '-' not in name:
        if name in benchmarks:
            short_name = name
            long_name = benchmarks[name]
        else:
            raise Exception(benchmarks_error(name))
    else:
        short_name = 'N/A'
        long_name = name

    return short_name, long_name, calc_info


def benchmark_atoms_and_calc(long_name, calc_info):
    """Create atoms and calculator ibject from long name and calculator
    info (new/old)
    """
    if calc_info == 'new':
        from gpaw.new.ase_interface import GPAW
    elif calc_info == 'old':
        from gpaw import GPAW
    else:
        raise Exception(f'Unknown calc info {calc_info}')

    system, parameter_sets = long_name.split('-')
    atoms = parse_system(system)
    parameters = parse_parameters(parameter_sets)
    if world.rank == 0:
        pp(parameters, indent=4, sort_dicts=True)
    atoms.calc = GPAW(**parameters, txt=f'{long_name}-{calc_info}.log')
    return atoms, atoms.calc


def gs_and_move_atoms(long_name, calc_info):
    """Main GPAW benchmark function

    Calculates one geometry step, and takes 0.1 * F step towards
    the minimum. The timings for first and second stops are recorded,
    to simulate a typical geometry relaxation step.
    """
    atoms, calc = benchmark_atoms_and_calc(long_name, calc_info)
    with Walltime('First step') as step1:
        E = atoms.get_potential_energy()
        F = atoms.get_forces()
    if abs(F).max() < 0.0001:
        S = atoms.get_stress(voigt=False)
        atoms.set_cell(atoms.cell @ (np.eye(3) - 0.02 * S), scale_atoms=True)
    else:
        atoms.positions += 0.1 * F
    atoms.wrap()
    with Walltime('Second step') as step2:
        atoms.get_potential_energy()
        F = atoms.get_forces()

    return {'energy': E,
            'forces': F.tolist(),
            **step1.todict(),
            **step2.todict()}


class Walltime:
    def __init__(self, name):
        self.name = name
        self.error = None
        self.max_rss = None

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            self.error = (exc_type, exc_value, exc_traceback)
        self.end = time()
        self.max_rss = maxrss()

    @property
    def walltime(self):
        return self.end - self.start

    def todict(self):
        return {self.name: {'walltime': self.walltime,
                            'error': self.error,
                            'max_rss': self.max_rss}}


class Benchmark(Walltime):
    def __init__(self, system_info, **kwargs):
        super().__init__('Benchmark')
        self.system_info = system_info
        self.results = None
        self.kwargs = kwargs

    def todict(self):
        dct = super().todict()
        dct[self.name].update({'system_info': self.system_info,
                               'results': self.results,
                               **self.kwargs})
        return dct

    def write_json(self, fname):
        Path(fname).write_text(dumps(self.todict()))


def benchmark_main(name):
    # Run the gs_and_move_atoms benchmars for 'name' where
    # name can be either a short name or a long name.
    short_name, long_name, calc_info = parse_name(name)

    if world.rank == 0:
        system_info = gather_system_information()
        print('Running benchmark', name)
    else:
        system_info = None

    benchmark_info = {'shortname': short_name,
                      'longname': long_name,
                      'calcinfo': calc_info}

    world.barrier()
    with Benchmark(system_info, **benchmark_info) as results:
        results.results = gs_and_move_atoms(long_name, calc_info)
    if world.rank == 0:
        # Finally, write all of the results to a json file
        results.write_json(f'{name}-benchmark.json')


def get_benchmarks(memory='8G', cores=16, gpus=0):
    for benchmark, long_name in benchmarks.items():
        requirements = benchmarks_reqs[benchmark]
        if gpus > 0:
            if gpus < requirements.get('mingpus', 1):
                continue
            if gpus > requirements.get('maxgpus', np.inf):
                continue
        else:
            if cores < requirements.get('mincores', 1):
                continue
            if cores > requirements.get('maxcores', np.inf):
                continue
        if parse_mem(memory) <= requirements.get('minmem', np.inf):
            continue
        yield benchmark


def sprint(s, summary=False):
    if len(s) > 60:
        if summary:
            print(
                ' '.join(
                    s.replace('\n', ' ').replace('\t', ' ').split()
                )[:60], '...')
        else:
            print()
            print(s)
    else:
        print(s.rstrip())


def mypp(dct, indent=0, summary=True):
    for key, value in dct.items():
        print(' ' * indent + key + ': ', end='')
        if isinstance(value, str):
            sprint(value, summary=summary)
        elif isinstance(value, dict):
            print()
            mypp(value, indent=indent + 4, summary=summary)
        else:
            print(value)


def load_benchmark(fname):
    # Load a json file
    return loads(Path(fname).read_text())


def view_benchmark(fname):
    mypp(load_benchmark(fname))


def parse_git_status(text):
    """Parse the branch from git status output
    """
    for line in text.split('\n'):
        if line.startswith('On branch'):
            return line.split()[-1]
    return '???'


def parse_processor(text):
    """Parse the processor model from lscpu
    """
    for line in text.split('\n'):
        if line.startswith('Model name:'):
            return line.split('Model name:')[-1].strip()
    return 'No "Model name:" found'


def parse_nvidia_smi(dct, out):
    """Parse output from nvidia-smi command.

    Gets the name of the GPU from out, and accumulates to dct
    how many there are."""
    if 'command not found' in out:
        return
    for line in out.split('\n'):
        if 'NVIDIA ' in line:
            def get_gpu():
                for n in line.split()[3:]:
                    if n in {'|', 'On', 'Off'}:
                        break
                    yield n
            dct[' '.join(get_gpu())] += 1


def parse_rocm_smi(dct, out):
    if 'command not found' in out:
        return

    raise NotImplementedError


def parse_gpu(nvidia, rocm):
    gpus = defaultdict(int)
    parse_nvidia_smi(gpus, nvidia)
    parse_rocm_smi(gpus, rocm)
    return ' '.join((f'{number}x ({name})'
                     if number > 1 else name) for name, number in gpus.items())


def benchmark_from_dict(dct):
    """Create a summary dictionary from the full json output of the benchmark.
    """
    dct = dct['Benchmark']
    results = dct['results']
    system_info = dct['system_info']

    summary = {'walltime': dct['walltime'],
               'shortname': dct['shortname'],
               'processor': parse_processor(system_info['processor']),
               'gpu': parse_gpu(system_info['nvidia-smi'],
                                system_info['rocm-smi']),
               'longname': dct['longname'],
               'hostname': system_info['hostname'].strip(),
               'calcinfo': dct['calcinfo'],
               'mpi-ranks': system_info['mpi-ranks'],
               'First step': results['First step']['walltime'],
               'Second step': results['Second step']['walltime'],
               'max_rss': dct['max_rss'],
               'githash': system_info['git-hash'].strip(),
               'branch': parse_git_status(system_info['git-status'])}
    return summary


def gather_benchmarks(directories, output_file):
    lst = []
    for fname in directories:
        try:
            dct = load_benchmark(fname)
            lst.append(benchmark_from_dict(dct))
        except Exception as e:
            print(str(e))
    Path(output_file).write_text(dumps(lst, indent=4))
    return lst
