from myqueue.workflow import run
from gpaw.test.big.test_systems.create import create_test_systems
from gpaw import GPAW, PW, MixerFull
from ase.optimize import BFGS


CORES = [1, 4, 8, 16, 14, 40, 48, 56, 72, 96, 120, 168]


def workflow():
    for name, (atoms, params) in create_test_systems().items():
        if name == 'biimtf':
            continue
        cores = len(atoms)**2 / 10
        # Find best match:
        _, cores = min((abs(cores - c), c) for c in CORES)
        run(function=calculate, args=[name, atoms, params],
            name=name,
            cores=cores,
            tmax='1d')
        # run(function=relax, args=[name, atoms, params],
        #     name=name + '-R',
        #     cores=cores,
        #     tmax='1d')


def calculate(name, atoms, params):
    """Do one-shot energy calculation."""
    atoms.calc = GPAW(**params,
                      mixer=MixerFull(),
                      txt=name + '.txt')
    atoms.get_potential_energy()


def relax(name, atoms, params):
    """Relax with PW-mode."""
    kwargs = {k: v for k, v in params.items() if k in ['charge', 'spinpol']}
    atoms.calc = GPAW(
        xc='PBE',
        mode=PW(600),
        kpts=dict(density=4.0),
        mixer=MixerFull(),
        txt=name + '-R.txt',
        **kwargs)
    BFGS(atoms,
         logfile=name + '-R.log',
         trajectory=name + '-R.traj').run(fmax=0.02)
