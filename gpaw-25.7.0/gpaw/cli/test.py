import sys
from pathlib import Path

from ase import Atoms
from ase.parallel import parprint

from .info import info
from gpaw import GPAW, PW, setup_paths
from gpaw.mpi import size


class CLICommand:
    """Test GPAW installation."""

    @staticmethod
    def add_arguments(parser):
        pass

    @staticmethod
    def run(args):
        info()
        test()


def test():
    for path in setup_paths:
        if Path(path).is_dir():
            break
    else:
        print("""Could not find any atomic PAW-data or pseudopotentials!

You need to set the GPAW_SETUP_PATH environment variable to point to
the directories where PAW dataset and basis files are stored.  See
https://gpaw.readthedocs.io/install.html#install-paw-datasets
for details.""", file=sys.stderr)
        return

    parprint(f'Doing a test calculation (cores: {size}): ... ',
             end='', flush=True)
    a = 2.5
    d = 0.9
    chain = Atoms('H', cell=[a, a, d], pbc=(False, False, True))
    chain.center()
    chain.calc = GPAW(mode=PW(200),
                      kpts=(1, 1, 8),
                      txt='test.txt')
    chain.get_forces()
    chain.get_stress()
    parprint('Done')
    if size == 1:
        print()
        print('Test parallel calculation with "gpaw -P 4 test".')
