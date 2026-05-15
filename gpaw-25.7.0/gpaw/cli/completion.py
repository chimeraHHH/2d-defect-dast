import sys
from pathlib import Path

from ase.cli.completion import update, CLICommand

from gpaw.cli.main import commands

# Path of the complete.py script:
path = Path(__file__).with_name('complete.py')

CLICommand.cmd = f'complete -o default -C "{sys.executable} {path}" gpaw'

if __name__ == '__main__':
    update(path, commands)
