"""Test GPAW in a venv."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


cmds = """\
python3 -m venv venv
. venv/bin/activate
pip install -U pip -qq
pip install pytest sphinx-rtd-theme coverage
pip install -q git+https://gitlab.com/ase/ase.git@master
git clone git@gitlab.com:gpaw/gpaw
cd gpaw
pip install -e .
coverage run -m pytest > test-1.out
coverage html
gpaw -P 2 python -m pytest -x > test-2.out
gpaw -P 4 python -m pytest -x > test-4.out
gpaw -P 8 python -m pytest -x > test-8.out"""


def run_tests():
    home = Path.cwd()
    root = Path('/tmp/gpaw-tests')
    if root.is_dir():
        sys.exit('Locked')
    root.mkdir()
    os.chdir(root)
    cmds2 = ' && '.join(cmd for cmd in cmds.splitlines()
                        if not cmd.startswith('#'))
    p = subprocess.run(cmds2, shell=True)
    if p.returncode == 0:
        status = 'ok'
        for n in [1, 2, 4, 8]:
            shutil.copy2(root / f'gpaw/test-{n}.out', home)
    else:
        print('FAILED!', file=sys.stdout)
        status = 'error'
    f = root.with_name('gpaw-test-' + status)
    if f.is_dir():
        shutil.rmtree(f)
    root.rename(f)
    return status


if __name__ == '__main__':
    run_tests()
