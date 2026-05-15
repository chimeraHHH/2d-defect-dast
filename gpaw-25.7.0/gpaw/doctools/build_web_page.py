"""Build GPAW's web-page."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


cmds = """\
python3 -m venv venv
. venv/bin/activate
pip install -U pip wheel -qq
pip install -q git+https://gitlab.com/ase/ase.git
git clone -q https://gitlab.com/gpaw/gpaw.git
cd gpaw
pip install -e .[docs]
python setup.py sdist
cd doc
make
mv build/html gpaw-web-page
cd ..
echo "from gpaw.utilities.urlcheck import test; test()"
python -m gpaw.utilities.urlcheck doc gpaw
echo done"""


def build():
    root = Path('/scratch/jensj/gpaw-docs')
    if root.is_dir():
        sys.exit('Locked')
    root.mkdir()
    os.chdir(root)
    cmds2 = ' && '.join(cmd for cmd in cmds.splitlines() if cmd[0] != '#')
    p = subprocess.run(cmds2, shell=True)
    if p.returncode == 0:
        status = 'ok'
    else:
        print('FAILED!', file=sys.stdout)
        status = 'error'
    f = root.with_name(f'gpaw-docs-{status}')
    if f.is_dir():
        shutil.rmtree(f)
    root.rename(f)
    return status


def build_all():
    assert build() == 'ok'
    tar = next(
        Path('/scratch/jensj/gpaw-docs-ok/gpaw/dist/').glob('gpaw-*.tar.gz'))
    webpage = Path('/scratch/jensj/gpaw-docs-ok/gpaw/doc/gpaw-web-page')
    home = Path.home() / 'web-pages'
    cmds = ' && '.join(
        [f'cp {tar} {webpage}',
         f'find {webpage} -name install.html | '
         f'xargs sed -i s/snapshot.tar.gz/{tar.name}/g',
         f'cd {webpage}/_sources/setups',  # backwards compatibility
         'cp setups.rst.txt setups.txt',  # with old install-data script
         f'cd {webpage.parent}',
         'tar -czf gpaw-web-page.tar.gz gpaw-web-page',
         f'cp gpaw-web-page.tar.gz {home}'])
    subprocess.run(cmds, shell=True, check=True)


if __name__ == '__main__':
    build_all()
