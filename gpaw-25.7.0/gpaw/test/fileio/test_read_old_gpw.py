import os
from pathlib import Path
from gpaw import GPAW
from ase.io import read


def test_fileio_read_old_gpw():
    dir = os.environ.get('GPW_TEST_FILES')
    if dir:
        for f in (Path(dir) / 'old').glob('*.gpw'):
            print(f)
            calc = GPAW(str(f))
            calc.get_fermi_level()
            read(f)
