from __future__ import annotations
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.io import write
from ase.units import Bohr
from gpaw.new.ase_interface import GPAW


def read_bader_charges(filename: str | Path = 'ACF.dat') -> np.ndarray:
    path = Path(filename)
    charges = []
    with path.open() as fd:
        for line in fd:
            words = line.split()
            if len(words) == 7:
                charges.append(float(words[4]))
    return np.array(charges)


def main(gpw_file_name: str):
    calc = GPAW(gpw_file_name)
    dens = calc.dft.densities()
    n_sR = dens.all_electron_densities(grid_spacing=0.05)
    # NOTE: Ignoring ASE's hint for **kwargs in write() because it is wrong:
    write('density.cube',
          calc.atoms,
          data=n_sR.data.sum(axis=0) * Bohr**3)  # type: ignore
    subprocess.run('bader -p all_atom density.cube'.split())
    ne = n_sR.integrate().sum()
    print(f'{ne:.6f} electrons')
    charges = calc.atoms.numbers - read_bader_charges()
    for symbol, charge in zip(calc.atoms.symbols, charges):
        print(f'{symbol:2} {charge:10.6f} |e|')


if __name__ == '__main__':
    main(sys.argv[1])
