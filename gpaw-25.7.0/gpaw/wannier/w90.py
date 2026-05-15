import subprocess
from pathlib import Path
from typing import Union, IO, Dict, Any, cast

from ase import Atoms
import numpy as np

from .overlaps import WannierOverlaps
from .functions import WannierFunctions
from gpaw.typing import Array3D


class Wannier90Error(Exception):
    """Wannier90 error."""


class Wannier90:
    def __init__(self,
                 prefix: str = 'wannier',
                 folder: Union[str, Path] = 'W90',
                 executable='wannier90.x'):
        self.prefix = prefix
        self.folder = Path(folder)
        self.executable = executable
        self.folder.mkdir(exist_ok=True)

    def run_wannier90(self, postprocess=False, world=None):
        args = [self.executable, self.prefix]
        if postprocess:
            args[1:1] = ['-pp']
        result = subprocess.run(args,
                                cwd=self.folder,
                                stdout=subprocess.PIPE)
        if b'Error:' in result.stdout:
            raise Wannier90Error(result.stdout.decode())

    def write_input_files(self,
                          overlaps: WannierOverlaps,
                          **kwargs) -> None:
        self.write_win(overlaps, **kwargs)
        self.write_mmn(overlaps)
        if overlaps.projections is not None:
            self.write_amn(overlaps.projections)

    def write_win(self,
                  overlaps: WannierOverlaps,

                  **kwargs) -> None:
        kwargs['num_bands'] = overlaps.nbands
        kwargs['num_wann'] = overlaps.nwannier
        kwargs['fermi_energy'] = overlaps.fermi_level
        kwargs['unit_cell_cart'] = overlaps.atoms.cell.tolist()
        kwargs['atoms_frac'] = [[symbol] + list(spos_c)
                                for symbol, spos_c
                                in zip(overlaps.atoms.symbols,
                                       overlaps.atoms.get_scaled_positions())]
        kwargs['mp_grid'] = tuple(overlaps.monkhorst_pack_size)
        kwargs['kpoints'] = overlaps.kpoints
        if overlaps.proj_indices_a:
            kwargs['guiding_centres'] = True
            centers = []
            for (x, y, z), indices in zip(overlaps.atoms.positions,
                                          overlaps.proj_indices_a):
                centers += [[f'c={x},{y},{z}: s']] * len(indices)
            kwargs['projections'] = centers

        with (self.folder / f'{self.prefix}.win').open('w') as fd:
            for key, val in kwargs.items():
                if isinstance(val, tuple):
                    print(f'{key} =', *val, file=fd)
                elif isinstance(val, (list, np.ndarray)):
                    print(f'begin {key}', file=fd)
                    for line in val:
                        print('   ', *line, file=fd)
                    print(f'end {key}', file=fd)
                else:
                    print(f'{key} = {val}', file=fd)

    def write_mmn(self,
                  overlaps: WannierOverlaps) -> None:
        size = overlaps.monkhorst_pack_size
        nbzkpts = cast(int, np.prod(size))
        nbands = overlaps.nbands

        directions = list(overlaps.directions)
        directions += [(-a, -b, -c) for (a, b, c) in directions]
        ndirections = len(directions)

        with (self.folder / f'{self.prefix}.mmn').open('w') as fd:
            print('Input generated from GPAW', file=fd)
            print(f'{nbands} {nbzkpts} {ndirections}', file=fd)

            for bz_index1 in range(nbzkpts):
                i1_c = np.unravel_index(bz_index1, size)
                for direction in directions:
                    i2_c = np.array(i1_c) + direction
                    bz_index2 = np.ravel_multi_index(i2_c,
                                                     size,
                                                     'wrap')  # type: ignore
                    d_c = (i2_c - i2_c % size) // size
                    print(bz_index1 + 1, bz_index2 + 1, *d_c, file=fd)
                    M_nn = overlaps.overlap(bz_index1, direction)
                    for M_n in M_nn.T:
                        for M in M_n:
                            print(f'{M.real} {M.imag}', file=fd)

    def write_amn(self,
                  proj_kmn: Array3D) -> None:
        nbzkpts, nproj, nbands = proj_kmn.shape

        with (self.folder / f'{self.prefix}.amn').open('w') as fd:
            print('Input generated from GPAW', file=fd)
            print(f'{nbands} {nbzkpts} {nproj}', file=fd)

            for bz_index, proj_mn in enumerate(proj_kmn):
                for m, proj_n in enumerate(proj_mn):
                    for n, P in enumerate(proj_n):
                        print(n + 1, m + 1, bz_index + 1, P.real, -P.imag,
                              file=fd)

    def read_result(self):
        with (self.folder / f'{self.prefix}.wout').open() as fd:
            w = read_wout_all(fd)
        return Wannier90Functions(w['atoms'], w['centers'])


class Wannier90Functions(WannierFunctions):
    def __init__(self,
                 atoms: Atoms,
                 centers):
        WannierFunctions.__init__(self, atoms, centers, 0.0, [])


def read_wout_all(fileobj: IO[str]) -> Dict[str, Any]:
    """Read atoms, wannier function centers and spreads."""
    lines = fileobj.readlines()

    for n, line in enumerate(lines):
        if line.strip().lower().startswith('lattice vectors (ang)'):
            break
    else:
        raise ValueError('Could not fine lattice vectors')

    cell = [[float(x) for x in line.split()[-3:]]
            for line in lines[n + 1:n + 4]]

    for n, line in enumerate(lines):
        if 'cartesian coordinate (ang)' in line.lower():
            break
    else:
        raise ValueError('Could not find coordinates')

    positions = []
    symbols = []
    n += 2
    while True:
        words = lines[n].split()
        if len(words) == 1:
            break
        positions.append([float(x) for x in words[-4:-1]])
        symbols.append(words[1])
        n += 1

    atoms = Atoms(symbols, positions, cell=cell, pbc=True)

    n = len(lines) - 1
    while n > 0:
        if lines[n].strip().lower().startswith('final state'):
            break
        n -= 1
    else:
        return {'atoms': atoms,
                'centers': np.zeros((0, 3)),
                'spreads': np.zeros((0,))}

    n += 1
    centers = []
    spreads = []
    while True:
        line = lines[n].strip()
        if line.startswith('WF'):
            centers.append([float(x)
                            for x in
                            line.split('(')[1].split(')')[0].split(',')])
            spreads.append(float(line.split()[-1]))
            n += 1
        else:
            break

    return {'atoms': atoms,
            'centers': np.array(centers),
            'spreads': np.array(spreads)}
