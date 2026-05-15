import argparse
from typing import List, Union

import numpy as np
from ase import Atoms
from gpaw.new.ase_interface import ASECalculator, GPAW
from gpaw.point_groups import SymmetryChecker, point_group_names
from gpaw.typing import Array1D, Array3D


class CubeCalc:
    """Wrap cube-file in a calculator."""
    def __init__(self, function: Array3D, atoms: Atoms):
        self.function = function
        self.atoms = atoms

    def get_pseudo_wave_function(self,
                                 band: int,
                                 spin: int) -> Array3D:
        return self.function

    def get_eigenvalues(self, spin: int) -> Array1D:
        return np.zeros(1)

    def get_number_of_spins(self):
        return 1


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(
        prog='python3 -m gpaw.point_groups',
        description='Analyse point-group of atoms and wave-functions.')
    add = parser.add_argument
    add('pg', metavar='point-group', choices=point_group_names,
        help='Name of point-group: C2, C2v, C3v, D2d, D3h, D5, D5h, '
        'Ico, Ih, Oh, Td or Th.')
    add('file', metavar='input-file',
        help='Cube-file, gpw-file or something else with atoms in it.')
    add('-c', '--center', help='Center specified as one or more atoms.  '
        'Use chemical symbols or sequence numbers.')
    add('-r', '--radius', default=2.5,
        help='Cutoff radius (in Å) used for wave function overlaps.')
    add('-b', '--bands', default=':', metavar='N1:N2',
        help='Band range.')
    add('-a', '--axes', default='',
        help='Example: "-a z=x,x=-y".')
    args = parser.parse_intermixed_args(argv)

    calc: Union[None, ASECalculator, CubeCalc]

    if args.file.endswith('.gpw'):
        calc = GPAW(args.file)
        atoms = calc.atoms
        n1, n2 = (int(x) if x else 0 for x in args.bands.split(':'))
    elif args.file.endswith('.cube'):
        from ase.io.cube import read_cube
        with open(args.file) as fd:
            dct = read_cube(fd)
        calc = CubeCalc(dct['data'], dct['atoms'])
        atoms = dct['atoms']
        n1 = 0
        n2 = 1
    else:
        from ase.io import read
        atoms_maybe = read(args.file)
        assert isinstance(atoms_maybe, Atoms)
        atoms = atoms_maybe
        calc = None

    if args.center:
        symbols = set(args.center.split(','))
        center = np.zeros(3)
        n = 0
        for a, (symbol, position) in enumerate(zip(atoms.symbols,
                                                   atoms.positions)):
            if symbol in symbols or str(a) in symbols:
                center += position
                n += 1
        center /= n
    else:
        center = atoms.cell.sum(0) / 2
    print('Center:', center, f'(atoms: {n})')

    radius = float(args.radius)

    kwargs = {}
    for axis in args.axes.split(',') if args.axes else []:
        axis1, axis2 = axis.split('=')
        kwargs[axis1] = axis2

    checker = SymmetryChecker(args.pg, center, radius, **kwargs)

    ok = checker.check_atoms(atoms)
    print(f'{args.pg}-symmetry:', 'Yes' if ok else 'No')

    if calc:
        nspins = calc.get_number_of_spins()
        for spin in range(nspins):
            if nspins == 2:
                print('Spin', ['up', 'down'][spin])
            checker.check_calculation(calc, n1, n2, spin=spin)
