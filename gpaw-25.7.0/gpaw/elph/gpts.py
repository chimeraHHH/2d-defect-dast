from __future__ import annotations

import numpy as np
from ase import Atoms
from gpaw.utilities.gpts import get_number_of_grid_points
from gpaw.wavefunctions.lcao import LCAO
from gpaw.wavefunctions.pw import PW


def obtain_gpts_suggestion(cell_cv, ecut, h, print_suggestions=False):
    """Compare PW and LCAO gpts and returns tighter one.

    When two separate simulations using different calculators need to use the
    same real space grid, like in the electron-phonon coupling case, it might
    be necessary to specify gpts=(...) explicitly. This tool helps with finding
    the correct values for a combination of PW and LCAO mode calculations.

    Parameters
    ----------
    cell_cv: np.ndarray
        cell vector array
    ecut: int, float
        planewave cutoff to be used in PW mode
    h: float
        intended maximal grid spacing in LCAO mode
    print_suggestions: bool
       if True, prints human readable information
    """
    Npw_c = get_number_of_grid_points(cell_cv, mode=PW(ecut))
    Nlcao_c = get_number_of_grid_points(cell_cv, h=h, mode=LCAO())

    Nopt_c = np.maximum(Nlcao_c, (Npw_c / 4 + 0.5).astype(int) * 4)

    if print_suggestions:
        print(f'PW({ecut:3.0f}) -> gpts={list(Npw_c)}')
        print(f'LCAO, h={h:1.3f} -> gpts={list(Nlcao_c)}')
        print(f'Recommended for elph: gpts={list(Nopt_c)}')

        if np.all(Npw_c == Nlcao_c):
            print('  Both sets of gpts the same. No action needed.')
        if np.any(Npw_c < Nopt_c):
            print(f'  Add "gpts={Nopt_c.tolist()}" to PW mode calculator.')
        if np.any(Nlcao_c < Nopt_c):
            print(f'  Use "gpts={list(Nopt_c)}" instead of "h={h:1.3f}" ' +
                  'in LCAO calculator.')

    return Nopt_c


def main(argv: list[str] = None) -> None:
    import argparse

    from ase.io import read

    parser = argparse.ArgumentParser(
        prog='python3 -m gpaw.utilities.gpts',
        description='Calculate optimal gpts size between PW and LCAO mode.')
    add = parser.add_argument
    add('file', metavar='input-file',
        help='ASE readable structure file.')
    add('-e', '--ecut', type=float,
        help='Cutoff "ecut" used in PW mode.')
    add('-g', '--grid-spacing', type=float,
        help='Maximal grid spacing "h" used in LCAO mode.')
    add('-s', '--super-cell', default='1,1,1',
        help='Supercell size.  Example: "-s 2,2,2".')

    args = parser.parse_intermixed_args(argv)
    if not args.ecut or not args.grid_spacing:
        parser.print_help()
        raise SystemExit

    sc = []
    for value in args.super_cell.split(','):
        sc.append(int(value))

    atoms = read(args.file)
    assert isinstance(atoms, Atoms)
    atoms_sc = atoms * sc
    cell_cv = atoms_sc.get_cell()
    obtain_gpts_suggestion(cell_cv, args.ecut, args.grid_spacing, True)


if __name__ == '__main__':
    main()
