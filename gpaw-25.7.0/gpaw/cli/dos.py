"""CLI-code for dos-subcommand."""
from pathlib import Path
from typing import Union, List, Tuple, Optional

import numpy as np

from gpaw import GPAW
from gpaw.setup import Setup
from gpaw.spherical_harmonics import names as ylmnames
from gpaw.dos import DOSCalculator


class CLICommand:
    """Calculate (projected) density of states from gpw-file."""

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('gpw', metavar='gpw-file')
        add('-p', '--plot', action='store_true',
            help='Plot the DOS.')
        add('-i', '--integrated', action='store_true',
            help='Calculate integrated DOS.')
        add('-w', '--width', type=float, default=0.1,
            help='Width of Gaussian.  Use 0.0 to use linear tetrahedron '
            'interpolation.')
        add('-a', '--atom', help='Project onto atoms: "Cu-spd,H-s" or use '
            'atom indices "12-spdf".  Particular m-values can be obtained '
            'like this: "N-p0,N-p1,N-p2. For p-orbitals, m=0,1,2 translates '
            'to y, z and x. For d-orbitals, m=0,1,2,3,4 translates '
            'to xy, yz, 3z2-r2, zx and x2-y2.')
        add('-t', '--total', action='store_true',
            help='Show both PDOS and total DOS.')
        add('-r', '--range', nargs=2, metavar=('emin', 'emax'),
            help='Energy range in eV relative to Fermi level.')
        add('-n', '--points', type=int, default=400, help='Number of points.')
        add('--soc', action='store_true',
            help='Include spin-orbit coupling.')

    @staticmethod
    def run(args):
        if args.range is None:
            emin = None
            emax = None
        else:
            emin, emax = (float(e) for e in args.range)
        dos(args.gpw,
            plot=args.plot,
            width=args.width,
            integrated=args.integrated,
            projection=args.atom,
            soc=args.soc,
            emin=emin,
            emax=emax,
            npoints=args.points,
            show_total=args.total)


def parse_projection_string(projection: str,
                            symbols: List[str],
                            setups: List[Setup]
                            ) -> List[Tuple[str,
                                            List[Tuple[int,
                                                       int,
                                                       Union[None, int]]]]]:
    """Create labels and lists of (a, l, m)-tuples.

    Example:

    >>> from gpaw.setup import create_setup
    >>> setup = create_setup('Li')
    >>> s, py = parse_projection_string('Li-sp0',
    ...                                 ['Li', 'Li'],
    ...                                 [setup, setup])
    >>> s
    ('Li-s', [(0, 0, None), (1, 0, None)])
    >>> py
    ('Li-p(y)', [(0, 1, 0), (1, 1, 0)])

    * "Li-s" will have contributions from l=0 and atoms 0 and 1
    * "Li-p(y)" will have contributions from l=1, m=0 and atoms 0 and 1

    """
    result: List[Tuple[str, List[Tuple[int, int, Union[None, int]]]]] = []
    for proj in projection.split(','):
        s, ll = proj.split('-')
        if s.isdigit():
            A = [int(s)]
            s = '#' + s
        else:
            A = [a for a, symbol in
                 enumerate(symbols)
                 if symbol == s]
            if not A:
                raise ValueError('No such atom: ' + s)
        for l, m in parse_lm_string(ll):
            label = s + '-' + 'spdfg'[l]
            if m is not None:
                name = ylmnames[l][m]
                label += f'({name})'
            result.append((label, [(a, l, m) for a in A]))

    return result


def parse_lm_string(s: str) -> List[Tuple[int, Union[None, int]]]:
    """Parse 'spdf' kind of string to numbers.

    Return list of (l, m) tuples with m=None if not specified:

    >>> parse_lm_string('sp')
    [(0, None), (1, None)]
    >>> parse_lm_string('p0p1p2')
    [(1, 0), (1, 1), (1, 2)]
    """
    result = []
    while s:
        l = 'spdfg'.index(s[0])
        m: Union[None, int]
        if s[1:2].isnumeric():
            m = int(s[1:2])
            s = s[2:]
        else:
            m = None
            s = s[1:]
        result.append((l, m))
    return result


def dos(filename: Union[Path, str],
        *,
        plot=False,
        width=0.1,
        integrated=False,
        projection=None,
        soc=False,
        emin=None,
        emax=None,
        npoints=200,
        show_total=None):
    """Calculate density of states.

    filename: str
        Name of restart-file.
    plot: bool
        Show a plot.
    width: float
        Width of Gaussians.
    integrated: bool
        Calculate integrated DOS.

    """
    from ase.spectrum.dosdata import GridDOSData
    from ase.spectrum.doscollection import GridDOSCollection

    calc = GPAW(filename)

    doscalc = DOSCalculator.from_calculator(calc, soc)
    energies = doscalc.get_energies(emin, emax, npoints)
    nspins = doscalc.nspins
    spinlabels = [''] if nspins == 1 else [' up', ' dn']
    spins: List[Optional[int]] = [None] if nspins == 1 else [0, 1]

    dosobjs = GridDOSCollection([], energies)

    # Note: ignoring wrong GridDOSCollection.__add__ hint below.

    if projection is None or show_total:
        for spin, label in zip(spins, spinlabels):
            dos = doscalc.raw_dos(energies, spin=spin, width=width)
            dosobjs += GridDOSData(energies, dos,  # type: ignore
                                   {'label': 'DOS' + label})

    if projection is not None:
        symbols = calc.atoms.get_chemical_symbols()
        projs = parse_projection_string(
            projection, symbols, calc.setups)
        for label, contributions in projs:
            for spin, spinlabel in zip(spins, spinlabels):
                dos = np.zeros_like(energies)
                for a, l, m in contributions:
                    dos += doscalc.raw_pdos(energies,
                                            a, l, m,
                                            spin=spin,
                                            width=width)
                dosobjs += GridDOSData(energies, dos,  # type: ignore
                                       {'label': label + spinlabel})

    if integrated:
        de = energies[1] - energies[0]
        energies = energies + de / 2
        dosobjs = GridDOSCollection(
            [GridDOSData(energies,
                         np.cumsum(obj.get_weights()) * de,
                         obj.info)
             for obj in dosobjs])
        ylabel = 'iDOS [electrons]'
    else:
        ylabel = 'DOS [electrons/eV]'

    if plot:
        ax = dosobjs.plot()
        ax.set_xlabel(r'$\epsilon-\epsilon_F$ [eV]')
        ax.set_ylabel(ylabel)
        import matplotlib.pyplot as plt
        plt.show()

    return dosobjs
