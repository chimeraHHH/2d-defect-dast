from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from operator import methodcaller
from os import PathLike
from pathlib import Path
from typing import NamedTuple

from ..basis_data import Basis, parse_basis_name
from ..setup_data import SetupData, read_maybe_unzipping, search_for_file
from ..typing import Self
from .all_electron import ValenceData
from .basis import BasisMaker


BasisGetter = Callable[[BasisMaker], Basis]


def read_setup_and_generate_basis(
        setup: str | PathLike,
        getter: BasisGetter, /,
        **kwargs) -> tuple[SetupData, Basis]:
    setupdata = read_setupdata(setup)
    valdata = ValenceData.from_setupdata_onthefly_potentials(setupdata)
    return setupdata, getter(BasisMaker(valdata, **kwargs))


class BasisInfo(NamedTuple):
    zetacount: int
    polcount: int
    name: str | None = None

    @classmethod
    def from_name(cls, name: str) -> Self:
        zc, pc = parse_basis_name(name)
        return cls(zc, pc, name)


def parse_j_values(j: str) -> list[int]:
    return [int(value) for value in j.split(',')]


def parse_tail_norm(tail: str) -> list[float]:
    return [float(value) for value in tail.split(',')]


def add_common_args(add: Callable) -> None:
    add('-t', '--type',
        default='dzp', metavar='<type>', type=BasisInfo.from_name,
        help='type of basis (e.g.: sz, dzp, qztp, 4z3p) '
        '[default: %(default)s]')
    add('-E', '--energy-shift',
        default=.1, metavar='<energy>', type=float,
        help='use given energy shift to determine cutoff '
        '[default/eV: %(default)s]')
    add('-T', '--tail-norm',
        default=[0.16, 0.3, 0.6], dest='tailnorm',
        metavar='<norm>[,<norm>[,...]]', type=parse_tail_norm,
        help='use the given fractions to define the split-valence cutoffs '
        '[default: %(default)s]')
    add('--rcut-max',
        default=16., metavar='<rcut>', type=float,
        help='max cutoff for confined atomic orbitals.  '
        'This option has no effect on orbitals with smaller cutoff '
        '[default/Bohr: %(default)s]')
    add('--rcut-pol-rel', default=1.0, metavar='<rcut>', type=float,
        help='polarization function cutoff relative to largest '
        'single-zeta cutoff [default: %(default)s]')
    add('--rchar-pol-rel', metavar='<rchar>', type=float,
        help='characteristic radius of Gaussian when not using interpolation '
        'scheme, relative to rcut')
    add('--vconf-amplitude', default=12., metavar='<alpha>', type=float,
        help='set proportionality constant of smooth '
        'confinement potential [default: %(default)s]')
    add('--vconf-rstart-rel', default=.6, metavar='<ri/rc>', type=float,
        help='set inner cutoff for smooth confinement potential '
        'relative to hard cutoff [default: %(default)s]')
    add('--vconf-sharp-confinement', action='store_true',
        help='use sharp rather than smooth confinement potential')
    add('--lpol', metavar='<l>', type=int,
        help='angular momentum quantum number of polarization function.  '
        'Default behaviour is to take the lowest l which is not '
        'among the valence states')
    add('--jvalues', metavar='<j>[,<j>[,...]]', type=parse_j_values,
        help='explicitly specify which states to include.  '
        'Numbering corresponds to generator\'s valence state ordering.  '
        'For example: 0,1,2')


def read_setupdata(path: str | PathLike) -> SetupData:
    setupdata = SetupData(symbol=None, xcsetupname=None, readxml=False)
    setupdata.read_xml(read_maybe_unzipping(Path(path)))
    return setupdata


def get_basis_getter(args: Namespace) -> BasisGetter:
    if args.vconf_sharp_confinement:
        vconf_args = None
    else:
        vconf_args = args.vconf_amplitude, args.vconf_rstart_rel
    return methodcaller('generate', args.type.zetacount, args.type.polcount,
                        tailnorm=args.tailnorm,
                        energysplit=args.energy_shift,
                        rcutpol_rel=args.rcut_pol_rel,
                        rcutmax=args.rcut_max,
                        rcharpol_rel=args.rchar_pol_rel,
                        vconf_args=vconf_args,
                        l_pol=args.lpol,
                        jvalues=args.jvalues)


def main(args: Namespace) -> None:
    get_basis = get_basis_getter(args)
    tokens = []
    if args.name:
        tokens.append(args.name)
    tokens += [args.type.name, 'basis']
    for filename in args.file:
        print(f'Generating basis set for {filename!r}')
        if args.search:
            found_filename, _ = search_for_file(filename)
            print(f'Search result: {filename!r} -> {found_filename!r}')
            filename = found_filename
        setupdata, basis = read_setup_and_generate_basis(filename, get_basis)
        # Should the setupname be added as part of the name, too?
        # Probably not, since we don't include the xcname either.
        # But I suppose it depends more on the runtime behaviour when
        # GPAW actually picks setups/basis sets for a calculation.
        outputfile = '.'.join([setupdata.symbol, *tokens])
        with open(outputfile, 'w') as fd:
            basis.write_to(fd)


class CLICommand:
    """Create basis sets from setup files."""

    @staticmethod
    def add_arguments(parser) -> None:
        add = parser.add_argument
        add('file', metavar='<filename>', nargs='+', help='setup data file')
        add('--name',
            metavar='<name>',
            help='basis name to be included in output filename')
        add('-s', '--search',
            action='store_true',
            help='instead of treating <filename> as paths, '
            'search the installed datasets for them')
        add_common_args(add)

    run = staticmethod(main)
