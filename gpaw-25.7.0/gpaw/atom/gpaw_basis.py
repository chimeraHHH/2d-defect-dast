from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Sequence

from .basis import BasisMaker
from .basisfromfile import (BasisGetter, add_common_args,
                            get_basis_getter, read_setup_and_generate_basis)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Generate LCAO basis sets for '
                            'the specified elements.')
    add = parser.add_argument
    add('symbols', metavar='<symbol>', nargs='+', help='chemical symbols')
    add('--version', action='version', version='%(prog)s 0.1')
    add('-n', '--name', default=None, metavar='<name>',
        help='name of generated basis files')
    add('-f', '--xcfunctional', default='PBE', metavar='<XC>',
        help='exchange-Correlation functional [default: %(default)s]')
    add_common_args(add)
    add('--save-setup', action='store_true',
        help='save setup to file')
    return parser


def main(args: Sequence[str] | None = None) -> None:
    def generate_basis_set(symbol_or_path: str, getter: BasisGetter, /,
                           **kwargs) -> None:
        if '.' in symbol_or_path:  # symbol is actually a path
            _, basis = read_setup_and_generate_basis(
                symbol_or_path, getter, **kwargs)
        else:
            bm = BasisMaker.from_symbol(symbol_or_path, **kwargs)
            basis = getter(bm)
        basis.write_xml()

    parser = get_parser()
    arguments = parser.parse_args(args)
    get_basis = get_basis_getter(arguments)

    for symbol in arguments.symbols:
        generate_basis_set(symbol, get_basis,
                           name=arguments.name,
                           xc=arguments.xcfunctional,
                           save_setup=arguments.save_setup)
