from __future__ import annotations

from dataclasses import dataclass, field, replace
import xml.sax

import numpy as np

from gpaw.setup_data import search_for_file
from gpaw.atom.radialgd import RadialGridDescriptor, radial_grid_descriptor


_basis_letter2number = {'s': 1, 'd': 2, 't': 3, 'q': 4}
_basis_number2letter = 'Xsdtq56789'


def parse_basis_name(name):
    """Parse any basis type identifier: 'sz', 'dzp', 'qztp', '4z3p', ... """

    zetacount = _basis_letter2number.get(name[0])
    if zetacount is None:
        zetacount = int(name[0])
    assert name[1] == 'z'

    if len(name) == 2:
        polcount = 0
    elif len(name) == 3:
        assert name[-1] == 'p'
        polcount = 1
    else:
        assert len(name) == 4 and name[-1] == 'p'
        polcount = _basis_letter2number.get(name[2])
        if polcount is None:
            polcount = int(name[2])

    return zetacount, polcount


def parse_basis_filename(filename: str):
    tokens = filename.split('.')
    if tokens[-1] == 'gz':
        tokens.pop()

    if tokens[-1] != 'basis':
        raise RuntimeError('Expected <symbol>[.<name>].basis[.gz], '
                           'got {filename!r}')

    symbol = tokens[0]
    name = '.'.join(tokens[1:-1])
    if not name:
        return symbol, None
    return symbol, name


def get_basis_name(zetacount, polarizationcount):
    zetachar = _basis_number2letter[zetacount]
    if polarizationcount == 0:
        return '%sz' % zetachar
    elif polarizationcount == 1:
        return '%szp' % zetachar
    else:
        polarizationchar = _basis_number2letter[polarizationcount]
        return f'{zetachar}z{polarizationchar}p'


@dataclass(eq=False, frozen=True)
class Basis:
    symbol: str
    name: str | None = None
    rgd: RadialGridDescriptor | None = None

    bf_j: list[BasisFunction] = field(default_factory=list)
    ribf_j: list[BasisFunction] = field(default_factory=list)
    generatorattrs: dict = field(default_factory=dict)
    generatordata: str = ''
    filename: str | None = None

    @classmethod
    def find(cls, symbol, name, world=None):
        return cls.read_xml(symbol, name, world=world)

    @classmethod
    def read_path(cls, symbol, name, path, world=None):
        return cls.read_xml(symbol, name, filename=path, world=world)

    @classmethod
    def read_xml(cls, symbol, name, filename=None, world=None):
        parser = BasisSetXMLParser(symbol, name)
        return parser.parse(filename, world=world)

    @property
    def nao(self):  # implement as a property so we don't have to
        # catch all the places where Basis objects are modified without
        # updating it.  (we can do that later)
        return sum([2 * bf.l + 1 for bf in self.bf_j])

    @property
    def nrio(self):
        return sum([2 * ribf.l + 1 for ribf in self.ribf_j])

    def get_grid_descriptor(self):
        return self.rgd

    def tosplines(self):
        return [self.rgd.spline(bf.phit_g, bf.rc, bf.l, points=400)
                for bf in self.bf_j]

    def ritosplines(self):
        return [self.rgd.spline(ribf.phit_g, ribf.rc, ribf.l, points=400)
                for ribf in self.ribf_j]

    def write_xml(self):
        """Write basis functions to file.

        Writes all basis functions in the given list of basis functions
        to the file "<symbol>.<name>.basis".
        """
        if self.name is None:
            filename = '%s.basis' % self.symbol
        else:
            filename = f'{self.symbol}.{self.name}.basis'

        with open(filename, 'w') as fd:
            self.write_to(fd)

    def write_to(self, fd):
        write = fd.write
        write('<paw_basis version="0.1">\n')

        generatorattrs = ' '.join([f'{key}="{value}"'
                                   for key, value
                                   in self.generatorattrs.items()])
        write('  <generator %s>' % generatorattrs)
        for line in self.generatordata.split('\n'):
            write('\n    ' + line)
        write('\n  </generator>\n')

        write('  ' + self.rgd.xml())

        # Write both the basis functions and auxiliary ones
        for bfs in [self.bf_j, self.ribf_j]:
            for bf in bfs:
                write(bf.xml(indentation='  '))

        write('</paw_basis>\n')

    def reduce(self, name):
        """Reduce the number of basis functions and return new Basis.

        Example: basis.reduce('sz') will remove all non single-zeta
        and polarization functions."""

        zeta, pol = parse_basis_name(name)
        newbf_j = []
        N = {}
        p = 0
        for bf in self.bf_j:
            if 'polarization' in bf.type:
                if p < pol:
                    newbf_j.append(bf)
                    p += 1
            else:
                nl = (int(bf.type[0]), 'spdf'.index(bf.type[1]))
                if nl not in N:
                    N[nl] = 0
                if N[nl] < zeta:
                    newbf_j.append(bf)
                    N[nl] += 1
        return replace(self, bf_j=newbf_j)

    def get_description(self):
        title = 'LCAO basis set for %s:' % self.symbol
        if self.name is not None:
            name = 'Name: ' + self.name
        else:
            name = 'This basis set does not have a name'
        if self.filename is None:
            fileinfo = 'This basis set was not loaded from a file'
        else:
            fileinfo = 'File: ' + self.filename
        nj = len(self.bf_j)
        count1 = 'Number of radial functions: %d' % nj
        count2 = 'Number of spherical harmonics: %d' % self.nao

        bf_lines = []
        for bf in self.bf_j:
            line = '  l=%d, rc=%.4f Bohr: %s' % (bf.l, bf.rc, bf.type)
            bf_lines.append(line)

        lines = [title, name, fileinfo, count1, count2]
        lines.extend(bf_lines)
        lines.append(f'Number of RI-basis functions {self.nrio}')
        for ribf in self.ribf_j:
            lines.append('l=%d %s' % (ribf.l, ribf.type))

        return '\n  '.join(lines)


@dataclass
class BasisFunction:
    """Encapsulates various basis function data."""

    n: int | None = None
    l: int | None = None
    rc: float | None = None
    phit_g: np.ndarray | None = None
    type: str | None = None

    @property
    def name(self):
        if self.n is None or self.n < 0:
            return f'l={self.l} {self.type}'

        lname = 'spdf'[self.l]
        return f'{self.n}{lname} {type}'

    def __repr__(self, gridid=None):
        txt = '<basis_function '
        if self.n is not None:
            txt += 'n="%d" ' % self.n
        txt += (f'l="{self.l}" rc="{self.rc}" type="{self.type}"')
        if gridid is not None:
            txt += ' grid="%s"' % gridid
        return txt + '>'

    def xml(self, gridid='grid1', indentation=''):
        txt = indentation + self.__repr__(gridid) + '\n'
        txt += indentation + '  ' + ' '.join(str(x) for x in self.phit_g)
        txt += '\n' + indentation + '</basis_function>\n'
        return txt


class BasisSetXMLParser(xml.sax.handler.ContentHandler):
    def __init__(self, symbol, name):
        super().__init__()
        self.symbol = symbol
        self.name = name

        self.type = None
        self.rc = None
        self.data = None
        self.l = None
        self.bf_j = []
        self.ribf_j = []

        self._dct = {}

    def parse(self, filename=None, world=None):
        """Read from symbol.name.basis file.

        Example of filename: N.dzp.basis.  Use sz(dzp) to read
        the sz-part from the N.dzp.basis file."""
        from gpaw.setup_data import read_maybe_unzipping

        if '(' in self.name:
            assert self.name.endswith(')')
            reduced, name = self.name.split('(')
            name = name[:-1]
        else:
            name = self.name
            reduced = None
        fullname = f'{self.symbol}.{name}.basis'
        if filename is None:
            filename, source = search_for_file(fullname, world=world)
        else:
            source = read_maybe_unzipping(filename)

        self.filename = filename
        self.data = None
        xml.sax.parseString(source, self)

        basis = Basis(symbol=self.symbol, name=self.name, filename=filename,
                      bf_j=[*self.bf_j], ribf_j=[*self.ribf_j],
                      **self._dct)

        if reduced:
            basis = basis.reduce(reduced)

        return basis

    def startElement(self, name, attrs):
        dct = self._dct
        # For name == 'paw_basis' we can save attrs['version'], too.
        if name == 'generator':
            dct['generatorattrs'] = dict(attrs)
            self.data = []
        elif name == 'radial_grid':
            dct['rgd'] = radial_grid_descriptor(**attrs)
        elif name == 'basis_function':
            self.l = int(attrs['l'])
            self.rc = float(attrs['rc'])
            self.type = attrs.get('type')
            self.data = []
            if 'n' in attrs:
                self.n = int(attrs['n'])
            elif self.type[0].isdigit():
                self.n = int(self.type[0])
            else:
                self.n = None

    def characters(self, data):
        if self.data is not None:
            self.data.append(data)

    def endElement(self, name):
        if name == 'basis_function':
            phit_g = np.array([float(x) for x in ''.join(self.data).split()])
            bf = BasisFunction(self.n, self.l, self.rc, phit_g, self.type)
            # Also auxiliary basis functions are added here. They are
            # distinguished by their type='auxiliary'.

            if bf.type == 'auxiliary':
                self.ribf_j.append(bf)
            else:
                self.bf_j.append(bf)

        elif name == 'generator':
            self._dct['generatordata'] = ''.join([line for line in self.data])


class BasisPlotter:
    def __init__(self, premultiply=True, normalize=False,
                 show=False, save=False, ext='png'):
        self.premultiply = premultiply
        self.show = show
        self.save = save
        self.ext = ext
        self.default_filename = '%(symbol)s.%(name)s.' + ext

        self.title = 'Basis functions: %(symbol)s %(name)s'
        self.xlabel = 'radius [Bohr]'
        ylabel = r'\Phi(r)'
        if premultiply:
            ylabel = 'r' + ylabel
        self.ylabel = f'${ylabel}$'

        self.normalize = normalize

    def plot(self, basis, filename=None, ax=None, **plot_args):
        if ax is None:
            from matplotlib import pyplot as plt

            ax = plt.figure().gca()

        if plot_args is None:
            plot_args = {}
        r_g = basis.rgd.r_g

        print('Element  :', basis.symbol)
        print('Name     :', basis.name)
        print()
        print('Basis functions')
        print('---------------')

        norm_j = []
        for j, bf in enumerate(basis.bf_j):
            ng = len(bf.phit_g)
            rphit_g = r_g[:ng] * bf.phit_g
            norm = (rphit_g**2 * basis.rgd.dr_g[:ng]).sum()
            norm_j.append(norm)
            print(bf.type, '[norm=%0.4f]' % norm)

        print()
        print('Generator')
        for key, item in basis.generatorattrs.items():
            print('   ', key, ':', item)
        print()
        print('Generator data')
        print(basis.generatordata)

        if self.premultiply:
            factor = r_g
        else:
            factor = np.ones_like(r_g)

        dashes_l = [(), (6, 3), (4, 1, 1, 1), (1, 1)]

        for norm, bf in zip(norm_j, basis.bf_j):
            ng = len(bf.phit_g)
            y_g = bf.phit_g * factor[:ng]
            if self.normalize:
                y_g /= norm
            ax.plot(r_g[:ng], y_g, label=bf.type[:12],
                    dashes=dashes_l[bf.l], lw=2,
                    **plot_args)
        axis = ax.axis()
        rc = max([bf.rc for bf in basis.bf_j])
        newaxis = [0., rc, axis[2], axis[3]]
        ax.axis(newaxis)
        ax.legend()
        ax.set_title(self.title % basis.__dict__)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        if filename is None:
            filename = self.default_filename
        if self.save:
            ax.get_figure().savefig(filename % basis.__dict__)

        if self.show:
            plt.show()

        return ax


class CLICommand:
    """Plot basis set from FILE."""

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('file', metavar='FILE')
        parser.add_argument(
            '--write', metavar='FILE',
            help='write plot to file inferring format from file extension.')

    @staticmethod
    def run(args):
        from pathlib import Path
        import matplotlib.pyplot as plt
        path = Path(args.file)

        # It is not particularly beautiful that we get the symbol and type
        # from the filename.  It would be better for that information
        # to be stored in the file, but it isn't.
        symbol, name = parse_basis_filename(path.name)
        basis = Basis.read_path(symbol, name, path=path)

        plotter = BasisPlotter()
        ax = plotter.plot(basis)

        if args.write:
            ax.get_figure().savefig(args.write)
        else:
            plt.show()
