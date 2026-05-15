from __future__ import annotations

import hashlib
import os
import re
import xml.sax
from glob import glob
from math import pi, sqrt
from pathlib import Path
from typing import IO, Tuple

import numpy as np
from ase.data import atomic_names, atomic_numbers
from ase.units import Bohr, Ha

from gpaw import setup_paths
from gpaw.atom.radialgd import (AbinitRadialGridDescriptor,
                                AERadialGridDescriptor)
from gpaw.atom.shapefunc import shape_functions
from gpaw.mpi import broadcast
from gpaw.xc.pawcorrection import PAWXCCorrection


class SetupData:
    """Container class for persistent setup attributes and XML I/O."""
    def __init__(self, symbol, xcsetupname,
                 name='paw', readxml=True,
                 zero_reference=False, world=None,
                 generator_version=None):
        self.symbol = symbol
        self.setupname = xcsetupname
        self.name = name
        self.zero_reference = zero_reference
        self.generator_version = generator_version

        self.filename = None  # full path if this setup was loaded from file
        self.fingerprint = None  # hash value of file data if applicable

        self.Z = None
        self.Nc = None
        self.Nv = None

        # Quantum numbers, energies
        self.n_j = []
        self.l_j = []
        self.l_orb_J = self.l_j  # pointer to same list!
        self.f_j = []
        self.eps_j = []
        self.e_kin_jj = None  # <phi | T | phi> - <phit | T | phit>

        self.rgd = None

        # Parameters for compensation charge expansion functions:
        self.shape_function = {'type': 'undefined', 'rc': np.nan}

        # State identifier, like "X-2s" or "X-p1", where X is chemical symbol,
        # for bound and unbound states
        self.id_j = []

        # Partial waves, projectors
        self.phi_jg = []
        self.phit_jg = []
        self.pt_jg = []
        self.rcut_j = []

        # Densities, potentials
        self.nc_g = None
        self.nct_g = None
        self.nvt_g = None
        self.vbar_g = None
        self.vt_g = None

        # Kinetic energy densities of core electrons
        self.tauc_g = None
        self.tauct_g = None

        # Reference energies
        self.e_kinetic = 0.0
        self.e_xc = 0.0
        self.e_electrostatic = 0.0
        self.e_total = 0.0
        self.e_kinetic_core = 0.0

        # Generator may store description of setup in these
        self.type = None
        self.generatorattrs = []
        self.generatordata = ''

        # Optional quantities, normally not used
        self.X_p = None
        self.X_wp = {}
        self.X_pg = None
        self.ExxC = None
        self.ExxC_w = {}
        self.X_gamma = None
        self.extra_xc_data = {}
        self.phicorehole_g = None
        self.fcorehole = 0.0
        self.lcorehole = None
        self.ncorehole = None
        self.core_hole_e = None
        self.core_hole_e_kin = None
        self.has_corehole = False

        # Parameters for zero-potential:
        self.l0 = None
        self.e0 = None
        self.r0 = None
        self.nderiv0 = None

        self.orbital_free = False  # orbital-free DFT

        self.version = None

        if readxml:
            self.read_xml(world=world)

    @classmethod
    def find_and_read_path(cls, symbol, xctype,
                           setuptype='paw', world=None):

        setupdata = SetupData(symbol, xctype,
                              name=setuptype,
                              readxml=False,
                              world=world)

        setupdata.filename, source = search_for_file(setupdata.stdfilename,
                                                     world=world)
        PAWXMLParser(setupdata).parse(source=source, world=world)

        nj = len(setupdata.l_j)
        setupdata.e_kin_jj.shape = (nj, nj)

        return setupdata

    @property
    def stdfilename(self):
        """Default filename if this setup is written."""
        assert self.symbol is not None
        assert self.setupname is not None
        if self.name is None or self.name == 'paw':
            return f'{self.symbol}.{self.setupname}'
        else:
            return f'{self.symbol}.{self.name}.{self.setupname}'

    def __repr__(self):
        return ('{0}({symbol!r}, {setupname!r}, name={name!r}, '
                'generator_version={generator_version!r}, ...)'
                .format(self.__class__.__name__, **vars(self)))

    def append(self, n, l, f, e, rcut, phi_g, phit_g, pt_g):
        self.n_j.append(n)
        self.l_j.append(l)
        self.f_j.append(f)
        self.eps_j.append(e)
        self.rcut_j.append(rcut)
        self.phi_jg.append(phi_g)
        self.phit_jg.append(phit_g)
        self.pt_jg.append(pt_g)

    # XXX delete me
    def read_xml(self, source=None, world=None):
        PAWXMLParser(self).parse(source=source, world=world)
        nj = len(self.l_j)
        self.e_kin_jj.shape = (nj, nj)

    def is_compatible(self, xc):
        return xc.get_setup_name() == self.setupname

    def print_info(self, text, setup):
        if self.phicorehole_g is None:
            text(self.symbol + ':')
        else:
            text(f'{self.symbol}:  # ({self.fcorehole:.1f} core hole)')
        text('  name:', atomic_names[atomic_numbers[self.symbol]])
        text('  id:', self.fingerprint)
        text('  Z:', self.Z)
        text('  valence:', self.Nv)
        if self.phicorehole_g is None:
            text('  core: %d' % self.Nc)
        else:
            text(f'  core: {self.Nc:.1f}')
        text('  charge:', self.Z - self.Nv - self.Nc)
        if setup.hubbard_u is not None:
            description = ''.join([f'  {line}' for line
                                   in setup.hubbard_u.descriptions()])
            text(description)
        text('  file:', self.filename)
        sf = self.shape_function
        text(f'  compensation charges: {{type: {sf["type"]},\n'
             f'                         rc: {sf["rc"] * Bohr:.2f},\n'
             f'                         lmax: {setup.lmax}}}')
        text(f'  cutoffs: {{filter: {setup.rcutfilter * Bohr:.2f},\n'
             f'            core: {setup.rcore * Bohr:.2f}}}')
        text('  projectors:')
        text('    #              energy  rcut')
        j = 0
        for n, l, f, eps in zip(self.n_j, self.l_j, self.f_j, self.eps_j):
            if n > 0:
                f = f'({f:.2f})'
                text('    - %d%s%-5s %9.3f   %5.3f' % (
                    n, 'spdf'[l], f, eps * Ha, self.rcut_j[j] * Bohr))
            else:
                text('    -  {}       {:9.3f}   {:5.3f}'.format(
                    'spdf'[l], eps * Ha, self.rcut_j[j] * Bohr))
            j += 1
        text()

    def create_compensation_charge_functions(self, lmax):
        """Create shape functions used to expand compensation charges."""
        g_lg = shape_functions(self.rgd, **self.shape_function, lmax=lmax)
        return g_lg

    def get_smooth_core_density_integral(self, Delta0):
        return -Delta0 * sqrt(4 * pi) - self.Z + self.Nc

    def get_overlap_correction(self, Delta0_ii):
        return sqrt(4.0 * pi) * Delta0_ii

    def get_linear_kinetic_correction(self, T0_qp):
        e_kin_jj = self.e_kin_jj
        nj = len(e_kin_jj)
        K_q = []
        for j1 in range(nj):
            for j2 in range(j1, nj):
                K_q.append(e_kin_jj[j1, j2])
        K_p = sqrt(4 * pi) * np.dot(K_q, T0_qp)
        return K_p

    def find_core_density_cutoff(self, nc_g):
        if self.Nc == 0:
            return 1.0
        else:
            rgd = self.rgd
            N = 0.0
            g = self.rgd.N - 1
            while N < 1e-7:
                N += sqrt(4 * pi) * nc_g[g] * rgd.r_g[g]**2 * rgd.dr_g[g]
                g -= 1
            return rgd.r_g[g]

    def get_xc_correction(self, rgd, xc, gcut2, lcut):
        phicorehole_g = self.phicorehole_g
        if phicorehole_g is not None:
            phicorehole_g = phicorehole_g[:gcut2].copy()

        xc_correction = PAWXCCorrection(
            [phi_g[:gcut2] for phi_g in self.phi_jg],
            [phit_g[:gcut2] for phit_g in self.phit_jg],
            self.nc_g[:gcut2] / sqrt(4 * pi),
            self.nct_g[:gcut2] / sqrt(4 * pi),
            rgd,
            list(enumerate(self.l_j)),
            min(2 * lcut, 4),
            self.e_xc,
            phicorehole_g,
            self.fcorehole,
            None if self.tauc_g is None else self.tauc_g[:gcut2].copy(),
            None if self.tauct_g is None else self.tauct_g[:gcut2].copy())

        return xc_correction

    def write_xml(self, path=None) -> None:
        if path is None:
            path = self.stdfilename

        with open(path, 'w') as fd:
            self._write_xml(fd)

    def _write_xml(self, xml: IO[str]) -> None:
        l_j = self.l_j

        print('<?xml version="1.0"?>', file=xml)
        print(f'<paw_dataset version="{self.version}">',
              file=xml)
        name = atomic_names[atomic_numbers[self.symbol]].title()
        comment1 = name + ' setup for the Projector Augmented Wave method.'
        comment2 = 'Units: Hartree and Bohr radii.'
        comment2 += ' ' * (len(comment1) - len(comment2))
        print('  <!--', comment1, '-->', file=xml)
        print('  <!--', comment2, '-->', file=xml)

        print(('  <atom symbol="%s" Z="%r" core="%s" valence="%s"/>' %
               (self.symbol, self.Z, self.Nc, self.Nv)), file=xml)
        if self.orbital_free:
            type = 'OFDFT'
            name = self.setupname
        elif self.setupname == 'LDA':
            type = 'LDA'
            name = 'PW'
        else:
            type = 'GGA'
            name = self.setupname
        print(f'  <xc_functional type="{type}" name="{name}"/>',
              file=xml)
        gen_attrs = ' '.join([f'{key}="{value}"' for key, value
                              in self.generatorattrs])
        print(f'  <generator {gen_attrs}>', file=xml)
        print(f'    {self.generatordata}', file=xml)
        print('  </generator>', file=xml)
        print(f'  <ae_energy kinetic="{self.e_kinetic}" xc="{self.e_xc}"',
              file=xml)
        print('             electrostatic="%s" total="%s"/>' %
              (self.e_electrostatic, self.e_total), file=xml)

        print(f'  <core_energy kinetic="{self.e_kinetic_core}"/>', file=xml)
        print('  <valence_states>', file=xml)
        line1 = '    <state n="%d" l="%d" f="%s" rc="%s" e="%s" id="%s"/>'
        line2 = '    <state       l="%d"        rc="%s" e="%s" id="%s"/>'

        for id, l, n, f, e, rc in zip(self.id_j, l_j, self.n_j, self.f_j,
                                      self.eps_j, self.rcut_j):
            if n > 0:
                print(line1 % (n, l, f, rc, e, id), file=xml)
            else:
                print(line2 % (l, rc, e, id), file=xml)
        print('  </valence_states>', file=xml)

        print(self.rgd.xml('g1'), file=xml)

        print('  <shape_function type="{type}" rc="{rc}"/>'
              .format(**self.shape_function), file=xml)

        if self.r0 is None:
            # Old setups:
            xml.write('  <zero_potential grid="g1">\n')
        elif self.l0 is None:
            xml.write('  <zero_potential type="polynomial" ' +
                      'nderiv="%d" r0="%r" grid="g1">\n' %
                      (self.nderiv0, self.r0))
        else:
            xml.write(('  <zero_potential type="%s" ' +
                       'e0="%r" nderiv="%d" r0="%r" grid="g1">\n') %
                      ('spdfg'[self.l0], self.e0, self.nderiv0, self.r0))

        for x in self.vbar_g:
            print(x, end=' ', file=xml)
        print('\n  </zero_potential>', file=xml)

        if self.has_corehole:
            print((('  <core_hole_state state="%d%s" ' +
                    'removed="%s" eig="%s" ekin="%s">') %
                   (self.ncorehole, 'spdf'[self.lcorehole],
                    self.fcorehole,
                    self.core_hole_e, self.core_hole_e_kin)), file=xml)
            for x in self.phicorehole_g:
                print(x, end=' ', file=xml)
            print('\n  </core_hole_state>', file=xml)

        for name, a in [('ae_core_density', self.nc_g),
                        ('pseudo_core_density', self.nct_g),
                        ('ae_core_kinetic_energy_density', self.tauc_g),
                        ('pseudo_core_kinetic_energy_density', self.tauct_g)]:
            print(f'  <{name} grid="g1">\n    ', end=' ', file=xml)
            for x in a:
                print(x, end=' ', file=xml)
            print(f'\n  </{name}>', file=xml)

        # Print xc-specific data to setup file (used so for KLI and GLLB)
        for name, a in self.extra_xc_data.items():
            newname = 'GLLB_' + name
            print(f'  <{newname} grid="g1">\n    ', end=' ', file=xml)
            for x in a:
                print(x, end=' ', file=xml)
            print(f'\n  </{newname}>', file=xml)

        for id, l, u, s, q, in zip(self.id_j, l_j, self.phi_jg, self.phit_jg,
                                   self.pt_jg):
            for name, a in [('ae_partial_wave', u),
                            ('pseudo_partial_wave', s),
                            ('projector_function', q)]:
                print(f'  <{name} state="{id}" grid="g1">\n    ',
                      end=' ', file=xml)
                for x in a:
                    print(x, end=' ', file=xml)
                print(f'\n  </{name}>', file=xml)

        if self.vt_g is not None:
            xml.write('  <pseudo_potential grid="g1">\n')
            for x in self.vt_g:
                print(x, end=' ', file=xml)
            print('\n  </pseudo_potential>', file=xml)

        print('  <kinetic_energy_differences>', end=' ', file=xml)
        nj = len(self.e_kin_jj)
        for j1 in range(nj):
            print('\n    ', end=' ', file=xml)
            for j2 in range(nj):
                print(self.e_kin_jj[j1, j2], end=' ', file=xml)
        print('\n  </kinetic_energy_differences>', file=xml)

        if self.X_p is not None:
            print('  <exact_exchange_X_matrix>\n    ', end=' ', file=xml)
            for x in self.X_p:
                print(x, end=' ', file=xml)
            print('\n  </exact_exchange_X_matrix>', file=xml)

            print(f'  <exact_exchange core-core="{self.ExxC}"/>', file=xml)
            for omega, Ecc in self.ExxC_w.items():
                print(f'  <erfc_exchange omega="{omega}" core-core="{Ecc}"/>',
                      file=xml)
                print(f'  <erfc_exchange_X_matrix omega="{omega}" X_p="',
                      end=' ', file=xml)
                for x in self.X_wp[omega]:
                    print(x, end=' ', file=xml)
                print('"/>', file=xml)

        if self.X_pg is not None:
            print('  <yukawa_exchange_X_matrix>\n    ', end=' ', file=xml)
            for x in self.X_pg:
                print(x, end=' ', file=xml)
            print('\n  </yukawa_exchange_X_matrix>', file=xml)
            print(f'  <yukawa_exchange gamma="{self.X_gamma}"/>', file=xml)
        print('</paw_dataset>', file=xml)

    def build(self, xcfunc, lmax, basis, filter=None,
              backwards_compatible=True):
        from gpaw.setup import Setup
        setup = Setup(self, xcfunc, lmax, basis, filter,
                      backwards_compatible=backwards_compatible)
        return setup


def read_maybe_unzipping(path: Path | str) -> bytes:
    import gzip
    if Path(path).suffix == '.gz':
        with gzip.open(path) as fd:
            return fd.read()

    with open(path, 'rb') as fd:
        return fd.read()


def search_for_file(name: str, world=None) -> Tuple[str, bytes]:
    """Traverse gpaw setup paths to find file.

    Returns the file path and file contents.  If the file is not
    found, raises RuntimeError."""

    if world is None or world.rank == 0:
        source = b''
        filename = None
        for path in setup_paths:
            pattern = os.path.join(path, name)
            filenames = glob(pattern) + glob(f'{pattern}.gz')
            if filenames:
                # The globbing is a hack to grab the 'newest' version if
                # the files are somehow version numbered; then we want the
                # last/newest of the results (used with SG15).  (User must
                # instantiate (UPF)SetupData directly to override.)
                filename = max(filenames)
                source = read_maybe_unzipping(filename)
                break

    if world is not None:
        if world.rank == 0:
            broadcast((filename, source), 0, world)
        else:
            filename, source = broadcast(None, 0, world)

    if filename is None:
        if name.endswith('basis'):
            _type = 'basis set'
        else:
            _type = 'PAW dataset'
        err = f'Could not find required {_type} file "{name}".'
        helpful_message = """
You need to set the GPAW_SETUP_PATH environment variable to point to
the directories where PAW dataset and basis files are stored.  See
https://gpaw.readthedocs.io/install.html#install-paw-datasets
for details."""
        raise FileNotFoundError(f'{err}\n{helpful_message}\n')

    return filename, source


class PAWXMLParser(xml.sax.handler.ContentHandler):
    def __init__(self, setup):
        xml.sax.handler.ContentHandler.__init__(self)
        self.setup = setup
        self.id = None
        self.data = None

    def parse(self, source=None, world=None):
        setup = self.setup
        if source is None:
            setup.filename, source = search_for_file(setup.stdfilename, world)

        setup.fingerprint = hashlib.md5(source).hexdigest()

        # XXXX There must be a better way!
        # We don't want to look at the dtd now.  Remove it:
        source = re.compile(b'<!DOCTYPE .*?>', re.DOTALL).sub(b'', source, 1)
        xml.sax.parseString(source, self)

        if setup.zero_reference:
            setup.e_total = 0.0
            setup.e_kinetic = 0.0
            setup.e_electrostatic = 0.0
            setup.e_xc = 0.0

    def startElement(self, name, attrs):
        setup = self.setup
        if name == 'paw_setup' or name == 'paw_dataset':
            setup.version = attrs['version']
            assert [int(v) for v in setup.version.split('.')] >= [0, 4]
        if name == 'atom':
            Z = float(attrs['Z'])
            setup.Z = Z
            assert setup.symbol is None or setup.symbol == attrs['symbol']
            setup.symbol = attrs['symbol']
            assert setup.Z == Z
            setup.Nc = float(attrs['core'])
            Nv = float(attrs['valence'])
            setup.Nv = int(Nv)
            assert setup.Nv == Nv
        elif name == 'xc_functional':
            if attrs['type'] == 'LDA':
                setup.xcname = 'LDA'
            else:
                setup.xcname = attrs['name']
                if attrs['type'] == 'OFDFT':
                    setup.orbital_free = True
                else:
                    assert attrs['type'] == 'GGA'
            assert setup.setupname is None or setup.setupname == setup.xcname
            setup.setupname = setup.xcname
        elif name == 'ae_energy':
            setup.e_total = float(attrs['total'])
            setup.e_kinetic = float(attrs['kinetic'])
            setup.e_electrostatic = float(attrs['electrostatic'])
            setup.e_xc = float(attrs['xc'])
        elif name == 'core_energy':
            setup.e_kinetic_core = float(attrs['kinetic'])
        elif name == 'state':
            setup.n_j.append(int(attrs.get('n', -1)))
            setup.l_j.append(int(attrs['l']))
            setup.f_j.append(float(attrs.get('f', 0)))
            setup.eps_j.append(float(attrs['e']))
            setup.rcut_j.append(float(attrs.get('rc', -1)))
            setup.id_j.append(attrs['id'])
            # Compatibility with old setups:
            version = [int(v) for v in setup.version.split('.')]
            if version < [0, 6] and setup.f_j[-1] == 0:
                setup.n_j[-1] = -1
        elif name == 'radial_grid':
            if attrs['eq'] == 'r=a*i/(n-i)':
                beta = float(attrs['a'])
                ng = int(attrs['n'])
                setup.rgd = AERadialGridDescriptor(beta / ng, 1.0 / ng, ng)
            elif attrs['eq'] == 'r=a*i/(1-b*i)':
                a = float(attrs['a'])
                b = float(attrs['b'])
                N = int(attrs['n'])
                setup.rgd = AERadialGridDescriptor(a, b, N)
            elif attrs['eq'] == 'r=a*(exp(d*i)-1)':
                a = float(attrs['a'])
                d = float(attrs['d'])
                istart = int(attrs['istart'])
                iend = int(attrs['iend'])
                assert istart == 0
                setup.rgd = AbinitRadialGridDescriptor(a, d, iend + 1)
            else:
                raise ValueError('Unknown grid:' + attrs['eq'])
        elif name == 'shape_function':
            assert attrs['type'] in {'gauss', 'sinc', 'bessel'}
            setup.shape_function = {'type': attrs['type'],
                                    'rc': float(attrs['rc'])}
        elif name in ['ae_core_density', 'pseudo_core_density',
                      'localized_potential', 'yukawa_exchange_X_matrix',
                      'kinetic_energy_differences', 'exact_exchange_X_matrix',
                      'ae_core_kinetic_energy_density',
                      'pseudo_core_kinetic_energy_density',
                      'pseudo_potential']:
            self.data = []
        elif name.startswith('GLLB_'):
            self.data = []
        elif name in ['ae_partial_wave', 'pseudo_partial_wave']:
            self.data = []
            self.id = attrs['state']
        elif name == 'projector_function':
            self.id = attrs['state']
            self.data = []
        elif name == 'erfc_exchange':
            setup.ExxC_w[float(attrs['omega'])] = float(attrs['core-core'])
        elif name == 'exact_exchange':
            setup.ExxC = float(attrs['core-core'])
        elif name == 'erfc_exchange_X_matrix':
            X_p = np.array([float(x) for x in ''.join(attrs['X_p']).split()])
            setup.X_wp[float(attrs['omega'])] = X_p
        elif name == 'yukawa_exchange':
            setup.X_gamma = float(attrs['gamma'])
        elif name == 'core_hole_state':
            setup.has_corehole = True
            setup.fcorehole = float(attrs['removed'])
            full_state = attrs['state']
            state_l = full_state.lstrip('0123456789')
            assert state_l
            state_n = full_state[:-len(state_l)]
            assert state_n
            setup.ncorehole = int(state_n)
            setup.lcorehole = 'spdf'.find(state_l)
            setup.core_hole_e = float(attrs['eig'])
            setup.core_hole_e_kin = float(attrs['ekin'])
            self.data = []
        elif name == 'zero_potential':
            if 'type' in attrs:
                setup.r0 = float(attrs['r0'])
                setup.nderiv0 = int(attrs['nderiv'])
                if attrs['type'] == 'polynomial':
                    setup.e0 = None
                    setup.l0 = None
                else:
                    setup.e0 = float(attrs['e0'])
                    setup.l0 = 'spdfg'.find(attrs['type'])
            self.data = []
        elif name == 'generator':
            setup.type = attrs['type']
            setup.generator_version = int(attrs.get('version', '1'))
        else:
            self.data = None

    def characters(self, data):
        if self.data is not None:
            self.data.append(data)

    def endElement(self, name):
        setup = self.setup
        if self.data is None:
            return
        x_g = np.array([float(x) for x in ''.join(self.data).split()])
        if name == 'ae_core_density':
            setup.nc_g = x_g
        elif name == 'pseudo_core_density':
            setup.nct_g = x_g
        elif name == 'kinetic_energy_differences':
            setup.e_kin_jj = x_g
        elif name == 'ae_core_kinetic_energy_density':
            setup.tauc_g = x_g
        elif name == 'pseudo_valence_density':
            setup.nvt_g = x_g
        elif name == 'pseudo_core_kinetic_energy_density':
            setup.tauct_g = x_g
        elif name in ['localized_potential', 'zero_potential']:  # XXX
            setup.vbar_g = x_g
        elif name in ['pseudo_potential']:
            setup.vt_g = x_g
        elif name.startswith('GLLB_'):
            # Add setup tags starting with GLLB_ to extra_xc_data. Remove
            # GLLB_ from front of string:
            if name == 'GLLB_w_j':
                v1, v2 = (int(x) for x in self.setup.version.split('.'))
                if (v1, v2) < (0, 8):
                    # Order was wrong in old generator:
                    w_j = {}
                    j_old = 0
                    for l in range(4):
                        for j_new, (n1, l1) in enumerate(zip(setup.n_j,
                                                             setup.l_j)):
                            if l == l1:
                                w_j[j_new] = x_g[j_old]
                                j_old += 1
                    x_g = [w_j[j] for j in range(len(w_j))]
            setup.extra_xc_data[name[5:]] = x_g

        elif name == 'ae_partial_wave':
            j = len(setup.phi_jg)
            assert self.id == setup.id_j[j]
            setup.phi_jg.append(x_g)
        elif name == 'pseudo_partial_wave':
            j = len(setup.phit_jg)
            assert self.id == setup.id_j[j]
            setup.phit_jg.append(x_g)
        elif name == 'projector_function':
            j = len(setup.pt_jg)
            assert self.id == setup.id_j[j]
            setup.pt_jg.append(x_g)
        elif name == 'exact_exchange_X_matrix':
            setup.X_p = x_g
        elif name == 'yukawa_exchange_X_matrix':
            setup.X_pg = x_g
        elif name == 'core_hole_state':
            setup.phicorehole_g = x_g
