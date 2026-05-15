"""This module defines a linear response TDDFT-class."""
import numbers
import sys
from math import sqrt
from typing import Dict, Any
import numpy as np

from ase.units import Hartree
from ase.utils.timing import Timer

import gpaw.mpi as mpi
from gpaw.xc import XC
from gpaw.lrtddft.excitation import Excitation, ExcitationList, get_filehandle
from gpaw.lrtddft.kssingle import KSSingles
from gpaw.lrtddft.omega_matrix import OmegaMatrix
from gpaw.lrtddft.apmb import ApmB
from gpaw.lrtddft.spectrum import spectrum

__all__ = ['LrTDDFT', 'photoabsorption_spectrum', 'spectrum']


class LrTDDFT(ExcitationList):
    """Linear Response TDDFT excitation list class

    Input parameters:

    calculator:
    the calculator object after a ground state calculation

    nspins:
    number of spins considered in the calculation
    Note: Valid only for unpolarised ground state calculation

    eps:
    Minimal occupation difference for a transition (default 0.001)

    istart:
    First occupied state to consider
    jend:
    Last unoccupied state to consider

    xc:
    Exchange-Correlation approximation in the Kernel
    derivative_level:
    0: use Exc, 1: use vxc, 2: use fxc  if available

    filename:
    read from a file
    """

    default_parameters: Dict[str, Any] = {
        'nspins': None,
        'restrict': {},
        'xc': None,
        'derivative_level': 1,
        'numscale': 0.00001,
        'filename': None,
        'finegrid': 2,
        'force_ApmB': False,  # for tests
        'eh_comm': None,  # parallelization over eh-pairs
        'poisson': None}  # use calculator's Poisson

    def __init__(self, calculator=None, log=None, txt='-', **kwargs):

        self.energy_to_eV_scale = Hartree
        self.timer = Timer()
        self.diagonalized = False

        self.set(**kwargs)
        self.calculator = calculator

        ExcitationList.__init__(self, log=log, txt=txt)

        if self.eh_comm is None:
            self.eh_comm = mpi.serial_comm
        elif isinstance(self.eh_comm, (mpi.world.__class__,
                                       mpi.serial_comm.__class__)):
            # Correct type already.
            pass
        else:
            # world should be a list of ranks:
            self.eh_comm = mpi.world.new_communicator(
                np.asarray(self.eh_comm))

        if calculator is not None and calculator.initialized:
            # XXXX not ready for k-points
            assert len(calculator.wfs.kd.bzk_kc) == 1
            if calculator.wfs.mode not in ['fd', 'lcao']:
                raise RuntimeError('LrTDDFT supports only fd and lcao modes')
            if calculator.wfs.mode != 'lcao':
                calculator.converge_wave_functions()
            if calculator.density.nct_G is None:
                spos_ac = calculator.initialize_positions()
                calculator.wfs.initialize(calculator.density,
                                          calculator.hamiltonian, spos_ac)

            self.forced_update()

    @property
    def calculator(self):
        return self._calc

    @calculator.setter
    def calculator(self, calc):
        self._calc = calc

        if self.xc is None and calc is not None:
            if calc.initialized:
                self.xc = calc.hamiltonian.xc
            else:
                self.xc = calc.parameters['xc']

    def set(self, **kwargs):
        """Change parameters."""
        changed = []
        for key, value in LrTDDFT.default_parameters.items():
            if hasattr(self, key):
                value = getattr(self, key)  # do not overwrite
            setattr(self, key, kwargs.pop(key, value))
            if value != getattr(self, key):
                changed.append(key)

        for key in kwargs:
            raise KeyError('Unknown key ' + key)

        return changed

    def analyse(self, what=None, out=None, min=0.1):
        """Print info about the transitions.

        Parameters:
          1. what: I list of excitation indicees, None means all
          2. out : I where to send the output, None means sys.stdout
          3. min : I minimal contribution to list (0<min<1)
        """
        if what is None:
            what = range(len(self))
        elif isinstance(what, numbers.Integral):
            what = [what]

        if out is None:
            out = sys.stdout

        for i in what:
            print(str(i) + ':', self[i].analyse(min=min), file=out)

    def calculate(self, atoms):
        self.calculator = atoms.calc
        if not hasattr(self, 'Om') or self.calculator.check_state(atoms):
            self.calculator.get_potential_energy(atoms)
            self.forced_update()
        return self

    def forced_update(self):
        """Recalc yourself."""
        Om = OmegaMatrix
        name = 'LrTDDFT'

        hybrid = False
        if self.xc and self.xc != 'RPA':
            xc = self.xc
            if isinstance(self.xc, str):
                xc = XC(self.xc)
            hybrid = hasattr(xc, 'hybrid') and xc.hybrid > 0.0

        if hybrid or self.force_ApmB:
            Om = ApmB
            name = 'LrTDDFThyb'

        kss = KSSingles(restrict=self.restrict,
                        log=self.log)
        atoms = self.calculator.get_atoms()
        kss.calculate(atoms, self.nspins)

        self.Om = Om(self.calculator, kss,
                     self.xc, self.derivative_level, self.numscale,
                     finegrid=self.finegrid, eh_comm=self.eh_comm,
                     poisson=self.poisson, log=self.log)
        self.name = name

    def diagonalize(self, **kwargs):
        """Diagonalize and save new Eigenvalues and Eigenvectors"""
        self.set(**kwargs)
        self.timer.start('diagonalize')
        self.timer.start('omega')
        self.Om.diagonalize(kwargs.pop('restrict', {}))
        self.timer.stop('omega')
        self.diagonalized = True

        # remove old stuff
        self.timer.start('clean')
        while len(self):
            self.pop()
        self.timer.stop('clean')

        self.log('LrTDDFT digonalized:')
        self.timer.start('build')
        for j in range(len(self.Om.kss)):
            self.append(LrTDDFTExcitation(self.Om, j))
            self.log(' ', str(self[-1]))
        self.timer.stop('build')
        self.timer.stop('diagonalize')

    @classmethod
    def read(cls, filename=None, fh=None, restrict={}, log=None, txt=None):
        """Read myself from a file"""
        lr = cls(log=log, txt=txt)
        timer = lr.timer
        timer.start('name')
        if fh is None:
            f = get_filehandle(lr, filename)
        else:
            f = fh
        timer.stop('name')

        timer.start('header')
        # get my name
        s = f.readline().strip()
        lr.name = s.split()[1]

        xc = f.readline().strip().split()[0]
        if xc == 'RPA':
            lr.xc = xc
        else:
            lr.xc = XC(xc)
        values = f.readline().split()
        eps = float(values[0])
        if len(values) > 1:
            lr.derivative_level = int(values[1])
            lr.numscale = float(values[2])
            lr.finegrid = int(values[3])
        else:
            # old writing style, use old defaults
            lr.numscale = 0.001
        timer.stop('header')

        timer.start('init_kss')
        kss = KSSingles.read(fh=f, log=log)
        assert eps == kss.restrict['eps']
        lr.restrict = kss.restrict.values
        timer.stop('init_kss')
        timer.start('init_obj')
        if lr.name == 'LrTDDFT':
            lr.Om = OmegaMatrix(kss=kss, filehandle=f, log=lr.log)
        else:
            lr.Om = ApmB(kss=kss, filehandle=f, log=lr.log)
        timer.stop('init_obj')

        if not len(restrict):
            timer.start('read diagonalized')
            # check if already diagonalized
            p = f.tell()
            s = f.readline()
            if s != '# Eigenvalues\n' or len(restrict):
                # no further info or selection of
                # Kohn-Sham states changed
                # go back to previous position
                f.seek(p)
            else:
                lr.diagonalized = True
                # load the eigenvalues
                n = int(f.readline().split()[0])
                for i in range(n):
                    lr.append(LrTDDFTExcitation(string=f.readline()))
                # load the eigenvectors
                timer.start('read eigenvectors')
                f.readline()
                for i in range(n):
                    lr[i].f = np.array([float(x) for x in
                                        f.readline().split()])
                    lr[i].kss = lr.kss
                timer.stop('read eigenvectors')
            timer.stop('read diagonalized')
        else:
            timer.start('diagonalize')
            lr.diagonalize(restrict=restrict)
            timer.stop('diagonalize')

        if fh is None:
            f.close()

        return lr

    @property
    def kss(self):
        return self.Om.kss

    def singlets_triplets(self):
        """Split yourself into a singlet and triplet object"""

        slr = LrTDDFT(nspins=self.nspins, xc=self.xc,
                      restrict=self.kss.restrict.values,
                      derivative_level=self.derivative_level,
                      numscale=self.numscale)
        tlr = LrTDDFT(nspins=self.nspins, xc=self.xc,
                      restrict=self.kss.restrict.values,
                      derivative_level=self.derivative_level,
                      numscale=self.numscale)
        slr.Om, tlr.Om = self.Om.singlets_triplets()

        return slr, tlr

    def single_pole_approximation(self, i, j):
        """Return the excitation according to the
        single pole approximation. See e.g.:
        Grabo et al, Theochem 501 (2000) 353-367
        """
        for ij, kss in enumerate(self.kss):
            if kss.i == i and kss.j == j:
                return sqrt(self.Om.full[ij][ij]) * Hartree
                return self.Om.full[ij][ij] / kss.energy * Hartree

    def __str__(self):
        string = ExcitationList.__str__(self)
        string += '# derived from:\n'
        string += self.Om.kss.__str__()
        return string

    def write(self, filename=None, fh=None):
        """Write current state to a file.

        'filename' is the filename. If the filename ends in .gz,
        the file is automatically saved in compressed gzip format.

        'fh' is a filehandle. This can be used to write into already
        opened files.
        """

        if self.calculator is None:
            rank = mpi.world.rank
        else:
            rank = self.calculator.wfs.world.rank

        if rank == 0:
            if fh is None:
                f = get_filehandle(self, filename, mode='w')
            else:
                f = fh

            f.write('# ' + self.name + '\n')
            if isinstance(self.xc, str):
                xc = self.xc
            else:
                xc = self.xc.tostring()
            if xc is None:
                xc = 'RPA'
            if self.calculator is not None:
                xc += ' ' + self.calculator.get_xc_functional()
            f.write(xc + '\n')
            f.write('%g %d %g %d' % (self.kss.restrict['eps'],
                                     int(self.derivative_level),
                                     self.numscale, int(self.finegrid)) + '\n')
            self.kss.write(fh=f)
            self.Om.write(fh=f)

            if len(self):
                f.write('# Eigenvalues\n')
                f.write(f'{len(self)}\n')
                for ex in self:
                    f.write(ex.outstring())
                f.write('# Eigenvectors\n')
                for ex in self:
                    for w in ex.f:
                        f.write('%g ' % w)
                    f.write('\n')

            if fh is None:
                f.close()
        mpi.world.barrier()

    def overlap(self, ov_nn, other):
        """Matrix element overlap determined from pair density overlaps.

        Parameters
        ----------
        ov_nn: array
            Wave function overlap factors from a displaced calculator.

            Index 0 corresponds to our own wavefunctions and
            index 1 to the others wavefunctions

        Returns
        -------
        ov_pp: array
            Overlap
        """
        # XXX ov_pp = self.kss.overlap(ov_nn, other.kss)
        ov_pp = self.Om.kss.overlap(ov_nn, other.Om.kss)
        self.diagonalize()
        other.diagonalize()
        # ov[pLm, pLo] = Om[pLm, :pKm]* ov[:pKm, pLo]
        return np.dot(self.Om.eigenvectors.conj(),
                      # ov[pKm, pLo] = ov[pKm, :pKo] Om[pLo, :pKo].T
                      np.dot(ov_pp, other.Om.eigenvectors.T))

    def __getitem__(self, i):
        if not self.diagonalized:
            self.diagonalize()
        return list.__getitem__(self, i)

    def __iter__(self):
        if not self.diagonalized:
            self.diagonalize()
        return list.__iter__(self)

    def __len__(self):
        if not self.diagonalized:
            self.diagonalize()
        return list.__len__(self)

    def __del__(self):
        try:
            self.timer.write(self.log.fd)
        except AttributeError:
            pass


class LrTDDFTExcitation(Excitation):

    def __init__(self, Om=None, i=None,
                 e=None, m=None, string=None):

        # multiplicity comes from Kohn-Sham contributions
        self.fij = 1

        if string is not None:
            self.fromstring(string)
            return None

        # define from the diagonalized Omega matrix
        if Om is not None:
            if i is None:
                raise RuntimeError

            ev = Om.eigenvalues[i]
            if ev < 0:
                # we reached an instability, mark it with a negative value
                self.energy = -sqrt(-ev)
            else:
                self.energy = sqrt(ev)
            self.f = Om.eigenvectors[i]
            self.kss = Om.kss

            self.kss.set_arrays()
            self.me = np.dot(self.f, self.kss.me)
            erat_k = np.sqrt(self.kss.energies / self.energy)
            wght_k = np.sqrt(self.kss.fij) * self.f
            ew_k = erat_k * wght_k
            self.mur = np.dot(ew_k, self.kss.mur)
            if self.kss.muv is not None:
                self.muv = np.dot(ew_k, self.kss.muv)
            else:
                self.muv = None
            if self.kss.magn is not None:
                self.magn = np.dot(wght_k / erat_k, self.kss.magn)
            else:
                self.magn = None

            return

        # define from energy and matrix element
        if e is not None:
            self.energy = e
            if m is None:
                raise RuntimeError
            self.me = m
            return

        raise RuntimeError

    def density_change(self, paw):
        """get the density change associated with this transition"""
        raise NotImplementedError

    def fromstring(self, string):
        l = string.split()
        self.energy = float(l.pop(0))
        if len(l) == 3:  # old writing style
            self.me = np.array([float(l.pop(0)) for i in range(3)])
        else:
            self.mur = np.array([float(l.pop(0)) for i in range(3)])
            self.me = - self.mur * sqrt(self.energy)
            self.muv = np.array([float(l.pop(0)) for i in range(3)])
            self.magn = np.array([float(l.pop(0)) for i in range(3)])

    def outstring(self):
        str = '%g ' % self.energy
        str += '  '
        for m in self.mur:
            str += '%12.4e' % m
        str += '  '
        for m in self.muv:
            str += '%12.4e' % m
        str += '  '
        for m in self.magn:
            str += '%12.4e' % m
        str += '\n'
        return str

    def __str__(self):
        m2 = np.sum(self.me * self.me)
        m = sqrt(m2)
        if m > 0:
            me = self.me / m
        else:
            me = self.me
        str = '<LrTDDFTExcitation> om=%g[eV] |me|=%g (%.2f,%.2f,%.2f)' % \
              (self.energy * Hartree, m, me[0], me[1], me[2])
        return str

    def analyse(self, min=.1):
        """Return an analysis string of the excitation"""
        osc = self.get_oscillator_strength()
        s = ('E=%.3f' % (self.energy * Hartree) + ' eV, ' +
             'f=%.5g' % osc[0] + ', (%.5g,%.5g,%.5g) ' %
             (osc[1], osc[2], osc[3]) + '\n')
        # 'R=%.5g' % self.get_rotatory_strength() + ' cgs\n')

        def sqr(x):
            return x * x
        spin = ['u', 'd']
        min2 = sqr(min)
        rest = np.sum(self.f**2)
        for f, k in zip(self.f, self.kss):
            f2 = sqr(f)
            if f2 > min2:
                s += '  %d->%d ' % (k.i, k.j) + spin[k.pspin] + ' '
                s += '%.3g \n' % f2
                rest -= f2
        s += '  rest=%.3g' % rest
        return s


def photoabsorption_spectrum(excitation_list, spectrum_file=None,
                             e_min=None, e_max=None, delta_e=None,
                             energyunit='eV',
                             folding='Gauss', width=0.1, comment=None):
    """Uniform absorption spectrum interface

    Parameters:
    ================= ===================================================
    ``exlist``        ExcitationList
    ``spectrum_file`` File name for the output file, STDOUT if not given
    ``e_min``         min. energy, set to cover all energies if not given
    ``e_max``         max. energy, set to cover all energies if not given
    ``delta_e``       energy spacing
    ``energyunit``    Energy unit, default 'eV'
    ``folding``       Gauss (default) or Lorentz
    ``width``         folding width in terms of the chosen energyunit
    ================= ===================================================
    all energies in [eV]
    """

    spectrum(exlist=excitation_list, filename=spectrum_file,
             emin=e_min, emax=e_max,
             de=delta_e, energyunit=energyunit,
             folding=folding, width=width,
             comment=comment)
