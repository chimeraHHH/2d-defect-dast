"""Omega matrix for functionals with Hartree-Fock exchange.

"""
from math import sqrt

from ase.units import Hartree
from ase.utils.timing import Timer
from ase.utils import IOContext
import numpy as np
from numpy.linalg import inv
from scipy.linalg import eigh

from gpaw import debug
import gpaw.mpi as mpi
from gpaw.lrtddft.omega_matrix import OmegaMatrix
from gpaw.pair_density import PairDensity
from gpaw.helmholtz import HelmholtzSolver
from gpaw.utilities.blas import mmm


class ApmB(OmegaMatrix):

    """Omega matrix for functionals with Hartree-Fock exchange.

    """

    def get_full(self):

        hybrid = ((self.xc is not None) and
                  hasattr(self.xc, 'hybrid') and
                  (self.xc.hybrid > 0.0))
        if self.fullkss.npspins < 2 and hybrid:
            raise RuntimeError('Does not work spin-unpolarized ' +
                               'with hybrids (use nspins=2)')

        if hasattr(self.xc, 'rsf') and (self.xc.rsf == 'Yukawa'):
            self.screened_poissonsolver = HelmholtzSolver(
                k2=-self.xc.omega**2, eps=1e-11, nn=3)
            self.screened_poissonsolver.set_grid_descriptor(self.gd)
        self.paw.timer.start('ApmB RPA')
        self.ApB = self.Om
        self.AmB = self.get_rpa()
        self.paw.timer.stop()

        if self.xc is not None:
            self.paw.timer.start('ApmB XC')
            self.get_xc()  # inherited from OmegaMatrix
            self.paw.timer.stop()

    def get_rpa(self):
        """Calculate RPA and Hartree-fock part of the A+-B matrices."""

        # shorthands
        kss = self.fullkss
        finegrid = self.finegrid
        yukawa = hasattr(self.xc, 'rsf') and (self.xc.rsf == 'Yukawa')

        # calculate omega matrix
        nij = len(kss)
        self.log('RPAhyb', nij, 'transitions')

        AmB = np.zeros((nij, nij))
        ApB = self.ApB

        # storage place for Coulomb integrals
        integrals = {}
        if yukawa:
            rsf_integrals = {}
        # setup things for IVOs
        if (hasattr(self.xc, 'excitation') and
           (self.xc.excitation is not None or self.xc.excited != 0)):
            sin_tri_weight = 1
            if self.xc.excitation is not None:
                ex_type = self.xc.excitation.lower()
                if ex_type == 'singlet':
                    sin_tri_weight = 2
                elif ex_type == 'triplet':
                    sin_tri_weight = 0
            homo = int(self.paw.get_number_of_electrons() // 2)
            ivo_l = homo - self.xc.excited - 1
        else:
            ivo_l = None

        for ij in range(nij):
            self.log('RPAhyb kss[' + '%d' % ij + ']=', kss[ij])

            timer = Timer()
            timer.start('init')
            timer2 = Timer()

            # smooth density including compensation charges
            timer2.start('with_compensation_charges 0')
            rhot_p = kss[ij].with_compensation_charges(
                finegrid != 0)
            timer2.stop()

            # integrate with 1/|r_1-r_2|
            timer2.start('poisson')
            phit_p = np.zeros(rhot_p.shape, rhot_p.dtype)
            self.poisson.solve(phit_p, rhot_p, charge=None)
            timer2.stop()

            timer.stop()
            t0 = timer.get_time('init')
            timer.start(ij)

            if finegrid == 1:
                rhot = kss[ij].with_compensation_charges()
                phit = self.gd.zeros()
                self.restrict(phit_p, phit)
            else:
                phit = phit_p
                rhot = rhot_p

            for kq in range(ij, nij):
                if kq != ij:
                    # smooth density including compensation charges
                    timer2.start('kq with_compensation_charges')
                    rhot = kss[kq].with_compensation_charges(
                        finegrid == 2)
                    timer2.stop()
                pre = self.weight_Kijkq(ij, kq)

                timer2.start('integrate')
                I = self.Coulomb_integral_kss(kss[ij], kss[kq], phit, rhot)
                if kss[ij].spin == kss[kq].spin:
                    name = self.Coulomb_integral_name(kss[ij].i, kss[ij].j,
                                                      kss[kq].i, kss[kq].j,
                                                      kss[ij].spin)
                    integrals[name] = I
                ApB[ij, kq] = pre * I
                timer2.stop()

                if ij == kq:
                    epsij = kss[ij].get_energy() / kss[ij].get_weight()
                    AmB[ij, kq] += epsij
                    ApB[ij, kq] += epsij

            timer.stop()
# timer2.write()
            if ij < (nij - 1):
                self.log('RPAhyb estimated time left',
                         self.time_left(timer, t0, ij, nij))

        # add HF parts and apply symmetry
        if hasattr(self.xc, 'hybrid'):
            weight = self.xc.hybrid
        else:
            weight = 0.0
        for ij in range(nij):
            self.log('HF kss[' + '%d' % ij + ']')
            timer = Timer()
            timer.start('init')
            timer.stop()
            t0 = timer.get_time('init')
            timer.start(ij)

            i = kss[ij].i
            j = kss[ij].j
            s = kss[ij].spin
            for kq in range(ij, nij):
                if kss[ij].pspin == kss[kq].pspin:
                    k = kss[kq].i
                    q = kss[kq].j
                    ikjq = self.Coulomb_integral_ijkq(i, k, j, q, s, integrals)
                    iqkj = self.Coulomb_integral_ijkq(i, q, k, j, s, integrals)
                    if yukawa:  # Yukawa integrals might be caches
                        ikjq -= self.Coulomb_integral_ijkq(
                            i, k, j, q, s, rsf_integrals, yukawa)
                        iqkj -= self.Coulomb_integral_ijkq(
                            i, q, k, j, s, rsf_integrals, yukawa)
                    ApB[ij, kq] -= weight * (ikjq + iqkj)
                    AmB[ij, kq] -= weight * (ikjq - iqkj)

                ApB[kq, ij] = ApB[ij, kq]
                AmB[kq, ij] = AmB[ij, kq]

            timer.stop()
            if ij < (nij - 1):
                self.log('HF estimated time left',
                         self.time_left(timer, t0, ij, nij))

        if ivo_l is not None:
            # IVO RPA after Berman, Kaldor, Chem. Phys. 43 (3) 1979
            # doi: 10.1016/0301-0104(79)85205-2
            l = ivo_l
            for ij in range(nij):
                i = kss[ij].i
                j = kss[ij].j
                s = kss[ij].spin
                for kq in range(ij, nij):
                    if kss[kq].i == i and kss[ij].pspin == kss[kq].pspin:
                        k = kss[kq].i
                        q = kss[kq].j
                        jqll = self.Coulomb_integral_ijkq(j, q, l, l, s,
                                                          integrals)
                        jllq = self.Coulomb_integral_ijkq(j, l, l, q, s,
                                                          integrals)
                        if yukawa:
                            jqll -= self.Coulomb_integral_ijkq(j, q, l, l, s,
                                                               rsf_integrals,
                                                               yukawa)
                            jllq -= self.Coulomb_integral_ijkq(j, l, l, q, s,
                                                               rsf_integrals,
                                                               yukawa)
                        jllq *= sin_tri_weight
                        ApB[ij, kq] += weight * (jqll - jllq)
                        AmB[ij, kq] += weight * (jqll - jllq)
                        ApB[kq, ij] = ApB[ij, kq]
                        AmB[kq, ij] = AmB[ij, kq]
        return AmB

    def Coulomb_integral_name(self, i, j, k, l, spin):
        """return a unique name considering the Coulomb integral
        symmetry"""
        def ij_name(i, j):
            return str(max(i, j)) + ' ' + str(min(i, j))

        # maximal gives the first
        if max(i, j) >= max(k, l):
            base = ij_name(i, j) + ' ' + ij_name(k, l)
        else:
            base = ij_name(k, l) + ' ' + ij_name(i, j)
        return base + ' ' + str(spin)

    def Coulomb_integral_ijkq(self, i, j, k, q, spin, integrals,
                              yukawa=False):
        name = self.Coulomb_integral_name(i, j, k, q, spin)
        if name in integrals:
            return integrals[name]

        # create the Kohn-Sham singles
        kss_ij = PairDensity(self.paw)
        kss_ij.initialize(self.paw.wfs.kpt_u[spin], i, j)
        kss_kq = PairDensity(self.paw)
        kss_kq.initialize(self.paw.wfs.kpt_u[spin], k, q)

        rhot_p = kss_ij.with_compensation_charges(
            self.finegrid != 0)
        phit_p = np.zeros(rhot_p.shape, rhot_p.dtype)
        if yukawa:
            self.screened_poissonsolver.solve(phit_p, rhot_p, charge=None)
        else:
            self.poisson.solve(phit_p, rhot_p, charge=None)

        if self.finegrid == 1:
            phit = self.gd.zeros()
            self.restrict(phit_p, phit)
        else:
            phit = phit_p

        rhot = kss_kq.with_compensation_charges(
            self.finegrid == 2)

        integrals[name] = self.Coulomb_integral_kss(kss_ij, kss_kq,
                                                    phit, rhot,
                                                    yukawa=yukawa)
        return integrals[name]

    def timestring(self, t):
        ti = int(t + .5)
        td = int(ti // 86400)
        st = ''
        if td > 0:
            st += '%d' % td + 'd'
            ti -= td * 86400
        th = int(ti // 3600)
        if th > 0:
            st += '%d' % th + 'h'
            ti -= th * 3600
        tm = int(ti // 60)
        if tm > 0:
            st += '%d' % tm + 'm'
            ti -= tm * 60
        st += '%d' % ti + 's'
        return st

    def mapAB(self, restrict={}):
        """Map A+B, A-B matrices according to constraints."""

        map, self.kss = self.get_map(restrict)
        if map is None:
            ApB = self.ApB.copy()
            AmB = self.AmB.copy()
        else:
            nij = len(self.kss)
            ApB = np.empty((nij, nij))
            AmB = np.empty((nij, nij))
            for ij in range(nij):
                for kq in range(nij):
                    ApB[ij, kq] = self.ApB[map[ij], map[kq]]
                    AmB[ij, kq] = self.AmB[map[ij], map[kq]]

        return ApB, AmB

    def diagonalize(self, restrict={}, TDA=False):
        """Evaluate Eigenvectors and Eigenvalues"""

        ApB, AmB = self.mapAB(restrict)
        nij = len(self.kss)

        if TDA:
            # Tamm-Dancoff approximation (B=0)
            eigenvalues, evecs = eigh(0.5 * (ApB + AmB))
            self.eigenvectors = evecs.T
            self.eigenvalues = eigenvalues ** 2
        else:
            # the occupation matrix
            C = np.empty((nij,))
            for ij in range(nij):
                C[ij] = 1. / self.kss[ij].fij

            S = C * inv(AmB) * C
            S = sqrt_matrix(inv(S).copy())

            # get Omega matrix
            M = np.zeros(ApB.shape)
            mmm(1.0, S, 'N', ApB, 'N', 0.0, M)
            self.eigenvectors = np.zeros(ApB.shape)
            mmm(1.0, M, 'N', S, 'N', 0.0, self.eigenvectors)

            self.eigenvalues, self.eigenvectors.T[:] = eigh(self.eigenvectors)

    def read(self, filename=None, fh=None):
        """Read myself from a file"""
        if mpi.rank == 0:
            with IOContext() as io:
                if fh is None:
                    fd = io.openfile(filename, 'r', comm=self.paw.world)
                else:
                    fd = fh
                fd.readline()
                nij = int(fd.readline())
                ApB = np.zeros((nij, nij))
                for ij in range(nij):
                    l = fd.readline().split()
                    for kq in range(ij, nij):
                        ApB[ij, kq] = float(l[kq - ij])
                        ApB[kq, ij] = ApB[ij, kq]
                self.ApB = ApB

                fd.readline()
                nij = int(fd.readline())
                AmB = np.zeros((nij, nij))
                for ij in range(nij):
                    l = fd.readline().split()
                    for kq in range(ij, nij):
                        AmB[ij, kq] = float(l[kq - ij])
                        AmB[kq, ij] = AmB[ij, kq]
                self.AmB = AmB

    def weight_Kijkq(self, ij, kq):
        """weight for the coupling matrix terms"""
        return 2.

    def write(self, filename=None, fh=None):
        """Write current state to a file."""
        if mpi.rank == 0:
            with IOContext() as io:
                if fh is None:
                    fd = io.openfile(filename, 'r', comm=self.paw.world)
                else:
                    fd = fh
                fd.write('# A+B\n')
                nij = len(self.fullkss)
                fd.write('%d\n' % nij)
                for ij in range(nij):
                    for kq in range(ij, nij):
                        fd.write(' %g' % self.ApB[ij, kq])
                    fd.write('\n')

                fd.write('# A-B\n')
                nij = len(self.fullkss)
                fd.write('%d\n' % nij)
                for ij in range(nij):
                    for kq in range(ij, nij):
                        fd.write(' %g' % self.AmB[ij, kq])
                    fd.write('\n')

    def __str__(self):
        string = '<ApmB> '
        if hasattr(self, 'eigenvalues'):
            string += 'dimension ' + ('%d' % len(self.eigenvalues))
            string += '\neigenvalues: '
            for ev in self.eigenvalues:
                string += ' ' + ('%f' % (sqrt(ev) * Hartree))
        return string


def sqrt_matrix(a, preserve=False):
    """Get the sqrt of a symmetric matrix a (diagonalize is used).
    The matrix is kept if preserve=True, a=sqrt(a) otherwise."""
    n = len(a)
    if debug:
        assert a.flags.contiguous
        assert a.dtype == float
        assert a.shape == (n, n)
    if preserve:
        b = a.copy()
    else:
        b = a

    # diagonalize to get the form b = Z * D * Z^T
    # where D is diagonal
    D = np.empty((n,))
    D, b.T[:] = eigh(b, lower=True)
    ZT = b.copy()
    Z = np.transpose(b)

    # c = Z * sqrt(D)
    c = Z * np.sqrt(D)

    # sqrt(b) = c * Z^T
    mmm(1.0, np.ascontiguousarray(c), 'N', ZT, 'N', 0.0, b)
    return b
