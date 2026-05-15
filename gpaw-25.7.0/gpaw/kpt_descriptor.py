# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""K-point descriptor."""

from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
from ase.calculators.calculator import kptdensity2monkhorstpack
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset, monkhorst_pack

import gpaw.cgpaw as cgpaw
import gpaw.mpi as mpi
from gpaw import KPointError
from gpaw.typing import Array1D
from gpaw.kpoint import KPoint


def to1bz(bzk_kc, cell_cv):
    """Wrap k-points to 1. BZ.

    Return k-points wrapped to the 1. BZ.

    bzk_kc: (n,3) ndarray
        Array of k-points in units of the reciprocal lattice vectors.
    cell_cv: (3,3) ndarray
        Unit cell.
    """

    B_cv = 2.0 * np.pi * np.linalg.inv(cell_cv).T
    K_kv = np.dot(bzk_kc, B_cv)
    N_xc = np.indices((3, 3, 3)).reshape((3, 27)).T - 1
    G_xv = np.dot(N_xc, B_cv)

    bz1k_kc = bzk_kc.copy()

    # Find the closest reciprocal lattice vector:
    for k, K_v in enumerate(K_kv):
        # If a k-point has the same distance to several reciprocal
        # lattice vectors, we don't want to pick a random one on the
        # basis of numerical noise, so we round off the differences
        # between the shortest distances to 6 decimals and chose the
        # one with the lowest index.
        d = ((G_xv - K_v)**2).sum(1)
        x = (d - d.min()).round(6).argmin()
        bz1k_kc[k] -= N_xc[x]

    return bz1k_kc


def kpts2sizeandoffsets(size=None, density=None, gamma=None, even=None,
                        atoms=None):
    """Helper function for selecting k-points.

    Use either size or density.

    size: 3 ints
        Number of k-points.
    density: float
        K-point density in units of k-points per Ang^-1.
    gamma: None or bool
        Should the Gamma-point be included?  Yes / no / don't care:
        True / False / None.
    even: None or bool
        Should the number of k-points be even?  Yes / no / don't care:
        True / False / None.
    atoms: Atoms object
        Needed for calculating k-point density.

    """

    if size is None:
        if density is None:
            size = [1, 1, 1]
        else:
            size = kptdensity2monkhorstpack(atoms, density, even)

    offsets = [0, 0, 0]

    if gamma is not None:
        for i, s in enumerate(size):
            if atoms.pbc[i] and s % 2 != bool(gamma):
                offsets[i] = 0.5 / s

    return size, offsets


class KPointDescriptor:
    """Descriptor-class for k-points."""

    def __init__(self, kpts, nspins: int = 1):
        """Construct descriptor object for kpoint/spin combinations (ks-pair).

        Parameters:

        kpts: None, sequence of 3 ints, or (n,3)-shaped array
            Specification of the k-point grid. None=Gamma, list of
            ints=Monkhorst-Pack, ndarray=user specified.
        nspins: int
            Number of spins.

        Attributes
        ===================  =================================================
        ``N_c``               Number of k-points in the different directions.
        ``nspins``            Number of spins in total.
        ``mynspins``          Number of spins on this CPU.
        ``nibzkpts``          Number of irreducible kpoints in 1st BZ.
        ``mynks``             Number of k-point/spin combinations on this CPU.
        ``gamma``             Boolean indicator for gamma point calculation.
        ``comm``              MPI-communicator for kpoint distribution.
        ``weight_k``          Weights of each k-point
        ``ibzk_kc``           Unknown
        ``ibzk_qc``           Unknown
        ``sym_k``             Unknown
        ``time_reversal_k``   Unknown
        ``bz2ibz_k``          Unknown
        ``ibz2bz_k``          Unknown
        ``bz2bz_ks``          Unknown
        ``symmetry``          Object representing symmetries
        ===================  =================================================
        """

        self.N_c: Optional[Array1D] = None
        self.offset_c: Optional[Array1D] = None

        if kpts is None:
            self.bzk_kc = np.zeros((1, 3))
            self.N_c = np.array((1, 1, 1), dtype=int)
            self.offset_c = np.zeros(3)
        else:
            kpts = np.asarray(kpts)
            if kpts.ndim == 1:
                self.N_c = np.array(kpts, dtype=int)
                self.bzk_kc = monkhorst_pack(self.N_c)
                self.offset_c = np.zeros(3)
            else:
                self.bzk_kc = np.array(kpts, dtype=float)
                try:
                    self.N_c, self.offset_c = \
                        get_monkhorst_pack_size_and_offset(self.bzk_kc)
                except ValueError:
                    pass
        self.nspins = nspins
        self.nbzkpts = len(self.bzk_kc)

        # Gamma-point calculation?
        self.gamma = self.nbzkpts == 1 and not self.bzk_kc.any()

        # Point group and time-reversal symmetry neglected:
        self.weight_k = np.ones(self.nbzkpts) / self.nbzkpts
        self.ibzk_kc = self.bzk_kc.copy()
        self.sym_k = np.zeros(self.nbzkpts, int)
        self.time_reversal_k = np.zeros(self.nbzkpts, bool)
        self.bz2ibz_k = np.arange(self.nbzkpts)
        self.ibz2bz_k = np.arange(self.nbzkpts)
        self.bz2bz_ks = np.arange(self.nbzkpts)[:, np.newaxis]
        self.nibzkpts = self.nbzkpts
        self.refine_info = None
        self.monkhorst = (self.N_c is not None)

        self.set_communicator(mpi.serial_comm)

    def __str__(self):
        s = str(self.symmetry)

        if self.refine_info is not None:
            s += '\n' + str(self.refine_info)

        if -1 in self.bz2bz_ks:
            s += 'Note: your k-points are not as symmetric as your crystal!\n'

        if self.gamma:
            s += '\n1 k-point (Gamma)'
        else:
            s += '\n%d k-points' % self.nbzkpts
            if self.monkhorst:
                s += ': %d x %d x %d Monkhorst-Pack grid' % tuple(self.N_c)
                if self.offset_c.any():
                    s += ' + ['
                    for x in self.offset_c:
                        if x != 0 and abs(round(1 / x) - 1 / x) < 1e-12:
                            s += '1/%d,' % round(1 / x)
                        else:
                            s += '%f,' % x
                    s = s[:-1] + ']'

        s += ('\n%d k-point%s in the irreducible part of the Brillouin zone\n'
              % (self.nibzkpts, ' s'[1:self.nibzkpts]))

        if self.monkhorst:
            w_k = self.weight_k * self.nbzkpts
            assert np.allclose(w_k, w_k.round())
            w_k = w_k.round()

        s += '       k-points in crystal coordinates                weights\n'
        for k in range(self.nibzkpts):
            if k < 10 or k == self.nibzkpts - 1:
                if self.monkhorst:
                    s += ('%4d:   %12.8f  %12.8f  %12.8f     %6d/%d\n' %
                          ((k,) + tuple(self.ibzk_kc[k]) +
                           (w_k[k], self.nbzkpts)))
                else:
                    s += ('%4d:   %12.8f  %12.8f  %12.8f     %12.8f\n' %
                          ((k,) + tuple(self.ibzk_kc[k]) +
                           (self.weight_k[k],)))
            elif k == 10:
                s += '          ...\n'
        return s

    def set_symmetry(self, atoms, symmetry, comm=None):
        """Create symmetry object and construct irreducible Brillouin zone.

        atoms: Atoms object
            Defines atom positions and types and also unit cell and
            boundary conditions.
        symmetry: Symmetry object
            Symmetry object.
        """

        self.symmetry = symmetry

        # XXX we pass the whole atoms object just to complain if its PBCs
        # are not how we like them
        for c, periodic in enumerate(atoms.pbc):
            if not periodic and not np.allclose(self.bzk_kc[:, c], 0.0):
                raise ValueError('K-points can only be used with PBCs!')

        if symmetry.time_reversal or symmetry.point_group:
            (self.ibzk_kc, self.weight_k,
             self.sym_k,
             self.time_reversal_k,
             self.bz2ibz_k,
             self.ibz2bz_k,
             self.bz2bz_ks) = symmetry.reduce(self.bzk_kc, comm)

        # Number of irreducible k-points and k-point/spin combinations.
        self.nibzkpts = len(self.ibzk_kc)

    def set_communicator(self, comm):
        """Set k-point communicator."""

        # Ranks < self.rank0 have mynks0 k-point/spin combinations and
        # ranks >= self.rank0 have mynks0+1 k-point/spin combinations.
        mynk0, x = divmod(self.nibzkpts, comm.size)
        self.rank0 = comm.size - x
        self.comm = comm

        # My number and offset of k-point/spin combinations
        self.mynk = self.get_count()
        self.k0 = self.get_offset()

        self.ibzk_qc = self.ibzk_kc[self.k0:self.k0 + self.mynk]
        self.weight_q = self.weight_k[self.k0:self.k0 + self.mynk]

    def copy(self, comm=mpi.serial_comm):
        """Create a copy with shared symmetry object."""
        kd = KPointDescriptor(self.bzk_kc, self.nspins)
        kd.weight_k = self.weight_k
        kd.ibzk_kc = self.ibzk_kc
        kd.sym_k = self.sym_k
        kd.time_reversal_k = self.time_reversal_k
        kd.bz2ibz_k = self.bz2ibz_k
        kd.ibz2bz_k = self.ibz2bz_k
        kd.bz2bz_ks = self.bz2bz_ks
        kd.symmetry = self.symmetry
        kd.nibzkpts = self.nibzkpts
        kd.set_communicator(comm)
        return kd

    def create_k_points(self, sdisp_cd, collinear):
        """Return a list of KPoints."""

        kpt_qs = []

        for k in range(self.k0, self.k0 + self.mynk):
            q = k - self.k0
            weightk = self.weight_k[k]
            weight = weightk * 2 / self.nspins
            if self.gamma:
                phase_cd = np.ones((3, 2), complex)
            else:
                phase_cd = np.exp(2j * np.pi *
                                  sdisp_cd * self.ibzk_kc[k, :, np.newaxis])
            if collinear:
                spins = range(self.nspins)
            else:
                spins = [None]
                weight *= 0.5
            kpt_qs.append([KPoint(weightk, weight, s, k, q, phase_cd)
                           for s in spins])

        return kpt_qs

    def collect(self, a_ux, broadcast: bool):
        """Collect distributed data to all."""

        xshape = a_ux.shape[1:]
        a_qsx = a_ux.reshape((-1, self.nspins) + xshape)
        if self.comm.rank == 0 or broadcast:
            a_ksx = np.empty((self.nibzkpts, self.nspins) + xshape, a_ux.dtype)

        if self.comm.rank > 0:
            self.comm.send(a_qsx, 0)
        else:
            k1 = self.get_count(0)
            a_ksx[0:k1] = a_qsx
            requests = []
            for rank in range(1, self.comm.size):
                k2 = k1 + self.get_count(rank)
                requests.append(self.comm.receive(a_ksx[k1:k2], rank,
                                                  block=False))
                k1 = k2
            assert k1 == self.nibzkpts
            self.comm.waitall(requests)

        if broadcast:
            self.comm.broadcast(a_ksx, 0)

        if self.comm.rank == 0 or broadcast:
            return a_ksx.transpose((1, 0, 2))

    def transform_wave_function(self, psit_G, k, index_G=None, phase_G=None):
        """Transform wave function from IBZ to BZ.

        k is the index of the desired k-point in the full BZ.
        """

        s = self.sym_k[k]
        time_reversal = self.time_reversal_k[k]
        op_cc = np.linalg.inv(self.symmetry.op_scc[s]).round().astype(int)

        # Identity
        if (np.abs(op_cc - np.eye(3, dtype=int)) < 1e-10).all():
            if time_reversal:
                return psit_G.conj()
            else:
                return psit_G
        # General point group symmetry
        else:
            ik = self.bz2ibz_k[k]
            kibz_c = self.ibzk_kc[ik]
            b_g = np.zeros_like(psit_G)
            kbz_c = np.dot(self.symmetry.op_scc[s], kibz_c)
            if index_G is not None:
                assert index_G.shape == psit_G.shape == phase_G.shape
                cgpaw.symmetrize_with_index(psit_G, b_g, index_G, phase_G)
            else:
                cgpaw.symmetrize_wavefunction(psit_G, b_g, op_cc.copy(),
                                              np.ascontiguousarray(kibz_c),
                                              kbz_c)

            if time_reversal:
                return b_g.conj()
            else:
                return b_g

    def get_transform_wavefunction_index(self, nG, k):
        """Get the "wavefunction transform index".

        This is a permutation of the numbers 1, 2, .. N which
        associates k + q to some k, and where N is the total
        number of grid points as specified by nG which is a
        3D tuple.

        Returns index_G and phase_G which are one-dimensional
        arrays on the grid."""

        s = self.sym_k[k]
        op_cc = np.linalg.inv(self.symmetry.op_scc[s]).round().astype(int)

        # General point group symmetry
        if (np.abs(op_cc - np.eye(3, dtype=int)) < 1e-10).all():
            nG0 = np.prod(nG)
            index_G = np.arange(nG0).reshape(nG)
            phase_G = np.ones(nG)
        else:
            ik = self.bz2ibz_k[k]
            kibz_c = self.ibzk_kc[ik]
            index_G = np.zeros(nG, dtype=int)
            phase_G = np.zeros(nG, dtype=complex)

            kbz_c = np.dot(self.symmetry.op_scc[s], kibz_c)
            cgpaw.symmetrize_return_index(index_G, phase_G, op_cc.copy(),
                                          np.ascontiguousarray(kibz_c),
                                          kbz_c)
        return index_G, phase_G

    def find_k_plus_q(self, q_c, kpts_k: Sequence[int] = None) -> list[int]:
        """Find the indices of k+q for all kpoints in the Brillouin zone.

        In case that k+q is outside the BZ, the k-point inside the BZ
        corresponding to k+q is given.

        Parameters
        ----------
        q_c: np.ndarray
            Coordinates for the q-vector in units of the reciprocal
            lattice vectors.
        kpts_k:
            Restrict search to specified k-points.

        """
        k_x = kpts_k
        if k_x is None:
            return self.find_k_plus_q(q_c, range(self.nbzkpts))

        i_x = []
        for k in k_x:
            kpt_c = self.bzk_kc[k] + q_c
            d_kc = kpt_c - self.bzk_kc
            d_k = abs(d_kc - d_kc.round()).sum(1)
            i = d_k.argmin()
            if d_k[i] > 1e-8:
                raise KPointError('Could not find k+q!')
            i_x.append(i)

        return i_x

    def get_bz_q_points(self, first=False):
        """Return the q=k1-k2. q-mesh is always Gamma-centered."""
        shift_c = 0.5 * ((self.N_c + 1) % 2) / self.N_c
        bzq_qc = monkhorst_pack(self.N_c) + shift_c
        if first:
            return to1bz(bzq_qc, self.symmetry.cell_cv)
        else:
            return bzq_qc

    def get_ibz_q_points(self, bzq_qc, op_scc):
        """Return ibz q points and the corresponding symmetry operations that
        work for k-mesh as well."""

        ibzq_qc_tmp = []
        ibzq_qc_tmp.append(bzq_qc[-1])
        weight_tmp = [0]

        for i, op_cc in enumerate(op_scc):
            if np.abs(op_cc - np.eye(3)).sum() < 1e-8:
                identity_iop = i
                break

        ibzq_q_tmp = {}
        iop_q = {}
        timerev_q = {}
        diff_qc = {}

        for i in range(len(bzq_qc) - 1, -1, -1):  # loop opposite to kpoint
            try:
                ibzk, iop, timerev, diff_c = self.find_ibzkpt(
                    op_scc, ibzq_qc_tmp, bzq_qc[i])
                find = False
                for ii, iop1 in enumerate(self.sym_k):
                    if iop1 == iop and self.time_reversal_k[ii] == timerev:
                        find = True
                        break
                if not find:
                    raise ValueError('cant find k!')

                ibzq_q_tmp[i] = ibzk
                weight_tmp[ibzk] += 1.
                iop_q[i] = iop
                timerev_q[i] = timerev
                diff_qc[i] = diff_c
            except ValueError:
                ibzq_qc_tmp.append(bzq_qc[i])
                weight_tmp.append(1.)
                ibzq_q_tmp[i] = len(ibzq_qc_tmp) - 1
                iop_q[i] = identity_iop
                timerev_q[i] = False
                diff_qc[i] = np.zeros(3)

        # reverse the order.
        nq = len(ibzq_qc_tmp)
        ibzq_qc = np.zeros((nq, 3))
        ibzq_q = np.zeros(len(bzq_qc), dtype=int)
        for i in range(nq):
            ibzq_qc[i] = ibzq_qc_tmp[nq - i - 1]
        for i in range(len(bzq_qc)):
            ibzq_q[i] = nq - ibzq_q_tmp[i] - 1
        self.q_weights = np.array(weight_tmp[::-1]) / len(bzq_qc)
        return ibzq_qc, ibzq_q, iop_q, timerev_q, diff_qc

    def find_ibzkpt(self, symrel, ibzk_kc, bzk_c):
        """Find index in IBZ and related symmetry operations."""
        find = False
        ibzkpt = 0
        iop = 0
        timerev = False

        for sign in (1, -1):
            for ioptmp, op in enumerate(symrel):
                for i, ibzk in enumerate(ibzk_kc):
                    diff_c = bzk_c - sign * np.dot(op, ibzk)
                    if (np.abs(diff_c - diff_c.round()) < 1e-8).all():
                        ibzkpt = i
                        iop = ioptmp
                        find = True
                        if sign == -1:
                            timerev = True
                        break
                if find:
                    break
            if find:
                break

        if not find:
            raise ValueError('Cant find corresponding IBZ kpoint!')
        return ibzkpt, iop, timerev, diff_c.round()

    def where_is_q(self, q_c, bzq_qc):
        """Find the index of q points in BZ."""
        d_qc = q_c - bzq_qc
        d_q = abs(d_qc - d_qc.round()).sum(1)
        q = d_q.argmin()
        if d_q[q] > 1e-8:
            raise KPointError('Could not find q!')
        return q

    def get_count(self, rank=None):
        """Return the number of ks-pairs which belong to a given rank."""

        if rank is None:
            rank = self.comm.rank
        assert rank in range(self.comm.size)
        mynk0 = self.nibzkpts // self.comm.size
        mynk = mynk0
        if rank >= self.rank0:
            mynk += 1
        return mynk

    def get_offset(self, rank=None):
        """Return the offset of the first ks-pair on a given rank."""

        if rank is None:
            rank = self.comm.rank
        assert rank in range(self.comm.size)
        mynk0 = self.nibzkpts // self.comm.size
        k0 = rank * mynk0
        if rank >= self.rank0:
            k0 += rank - self.rank0
        return k0

    def get_rank_and_index(self, k):
        """Find rank and local index of k-point/spin combination."""

        rank, q = self.who_has(k)
        return rank, q

    def get_indices(self, rank=None):
        """Return the global ks-pair indices which belong to a given rank."""

        k1 = self.get_offset(rank)
        k2 = k1 + self.get_count(rank)
        return np.arange(k1, k2)

    def who_has(self, k):
        """Convert global index to rank information and local index."""

        mynk0 = self.nibzkpts // self.comm.size
        if k < mynk0 * self.rank0:
            rank, q = divmod(k, mynk0)
        else:
            rank, q = divmod(k - mynk0 * self.rank0, mynk0 + 1)
            rank += self.rank0
        return rank, q

    def write(self, writer):
        writer.write('ibzkpts', self.ibzk_kc)
        writer.write('bzkpts', self.bzk_kc)
        writer.write('bz2ibz', self.bz2ibz_k)
        writer.write('weights', self.weight_k)
        writer.write('rotations', self.symmetry.op_scc)
        writer.write('translations', self.symmetry.ft_sc)
        writer.write('atommap', self.symmetry.a_sa)
