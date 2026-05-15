from math import pi

import numpy as np
from ase.units import Bohr

import gpaw.cgpaw as cgpaw
from gpaw import debug
from gpaw.grid_descriptor import GridDescriptor, GridBoundsError
from gpaw.gpu import cupy_is_fake
from gpaw.new import trace
from gpaw.utilities import smallest_safe_grid_spacing

"""

===  =================================================
 M   Global localized function number.
 W   Global volume number.
 G   Global grid point number.
 g   Local (inside sphere) grid point number.
 i   Index into list of current spheres for current G.
===  =================================================

l
m
b
w

Global grid point number (*G*) for a 7*6 grid::
   -------------
  |5 . . . . . .|
  |4 . . . . . .|
  |3 9 . . . . .|
  |2 8 . . . . .|
  |1 7 . . . . .|
  |0 6 . . . . .|
   -------------

For this example *G* runs from 0 to 41.

Here is a sphere inside the box with grid points (*g*) numbered from 0
to 7::

   -------------
  |. . . . . . .|
  |. . . . 5 . .|
  |. . . 1 4 7 .|
  |. . . 0 3 6 .|
  |. . . . 2 . .|
  |. . . . . . .|
   -------------

~  _  ^  ~  ~
p  v  g  n  F
 i     L  c  M

i
d  d  d  d  d
s     s
   s     s
"""


class Sphere:
    def __init__(self, spline_j):
        self.spline_j = spline_j
        self.spos_c = None
        self.rank = None  # Rank corresponding to center
        self.ranks = None  # Ranks with at least some overlap
        self.Mmax = None
        self.A_wgm = None
        self.G_wb = None
        self.M_w = None
        self.sdisp_wc = None
        self.normalized = False
        self.I_M = None

    def set_position(self, spos_c, gd, cut):
        if self.spos_c is not None and not (self.spos_c - spos_c).any():
            return False

        self.A_wgm = []
        self.G_wb = []
        self.M_w = []
        self.sdisp_wc = []

        ng = 0
        M = 0
        for spline in self.spline_j:
            rcut = spline.get_cutoff()
            l = spline.get_angular_momentum_number()
            for beg_c, end_c, sdisp_c in gd.get_boxes(spos_c, rcut, cut):
                pos_v = np.dot(spos_c - sdisp_c, gd.cell_cv)
                A_gm, G_b = cgpaw.spline_to_grid(
                    spline.spline,
                    beg_c, end_c, pos_v,
                    np.ascontiguousarray(gd.h_cv),
                    gd.n_c, gd.beg_c)
                if len(G_b) > 0:
                    self.A_wgm.append(A_gm)
                    self.G_wb.append(G_b)
                    self.M_w.append(M)
                    self.sdisp_wc.append(sdisp_c)
                    ng += A_gm.shape[0]
                    assert A_gm.shape[0] > 0
            M += 2 * l + 1

        self.Mmax = M

        self.rank = gd.get_rank_from_position(spos_c)
        if ng == 0:
            self.ranks = None  # What about making empty lists instead?
            self.A_wgm = None
            self.G_wb = None
            self.M_w = None
            self.sdisp_wc = None

        self.spos_c = spos_c.copy()
        self.normalized = False
        return True

    def get_function_count(self):
        return sum([2 * spline.get_angular_momentum_number() + 1
                    for spline in self.spline_j])

    def normalize(self, integral, a, dv, comm):
        """Normalize localized functions."""
        if self.normalized or integral < 1e-15:
            self.normalized = True
            yield None
            yield None
            yield None
            return

        I_M = np.zeros(self.Mmax)

        nw = len(self.A_wgm) // len(self.spline_j)
        assert nw * len(self.spline_j) == len(self.A_wgm)

        for M, A_gm in zip(self.M_w, self.A_wgm):
            I_m = A_gm.sum(axis=0)
            I_M[M:M + len(I_m)] += I_m * dv

        requests = []
        if len(self.ranks) > 0:
            I_rM = np.empty((len(self.ranks), self.Mmax))
            for r, J_M in zip(self.ranks, I_rM):
                requests.append(comm.receive(J_M, r, a, False))
        if self.rank != comm.rank:
            requests.append(comm.send(I_M, self.rank, a, False))

        yield None

        for request in requests:
            comm.wait(request)

        requests = []
        if len(self.ranks) > 0:
            I_M += I_rM.sum(axis=0)
            for r in self.ranks:
                requests.append(comm.send(I_M, r, a, False))
        if self.rank != comm.rank:
            requests.append(comm.receive(I_M, self.rank, a, False))

        yield None

        for request in requests:
            comm.wait(request)

        w = 0
        for M, A_gm in zip(self.M_w, self.A_wgm):
            if M == 0:
                A_gm *= integral / I_M[0]
            else:
                A_gm -= (I_M[M:M + A_gm.shape[1]] / integral *
                         self.A_wgm[w % nw])
            w += 1
        self.normalized = True
        self.I_M = I_M
        yield None

    def estimate_gridpointcount(self, gd):
        points = 0.0
        for spline in self.spline_j:
            l = spline.get_angular_momentum_number()
            rc = spline.get_cutoff()
            points += 4.0 * np.pi / 3.0 * rc**3 / gd.dv * (2 * l + 1)
        return points


# Quick hack: base class to share basic functionality across LFC classes
class BaseLFC:
    def dict(self, shape=(), derivative=False, zero=False):
        if isinstance(shape, int):
            shape = (shape,)
        if derivative:
            assert not zero
            c_axiv = {}
            for a in self.my_atom_indices:
                ni = self.get_function_count(a)
                c_axiv[a] = np.empty(shape + (ni, 3), self.dtype)
            return c_axiv
        else:
            c_axi = {}
            for a in self.my_atom_indices:
                ni = self.get_function_count(a)
                c_axi[a] = np.empty(shape + (ni,), self.dtype)
                if zero:
                    c_axi[a][:] = 0.0
            return c_axi

    def estimate_memory(self, mem):
        points = 0
        for sphere in self.sphere_a:
            points += sphere.estimate_gridpointcount(self.gd)
        nbytes = points * mem.floatsize
        mem.setsize(nbytes / self.gd.comm.size)  # Assume equal distribution


class LocalizedFunctionsCollection(BaseLFC):
    """LocalizedFunctionsCollection

    Utilizes that localized functions can be stored on a spherical subset of
    the uniform grid, as opposed to LocalizedFunctionsCollection which is just
    a wrapper around the old localized_functions which use rectangular grids.

    """
    def __init__(self, gd, spline_aj, kd=None, cut=False, dtype=float,
                 integral=None, forces=None, xp=np):
        self.gd = gd
        self.kd = kd
        self.sphere_a = [Sphere(spline_j) for spline_j in spline_aj]
        self.cut = cut
        self.dtype = dtype
        self.Mmax = None
        self.xp = xp

        if kd is None:
            self.ibzk_qc = np.zeros((1, 3))
            self.gamma = True
        else:
            self.ibzk_qc = kd.ibzk_qc
            self.gamma = kd.gamma

        # Global or local M-indices?
        self.use_global_indices = False

        if integral is not None:
            if isinstance(integral, (float, int)):
                self.integral_a = np.empty(len(spline_aj))
                self.integral_a[:] = integral
            else:
                self.integral_a = np.array(integral)
        else:
            self.integral_a = None

        self.my_atom_indices = None
        self.lfc = None

    def set_positions(self, spos_ac, atom_partition=None) -> bool:
        """Set positions and return True if any atoms have migrated to
        another rank.
        """
        assert len(spos_ac) == len(self.sphere_a)
        spos_ac = np.asarray(spos_ac)
        movement = False
        old_ranks = [sphere.rank for sphere in self.sphere_a]
        for a, (spos_c, sphere) in enumerate(zip(spos_ac, self.sphere_a)):
            try:
                movement |= sphere.set_position(spos_c, self.gd, self.cut)
            except GridBoundsError as e:
                e.args = (f'Atom {a} too close to edge: {e}',)
                raise

        if self.my_atom_indices is None:
            self._update(spos_ac)
            return False

        if movement:
            self._update(spos_ac)
            for rank, sphere in zip(old_ranks, self.sphere_a):
                if rank != sphere.rank:
                    return True
        return False

    def _update(self, spos_ac):
        nB = 0
        nW = 0
        self.my_atom_indices = []
        self.atom_indices = []
        M = 0
        self.M_a = []
        for a, sphere in enumerate(self.sphere_a):
            self.M_a.append(M)
            if sphere.rank == self.gd.comm.rank:
                self.my_atom_indices.append(a)
            G_wb = sphere.G_wb
            if G_wb:
                nB += sum([len(G_b) for G_b in G_wb])
                nW += len(G_wb)
                self.atom_indices.append(a)

                if not self.use_global_indices:
                    M += sphere.Mmax

            if self.use_global_indices:
                M += sphere.Mmax

        self.Mmax = M

        natoms = len(spos_ac)
        # Holm-Nielsen check:
        if (self.gd.comm.sum_scalar(float(sum(self.my_atom_indices))) !=
            natoms * (natoms - 1) // 2):
            raise ValueError('Holm-Nielsen check failed.  Grid might be '
                             'too coarse.  Use h < %.3f'
                             % (smallest_safe_grid_spacing * Bohr))

        self.M_W = np.empty(nW, np.intc)
        self.G_B = np.empty(nB, np.intc)
        self.W_B = np.empty(nB, np.intc)
        self.A_Wgm = []
        sdisp_Wc = np.empty((nW, 3), int)
        self.pos_Wv = np.empty((nW, 3))

        B1 = 0
        W = 0
        for a in self.atom_indices:
            sphere = self.sphere_a[a]
            self.A_Wgm.extend(sphere.A_wgm)
            nw = len(sphere.M_w)
            self.M_W[W:W + nw] = self.M_a[a] + np.array(sphere.M_w)
            sdisp_Wc[W:W + nw] = sphere.sdisp_wc
            self.pos_Wv[W:W + nw] = np.dot(spos_ac[a] -
                                           np.array(sphere.sdisp_wc),
                                           self.gd.cell_cv)
            for G_b in sphere.G_wb:
                B2 = B1 + len(G_b)
                self.G_B[B1:B2] = G_b
                self.W_B[B1:B2:2] = W
                self.W_B[B1 + 1:B2 + 1:2] = -W - 1
                B1 = B2
                W += 1
        assert B1 == nB

        if self.gamma:
            if self.dtype == float:
                self.phase_qW = np.empty((0, nW), complex)
            else:
                # TDDFT calculation:
                self.phase_qW = np.ones((1, nW), complex)
        else:
            self.phase_qW = np.exp(2j * pi * np.inner(self.ibzk_qc, sdisp_Wc))

        indices = np.argsort(self.G_B, kind='mergesort')
        self.G_B = self.G_B[indices]
        self.W_B = self.W_B[indices]

        # Find out which ranks have a piece of the
        # localized functions:
        x_a = np.zeros(natoms, bool)
        x_a[self.atom_indices] = True
        x_a[self.my_atom_indices] = False
        x_ra = np.empty((self.gd.comm.size, natoms), bool)
        self.gd.comm.all_gather(x_a, x_ra)
        for a in self.atom_indices:
            sphere = self.sphere_a[a]
            if sphere.rank == self.gd.comm.rank:
                sphere.ranks = x_ra[:, a].nonzero()[0]
            else:
                sphere.ranks = []

        if self.integral_a is not None:
            iterators = []
            for a in self.atom_indices:
                iterator = self.sphere_a[a].normalize(self.integral_a[a], a,
                                                      self.gd.dv,
                                                      self.gd.comm)
                iterators.append(iterator)
            for i in range(3):
                for iterator in iterators:
                    next(iterator)

        self.lfc = cgpaw.LFC(self.A_Wgm, self.M_W, self.G_B, self.W_B,
                             self.gd.dv, self.phase_qW, self.xp is not np)

        return sdisp_Wc

    def M_to_ai(self, src_xM, dst_axi):
        xshape = src_xM.shape[:-1]
        src_xM = src_xM.reshape(np.prod(xshape), self.Mmax)
        for a in self.my_atom_indices:
            M1 = self.M_a[a]
            M2 = M1 + self.sphere_a[a].Mmax
            dst_axi[a] = src_xM[:, M1:M2].copy()

    def ai_to_M(self, src_axi, dst_xM):
        xshape = dst_xM.shape[:-1]
        dst_xM = dst_xM.reshape(np.prod(xshape), self.Mmax)
        for a in self.my_atom_indices:
            M1 = self.M_a[a]
            M2 = M1 + self.sphere_a[a].Mmax
            dst_xM[:, M1:M2] = src_axi[a]

    def add(self, a_xG, c_axi=1.0, q=-1):
        """Add localized functions to extended arrays.

        ::

                   --  a     a
          a (G) += >  c   Phi (G)
           x       --  xi    i
                   a,i
        """

        assert not self.use_global_indices
        if q == -1:
            assert self.dtype == float

        if isinstance(c_axi, float):
            assert q == -1 and a_xG.ndim == 3
            c_xM = self.xp.empty(self.Mmax)
            c_xM.fill(c_axi)
            if self.xp is np:
                self.lfc.add(c_xM, a_xG, q)
            elif cupy_is_fake:
                self.lfc.add(c_xM._data, a_xG._data, q)
            else:
                self.lfc.add_gpu(c_xM.data.ptr,
                                 c_xM.shape,
                                 a_xG.data.ptr,
                                 a_xG.shape, q)
            return

        dtype = a_xG.dtype

        if debug:
            assert a_xG.ndim >= 3
            assert sorted(c_axi.keys()) == self.my_atom_indices
            for c_xi in c_axi.values():
                assert c_xi.dtype == dtype

        comm = self.gd.comm
        xshape = a_xG.shape[:-3]
        assert len(xshape) <= 1
        requests = []
        M1 = 0
        b_axi = {}
        for a in self.atom_indices:
            c_xi = c_axi.get(a)
            sphere = self.sphere_a[a]
            M2 = M1 + sphere.Mmax
            if c_xi is None:
                c_xi = self.xp.empty(xshape + (sphere.Mmax,), dtype)
                b_axi[a] = c_xi
                requests.append(comm.receive(c_xi, sphere.rank, a, False))
            else:
                for r in sphere.ranks:
                    requests.append(comm.send(c_xi.copy(), r, a, False))

            M1 = M2

        for request in requests:
            comm.wait(request)

        c_xM = self.xp.empty(xshape + (self.Mmax,), dtype)
        M1 = 0
        for a in self.atom_indices:
            c_xi = c_axi.get(a)
            sphere = self.sphere_a[a]
            M2 = M1 + sphere.Mmax
            if c_xi is None:
                c_xi = b_axi[a]
            c_xM[..., M1:M2] = c_xi
            M1 = M2

        if self.xp is np:
            self.lfc.add(c_xM, a_xG, q)
        elif cupy_is_fake:
            self.lfc.add(c_xM._data, a_xG._data, q)
        else:
            self.lfc.add_gpu(c_xM.data.ptr,
                             c_xM.shape,
                             a_xG.data.ptr,
                             a_xG.shape, q)

    def add_derivative(self, a, v, a_xG, c_axi=1.0, q=-1):
        """Add derivative of localized functions on atom to extended arrays.

        Parameters:

        a: int
            Atomic index of the derivative
        v: int
            Cartesian coordinate of the derivative (0, 1 or 2)

        This function adds the following sum to the extended arrays::

                   --  a      a
          a (G) += >  c   dPhi  (G)
           x       --  xi     iv
                    i

        where::

              a        d     a
          dPhi  (G) =  -- Phi (g)
              iv       dv    i

        is the derivative of the Phi^a and v is either x, y, or z.

        """

        assert v in [0, 1, 2]
        assert not self.use_global_indices

        if q == -1:
            assert self.dtype == float

        if isinstance(c_axi, float):
            assert q == -1
            c_xM = np.empty(self.Mmax)
            c_xM.fill(c_axi)
            self.lfc.add(c_xM, a_xG, q)
            return

        dtype = a_xG.dtype

        if debug:
            assert a_xG.ndim >= 3
            assert dtype == self.dtype
            if isinstance(c_axi, dict):
                assert sorted(c_axi.keys()) == self.my_atom_indices
            for c_xi in c_axi.values():
                assert c_xi.dtype == dtype

        cspline_M = []
        for a_ in self.atom_indices:
            for spline in self.sphere_a[a_].spline_j:
                nm = 2 * spline.get_angular_momentum_number() + 1
                cspline_M.extend([spline.spline] * nm)

        # Temp solution - set all coefficient to zero except for those at
        # atom a

        # Coefficient indices for atom a
        M1 = self.M_a[a]
        M2 = M1 + self.sphere_a[a].Mmax

        if isinstance(c_axi, float):
            assert q == -1
            c_xM = np.zeros(self.Mmax)
            c_xM[..., M1:M2] = c_axi
        else:
            xshape = a_xG.shape[:-3]
            c_xM = np.zeros(xshape + (self.Mmax,), dtype)
            c_xM[..., M1:M2] = c_axi[a]

        gd = self.gd

        self.lfc.add_derivative(c_xM, a_xG, np.ascontiguousarray(gd.h_cv),
                                gd.n_c, cspline_M,
                                gd.beg_c, self.pos_Wv, a, v, q)

    def integrate(self, a_xG, c_axi, q=-1, add_to=False):
        """Calculate integrals of arrays times localized functions.

        ::

                   /             a*
          c_axi =  | dG a (G) Phi  (G)
                   /     x       i

        """
        assert not add_to
        assert not self.use_global_indices
        if q == -1:
            assert self.dtype == float

        xshape = a_xG.shape[:-3]
        dtype = a_xG.dtype

        if debug:
            assert self.dtype == dtype
            assert a_xG.ndim >= 3
            assert sorted(c_axi.keys()) == self.my_atom_indices
            for c_xi in c_axi.values():
                assert c_xi.shape[:-1] == xshape

        comm = self.gd.comm

        c_xM = self.xp.zeros(xshape + (self.Mmax,), dtype)
        if self.xp is np:
            self.lfc.integrate(a_xG, c_xM, q)
        elif cupy_is_fake:
            self.lfc.integrate(a_xG._data, c_xM._data, q)
        else:
            self.lfc.integrate_gpu(a_xG.data.ptr,
                                   a_xG.shape,
                                   c_xM.data.ptr,
                                   c_xM.shape, q)
            c_xM *= self.gd.dv

        rank = comm.rank
        srequests = []
        rrequests = []
        c_arxi = {}
        b_axi = {}
        M1 = 0
        for a in self.atom_indices:
            sphere = self.sphere_a[a]
            M2 = M1 + sphere.Mmax
            if sphere.rank != rank:
                c_xi = c_xM[..., M1:M2].copy()
                b_axi[a] = c_xi
                srequests.append(comm.send(c_xi,
                                           sphere.rank, a, False))
            else:
                if len(sphere.ranks) > 0:
                    c_rxi = self.xp.empty(
                        sphere.ranks.shape + xshape + (M2 - M1,),
                        dtype)
                    c_arxi[a] = c_rxi
                    for r, b_xi in zip(sphere.ranks, c_rxi):
                        rrequests.append(comm.receive(b_xi, r, a, False))
            M1 = M2

        for request in rrequests:
            comm.wait(request)

        M1 = 0
        for a in self.atom_indices:
            c_xi = c_axi.get(a)
            sphere = self.sphere_a[a]
            M2 = M1 + sphere.Mmax
            if c_xi is not None:
                if len(sphere.ranks) > 0:
                    if c_xM.shape[-1] > M1:
                        c_xi[:] = c_xM[..., M1:M2] + c_arxi[a].sum(axis=0)
                elif c_xM.shape[-1] > M1:
                    c_xi[:] = c_xM[..., M1:M2]
            M1 = M2

        for request in srequests:
            comm.wait(request)

    def derivative(self, a_xG, c_axiv, q=-1):
        """Calculate x-, y-, and z-derivatives of localized function integrals.

        ::

                    /              a*
          c_axiv =  | dG a (G) dPhi  (G)
                    /     x        iv

        where::


              a        d     a
          dPhi  (G) =  -- Phi (g)
              iv       dv    i


        and v is either x, y, or z, and R^a_v is the center of Phi^a.

        Notice that d Phi^a_i / dR^a_v == - d Phi^a_i / d v.

        """

        assert not self.use_global_indices

        if debug:
            assert a_xG.ndim >= 3
            assert sorted(c_axiv.keys()) == self.my_atom_indices

        if self.integral_a is not None:
            # assert q == -1
            assert a_xG.ndim == 3
            assert a_xG.dtype == float
            self._normalized_derivative(a_xG, c_axiv)
            return

        dtype = a_xG.dtype

        xshape = a_xG.shape[:-3]
        c_xMv = np.zeros(xshape + (self.Mmax, 3), dtype)
        cspline_M = []
        for a in self.atom_indices:
            for spline in self.sphere_a[a].spline_j:
                nm = 2 * spline.get_angular_momentum_number() + 1
                cspline_M.extend([spline.spline] * nm)

        gd = self.gd
        self.lfc.derivative(a_xG, c_xMv, np.ascontiguousarray(gd.h_cv),
                            gd.n_c, cspline_M,
                            gd.beg_c, self.pos_Wv, q)

        comm = self.gd.comm
        rank = comm.rank
        srequests = []
        rrequests = []
        c_arxiv = {}  # see also https://arXiv.org
        b_axiv = {}
        M1 = 0
        for a in self.atom_indices:
            sphere = self.sphere_a[a]
            M2 = M1 + sphere.Mmax
            if sphere.rank != rank:
                c_xiv = c_xMv[..., M1:M2, :].copy()
                b_axiv[a] = c_xiv
                srequests.append(comm.send(c_xiv,
                                           sphere.rank, a, False))
            else:
                if len(sphere.ranks) > 0:
                    c_rxiv = np.empty(sphere.ranks.shape + xshape +
                                      (M2 - M1, 3), dtype)
                    c_arxiv[a] = c_rxiv
                    for r, b_xiv in zip(sphere.ranks, c_rxiv):
                        rrequests.append(comm.receive(b_xiv, r, a, False))
            M1 = M2

        for request in rrequests:
            comm.wait(request)

        M1 = 0
        for a in self.atom_indices:
            c_xiv = c_axiv.get(a)
            sphere = self.sphere_a[a]
            M2 = M1 + sphere.Mmax
            if c_xiv is not None:
                if len(sphere.ranks) > 0:
                    c_xiv[:] = c_xMv[..., M1:M2, :] + c_arxiv[a].sum(axis=0)
                else:
                    c_xiv[:] = c_xMv[..., M1:M2, :]
            M1 = M2

        for request in srequests:
            comm.wait(request)

    def _normalized_derivative(self, a_G, c_aiv):
        """Calculate x-, y-, and z-derivatives of localized function integrals.

        Calculates the derivatives of this integral::

           a       /  _   _   a  -   _a
          A     =  | dr a(r) f  (r - R ),
           lm      /          lm

                    a
                  dA
                    lm
          c_aiv = ----,
                    a
                   R
                    v

        where v is either x, y, or z and i=l**2+m.  Note that the
        actual integrals used are normalized::

                      a
          ~a     a   I
          f   = f   ---,
           00    00  a
                    I
                     00

        and for l > 0::

                           a
                          I
          ~a     a     a   lm
          f   = f   - f   ---,
           lm    lm    00  a
                          I
                           00

        where

        ::

           a       /  _ -a  _   _a
          I     =  | dr f  (r - R ),
           lm      /     lm


        so the derivative look pretty ugly!
        """

        c_Mv = np.zeros((self.Mmax, 7))

        cspline_M = []
        for a in self.atom_indices:
            for spline in self.sphere_a[a].spline_j:
                nm = 2 * spline.get_angular_momentum_number() + 1
                cspline_M.extend([spline.spline] * nm)
        gd = self.gd
        self.lfc.normalized_derivative(a_G, c_Mv,
                                       np.ascontiguousarray(gd.h_cv), gd.n_c,
                                       cspline_M,
                                       gd.beg_c, self.pos_Wv)

        comm = self.gd.comm
        rank = comm.rank
        srequests = []
        rrequests = []
        c_ariv = {}
        b_aiv = {}
        M1 = 0
        for a in self.atom_indices:
            sphere = self.sphere_a[a]
            M2 = M1 + sphere.Mmax
            if sphere.rank != rank:
                c_iv = c_Mv[M1:M2].copy()
                b_aiv[a] = c_iv
                srequests.append(comm.send(c_iv, sphere.rank, a, False))
            else:
                if len(sphere.ranks) > 0:
                    c_riv = np.empty((len(sphere.ranks), M2 - M1, 7))
                    c_ariv[a] = c_riv
                    for r, b_iv in zip(sphere.ranks, c_riv):
                        rrequests.append(comm.receive(b_iv, r, a, False))
            M1 = M2

        for request in rrequests:
            comm.wait(request)

        M1 = 0
        for a in self.atom_indices:
            c_iv = c_aiv.get(a)
            sphere = self.sphere_a[a]
            M2 = M1 + sphere.Mmax
            if c_iv is not None:
                I = self.integral_a[a]
                if I > 1e-15:
                    if len(sphere.ranks) > 0:
                        c_Mv[M1:M2] += c_ariv[a].sum(axis=0)
                    I_L = sphere.I_M
                    I0 = I_L[0]
                    c_Lv = c_Mv[M1:M2, :3]
                    b_Lv = c_Mv[M1:M2, 3:6]
                    A0 = c_Mv[M1, 6]
                    c_iv[0, :] = (I / I0 * c_Lv[0] -
                                  I / I0**2 * b_Lv[0] * A0)
                    c_iv[1:, :] = (c_Lv[1:] -
                                   np.outer(I_L[1:] / I0, c_Lv[0]) -
                                   A0 / I0 * b_Lv[1:] +
                                   A0 / I0**2 * np.outer(I_L[1:], b_Lv[0]))
                else:
                    c_iv[:] = 0.0

            M1 = M2

        for request in srequests:
            comm.wait(request)

    def second_derivative(self, a_xG, c_axivv, q=-1):
        """Calculate second derivatives.

        Works only for this type of input for now::

              second_derivative(self, a_G, c_avv, q=-1)

        ::

                              2 a _ _a
                   /  _   _  d f (r-R )
          c_avv =  | dr a(r) ----------
                   /             a  a
                               dR dR
                                 i  j
        """

        assert not self.use_global_indices

        if debug:
            assert a_xG.ndim == 3
            assert a_xG.dtype == self.dtype
            assert sorted(c_axivv.keys()) == self.my_atom_indices

        dtype = a_xG.dtype

        c_Mvv = np.zeros((self.Mmax, 3, 3), dtype)

        cspline_M = []
        for a in self.atom_indices:
            # Works only for atoms with a single function
            assert len(self.sphere_a[a].spline_j) == 1
            spline = self.sphere_a[a].spline_j[0]
            # that is spherical symmetric
            assert spline.get_angular_momentum_number() == 0
            cspline_M.append(spline.spline)

        gd = self.gd

        self.lfc.second_derivative(a_xG, c_Mvv, np.ascontiguousarray(gd.h_cv),
                                   gd.n_c, cspline_M,
                                   gd.beg_c, self.pos_Wv, q)

        comm = self.gd.comm
        rank = comm.rank
        srequests = []
        rrequests = []
        c_arvv = {}
        b_avv = {}
        M1 = 0

        for a in self.atom_indices:
            sphere = self.sphere_a[a]
            M2 = M1 + sphere.Mmax
            if sphere.rank != rank:
                c_vv = c_Mvv[M1:M2].copy()
                b_avv[a] = c_vv
                srequests.append(comm.send(c_vv, sphere.rank, a, False))
            else:
                if len(sphere.ranks) > 0:
                    c_rvv = np.empty(sphere.ranks.shape + (3, 3), dtype)
                    c_arvv[a] = c_rvv
                    for r, b_vv in zip(sphere.ranks, c_rvv):
                        rrequests.append(comm.receive(b_vv, r, a, False))
            M1 = M2

        for request in rrequests:
            comm.wait(request)

        M1 = 0
        for a in self.atom_indices:
            c_vv = c_axivv.get(a)
            sphere = self.sphere_a[a]
            M2 = M1 + sphere.Mmax
            if c_vv is not None:
                if len(sphere.ranks) > 0:
                    c_vv[:] = c_Mvv[M1] + c_arvv[a].sum(axis=0)
                else:
                    c_vv[:] = c_Mvv[M1]
            M1 = M2

        for request in srequests:
            comm.wait(request)

    def griditer(self):
        """Iterate over grid points."""
        self.g_W = np.zeros(len(self.M_W), np.intc)
        self.current_lfindices = []
        G1 = 0
        for W, G in zip(self.W_B, self.G_B):
            G2 = G

            yield G1, G2

            self.g_W[self.current_lfindices] += G2 - G1

            if W >= 0:
                self.current_lfindices.append(W)
            else:
                self.current_lfindices.remove(-1 - W)

            G1 = G2

    def get_function_count(self, a):
        return self.sphere_a[a].get_function_count()


class BasisFunctions(LocalizedFunctionsCollection):
    def __init__(self, gd, spline_aj, kd=None, cut=False, dtype=float,
                 integral=None, forces=None, xp=np):
        LocalizedFunctionsCollection.__init__(self, gd, spline_aj,
                                              kd, cut,
                                              dtype, integral,
                                              forces, xp=xp)
        self.use_global_indices = True

        self.Mstart = None
        self.Mstop = None

    @trace
    def set_positions(self, spos_ac):
        LocalizedFunctionsCollection.set_positions(self, spos_ac)
        self.Mstart = 0
        self.Mstop = self.Mmax

    def _update(self, spos_ac):
        sdisp_Wc = LocalizedFunctionsCollection._update(self, spos_ac)

        if not self.gamma or self.dtype == complex:
            self.x_W, self.sdisp_xc = self.create_displacement_arrays(sdisp_Wc)
        return sdisp_Wc

    def create_displacement_arrays(self, sdisp_Wc=None):
        if sdisp_Wc is None:
            sdisp_Wc = np.empty((len(self.M_W), 3), int)

            W = 0
            for a in self.atom_indices:
                sphere = self.sphere_a[a]
                nw = len(sphere.M_w)
                sdisp_Wc[W:W + nw] = sphere.sdisp_wc
                W += nw

        if len(sdisp_Wc) > 0:
            n_c = sdisp_Wc.max(0) - sdisp_Wc.min(0)
        else:
            n_c = np.zeros(3, int)
        N_c = 2 * n_c + 1
        stride_c = np.array([N_c[1] * N_c[2], N_c[2], 1])
        x_W = np.dot(sdisp_Wc, stride_c).astype(np.intc)
        # use a neighbor list instead?
        x1 = np.dot(n_c, stride_c)
        sdisp_xc = np.zeros((x1 + 1, 3), int)
        r_x, sdisp_xc[:, 2] = divmod(np.arange(x1, 2 * x1 + 1), N_c[2])
        sdisp_xc.T[:2] = divmod(r_x, N_c[1])
        sdisp_xc -= n_c

        return x_W, sdisp_xc

    def set_matrix_distribution(self, Mstart, Mstop):
        assert self.Mmax is not None
        self.Mstart = Mstart
        self.Mstop = Mstop

    def add_to_density(self, nt_sG, f_asi):
        r"""Add linear combination of squared localized functions to density.

        :::

          ~        ---   a    a   2
          n (r) += >    f   [Φ(r)]
           s       ---   si   i
                   a,i

        """
        assert np.all(self.gd.n_c == nt_sG.shape[1:])
        nspins = len(nt_sG)
        f_sM = np.empty((nspins, self.Mmax))
        for a in self.atom_indices:
            sphere = self.sphere_a[a]
            M1 = self.M_a[a]
            M2 = M1 + sphere.Mmax
            f_sM[:, M1:M2] = f_asi[a]

        for nt_G, f_M in zip(nt_sG, f_sM):
            self.lfc.construct_density1(f_M, nt_G)

    def construct_density(self, rho_MM, nt_G, q):
        """Calculate electron density from density matrix.

        rho_MM: ndarray
            Density matrix.
        nt_G: ndarray
            Pseudo electron density.

        ::

          ~        --      *
          n(r) +=  >    Phi (r) rho     Phi (r)
                   --     M1       M1M2   M2
                  M1,M2
        """
        self.lfc.construct_density(rho_MM, nt_G, q, self.Mstart, self.Mstop)

    def integrate2(self, a_xG, c_xM, q=-1):
        """Calculate integrals of arrays times localized functions.

        ::

                               /       *
          c_xM += <Phi | a > = | dG Phi (G) a (G)
                      M   x    /       M     x
        """
        xshape, Gshape = a_xG.shape[:-3], a_xG.shape[-3:]
        Nx = int(np.prod(xshape))
        a_xG = a_xG.reshape((Nx,) + Gshape)
        c_xM = c_xM.reshape(Nx, -1)
        for a_G, c_M in zip(a_xG, c_xM):
            self.lfc.integrate(a_G, c_M, q)

    def calculate_potential_matrices(self, vt_G):
        """Calculate lower part of potential matrix.

        ::

                      /
            ~         |     *  _  ~ _        _   _
            V      =  |  Phi  (r) v(r) Phi  (r) dr    for  mu >= nu
             mu nu    |     mu            nu
                      /

        Overwrites the elements of the target matrix Vt_MM. """
        assert np.all(vt_G.shape == self.gd.n_c), (vt_G.shape, self.gd.n_c)
        if self.gamma and self.dtype == float:
            Vt_xMM = np.zeros((1, self.Mstop - self.Mstart, self.Mmax))
            self.lfc.calculate_potential_matrix(vt_G, Vt_xMM[0], -1,
                                                self.Mstart, self.Mstop)
        else:
            Vt_xMM = np.zeros((len(self.sdisp_xc),
                               self.Mstop - self.Mstart,
                               self.Mmax))
            self.lfc.calculate_potential_matrices(vt_G, Vt_xMM, self.x_W,
                                                  self.Mstart, self.Mstop)
        return Vt_xMM

    def calculate_potential_matrix(self, vt_G, Vt_MM, q):
        """Calculate lower part of potential matrix.

        ::

                      /
            ~         |     *  _  ~ _        _   _
            V      =  |  Phi  (r) v(r) Phi  (r) dr    for  mu >= nu
             mu nu    |     mu            nu
                      /

        Overwrites the elements of the target matrix Vt_MM. """
        Vt_MM[:] = 0.0
        self.lfc.calculate_potential_matrix(vt_G, Vt_MM, q,
                                            self.Mstart, self.Mstop)

    def lcao_to_grid(self,
                     C_xM: np.ndarray,
                     psit_xG: np.ndarray,
                     q: int,
                     block_size: int = 10) -> None:
        r"""Deploy basis functions onto grids according to coefficients.

        ::

                       ----
             ~   _     \                 _
            psi (r) +=  )    C     Phi  (r)
               n       /      n mu    mu
                       ----
                        mu
        """

        if C_xM.size == 0:
            return

        if psit_xG.dtype != self.dtype:
            raise TypeError(
                f'psit_xG has type {psit_xG.dtype}, '
                f'but expected one of {self.dtype}')

        if C_xM.dtype != self.dtype:
            raise TypeError(
                f'C_xM has type {C_xM.dtype}, '
                f'but expected one of {self.dtype}')

        xshape = C_xM.shape[:-1]
        assert psit_xG.shape[:-3] == xshape, (psit_xG.shape, xshape)

        C_xM = C_xM.reshape((-1,) + C_xM.shape[-1:])
        psit_xG = psit_xG.reshape((-1,) + psit_xG.shape[-3:])

        if self.gamma or len(C_xM) == 1:
            for C_M, psit_G in zip(C_xM, psit_xG):
                self.lfc.lcao_to_grid(C_M, psit_G, q)
        else:
            # Do sum over unit cells first followed by sum over bands
            # in blocks of block_size orbitals at the time:
            assert C_xM.flags.contiguous
            assert psit_xG.flags.contiguous
            self.lfc.lcao_to_grid_k(C_xM, psit_xG, q, block_size)

    def calculate_potential_matrix_derivative(self, vt_G, DVt_vMM, q):
        """Calculate derivatives of potential matrix elements.

        ::

                      /     *  _
                     |   Phi  (r)
           ~c        |      mu    ~ _        _   _
          DV      += |   -------- v(r) Phi  (r) dr
            mu nu    |     dr             nu
                    /        c

        Results are added to DVt_vMM.
        """
        cspline_M = []
        for a, sphere in enumerate(self.sphere_a):
            for j, spline in enumerate(sphere.spline_j):
                nm = 2 * spline.get_angular_momentum_number() + 1
                cspline_M.extend([spline.spline] * nm)
        gd = self.gd
        for v in range(3):
            self.lfc.calculate_potential_matrix_derivative(
                vt_G, DVt_vMM[v],
                np.ascontiguousarray(gd.h_cv),
                gd.n_c, q, v,
                np.array(cspline_M),
                gd.beg_c,
                self.pos_Wv,
                self.Mstart,
                self.Mstop)

    def calculate_force_contribution(self, vt_G, rhoT_MM, q):
        """Calculate derivatives of potential matrix elements.

        ::

                      /     *  _
                     |   Phi  (r)
           ~c        |      mu    ~ _        _   _
          DV      += |   -------- v(r) Phi  (r) dr
            mu nu    |     dr             nu
                    /        c

        Results are added to DVt_vMM.
        """
        assert np.all(vt_G.shape == self.gd.n_c)
        cspline_M = []
        for a, sphere in enumerate(self.sphere_a):
            for j, spline in enumerate(sphere.spline_j):
                nm = 2 * spline.get_angular_momentum_number() + 1
                cspline_M.extend([spline.spline] * nm)
        gd = self.gd
        Mstart = self.Mstart
        Mstop = self.Mstop
        F_vM = np.zeros((3, Mstop - Mstart))
        assert self.Mmax == rhoT_MM.shape[1]
        assert Mstop - Mstart == rhoT_MM.shape[0]
        assert rhoT_MM.flags.c_contiguous
        for v in range(3):
            self.lfc.calculate_potential_matrix_force_contribution(
                vt_G, rhoT_MM, F_vM[v],
                np.ascontiguousarray(gd.h_cv),
                gd.n_c, q, v,
                np.array(cspline_M),
                gd.beg_c,
                self.pos_Wv,
                Mstart,
                Mstop)

        F_av = np.zeros((len(self.M_a), 3))
        a = 0
        for a, M1 in enumerate(self.M_a):
            M1 -= Mstart
            M2 = M1 + self.sphere_a[a].Mmax
            if M2 < 0:
                continue
            M1 = max(0, M1)
            F_av[a, :] = 2.0 * F_vM[:, M1:M2].sum(axis=1)
        return F_av


def LFC(gd, spline_aj, kd=None,
        cut=False, dtype=float,
        integral=None, forces=False):
    if isinstance(gd, GridDescriptor):
        return LocalizedFunctionsCollection(gd, spline_aj, kd,
                                            cut, dtype, integral, forces)
    else:
        return gd.get_lfc(gd, spline_aj)


def test():
    from gpaw.grid_descriptor import GridDescriptor

    ngpts = 40
    h = 1 / ngpts
    N_c = (ngpts, ngpts, ngpts)
    a = h * ngpts
    gd = GridDescriptor(N_c, (a, a, a))

    from gpaw.spline import Spline
    a = np.array([1, 0.9, 0.8, 0.0])
    s = Spline.from_data(0, 0.2, a)
    x = LocalizedFunctionsCollection(gd, [[s], [s]])
    x.set_positions([(0.5, 0.45, 0.5), (0.5, 0.55, 0.5)])
    n_G = gd.zeros()
    x.add(n_G)
    import matplotlib.pyplot as plt
    plt.contourf(n_G[20, :, :])
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    test()
