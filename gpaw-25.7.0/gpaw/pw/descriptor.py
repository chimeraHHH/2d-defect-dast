import numbers
from math import pi
import numpy as np

import gpaw.cgpaw as cgpaw
import gpaw.fftw as fftw
from gpaw.utilities.blas import mmm, r2k, rk
from gpaw.gpu import cupy as cp


class PWDescriptor:
    ndim = 1  # all 3d G-vectors are stored in a 1d ndarray

    def __init__(self, ecut, gd, dtype=None, kd=None,
                 fftwflags=fftw.MEASURE, gammacentered=False,
                 _new=False):

        assert gd.pbc_c.all()

        self.gd = gd
        self.fftwflags = fftwflags

        N_c = gd.N_c
        self.comm = gd.comm

        ecut0 = 0.5 * pi**2 / (self.gd.h_cv**2).sum(1).max()
        if ecut is None:
            ecut = 0.9999 * ecut0
        else:
            assert ecut <= ecut0

        self.ecut = ecut

        if dtype is None:
            if kd is None or kd.gamma:
                dtype = float
            else:
                dtype = complex
        self.dtype = dtype
        self.gammacentered = gammacentered

        if dtype == float:
            Nr_c = N_c.copy()
            Nr_c[2] = N_c[2] // 2 + 1
            i_Qc = np.indices(Nr_c).transpose((1, 2, 3, 0))
            i_Qc[..., :2] += N_c[:2] // 2
            i_Qc[..., :2] %= N_c[:2]
            i_Qc[..., :2] -= N_c[:2] // 2
            self.tmp_Q = fftw.empty(Nr_c, complex)
            self.tmp_R = self.tmp_Q.view(float)[:, :, :N_c[2]]
        else:
            i_Qc = np.indices(N_c).transpose((1, 2, 3, 0))
            i_Qc += N_c // 2
            i_Qc %= N_c
            i_Qc -= N_c // 2
            self.tmp_Q = fftw.empty(N_c, complex)
            self.tmp_R = self.tmp_Q

        self.fftplan = fftw.create_plan(self.tmp_R, self.tmp_Q, -1, fftwflags)
        self.ifftplan = fftw.create_plan(self.tmp_Q, self.tmp_R, 1, fftwflags)

        # Calculate reciprocal lattice vectors:
        B_cv = 2.0 * pi * gd.icell_cv
        i_Qc.shape = (-1, 3)
        self.G_Qv = np.dot(i_Qc, B_cv)

        self.kd = kd
        if kd is None:
            self.K_qv = np.zeros((1, 3))
            self.only_one_k_point = True
        else:
            self.K_qv = np.dot(kd.ibzk_qc, B_cv)
            self.only_one_k_point = (kd.nbzkpts == 1)

        # Map from vectors inside sphere to fft grid:
        Q_Q = np.arange(len(i_Qc), dtype=np.int32)

        self.ng_q, self.Q_qG, G2_qG = \
            self.setup_pw_grid(i_Qc, Q_Q)

        self.ngmin = min(self.ng_q)
        self.ngmax = max(self.ng_q)

        if kd is not None:
            self.ngmin = kd.comm.min_scalar(self.ngmin)
            self.ngmax = kd.comm.max_scalar(self.ngmax)

        # Distribute things:
        S = gd.comm.size
        self.maxmyng = (self.ngmax + S - 1) // S
        ng1 = gd.comm.rank * self.maxmyng
        ng2 = ng1 + self.maxmyng

        self.G2_qG = []
        self.myQ_qG = []
        self.myng_q = []
        self.maxmyng_q = []
        for q, G2_G in enumerate(G2_qG):
            if _new:
                x = (self.ng_q[q] + S - 1) // S
                self.maxmyng_q.append(x)
                ng1 = gd.comm.rank * x
                ng2 = ng1 + x
            G2_G = G2_G[ng1:ng2].copy()
            G2_G.flags.writeable = False
            self.G2_qG.append(G2_G)
            myQ_G = self.Q_qG[q][ng1:ng2]
            self.myQ_qG.append(myQ_G)
            self.myng_q.append(len(myQ_G))

        if S > 1:
            self.tmp_G = np.empty(self.maxmyng * S, complex)
        else:
            self.tmp_G = None

        self._new = _new

    def setup_pw_grid(self, i_Qc, Q_Q):
        ng_q = []
        Q_qG = []
        G2_qG = []
        for q, K_v in enumerate(self.K_qv):
            G2_Q = ((self.G_Qv + K_v)**2).sum(axis=1)
            if self.gammacentered:
                mask_Q = ((self.G_Qv**2).sum(axis=1) <= 2 * self.ecut)
            else:
                mask_Q = (G2_Q <= 2 * self.ecut)

            if self.dtype == float:
                mask_Q &= ((i_Qc[:, 2] > 0) |
                           (i_Qc[:, 1] > 0) |
                           ((i_Qc[:, 0] >= 0) & (i_Qc[:, 1] == 0)))
            Q_G = Q_Q[mask_Q]
            Q_qG.append(Q_G)
            G2_qG.append(G2_Q[Q_G])
            ng = len(Q_G)
            ng_q.append(ng)

        return ng_q, Q_qG, G2_qG

    def get_reciprocal_vectors(self, q=0, add_q=True):
        """Returns reciprocal lattice vectors plus q, G + q,
        in xyz coordinates."""

        if add_q:
            q_v = self.K_qv[q]
            return self.G_Qv[self.myQ_qG[q]] + q_v
        return self.G_Qv[self.myQ_qG[q]]

    def __getstate__(self):
        return (self.ecut, self.gd, self.dtype, self.kd, self.fftwflags)

    def __setstate__(self, state):
        self.__init__(*state)

    def estimate_memory(self, mem):
        nbytes = (self.tmp_R.nbytes +
                  self.G_Qv.nbytes +
                  len(self.K_qv) * (self.ngmax * 4 +
                                    self.maxmyng * (8 + 4)))
        mem.subnode('Arrays', nbytes)

    def bytecount(self, dtype=float):
        return self.ngmax * 16

    def zeros(self, x=(), dtype=None, q=None, global_array=False):
        """Return zeroed array.

        The shape of the array will be x + (ng,) where ng is the number
        of G-vectors for on this core.  Different k-points will have
        different values for ng.  Therefore, the q index must be given,
        unless we are describibg a real-valued function."""

        a_xG = self.empty(x, dtype, q, global_array)
        a_xG.fill(0.0)
        return a_xG

    def empty(self, x=(), dtype=None, q=None, global_array=False):
        """Return empty array."""
        if dtype is not None:
            assert dtype == self.dtype
        if isinstance(x, numbers.Integral):
            x = (x,)
        if q is None:
            assert self.only_one_k_point
            q = 0
        if global_array:
            shape = x + (self.ng_q[q],)
        else:
            shape = x + (self.myng_q[q],)
        return np.empty(shape, complex)

    def fft(self, f_R, q=None, Q_G=None, local=False):
        """Fast Fourier transform.

        Returns c(G) for G<Gc::

                   --      -iG.R
            c(G) = > f(R) e
                   --
                   R

        If local=True, all cores will do an FFT without any
        collect/scatter.
        """

        if local:
            self.tmp_R[:] = f_R
        else:
            self.gd.collect(f_R, self.tmp_R)

        if self.gd.comm.rank == 0 or local:
            self.fftplan.execute()
            if Q_G is None:
                q = q or 0
                Q_G = self.Q_qG[q]
            f_G = self.tmp_Q.ravel()[Q_G]
            if local:
                return f_G
        else:
            f_G = None

        return self.scatter(f_G, q)

    def ifft(self, c_G, q=None, local=False, safe=True, distribute=True):
        """Inverse fast Fourier transform.

        Returns::

                   1 --        iG.R
            f(R) = - > c(G) e
                   N --
                     G

        If local=True, all cores will do an iFFT without any
        gather/distribute.
        """
        assert q is not None or self.only_one_k_point
        q = q or 0
        if not local:
            c_G = self.gather(c_G, q)
        comm = self.gd.comm
        scale = 1.0 / self.tmp_R.size
        if comm.rank == 0 or local:
            # Same as:
            #
            #    self.tmp_Q[:] = 0.0
            #    self.tmp_Q.ravel()[self.Q_qG[q]] = scale * c_G
            #
            # but much faster:
            Q_G = self.Q_qG[q]
            assert len(c_G) == len(Q_G)
            cgpaw.pw_insert(c_G, Q_G, scale, self.tmp_Q)
            if self.dtype == float:
                t = self.tmp_Q[:, :, 0]
                n, m = self.gd.N_c[:2] // 2 - 1
                t[0, -m:] = t[0, m:0:-1].conj()
                t[n:0:-1, -m:] = t[-n:, m:0:-1].conj()
                t[-n:, -m:] = t[n:0:-1, m:0:-1].conj()
                t[-n:, 0] = t[n:0:-1, 0].conj()
            self.ifftplan.execute()
        if comm.size == 1 or local or not distribute:
            if safe:
                return self.tmp_R.copy()
            return self.tmp_R
        return self.gd.distribute(self.tmp_R)

    def scatter(self, a_G, q=None):
        """Scatter coefficients from master to all cores."""
        comm = self.gd.comm
        if comm.size == 1:
            return a_G

        mya_G = np.empty(self.maxmyng, complex)
        comm.scatter(pad(a_G, self.maxmyng * comm.size), mya_G, 0)
        return mya_G[:self.myng_q[q or 0]]

    def gather(self, a_G, q=None):
        """Gather coefficients on master."""
        comm = self.gd.comm

        if comm.size == 1:
            return a_G

        mya_G = pad(a_G, self.maxmyng)
        if comm.rank == 0:
            a_G = self.tmp_G
        else:
            a_G = None
        comm.gather(mya_G, 0, a_G)
        if comm.rank == 0:
            return a_G[:self.ng_q[q or 0]]

    def alltoall1(self, a_rG, q):
        """Gather coefficients from a_rG[r] on rank r.

        On rank r, an array of all G-vector coefficients will be returned.
        These will be gathered from a_rG[r] on all the cores.
        """
        comm = self.gd.comm
        if comm.size == 1:
            return a_rG[0]
        N = len(a_rG)
        ng = self.ng_q[q]
        ssize_r = np.zeros(comm.size, int)
        ssize_r[:N] = self.myng_q[q]
        soffset_r = np.arange(comm.size) * self.myng_q[q]
        soffset_r[N:] = 0
        myng = self.maxmyng_q[q] if self._new else self.maxmyng
        roffset_r = (np.arange(comm.size) * myng).clip(max=ng)
        rsize_r = np.zeros(comm.size, int)
        if comm.rank < N:
            rsize_r[:-1] = roffset_r[1:] - roffset_r[:-1]
            rsize_r[-1] = ng - roffset_r[-1]
        b_G = self.tmp_G[:ng]
        comm.alltoallv(a_rG, ssize_r, soffset_r, b_G, rsize_r, roffset_r)
        if comm.rank < N:
            return b_G

    def alltoall2(self, a_G, q, b_rG):
        """Scatter all coefs. from rank r to B_rG[r] on other cores."""
        comm = self.gd.comm
        if comm.size == 1:
            b_rG[0] += a_G
            return
        N = len(b_rG)
        ng = self.ng_q[q]
        rsize_r = np.zeros(comm.size, int)
        rsize_r[:N] = self.myng_q[q]
        roffset_r = np.arange(comm.size) * self.myng_q[q]
        roffset_r[N:] = 0
        myng = self.maxmyng_q[q] if self._new else self.maxmyng
        soffset_r = (np.arange(comm.size) * myng).clip(max=ng)
        ssize_r = np.zeros(comm.size, int)
        if comm.rank < N:
            ssize_r[:-1] = soffset_r[1:] - soffset_r[:-1]
            ssize_r[-1] = ng - soffset_r[-1]
        tmp_rG = self.tmp_G[:b_rG.size].reshape(b_rG.shape)
        comm.alltoallv(a_G, ssize_r, soffset_r, tmp_rG, rsize_r, roffset_r)
        b_rG += tmp_rG

    def integrate(self, a_xg, b_yg=None,
                  global_integral=True, hermitian=False):
        """Integrate function(s) over domain.

        a_xg: ndarray
            Function(s) to be integrated.
        b_yg: ndarray
            If present, integrate a_xg.conj() * b_yg.
        global_integral: bool
            If the array(s) are distributed over several domains, then the
            total sum will be returned.  To get the local contribution
            only, use global_integral=False.
        hermitian: bool
            Result is hermitian.
        """

        if b_yg is None:
            # Only one array:
            assert self.dtype == float and self.gd.comm.size == 1
            return a_xg[..., 0].real * self.gd.dv

        if a_xg.ndim == 1:
            A_xg = a_xg.reshape((1, len(a_xg)))
        else:
            A_xg = a_xg
        if b_yg.ndim == 1:
            B_yg = b_yg.reshape((1, len(b_yg)))
        else:
            B_yg = b_yg

        alpha = self.gd.dv / self.gd.N_c.prod()

        if self.dtype == float:
            alpha *= 2
            A_xg = A_xg.view(float)
            B_yg = B_yg.view(float)

        result_yx = np.zeros((len(B_yg), len(A_xg)), self.dtype)

        if a_xg is b_yg:
            rk(alpha, A_xg, 0.0, result_yx)
        elif hermitian:
            r2k(0.5 * alpha, A_xg, B_yg, 0.0, result_yx)
        else:
            mmm(alpha, B_yg, 'N', A_xg, 'C', 0.0, result_yx)

        if self.dtype == float and self.gd.comm.rank == 0:
            correction_yx = np.outer(B_yg[:, 0], A_xg[:, 0])
            if hermitian:
                result_yx -= 0.25 * alpha * (correction_yx + correction_yx.T)
            else:
                result_yx -= 0.5 * alpha * correction_yx

        xshape = a_xg.shape[:-1]
        yshape = b_yg.shape[:-1]
        result = result_yx.T.reshape(xshape + yshape)

        if result.ndim == 0:
            if global_integral:
                return self.gd.comm.sum_scalar(result.item())
            return result.item()
        else:
            assert global_integral or self.gd.comm.size == 1
            self.gd.comm.sum(result.T)
            return result

    def interpolate(self, a_R, pd):
        if (pd.gd.N_c <= self.gd.N_c).any():
            raise ValueError('Too few points in target grid!')

        self.gd.collect(a_R, self.tmp_R[:])

        if self.gd.comm.rank == 0:
            self.fftplan.execute()

            a_Q = self.tmp_Q
            b_Q = pd.tmp_Q

            e0, e1, e2 = 1 - self.gd.N_c % 2  # even or odd size
            a0, a1, a2 = pd.gd.N_c // 2 - self.gd.N_c // 2
            b0, b1, b2 = self.gd.N_c + (a0, a1, a2)

            if self.dtype == float:
                b2 = (b2 - a2) // 2 + 1
                a2 = 0
                axes = (0, 1)
            else:
                axes = (0, 1, 2)

            b_Q[:] = 0.0
            b_Q[a0:b0, a1:b1, a2:b2] = np.fft.fftshift(a_Q, axes=axes)

            if e0:
                b_Q[a0, a1:b1, a2:b2] *= 0.5
                b_Q[b0, a1:b1, a2:b2] = b_Q[a0, a1:b1, a2:b2]
                b0 += 1
            if e1:
                b_Q[a0:b0, a1, a2:b2] *= 0.5
                b_Q[a0:b0, b1, a2:b2] = b_Q[a0:b0, a1, a2:b2]
                b1 += 1
            if self.dtype == complex:
                if e2:
                    b_Q[a0:b0, a1:b1, a2] *= 0.5
                    b_Q[a0:b0, a1:b1, b2] = b_Q[a0:b0, a1:b1, a2]
            else:
                if e2:
                    b_Q[a0:b0, a1:b1, b2 - 1] *= 0.5

            b_Q[:] = np.fft.ifftshift(b_Q, axes=axes)
            pd.ifftplan.execute()

            a_G = a_Q.ravel()[self.Q_qG[0]]
        else:
            a_G = None

        return (pd.gd.distribute(pd.tmp_R) * (1.0 / self.tmp_R.size),
                self.scatter(a_G))

    def restrict(self, a_R, pd):
        self.gd.collect(a_R, self.tmp_R[:])

        if self.gd.comm.rank == 0:
            a_Q = pd.tmp_Q
            b_Q = self.tmp_Q

            e0, e1, e2 = 1 - pd.gd.N_c % 2  # even or odd size
            a0, a1, a2 = self.gd.N_c // 2 - pd.gd.N_c // 2
            b0, b1, b2 = pd.gd.N_c // 2 + self.gd.N_c // 2 + 1

            if self.dtype == float:
                b2 = pd.gd.N_c[2] // 2 + 1
                a2 = 0
                axes = (0, 1)
            else:
                axes = (0, 1, 2)

            self.fftplan.execute()
            b_Q[:] = np.fft.fftshift(b_Q, axes=axes)

            if e0:
                b_Q[a0, a1:b1, a2:b2] += b_Q[b0 - 1, a1:b1, a2:b2]
                b_Q[a0, a1:b1, a2:b2] *= 0.5
                b0 -= 1
            if e1:
                b_Q[a0:b0, a1, a2:b2] += b_Q[a0:b0, b1 - 1, a2:b2]
                b_Q[a0:b0, a1, a2:b2] *= 0.5
                b1 -= 1
            if self.dtype == complex and e2:
                b_Q[a0:b0, a1:b1, a2] += b_Q[a0:b0, a1:b1, b2 - 1]
                b_Q[a0:b0, a1:b1, a2] *= 0.5
                b2 -= 1

            a_Q[:] = b_Q[a0:b0, a1:b1, a2:b2]
            a_Q[:] = np.fft.ifftshift(a_Q, axes=axes)
            a_G = a_Q.ravel()[pd.Q_qG[0]] / 8
            pd.ifftplan.execute()
        else:
            a_G = None

        return (pd.gd.distribute(pd.tmp_R) * (1.0 / self.tmp_R.size),
                pd.scatter(a_G))


class PWMapping:
    def __init__(self, pd1, pd2):
        """Mapping from pd1 to pd2."""
        N_c = np.array(pd1.tmp_Q.shape)
        N2_c = pd2.tmp_Q.shape
        Q1_G = pd1.Q_qG[0]
        Q1_Gc = np.empty((len(Q1_G), 3), int)
        Q1_Gc[:, 0], r_G = divmod(Q1_G, N_c[1] * N_c[2])
        Q1_Gc.T[1:] = divmod(r_G, N_c[2])
        if pd1.dtype == float:
            C = 2
        else:
            C = 3
        Q1_Gc[:, :C] += N_c[:C] // 2
        Q1_Gc[:, :C] %= N_c[:C]
        Q1_Gc[:, :C] -= N_c[:C] // 2
        Q1_Gc[:, :C] %= N2_c[:C]
        Q2_G = Q1_Gc[:, 2] + N2_c[2] * (Q1_Gc[:, 1] + N2_c[1] * Q1_Gc[:, 0])
        G2_Q = np.empty(N2_c, int).ravel()
        G2_Q[:] = -1
        G2_Q[pd2.myQ_qG[0]] = np.arange(pd2.myng_q[0])
        G2_G1 = G2_Q[Q2_G]

        mask_G1 = (G2_G1 != -1)
        self.G2_G1 = G2_G1[mask_G1]
        self.G1 = np.arange(pd1.ngmax)[mask_G1]

        self.pd1 = pd1
        self.pd2 = pd2

    def add_to1(self, a_G1, b_G2):
        """Do a += b * scale, where a is on pd1 and b on pd2."""
        scale = self.pd1.tmp_R.size / self.pd2.tmp_R.size

        if self.pd1.gd.comm.size == 1:
            a_G1 += b_G2[self.G2_G1] * scale
            return

        b_G1 = self.pd1.tmp_G
        b_G1[:] = 0.0
        b_G1[self.G1] = b_G2[self.G2_G1]
        self.pd1.gd.comm.sum(b_G1)
        ng1 = self.pd1.gd.comm.rank * self.pd1.maxmyng
        ng2 = ng1 + self.pd1.myng_q[0]
        a_G1 += b_G1[ng1:ng2] * scale

    def add_to2(self, a_G2, b_G1):
        """Do a += b * scale, where a is on pd2 and b on pd1."""
        myb_G1 = b_G1 * (self.pd2.tmp_R.size / self.pd1.tmp_R.size)
        if self.pd1.gd.comm.size == 1:
            a_G2[self.G2_G1] += myb_G1
            return

        b_G1 = self.pd1.tmp_G
        self.pd1.gd.comm.all_gather(pad(myb_G1, self.pd1.maxmyng), b_G1)
        a_G2[self.G2_G1] += b_G1[self.G1]


def count_reciprocal_vectors(ecut, gd, q_c):
    assert gd.comm.size == 1
    N_c = gd.N_c
    i_Qc = np.indices(N_c).transpose((1, 2, 3, 0))
    i_Qc += N_c // 2
    i_Qc %= N_c
    i_Qc -= N_c // 2

    B_cv = 2.0 * pi * gd.icell_cv
    i_Qc.shape = (-1, 3)
    Gpq_Qv = np.dot(i_Qc, B_cv) + np.dot(q_c, B_cv)

    G2_Q = (Gpq_Qv**2).sum(axis=1)
    return (G2_Q <= 2 * ecut).sum()


def pad(array, N):
    """Pad 1-d ndarray with zeros up to length N.

    >>> a = np.ones(2, complex)
    >>> pad(a, 3)
    array([1.+0.j, 1.+0.j, 0.+0.j])
    >>> pad(a, 2) is a
    True
    >>> pad(None, 7) is None
    True
    """
    if array is None:
        return None
    n = len(array)
    if n == N:
        return array
    if isinstance(array, np.ndarray):
        b = np.empty(N, complex)
    else:
        b = cp.empty(N, complex)
    b[:n] = array
    b[n:] = 0
    return b
