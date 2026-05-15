from math import pi

import gpaw.cgpaw as cgpaw
import numpy as np
from gpaw.lfc import BaseLFC
from gpaw.spherical_harmonics import Y, nablarlYL
from gpaw.ffbt import rescaled_fourier_bessel_transform
from gpaw.utilities.blas import mmm


class PWLFC(BaseLFC):
    def __init__(self, spline_aj, pd, blocksize=5000, comm=None):
        """Reciprocal-space plane-wave localized function collection.

        spline_aj: list of list of spline objects
            Splines.
        pd: PWDescriptor
            Plane-wave descriptor object.
        blocksize: int
            Block-size to use when looping over G-vectors.  Use None for
            doing all G-vectors in one big block.
        comm: communicator
            Communicator for operations that support parallelization
            over planewaves (only integrate so far)."""

        self.pd = pd
        self.spline_aj = spline_aj

        self.dtype = pd.dtype

        self.initialized = False

        # These will be filled in later:
        self.Y_qGL = []
        self.emiGR_qGa = []
        self.f_qGs = []
        self.l_s = None
        self.a_J = None
        self.s_J = None
        self.lmax = None

        if blocksize is not None:
            if pd.ngmax <= blocksize:
                # No need to block G-vectors
                blocksize = None
        self.blocksize = blocksize

        # These are set later in set_potitions():
        self.eikR_qa = None
        self.my_atom_indices = None
        self.my_indices = None
        self.pos_av = None
        self.nI = None

        if comm is None:
            comm = pd.gd.comm
        else:
            assert False
        self.comm = comm

    def initialize(self):
        """Initialize position-independent stuff."""
        if self.initialized:
            return

        splines = {}  # Dict[Spline, int]
        for spline_j in self.spline_aj:
            for spline in spline_j:
                if spline not in splines:
                    splines[spline] = len(splines)
        nsplines = len(splines)

        nJ = sum(len(spline_j) for spline_j in self.spline_aj)

        self.f_qGs = [np.empty((mynG, nsplines)) for mynG in self.pd.myng_q]
        self.l_s = np.empty(nsplines, np.int32)
        self.a_J = np.empty(nJ, np.int32)
        self.s_J = np.empty(nJ, np.int32)

        # Fourier transform radial functions:
        J = 0
        done = set()  # Set[Spline]
        for a, spline_j in enumerate(self.spline_aj):
            for spline in spline_j:
                s = splines[spline]  # get spline index
                if spline not in done:
                    f = rescaled_fourier_bessel_transform(spline)
                    for f_Gs, G2_G in zip(self.f_qGs, self.pd.G2_qG):
                        G_G = G2_G**0.5
                        f_Gs[:, s] = f.map(G_G)
                    self.l_s[s] = spline.get_angular_momentum_number()
                    done.add(spline)
                self.a_J[J] = a
                self.s_J[J] = s
                J += 1

        self.lmax = max(self.l_s, default=-1)

        # Spherical harmonics:
        for q, K_v in enumerate(self.pd.K_qv):
            G_Gv = self.pd.get_reciprocal_vectors(q=q)
            Y_GL = np.empty((len(G_Gv), (self.lmax + 1)**2))
            for L in range((self.lmax + 1)**2):
                Y_GL[:, L] = Y(L, *G_Gv.T)
            self.Y_qGL.append(Y_GL)

        self.initialized = True

    def estimate_memory(self, mem):
        splines = set()
        lmax = -1
        for spline_j in self.spline_aj:
            for spline in spline_j:
                splines.add(spline)
                l = spline.get_angular_momentum_number()
                lmax = max(lmax, l)
        nbytes = ((len(splines) + (lmax + 1)**2) *
                  sum(G2_G.nbytes for G2_G in self.pd.G2_qG))
        mem.subnode('Arrays', nbytes)

    def get_function_count(self, a):
        return sum(2 * spline.get_angular_momentum_number() + 1
                   for spline in self.spline_aj[a])

    def set_positions(self, spos_ac, atom_partition=None):
        self.initialize()
        kd = self.pd.kd
        if kd is None or kd.gamma:
            self.eikR_qa = np.ones((1, len(spos_ac)))
        else:
            self.eikR_qa = np.exp(2j * pi * np.dot(kd.ibzk_qc, spos_ac.T))

        self.pos_av = np.dot(spos_ac, self.pd.gd.cell_cv)

        del self.emiGR_qGa[:]
        G_Qv = self.pd.G_Qv
        for Q_G in self.pd.myQ_qG:
            GR_Ga = np.dot(G_Qv[Q_G], self.pos_av.T)
            self.emiGR_qGa.append(np.exp(-1j * GR_Ga))

        if atom_partition is None:
            assert self.comm.size == 1
            rank_a = np.zeros(len(spos_ac), int)
        else:
            rank_a = atom_partition.rank_a

        self.my_atom_indices = []
        self.my_indices = []
        I1 = 0
        for a, rank in enumerate(rank_a):
            I2 = I1 + self.get_function_count(a)
            if rank == self.comm.rank:
                self.my_atom_indices.append(a)
                self.my_indices.append((a, I1, I2))
            I1 = I2
        self.nI = I1

    def expand(self, q=-1, G1=0, G2=None, cc=False):
        """Expand functions in plane-waves.

        q: int
            k-point index.
        G1: int
            Start G-vector index.
        G2: int
            End G-vector index.
        cc: bool
            Complex conjugate.
        """
        if G2 is None:
            G2 = self.Y_qGL[q].shape[0]

        emiGR_Ga = self.emiGR_qGa[q][G1:G2]
        f_Gs = self.f_qGs[q][G1:G2]
        Y_GL = self.Y_qGL[q][G1:G2]

        if self.pd.dtype == complex:
            f_GI = np.empty((G2 - G1, self.nI), complex)
        else:
            # Special layout because BLAS does not have real-complex
            # multiplications.  f_GI(G,I) layout:
            #
            #    real(G1, 0),   real(G1, 1),   ...
            #    imag(G1, 0),   imag(G1, 1),   ...
            #    real(G1+1, 0), real(G1+1, 1), ...
            #    imag(G1+1, 0), imag(G1+1, 1), ...
            #    ...

            f_GI = np.empty((2 * (G2 - G1), self.nI))

        if True:
            # Fast C-code:
            cgpaw.pwlfc_expand_old(f_Gs, emiGR_Ga, Y_GL,
                                   self.l_s, self.a_J, self.s_J,
                                   cc, f_GI)
            return f_GI

        # Equivalent slow Python code:
        f_GI = np.empty((G2 - G1, self.nI), complex)
        I1 = 0
        for J, (a, s) in enumerate(zip(self.a_J, self.s_J)):
            l = self.l_s[s]
            I2 = I1 + 2 * l + 1
            f_GI[:, I1:I2] = (f_Gs[:, s] *
                              emiGR_Ga[:, a] *
                              Y_GL[:, l**2:(l + 1)**2].T *
                              (-1.0j)**l).T
            I1 = I2
        if cc:
            f_GI = f_GI.conj()
        if self.pd.dtype == float:
            f_GI = f_GI.T.copy().view(float).T.copy()

        return f_GI

    def block(self, q=-1, ensure_same_number_of_blocks=False):
        nG = self.Y_qGL[q].shape[0]
        B = self.blocksize
        if B:
            G1 = 0
            while G1 < nG:
                G2 = min(G1 + B, nG)
                yield G1, G2
                G1 = G2
            if ensure_same_number_of_blocks:
                # Make sure we yield the same number of times:
                nb = (self.pd.maxmyng + B - 1) // B
                mynb = (nG + B - 1) // B
                if mynb < nb:
                    yield nG, nG  # empty block
        else:
            yield 0, nG

    def add(self, a_xG, c_axi=1.0, q=-1, f0_IG=None):
        c_xI = np.empty(a_xG.shape[:-1] + (self.nI,), self.pd.dtype)

        if isinstance(c_axi, float):
            assert q == -1 and a_xG.ndim == 1
            c_xI[:] = c_axi
        else:
            assert q != -1 or self.pd.only_one_k_point
            if self.comm.size != 1:
                c_xI[:] = 0.0
            for a, I1, I2 in self.my_indices:
                c_xI[..., I1:I2] = c_axi[a] * self.eikR_qa[q][a].conj()
            if self.comm.size != 1:
                self.comm.sum(c_xI)

        nx = np.prod(c_xI.shape[:-1], dtype=int)
        c_xI = c_xI.reshape((nx, self.nI))
        a_xG = a_xG.reshape((nx, a_xG.shape[-1])).view(self.pd.dtype)

        for G1, G2 in self.block(q):
            if f0_IG is None:
                f_GI = self.expand(q, G1, G2, cc=False)
            else:
                1 / 0
                # f_IG = f0_IG

            if self.pd.dtype == float:
                # f_IG = f_IG.view(float)
                G1 *= 2
                G2 *= 2

            mmm(1.0 / self.pd.gd.dv, c_xI, 'N', f_GI, 'T',
                1.0, a_xG[:, G1:G2])

    def integrate(self, a_xG, c_axi=None, q=-1):
        c_xI = np.zeros(a_xG.shape[:-1] + (self.nI,), self.pd.dtype)

        nx = np.prod(c_xI.shape[:-1], dtype=int)
        b_xI = c_xI.reshape((nx, self.nI))
        a_xG = a_xG.reshape((nx, a_xG.shape[-1]))

        alpha = 1.0 / self.pd.gd.N_c.prod()
        if self.pd.dtype == float:
            alpha *= 2
            a_xG = a_xG.view(float)

        if c_axi is None:
            c_axi = self.dict(a_xG.shape[:-1])

        x = 0.0
        for G1, G2 in self.block(q):
            f_GI = self.expand(q, G1, G2, cc=self.pd.dtype == complex)
            if self.pd.dtype == float:
                if G1 == 0 and self.comm.rank == 0:
                    f_GI[0] *= 0.5
                G1 *= 2
                G2 *= 2
            mmm(alpha, a_xG[:, G1:G2], 'N', f_GI, 'N', x, b_xI)
            x = 1.0

        self.comm.sum(b_xI)
        for a, I1, I2 in self.my_indices:
            c_axi[a][:] = self.eikR_qa[q][a] * c_xI[..., I1:I2]

        return c_axi

    def matrix_elements(self, psit, out):
        P_ani = {a: P_in.T for a, P_in in out.items()}
        self.integrate(psit.array, P_ani, psit.kpt)

    def derivative(self, a_xG, c_axiv=None, q=-1):
        c_vxI = np.zeros((3,) + a_xG.shape[:-1] + (self.nI,), self.pd.dtype)
        nx = np.prod(c_vxI.shape[1:-1], dtype=int)
        b_vxI = c_vxI.reshape((3, nx, self.nI))
        a_xG = a_xG.reshape((nx, a_xG.shape[-1])).view(self.pd.dtype)

        alpha = 1.0 / self.pd.gd.N_c.prod()

        if c_axiv is None:
            c_axiv = self.dict(a_xG.shape[:-1], derivative=True)

        K_v = self.pd.K_qv[q]

        x = 0.0
        for G1, G2 in self.block(q):
            f_GI = self.expand(q, G1, G2, cc=True)
            G_Gv = self.pd.G_Qv[self.pd.myQ_qG[q][G1:G2]]
            if self.pd.dtype == float:
                d_GI = np.empty_like(f_GI)
                for v in range(3):
                    d_GI[::2] = f_GI[1::2] * G_Gv[:, v, np.newaxis]
                    d_GI[1::2] = f_GI[::2] * G_Gv[:, v, np.newaxis]
                    mmm(2 * alpha,
                        a_xG[:, 2 * G1:2 * G2], 'N',
                        d_GI, 'N',
                        x, b_vxI[v])
            else:
                for v in range(3):
                    mmm(-alpha,
                        a_xG[:, G1:G2], 'N',
                        f_GI * (G_Gv[:, v] + K_v[v])[:, np.newaxis], 'N',
                        x, b_vxI[v])
            x = 1.0

        self.comm.sum(c_vxI)

        for v in range(3):
            if self.pd.dtype == float:
                for a, I1, I2 in self.my_indices:
                    c_axiv[a][..., v] = c_vxI[v, ..., I1:I2]
            else:
                for a, I1, I2 in self.my_indices:
                    c_axiv[a][..., v] = (1.0j * self.eikR_qa[q][a] *
                                         c_vxI[v, ..., I1:I2])

        return c_axiv

    def stress_tensor_contribution(self, a_xG, c_axi=1.0, q=-1):
        cache = {}
        things = []
        I1 = 0
        lmax = 0
        for a, spline_j in enumerate(self.spline_aj):
            for spline in spline_j:
                if spline not in cache:
                    s = rescaled_fourier_bessel_transform(spline)
                    G_G = self.pd.G2_qG[q]**0.5
                    f_G = []
                    dfdGoG_G = []
                    for G in G_G:
                        f, dfdG = s.get_value_and_derivative(G)
                        if G < 1e-10:
                            G = 1.0
                        f_G.append(f)
                        dfdGoG_G.append(dfdG / G)
                    f_G = np.array(f_G)
                    dfdGoG_G = np.array(dfdGoG_G)
                    cache[spline] = (f_G, dfdGoG_G)
                else:
                    f_G, dfdGoG_G = cache[spline]
                l = spline.l
                lmax = max(l, lmax)
                I2 = I1 + 2 * l + 1
                things.append((a, l, I1, I2, f_G, dfdGoG_G))
                I1 = I2

        if isinstance(c_axi, float):
            c_axi = {a: c_axi for a in range(len(self.pos_av))}

        G0_Gv = self.pd.get_reciprocal_vectors(q=q)

        stress_vv = np.zeros((3, 3))
        for G1, G2 in self.block(q, ensure_same_number_of_blocks=True):
            G_Gv = G0_Gv[G1:G2]
            Z_LvG = np.array([nablarlYL(L, G_Gv.T)
                              for L in range((lmax + 1)**2)])
            aa_xG = a_xG[..., G1:G2]
            for v1 in range(3):
                for v2 in range(3):
                    stress_vv[v1, v2] += self._stress_tensor_contribution(
                        v1, v2, things, G1, G2, G_Gv, aa_xG, c_axi, q, Z_LvG)

        self.comm.sum(stress_vv)

        return stress_vv

    def _stress_tensor_contribution(self, v1, v2, things, G1, G2,
                                    G_Gv, a_xG, c_axi, q, Z_LvG):
        f_IG = np.empty((self.nI, G2 - G1), complex)
        emiGR_Ga = self.emiGR_qGa[q][G1:G2]
        Y_LG = self.Y_qGL[q].T
        for a, l, I1, I2, f_G, dfdGoG_G in things:
            L1 = l**2
            L2 = (l + 1)**2
            f_IG[I1:I2] = (emiGR_Ga[:, a] * (-1.0j)**l *
                           (dfdGoG_G[G1:G2] * G_Gv[:, v1] * G_Gv[:, v2] *
                            Y_LG[L1:L2, G1:G2] +
                            f_G[G1:G2] * G_Gv[:, v1] * Z_LvG[L1:L2, v2]))

        c_xI = np.zeros(a_xG.shape[:-1] + (self.nI,), self.pd.dtype)

        x = np.prod(c_xI.shape[:-1], dtype=int)
        b_xI = c_xI.reshape((x, self.nI))
        a_xG = a_xG.reshape((x, a_xG.shape[-1]))

        alpha = 1.0 / self.pd.gd.N_c.prod()
        if self.pd.dtype == float:
            alpha *= 2
            if G1 == 0 and self.pd.gd.comm.rank == 0:
                f_IG[:, 0] *= 0.5
            f_IG = f_IG.view(float)
            a_xG = a_xG.copy().view(float)

        mmm(alpha, a_xG, 'N', f_IG, 'C', 0.0, b_xI)
        self.comm.sum(b_xI)

        stress = 0.0
        for a, I1, I2 in self.my_indices:
            stress -= self.eikR_qa[q][a] * (c_axi[a] * c_xI[..., I1:I2]).sum()
        return stress.real
