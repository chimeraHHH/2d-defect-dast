import numpy as np

from gpaw.response import ResponseContext, ResponseGroundStateAdapter, timer
from gpaw.response.pw_parallelization import block_partition
from gpaw.utilities.blas import mmm


class KPoint:
    def __init__(self, s, K, n1, n2, blocksize, na, nb,
                 ut_nR, eps_n, f_n, P_ani, k_c):
        self.s = s    # spin index
        self.K = K    # BZ k-point index
        self.n1 = n1  # first band
        self.n2 = n2  # first band not included
        self.blocksize = blocksize
        self.na = na  # first band of block
        self.nb = nb  # first band of block not included
        self.ut_nR = ut_nR      # periodic part of wave functions in real-space
        self.eps_n = eps_n      # eigenvalues
        self.f_n = f_n          # occupation numbers
        self.P_ani = P_ani      # PAW projections
        self.k_c = k_c  # k-point coordinates


class KPointPair:
    """This class defines the kpoint-pair container object.

    Used for calculating pair quantities it contains two kpoints,
    and an associated set of Fourier components."""
    def __init__(self, kpt1, kpt2, Q_G):
        self.kpt1 = kpt1
        self.kpt2 = kpt2
        self.Q_G = Q_G

    def get_transition_energies(self):
        """Return the energy difference for specified bands."""
        kpt1 = self.kpt1
        kpt2 = self.kpt2
        deps_nm = kpt1.eps_n[:, np.newaxis] - kpt2.eps_n
        return deps_nm

    def get_occupation_differences(self):
        """Get difference in occupation factor between specified bands."""
        kpt1 = self.kpt1
        kpt2 = self.kpt2
        df_nm = kpt1.f_n[:, np.newaxis] - kpt2.f_n
        return df_nm


class KPointPairFactory:
    def __init__(self, gs, context):
        self.gs = gs
        self.context = context
        assert self.gs.kd.symmetry.symmorphic
        assert self.gs.world.size == 1

    @timer('Get a k-point')
    def get_k_point(self, s, K, n1, n2, blockcomm=None):
        """Return wave functions for a specific k-point and spin.

        s: int
            Spin index (0 or 1).
        K: int
            BZ k-point index.
        n1, n2: int
            Range of bands to include.
        """

        assert n1 <= n2

        gs = self.gs
        kd = gs.kd

        if blockcomm:
            nblocks = blockcomm.size
            rank = blockcomm.rank
        else:
            nblocks = 1
            rank = 0

        blocksize = (n2 - n1 + nblocks - 1) // nblocks
        na = min(n1 + rank * blocksize, n2)
        nb = min(na + blocksize, n2)

        ik = kd.bz2ibz_k[K]
        assert kd.comm.size == 1
        kpt = gs.kpt_qs[ik][s]

        assert n2 <= len(kpt.eps_n), \
            'Increase GS-nbands or decrease chi0-nbands!'
        eps_n = kpt.eps_n[n1:n2]
        f_n = kpt.f_n[n1:n2] / kpt.weight

        k_c = self.gs.ibz2bz[K].map_kpoint()

        with self.context.timer('load wfs'):
            psit_nG = kpt.psit_nG
            ut_nR = gs.gd.empty(nb - na, gs.dtype)
            for n in range(na, nb):
                ut_nR[n - na] = self.gs.ibz2bz[K].map_pseudo_wave(
                    gs.pd.ifft(psit_nG[n], ik))

        with self.context.timer('Load projections'):
            if nb - na > 0:
                proj = kpt.projections.new(nbands=nb - na, bcomm=None)
                proj.array[:] = kpt.projections.array[na:nb]
                proj = self.gs.ibz2bz[K].map_projections(proj)
                P_ani = [P_ni for _, P_ni in proj.items()]
            else:
                P_ani = []

        return KPoint(s, K, n1, n2, blocksize, na, nb,
                      ut_nR, eps_n, f_n, P_ani, k_c)

    @timer('Get kpoint pair')
    def get_kpoint_pair(self, qpd, s, K, n1, n2, m1, m2,
                        blockcomm=None, flipspin=False):
        assert m1 <= m2
        assert n1 <= n2

        kptfinder = self.gs.kpoints.kptfinder

        k_c = self.gs.kd.bzk_kc[K]
        K1 = kptfinder.find(k_c)
        K2 = kptfinder.find(k_c + qpd.q_c)
        s1 = s
        s2 = (s + flipspin) % 2

        with self.context.timer('get k-points'):
            kpt1 = self.get_k_point(s1, K1, n1, n2)
            kpt2 = self.get_k_point(s2, K2, m1, m2, blockcomm=blockcomm)

        with self.context.timer('fft indices'):
            Q_G = phase_shifted_fft_indices(kpt1.k_c, kpt2.k_c, qpd)

        return KPointPair(kpt1, kpt2, Q_G)

    def pair_calculator(self, blockcomm=None):
        # We have decoupled the actual pair density calculator
        # from the kpoint factory, but it's still handy to
        # keep this shortcut -- for now.
        if blockcomm is None:
            blockcomm, _ = block_partition(self.context.comm, nblocks=1)
        return ActualPairDensityCalculator(self, blockcomm)


class ActualPairDensityCalculator:
    def __init__(self, kptpair_factory, blockcomm):
        # it seems weird to use kptpair_factory only for this
        self.gs = kptpair_factory.gs
        self.context = kptpair_factory.context
        self.blockcomm = blockcomm
        self.ut_sKnvR = None  # gradient of wave functions for optical limit

    def get_optical_pair_density(self, qpd, kptpair, n_n, m_m, *,
                                 pawcorr, block=False):
        """Get the full optical pair density, including the optical limit head
        for q=0."""
        tmp_nmG = self.get_pair_density(qpd, kptpair, n_n, m_m,
                                        pawcorr=pawcorr, block=block)

        nG = qpd.ngmax
        # P = (x, y, z, G1, G2, ...)
        n_nmP = np.empty((len(n_n), len(m_m), nG + 2), dtype=tmp_nmG.dtype)
        n_nmP[:, :, 3:] = tmp_nmG[:, :, 1:]
        n_nmv = self.get_optical_pair_density_head(qpd, kptpair, n_n, m_m,
                                                   block=block)
        n_nmP[:, :, :3] = n_nmv

        return n_nmP

    @timer('get_pair_density')
    def get_pair_density(self, qpd, kptpair, n_n, m_m, *,
                         pawcorr, block=False):
        """Get pair density for a kpoint pair."""
        cpd = self.calculate_pair_density

        kpt1 = kptpair.kpt1
        kpt2 = kptpair.kpt2
        Q_G = kptpair.Q_G  # Fourier components of kpoint pair
        nG = len(Q_G)

        n_nmG = np.zeros((len(n_n), len(m_m), nG), qpd.dtype)

        for j, n in enumerate(n_n):
            Q_G = kptpair.Q_G
            with self.context.timer('conj'):
                ut1cc_R = kpt1.ut_nR[n - kpt1.na].conj()
            with self.context.timer('paw'):
                C1_aGi = pawcorr.multiply(kpt1.P_ani, band=n - kpt1.na)
                n_nmG[j] = cpd(ut1cc_R, C1_aGi, kpt2, qpd, Q_G, block=block)

        return n_nmG

    @timer('get_optical_pair_density_head')
    def get_optical_pair_density_head(self, qpd, kptpair, n_n, m_m,
                                      block=False):
        """Get the optical limit of the pair density head (G=0) for a k-pair.
        """
        assert np.allclose(qpd.q_c, 0.0), f"{qpd.q_c} is not the optical limit"

        kpt1 = kptpair.kpt1
        kpt2 = kptpair.kpt2

        # v = (x, y, z)
        n_nmv = np.zeros((len(n_n), len(m_m), 3), qpd.dtype)

        for j, n in enumerate(n_n):
            n_nmv[j] = self.calculate_optical_pair_density_head(n, m_m,
                                                                kpt1, kpt2,
                                                                block=block)

        return n_nmv

    @timer('Calculate pair-densities')
    def calculate_pair_density(self, ut1cc_R, C1_aGi, kpt2, qpd, Q_G,
                               block=True):
        """Calculate FFT of pair-densities and add PAW corrections.

        ut1cc_R: 3-d complex ndarray
            Complex conjugate of the periodic part of the left hand side
            wave function.
        C1_aGi: list of ndarrays
            PAW corrections for all atoms.
        kpt2: KPoint object
            Right hand side k-point object.
        qpd: SingleQPWDescriptor
            Plane-wave descriptor for q=k2-k1.
        Q_G: 1-d int ndarray
            Mapping from flattened 3-d FFT grid to 0.5(G+q)^2<ecut sphere.
        """
        dv = qpd.gd.dv
        n_mG = qpd.empty(kpt2.blocksize)
        myblocksize = kpt2.nb - kpt2.na

        for ut_R, n_G in zip(kpt2.ut_nR, n_mG):
            n_R = ut1cc_R * ut_R
            with self.context.timer('fft'):
                n_G[:] = qpd.fft(n_R, 0, Q_G) * dv
        # PAW corrections:
        with self.context.timer('gemm'):
            for C1_Gi, P2_mi in zip(C1_aGi, kpt2.P_ani):
                # gemm(1.0, C1_Gi, P2_mi, 1.0, n_mG[:myblocksize], 't')
                mmm(1.0, P2_mi, 'N', C1_Gi, 'T', 1.0, n_mG[:myblocksize])

        if not block or self.blockcomm.size == 1:
            return n_mG
        else:
            n_MG = qpd.empty(kpt2.blocksize * self.blockcomm.size)
            with self.context.timer('all_gather'):
                self.blockcomm.all_gather(n_mG, n_MG)
            return n_MG[:kpt2.n2 - kpt2.n1]

    @timer('Optical limit')
    def calculate_optical_pair_velocity(self, n, kpt1, kpt2, block=False):
        # This has the effect of caching at most one kpoint.
        # This caching will be efficient only if we are looping over kpoints
        # in a particular way.
        #
        # It would be better to refactor so this caching is handled explicitly
        # by the caller providing the right thing.
        #
        # See https://gitlab.com/gpaw/gpaw/-/issues/625
        if self.ut_sKnvR is None or kpt1.K not in self.ut_sKnvR[kpt1.s]:
            self.ut_sKnvR = self.calculate_derivatives(kpt1)

        gd = self.gs.gd
        k_v = 2 * np.pi * np.dot(kpt1.k_c, np.linalg.inv(gd.cell_cv).T)

        ut_vR = self.ut_sKnvR[kpt1.s][kpt1.K][n - kpt1.n1]
        atomdata_a = self.gs.pawdatasets.by_atom
        C_avi = [np.dot(atomdata.nabla_iiv.T, P_ni[n - kpt1.na])
                 for atomdata, P_ni in zip(atomdata_a, kpt1.P_ani)]

        blockbands = kpt2.nb - kpt2.na
        n0_mv = np.empty((kpt2.blocksize, 3), dtype=complex)
        nt_m = np.empty(kpt2.blocksize, dtype=complex)
        n0_mv[:blockbands] = -self.gs.gd.integrate(ut_vR,
                                                   kpt2.ut_nR).T
        nt_m[:blockbands] = self.gs.gd.integrate(kpt1.ut_nR[n - kpt1.na],
                                                 kpt2.ut_nR)

        n0_mv[:blockbands] += (1j * nt_m[:blockbands, np.newaxis] *
                               k_v[np.newaxis, :])

        for C_vi, P_mi in zip(C_avi, kpt2.P_ani):
            # gemm(1.0, C_vi, P_mi, 1.0, n0_mv[:blockbands], 'c')
            mmm(1.0, P_mi, 'N', C_vi, 'C', 1.0, n0_mv[:blockbands])

        if block and self.blockcomm.size > 1:
            n0_Mv = np.empty((kpt2.blocksize * self.blockcomm.size, 3),
                             dtype=complex)
            with self.context.timer('all_gather optical'):
                self.blockcomm.all_gather(n0_mv, n0_Mv)
            n0_mv = n0_Mv[:kpt2.n2 - kpt2.n1]

        return -1j * n0_mv

    def calculate_optical_pair_density_head(self, n, m_m, kpt1, kpt2,
                                            block=False):
        # Numerical threshold for the optical limit k dot p perturbation
        # theory expansion:
        threshold = 1

        eps1 = kpt1.eps_n[n - kpt1.n1]
        deps_m = (eps1 - kpt2.eps_n)[m_m - kpt2.n1]
        n0_mv = self.calculate_optical_pair_velocity(n, kpt1, kpt2,
                                                     block=block)

        deps_m = deps_m.copy()
        deps_m[deps_m == 0.0] = np.inf

        smallness_mv = np.abs(-1e-3 * n0_mv / deps_m[:, np.newaxis])
        inds_mv = (np.logical_and(np.inf > smallness_mv,
                                  smallness_mv > threshold))
        n0_mv *= - 1 / deps_m[:, np.newaxis]
        n0_mv[inds_mv] = 0

        return n0_mv

    @timer('Intraband')
    def intraband_pair_density(self, kpt, n_n):
        """Calculate intraband matrix elements of nabla"""
        # Bands and check for block parallelization
        na, nb, n1 = kpt.na, kpt.nb, kpt.n1
        vel_nv = np.zeros((nb - na, 3), dtype=complex)
        assert np.max(n_n) < nb, 'This is too many bands'
        assert np.min(n_n) >= na, 'This is too few bands'

        # Load kpoints
        gd = self.gs.gd
        k_v = 2 * np.pi * np.dot(kpt.k_c, np.linalg.inv(gd.cell_cv).T)
        atomdata_a = self.gs.pawdatasets.by_atom

        # Break bands into degenerate chunks
        degchunks_cn = []  # indexing c as chunk number
        for n in n_n:
            inds_n = np.nonzero(np.abs(kpt.eps_n[n - n1] -
                                       kpt.eps_n) < 1e-5)[0] + n1

            # Has this chunk already been computed?
            oldchunk = any([n in chunk for chunk in degchunks_cn])
            if not oldchunk:
                if not all([ind in n_n for ind in inds_n]):
                    raise RuntimeError(
                        'You are cutting over a degenerate band '
                        'using block parallelization.')
                degchunks_cn.append(inds_n)

        # Calculate matrix elements by diagonalizing each block
        for ind_n in degchunks_cn:
            deg = len(ind_n)
            ut_nvR = self.gs.gd.zeros((deg, 3), complex)
            vel_nnv = np.zeros((deg, deg, 3), dtype=complex)
            # States are included starting from kpt.na
            ut_nR = kpt.ut_nR[ind_n - na]

            # Get derivatives
            for ind, ut_vR in zip(ind_n, ut_nvR):
                ut_vR[:] = self.make_derivative(kpt.s, kpt.K,
                                                ind, ind + 1)[0]

            # Treat the whole degenerate chunk
            for n in range(deg):
                ut_vR = ut_nvR[n]
                C_avi = [np.dot(atomdata.nabla_iiv.T, P_ni[ind_n[n] - na])
                         for atomdata, P_ni in zip(atomdata_a, kpt.P_ani)]

                nabla0_nv = -self.gs.gd.integrate(ut_vR, ut_nR).T
                nt_n = self.gs.gd.integrate(ut_nR[n], ut_nR)
                nabla0_nv += 1j * nt_n[:, np.newaxis] * k_v[np.newaxis, :]

                for C_vi, P_ni in zip(C_avi, kpt.P_ani):
                    # gemm(1.0, C_vi, P_ni[ind_n - na], 1.0, nabla0_nv, 'c')
                    mmm(1.0, P_ni[ind_n - na], 'N', C_vi, 'C', 1.0, nabla0_nv)

                vel_nnv[n] = -1j * nabla0_nv

            for iv in range(3):
                vel, _ = np.linalg.eig(vel_nnv[..., iv])
                vel_nv[ind_n - na, iv] = vel  # Use eigenvalues

        return vel_nv[n_n - na]

    def calculate_derivatives(self, kpt):
        ut_sKnvR = [{}, {}]
        ut_nvR = self.make_derivative(kpt.s, kpt.K, kpt.n1, kpt.n2)
        ut_sKnvR[kpt.s][kpt.K] = ut_nvR

        return ut_sKnvR

    @timer('Derivatives')
    def make_derivative(self, s, K, n1, n2):
        gs = self.gs
        U_cc = gs.ibz2bz[K].U_cc
        A_cv = gs.gd.cell_cv
        M_vv = np.dot(np.dot(A_cv.T, U_cc.T), np.linalg.inv(A_cv).T)
        ik = gs.kd.bz2ibz_k[K]
        assert gs.kd.comm.size == 1
        kpt = gs.kpt_qs[ik][s]
        psit_nG = kpt.psit_nG
        iG_Gv = 1j * gs.pd.get_reciprocal_vectors(q=ik, add_q=False)
        ut_nvR = gs.gd.zeros((n2 - n1, 3), complex)
        for n in range(n1, n2):
            for v in range(3):
                ut_R = gs.ibz2bz[K].map_pseudo_wave(
                    gs.pd.ifft(iG_Gv[:, v] * psit_nG[n], ik))
                for v2 in range(3):
                    ut_nvR[n - n1, v2] += ut_R * M_vv[v, v2]

        return ut_nvR


def phase_shifted_fft_indices(k1_c, k2_c, qpd, coordinate_transformation=None):
    """Get phase shifted FFT indices for G-vectors inside the cutoff sphere.

    The output 1D FFT indices Q_G can be used to extract the plane-wave
    components G of the phase shifted Fourier transform

    n_kk'(G+q) = FFT_G[e^(-i[k+q-k']r) n_kk'(r)]

    where n_kk'(r) is some lattice periodic function and the wave vector
    difference k + q - k' is commensurate with the reciprocal lattice.
    """
    N_c = qpd.gd.N_c
    Q_G = qpd.Q_qG[0]
    q_c = qpd.q_c
    if coordinate_transformation:
        q_c = coordinate_transformation(q_c)

    shift_c = k1_c + q_c - k2_c
    assert np.allclose(shift_c.round(), shift_c)
    shift_c = shift_c.round().astype(int)

    if shift_c.any() or coordinate_transformation:
        # Get the 3D FFT grid indices (relative reciprocal space coordinates)
        # of the G-vectors inside the cutoff sphere
        i_cG = np.unravel_index(Q_G, N_c)
        if coordinate_transformation:
            i_cG = coordinate_transformation(i_cG)
        # Shift the 3D FFT grid indices to account for the Bloch-phase shift
        # e^(-i[k+q-k']r)
        i_cG += shift_c[:, np.newaxis]
        # Transform back the FFT grid to 1D FFT indices
        Q_G = np.ravel_multi_index(i_cG, N_c, 'wrap')

    return Q_G


def get_gs_and_context(calc, txt, world, timer):
    """Interface to initialize gs and context from old input arguments.
    Should be phased out in the future!"""
    from gpaw.calculator import GPAW as OldGPAW
    from gpaw.new.ase_interface import ASECalculator as NewGPAW

    context = ResponseContext(txt=txt, timer=timer, comm=world)

    if isinstance(calc, (OldGPAW, NewGPAW)):
        assert calc.wfs.world.size == 1
        gs = calc.gs_adapter()
    else:
        gs = ResponseGroundStateAdapter.from_gpw_file(gpw=calc)

    return gs, context
