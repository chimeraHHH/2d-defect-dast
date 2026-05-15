from functools import partial

import numpy as np
from ase.utils.timing import timer
from gpaw import debug
from gpaw.eigensolvers.diagonalizerbackend import (ScalapackDiagonalizer,
                                                   ElpaDiagonalizer,
                                                   ScipyDiagonalizer)
from gpaw.eigensolvers.eigensolver import Eigensolver
from gpaw.hybrids import HybridXC
from gpaw.matrix import matrix_matrix_multiply as mmm


class DummyArray:
    def __getitem__(self, x):
        return np.empty((0, 0))


class Davidson(Eigensolver):
    """Simple Davidson eigensolver

    It is expected that the trial wave functions are orthonormal
    and the integrals of projector functions and wave functions
    ``nucleus.P_uni`` are already calculated.

    Solution steps are:

    * Subspace diagonalization
    * Calculate all residuals
    * Add preconditioned residuals to the subspace and diagonalize
    """

    def __init__(
            self, niter=2):
        super().__init__()
        self.niter = niter
        self.diagonalizer_backend = None

        self.orthonormalization_required = False
        self.H_NN = DummyArray()
        self.S_NN = DummyArray()
        self.eps_N = DummyArray()

    def __repr__(self):
        return f'Davidson(niter={self.niter})'

    def todict(self):
        return {'name': 'davidson', 'niter': self.niter}

    def initialize(self, wfs, dist_backend='scalapack'):
        # dist_backend keyword exists due to this class having
        # no reference to the parallelization backend. As such,
        # the keyword must be specified by the user upon creation
        # of this object. Usually one would want this to be ELPA
        # if use_elpa in parallel is set to True.
        dist_diagonalizers = {
            'scalapack': ScalapackDiagonalizer,
            'elpa': ElpaDiagonalizer
        }

        super().initialize(wfs)
        slcomm, nrows, ncols, slsize = wfs.scalapack_parameters

        if wfs.gd.comm.rank == 0 and wfs.bd.comm.rank == 0:
            # Allocate arrays
            B = self.nbands
            self.H_NN = np.zeros((2 * B, 2 * B), self.dtype)
            self.S_NN = np.zeros((2 * B, 2 * B), self.dtype)
            self.eps_N = np.zeros(2 * B)

        if slsize is not None:
            self.diagonalizer_backend = dist_diagonalizers[dist_backend](
                arraysize=self.nbands * 2,
                grid_nrows=nrows,
                grid_ncols=ncols,
                scalapack_communicator=slcomm,
                dtype=self.dtype,
                blocksize=slsize)
        else:
            self.diagonalizer_backend = ScipyDiagonalizer(slcomm)

    def estimate_memory(self, mem, wfs):
        super().estimate_memory(mem, wfs)
        nbands = wfs.bd.nbands
        mem.subnode('H_nn', nbands * nbands * mem.itemsize[wfs.dtype])
        mem.subnode('S_nn', nbands * nbands * mem.itemsize[wfs.dtype])
        mem.subnode('H_2n2n', 4 * nbands * nbands * mem.itemsize[wfs.dtype])
        mem.subnode('S_2n2n', 4 * nbands * nbands * mem.itemsize[wfs.dtype])
        mem.subnode('eps_2n', 2 * nbands * mem.floatsize)

    @timer('Davidson')
    def iterate_one_k_point(self, ham, wfs, kpt, weights):
        """Do Davidson iterations for the kpoint"""
        if isinstance(ham.xc, HybridXC):
            niter = 1
        else:
            niter = self.niter

        bd = wfs.bd
        B = bd.nbands

        H_NN = self.H_NN
        S_NN = self.S_NN
        eps_N = self.eps_N

        def integrate(a_G):
            if wfs.collinear:
                return np.real(wfs.integrate(a_G, a_G, global_integral=False))
            return sum(
                np.real(wfs.integrate(b_G, b_G, global_integral=False))
                for b_G in a_G)

        self.subspace_diagonalize(ham, wfs, kpt)

        psit = kpt.psit
        psit2 = psit.new(buf=wfs.work_array)
        P = kpt.projections
        P2 = P.new()
        P3 = P.new()
        M = wfs.work_matrix_nn
        dS = wfs.setups.dS
        comm = wfs.gd.comm

        is_gridband_master = (comm.rank == 0) and (bd.comm.rank == 0)

        if bd.comm.size > 1:
            M0 = M.new(dist=(bd.comm, 1, 1))
        else:
            M0 = M

        if comm.rank == 0:
            e_N = bd.collect(kpt.eps_n)
            if e_N is not None:
                eps_N[:B] = e_N

        Ht = partial(wfs.apply_pseudo_hamiltonian, kpt, ham)

        if self.keep_htpsit:
            R = psit.new(buf=self.Htpsit_nG)
        else:
            R = psit.apply(Ht)

        self.calculate_residuals(kpt, wfs, ham, psit, P, kpt.eps_n, R, P2)

        precond = self.preconditioner

        for nit in range(niter):
            if nit == niter - 1:
                error = np.dot(weights, [integrate(R_G) for R_G in R.array])

            for psit_G, R_G, psit2_G in zip(psit.array, R.array, psit2.array):
                ekin = precond.calculate_kinetic_energy(psit_G, kpt)
                precond(R_G, kpt, ekin, out=psit2_G)

            # Calculate projections
            psit2.matrix_elements(wfs.pt, out=P2)

            def copy(M, C_nn):
                comm.sum(M.array, 0)
                if comm.rank == 0:
                    M.complex_conjugate()
                    M.redist(M0)
                    if bd.comm.rank == 0:
                        C_nn[:] = M0.array

            with self.timer('calc. matrices'):
                # <psi2 | H | psi2>
                psit2.matrix_elements(
                    operator=Ht, result=R, out=M, symmetric=True, cc=True)
                ham.dH(P2, out=P3)
                mmm(1.0, P2, 'N', P3, 'C', 1.0, M)  # , symmetric=True)
                copy(M, H_NN[B:, B:])

                # <psi2 | H | psi>
                R.matrix_elements(psit, out=M, cc=True)
                mmm(1.0, P3, 'N', P, 'C', 1.0, M)
                copy(M, H_NN[B:, :B])

                # <psi2 | S | psi2>
                psit2.matrix_elements(out=M, symmetric=True, cc=True)
                dS.apply(P2, out=P3)
                mmm(1.0, P2, 'N', P3, 'C', 1.0, M)
                copy(M, S_NN[B:, B:])

                # <psi2 | S | psi>
                psit2.matrix_elements(psit, out=M, cc=True)
                mmm(1.0, P3, 'N', P, 'C', 1.0, M)
                copy(M, S_NN[B:, :B])

            if is_gridband_master:
                H_NN[:B, :B] = np.diag(eps_N[:B])
                S_NN[:B, :B] = np.eye(B)

            with self.timer('diagonalize'):
                if is_gridband_master and debug:
                    H_NN[np.triu_indices(2 * B, 1)] = 42.0
                    S_NN[np.triu_indices(2 * B, 1)] = 42.0

                try:
                    self.diagonalizer_backend.diagonalize(
                        H_NN, S_NN, eps_N, debug=debug)
                except np.linalg.LinAlgError as ex:
                    raise ValueError(
                        'Too few plane waves or grid points') from ex
            if comm.rank == 0:
                bd.distribute(eps_N[:B], kpt.eps_n)
            comm.broadcast(kpt.eps_n, 0)

            with self.timer('rotate_psi'):
                if comm.rank == 0:
                    if bd.comm.rank == 0:
                        M0.array[:] = H_NN[:B, :B].T
                    M0.redist(M)
                comm.broadcast(M.array, 0)
                mmm(1.0, M, 'N', psit, 'N', 0.0, R)
                mmm(1.0, M, 'N', P, 'N', 0.0, P3)
                if comm.rank == 0:
                    if bd.comm.rank == 0:
                        M0.array[:] = H_NN[B:, :B].T
                    M0.redist(M)
                comm.broadcast(M.array, 0)
                mmm(1.0, M, 'N', psit2, 'N', 1.0, R)
                mmm(1.0, M, 'N', P2, 'N', 1.0, P3)
                psit[:] = R
                P, P3 = P3, P
                kpt.projections = P

            if nit < niter - 1:
                psit.apply(Ht, out=R)
                self.calculate_residuals(
                    kpt, wfs, ham, psit, P, kpt.eps_n, R, P2)

        error = wfs.gd.comm.sum_scalar(error)
        return error
