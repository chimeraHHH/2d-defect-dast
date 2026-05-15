"""
Helper class for LCAOETDM.

Handles orbital initialization, setting reference orbitals, applying the
unitary transformation, calculating the gradient, and getting canonical
representation.
"""

import numpy as np
from gpaw.lcao.eigensolver import DirectLCAO
from gpaw.directmin.functional.lcao import get_functional
from gpaw.utilities.tools import tri2full


class ETDMHelperLCAO(DirectLCAO):

    def __init__(self, wfs, dens, ham, nkpts, functional, diagonalizer=None,
                 orthonormalization='gramschmidt',
                 need_init_orbs=True):

        super(ETDMHelperLCAO, self).__init__(diagonalizer)
        super(ETDMHelperLCAO, self).initialize(wfs.gd, wfs.dtype,
                                               wfs.setups.nao, wfs.ksl)
        self.orthonormalization = orthonormalization
        self.need_init_orbs = need_init_orbs
        self.nkpts = nkpts
        self.func = get_functional(functional, wfs, dens, ham)
        self.reference_orbitals = {}
        self.initialize_orbitals(wfs, ham)

    def __repr__(self):
        pass

    def set_reference_orbitals(self, wfs, n_dim):
        for kpt in wfs.kpt_u:
            u = self.kpointval(kpt)
            self.reference_orbitals[u] = np.copy(kpt.C_nM[:n_dim[u]])

    def appy_transformation_kpt(self, wfs, u_mat, kpt, c_ref=None,
                                broadcast=True,
                                update_proj=True):
        """
        If c_ref are not provided then
        kpt.C_nM <- u_mat kpt.C_nM
        otherwise kpt.C_nM <- u_mat c_ref
        """

        dimens1 = u_mat.shape[1]
        dimens2 = u_mat.shape[0]

        if c_ref is None:
            kpt.C_nM[:dimens2] = u_mat @ kpt.C_nM[:dimens1]
        else:
            kpt.C_nM[:dimens2] = u_mat @ c_ref[:dimens1]

        if broadcast:
            with wfs.timer('Broadcast coefficients'):
                wfs.gd.comm.broadcast(kpt.C_nM, 0)
        if update_proj:
            with wfs.timer('Calculate projections'):
                wfs.atomic_correction.calculate_projections(wfs, kpt)

    def initialize_orbitals(self, wfs, ham):
        """
        If it is the first use of the scf then initialize
        coefficient matrix using eigensolver
        """

        orthname = self.orthonormalization
        need_canon_coef = \
            (not wfs.coefficients_read_from_file and self.need_init_orbs)
        if need_canon_coef or orthname == 'diag':
            super(ETDMHelperLCAO, self).iterate(ham, wfs)
        else:
            wfs.orthonormalize(type=orthname)
        wfs.coefficients_read_from_file = False
        self.need_init_orbs = False

    def calc_grad(self, wfs, ham, kpt, evecs, evals, matrix_exp,
                  representation, ind_up, constraints):
        """
        Gradient w.r.t. skew-Hermitian matrices
        """

        h_mm = self.calculate_hamiltonian_matrix(ham, wfs, kpt)
        # make matrix hermitian
        tri2full(h_mm)
        # calc gradient and eigenstate error
        g_mat, error = self.func.get_gradients(
            h_mm, kpt.C_nM, kpt.f_n, evecs, evals,
            kpt, wfs, wfs.timer, matrix_exp,
            representation, ind_up, constraints)

        return g_mat, error

    def update_to_canonical_orbitals(self, wfs, ham, kpt,
                                     update_ref_orbs_canonical, restart):
        """
        Choose canonical orbitals
        """

        h_mm = self.calculate_hamiltonian_matrix(ham, wfs, kpt)
        tri2full(h_mm)

        if self.func.name == 'ks':
            if update_ref_orbs_canonical or restart:
                # Diagonalize entire Hamiltonian matrix
                with wfs.timer('Diagonalize and rotate'):
                    kpt.C_nM, kpt.eps_n = rotate_subspace(h_mm, kpt.C_nM)
            else:
                # Diagonalize equally occupied subspaces separately
                f_unique = np.unique(kpt.f_n)
                for f in f_unique:
                    with wfs.timer('Diagonalize and rotate'):
                        kpt.C_nM[kpt.f_n == f, :], kpt.eps_n[kpt.f_n == f] = \
                            rotate_subspace(h_mm, kpt.C_nM[kpt.f_n == f, :])
        elif self.func.name == 'PZ-SIC':
            self.func.get_lagrange_matrices(
                h_mm, kpt.C_nM, kpt.f_n, kpt, wfs,
                update_eigenvalues=True)

        with wfs.timer('Calculate projections'):
            self.update_projections(wfs, kpt)

    def sort_orbitals(self, wfs, kpt, ind):
        """
        sort orbitals according to indices stored in ind
        """
        kpt.C_nM[np.arange(len(ind)), :] = kpt.C_nM[ind, :]
        self.update_projections(wfs, kpt)

    def update_projections(self, wfs, kpt):
        """
        calculate projections kpt.P_ani
        """

        wfs.atomic_correction.calculate_projections(wfs, kpt)

    def orbital_energies(self, wfs, ham, kpt):
        """
        diagonal elements of hamiltonian matrix in orbital representation
        """

        if self.func.name == 'ks':
            h_mm = self.calculate_hamiltonian_matrix(ham, wfs, kpt)
            tri2full(h_mm)
            # you probably need only diagonal terms?
            # wouldn't "for" be faster?
            h_mm = kpt.C_nM.conj() @ h_mm.conj() @ kpt.C_nM.T
            energies = h_mm.diagonal().real.copy()
        elif self.func.name == 'PZ-SIC':
            energies = self.func.lagr_diag_s[wfs.eigensolver.kpointval(kpt)]

        return energies

    def kpointval(self, kpt):
        return self.nkpts * kpt.s + kpt.q


def rotate_subspace(h_mm, c_nm):
    """
    choose canonical orbitals
    """
    l_nn = c_nm @ h_mm @ c_nm.conj().T
    # check if diagonal then don't rotate? it could save a bit of time
    eps, w = np.linalg.eigh(l_nn)
    return w.T.conj() @ c_nm, eps
