"""
Potentials for orbital density dependent energy functionals
"""

import numpy as np

from gpaw.directmin.tools import d_matrix
from gpaw.utilities import unpack_hermitian


class KSFDPW:
    """
    KS-DFT
    """
    def __init__(self, wfs, dens, ham):

        self.name = 'Zero'
        self.n_kps = wfs.kd.nibzkpts
        self.grad = {}
        self.total_sic = 0.0
        self.eks = 0.0
        self.changedocc = 0
        self.dens = dens
        self.ham = ham

    def get_energy_and_gradients_kpt(
            self, wfs, kpt, grad_knG=None, U_k=None, add_grad=False, ham=None):

        k = self.n_kps * kpt.s + kpt.q
        nbands = wfs.bd.nbands
        if U_k is not None:
            assert U_k[k].shape[0] == nbands

        wfs.timer.start('e/g grid calculations')
        self.grad[k] = wfs.empty(nbands, q=kpt.q)
        wfs.apply_pseudo_hamiltonian(kpt, ham, kpt.psit_nG, self.grad[k])

        c_axi = {}
        for a, P_xi in kpt.P_ani.items():
            try:
                dH_ii = unpack_hermitian(ham.dH_asp[a][kpt.s])
            except AttributeError:
                dH_ii = ham.potential.dH_asii[a][kpt.s]
            c_xi = np.dot(P_xi, dH_ii)
            c_axi[a] = c_xi

        # not sure about this:
        ham.xc.add_correction(kpt, kpt.psit_nG, self.grad[k],
                              kpt.P_ani, c_axi, n_x=None,
                              calculate_change=False)
        # add projectors to the H|psi_i>
        wfs.pt.add(self.grad[k], c_axi, kpt.q)
        # scale with occupation numbers
        for i, f in enumerate(kpt.f_n):
            self.grad[k][i] *= f

        if add_grad:
            if U_k is not None:
                grad_knG[k] += \
                    np.tensordot(U_k[k].conj(), self.grad[k], axes=1)
            else:
                grad_knG[k] += self.grad[k]
        else:
            if U_k is not None:
                self.grad[k][:] = np.tensordot(U_k[k].conj(),
                                               self.grad[k], axes=1)

        wfs.timer.stop('e/g grid calculations')

        return 0.0

    def get_energy_and_gradients_inner_loop(
            self, wfs, kpt, a_mat, evals, evec, ham=None, exstate=True):

        if not exstate:
            raise RuntimeError('Attempting to optimize unitary-invariant '
                               'energy in occupied-occupied rotation space.')

        ndim = wfs.bd.nbands

        k = self.n_kps * kpt.s + kpt.q
        self.grad[k] = np.zeros_like(kpt.psit_nG[:ndim])
        e_sic = self.get_energy_and_gradients_kpt(wfs, kpt, ham=ham)
        wfs.timer.start('Unitary gradients')
        l_odd = wfs.integrate(kpt.psit_nG[:ndim], self.grad[k][:ndim], True)
        f = np.ones(ndim)
        indz = np.absolute(l_odd) > 1.0e-4
        l_c = 2.0 * l_odd[indz]
        l_odd = f[:, np.newaxis] * l_odd.T.conj() - f * l_odd
        kappa = np.max(np.absolute(l_odd[indz]) / np.absolute(l_c))

        if a_mat is None:
            wfs.timer.stop('Unitary gradients')
            return 2.0 * l_odd.T.conj(), e_sic, kappa
        else:
            g_mat = evec.T.conj() @ l_odd.T.conj() @ evec
            g_mat = g_mat * d_matrix(evals)
            g_mat = evec @ g_mat @ evec.T.conj()
            for i in range(g_mat.shape[0]):
                g_mat[i][i] *= 0.5
            wfs.timer.stop('Unitary gradients')
            if a_mat.dtype == float:
                g_mat = g_mat.real
            return 2.0 * g_mat, e_sic, kappa
