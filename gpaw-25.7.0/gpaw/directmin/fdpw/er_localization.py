"""
Potentials for orbital density dependent energy functionals
"""

import numpy as np
from gpaw.utilities import pack_density, unpack_hermitian
from gpaw.lfc import LFC
from gpaw.transformers import Transformer
from gpaw.directmin.tools import d_matrix, get_n_occ
from gpaw.poisson import PoissonSolver
import gpaw.cgpaw as cgpaw


class ERLocalization:

    """
    Edmiston-Ruedenberg localisation functions.

    This class return -1/2 sum_i f_i (2integrals rr') n_i n_i'/ (r-r')
    and it is gradients

    """
    def __init__(self, wfs, dens,
                 sic_coarse_grid=True,
                 poisson_solver='FPS'):

        self.name = 'ER_SIC'
        # what we need from wfs
        self.setups = wfs.setups
        spos_ac = wfs.spos_ac
        self.cgd = wfs.gd

        # what we need from dens
        self.finegd = dens.finegd
        self.sic_coarse_grid = sic_coarse_grid

        if self.sic_coarse_grid:
            self.ghat = LFC(self.cgd,
                            [setup.ghat_l for setup
                             in self.setups],
                            integral=np.sqrt(4 * np.pi),
                            forces=True)
            self.ghat.set_positions(spos_ac)
        else:
            self.ghat = dens.ghat  # we usually solve poiss. on finegd

        if poisson_solver == 'FPS':
            self.poiss = PoissonSolver(use_charge_center=True,
                                       use_charged_periodic_corrections=True)
        elif poisson_solver == 'GS':
            self.poiss = PoissonSolver(name='fd',
                                       relax=poisson_solver,
                                       eps=1.0e-16,
                                       use_charge_center=True,
                                       use_charged_periodic_corrections=True)

        if self.sic_coarse_grid is True:
            self.poiss.set_grid_descriptor(self.cgd)
        else:
            self.poiss.set_grid_descriptor(self.finegd)

        self.interpolator = Transformer(self.cgd, self.finegd, 3)
        self.restrictor = Transformer(self.finegd, self.cgd, 3)
        self.dtype = wfs.dtype
        self.eigv_s = {}
        self.lagr_diag_s = {}
        self.e_sic_by_orbitals = {}
        self.counter = 0  # number of calls of this class

        self.n_kps = wfs.kd.nibzkpts

    def get_orbdens_compcharge_dm_kpt(self, wfs, kpt, n):

        if wfs.mode == 'pw':
            nt_G = wfs.pd.gd.zeros(global_array=True)
            psit_G = wfs.pd.alltoall1(kpt.psit.array[n:n + 1], kpt.q)
            if psit_G is not None:
                psit_R = wfs.pd.ifft(psit_G, kpt.q,
                                     local=True, safe=False)
                cgpaw.add_to_density(1.0, psit_R, nt_G)
            wfs.pd.gd.comm.sum(nt_G)
            nt_G = wfs.pd.gd.distribute(nt_G)
        else:
            nt_G = np.absolute(kpt.psit_nG[n] ** 2)

        # paw
        Q_aL = {}
        D_ap = {}
        for a, P_ni in kpt.P_ani.items():
            P_i = P_ni[n]
            D_ii = np.outer(P_i, P_i.conj()).real
            D_ap[a] = D_p = pack_density(D_ii)
            Q_aL[a] = np.dot(D_p, self.setups[a].Delta_pL)

        return nt_G, Q_aL, D_ap

    def get_energy_and_gradients_kpt(self, wfs, kpt, grad_knG):

        wfs.timer.start('SIC e/g grid calculations')
        k = self.n_kps * kpt.s + kpt.q
        n_occ = get_n_occ(kpt)[0]

        e_total_sic = np.array([])

        for i in range(n_occ):
            nt_G, Q_aL, D_ap = \
                self.get_orbdens_compcharge_dm_kpt(wfs, kpt, i)

            # calculate - SI Hartree
            e_sic, v_ht_g = \
                self.get_pseudo_pot(nt_G, Q_aL, i, kpoint=k)

            # calculate PAW corrections
            e_sic_paw_m, dH_ap = \
                self.get_paw_corrections(D_ap, v_ht_g)

            if wfs.mode == 'pw':
                v_ht_g = wfs.pd.gd.collect(v_ht_g, broadcast=True)
                Q_G = wfs.pd.Q_qG[kpt.q]
                psit_G = wfs.pd.alltoall1(kpt.psit_nG[i: i + 1], kpt.q)
                if psit_G is not None:
                    psit_R = wfs.pd.ifft(
                        psit_G, kpt.q, local=True, safe=False)
                    psit_R *= v_ht_g
                    wfs.pd.fftplan.execute()
                    vtpsit_G = wfs.pd.tmp_Q.ravel()[Q_G]
                else:
                    vtpsit_G = wfs.pd.tmp_G
                wfs.pd.alltoall2(vtpsit_G, kpt.q, grad_knG[k][i: i + 1])
                grad_knG[k][i] *= kpt.f_n[i]
            else:
                grad_knG[k][i] += kpt.psit_nG[i] * v_ht_g * kpt.f_n[i]

            # total sic:
            e_sic += e_sic_paw_m
            c_axi = {}
            for a in kpt.P_ani.keys():
                dH_ii = unpack_hermitian(dH_ap[a])
                c_xi = np.dot(kpt.P_ani[a][i], dH_ii)
                c_axi[a] = c_xi * kpt.f_n[i]
            # add projectors to
            wfs.pt.add(grad_knG[k][i], c_axi, kpt.q)

            e_total_sic = np.append(e_total_sic,
                                    kpt.f_n[i] * e_sic, axis=0)

        self.e_sic_by_orbitals[k] = np.copy(e_total_sic)
        wfs.timer.stop('SIC e/g grid calculations')

        return e_total_sic.sum()

    def get_pseudo_pot(self, nt, Q_aL, i, kpoint=None):

        if self.sic_coarse_grid is False:
            v_ht_g = self.finegd.zeros()
            nt_sg = self.finegd.zeros()
        else:
            v_ht_g = self.cgd.zeros()
            nt_sg = self.cgd.zeros()

        if self.sic_coarse_grid is False:
            self.interpolator.apply(nt, nt_sg)
            nt_sg *= self.cgd.integrate(nt) / \
                self.finegd.integrate(nt_sg)
        else:
            nt_sg = nt

        self.ghat.add(nt_sg, Q_aL)

        self.poiss.solve(v_ht_g, nt_sg, zero_initial_phi=False)

        if self.sic_coarse_grid is False:
            ec = 0.5 * self.finegd.integrate(nt_sg * v_ht_g)
        else:
            ec = 0.5 * self.cgd.integrate(nt_sg * v_ht_g)

        return np.array([-ec]), -1.0 * v_ht_g

    def get_paw_corrections(self, D_ap, vHt_g):

        # Hartree-PAW
        dH_ap = {}
        for a, D_p in D_ap.items():
            dH_ap[a] = np.zeros(len(D_p))

        ec = 0.0
        W_aL = self.ghat.dict()
        self.ghat.integrate(vHt_g, W_aL)
        for a, D_p in D_ap.items():
            setup = self.setups[a]
            M_p = np.dot(setup.M_pp, D_p)
            ec += np.dot(D_p, M_p)
            dH_ap[a] += (2.0 * M_p + np.dot(setup.Delta_pL, W_aL[a]))
        if self.sic_coarse_grid is False:
            ec = self.finegd.comm.sum_scalar(ec)
        else:
            ec = self.cgd.comm.sum_scalar(ec)

        return np.array([-ec]), dH_ap

    def get_energy_and_gradients_inner_loop(
            self, wfs, kpt, a_mat, evals, evec):

        e_sic, l_odd = \
            self.get_energy_and_hamiltonian_kpt(wfs, kpt)
        wfs.timer.start('Unitary gradients')
        f = np.ones(l_odd.shape[0])

        # calc_error:
        indz = np.absolute(l_odd) > 1.0e-4
        l_c = 2.0 * l_odd[indz]
        l_odd = f[:, np.newaxis] * l_odd.T.conj() - f * l_odd
        kappa = np.max(np.absolute(l_odd[indz]) / np.absolute(l_c))

        if a_mat is None:
            wfs.timer.stop('Unitary gradients')
            return l_odd.T.conj(), e_sic, kappa
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

    def get_energy_and_hamiltonian_kpt(self, wfs, kpt):

        n_occ = get_n_occ(kpt)[0]
        k = self.n_kps * kpt.s + kpt.q
        grad = {k: np.zeros_like(kpt.psit_nG[:n_occ])}

        e_sic = self.get_energy_and_gradients_kpt(wfs, kpt, grad)

        l_odd = wfs.integrate(kpt.psit_nG[:n_occ],
                              grad[k][:n_occ], True)

        return e_sic, l_odd
