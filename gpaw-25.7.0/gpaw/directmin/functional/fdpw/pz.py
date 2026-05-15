"""
Potentials for orbital density dependent energy functionals
"""

import numpy as np

from gpaw.directmin.tools import get_n_occ, d_matrix
from gpaw.lfc import LFC
from gpaw.poisson import PoissonSolver
from gpaw.transformers import Transformer
from gpaw.utilities import pack_density, unpack_hermitian
from gpaw.utilities.ewald import madelung
from gpaw.utilities.partition import AtomPartition
import gpaw.cgpaw as cgpaw


class PZSICFDPW:

    """
    Perdew-Zunger self-interaction corrections

    """
    def __init__(self, wfs, dens, ham, scaling_factor=(1.0, 1.0),
                 sic_coarse_grid=True, store_potentials=False,
                 poisson_solver='FPS'):

        self.name = 'PZ-SIC'
        # what we need from wfs
        self.setups = wfs.setups
        spos_ac = wfs.spos_ac
        self.cgd = wfs.gd

        # what we need from dens
        self.finegd = dens.finegd
        self.sic_coarse_grid = sic_coarse_grid
        self.pd2 = None
        self.pd3 = None
        self.corr = None
        if wfs.mode == 'pw':
            from gpaw.wavefunctions.pw import PWLFC

            assert self.sic_coarse_grid
            self.pd2 = dens.pd2
            self.pd3 = dens.pd3
            self.ghat = PWLFC(
                [setup.ghat_l for setup in self.setups], self.pd2)
            rank_a = wfs.gd.get_ranks_from_positions(spos_ac)
            atom_partition = AtomPartition(wfs.gd.comm, rank_a, name='gd')
            self.ghat.set_positions(spos_ac, atom_partition)
            self.corr = madelung(wfs.gd.cell_cv)
            self.corr_q = 1.0
            for nc in self.pd2.gd.N_c:
                self.corr_q *= nc
            self.corr_q *= self.corr

            self.G2 = self.pd2.G2_qG[0].copy()
            if self.pd2.gd.comm.rank == 0:
                self.G2[0] = 1.0
        else:
            if self.sic_coarse_grid:
                self.ghat = LFC(
                    self.cgd, [setup.ghat_l for setup in self.setups],
                    integral=np.sqrt(4 * np.pi), forces=True)
                self.ghat.set_positions(spos_ac)
            else:
                self.ghat = dens.ghat  # we usually solve poiss. on finegd

        # what we need from ham
        self.xc = ham.xc

        if poisson_solver == 'FPS':
            self.poiss = PoissonSolver(use_charge_center=True,
                                       use_charged_periodic_corrections=True)
        elif poisson_solver == 'GS':
            self.poiss = PoissonSolver(
                name='fd', relax=poisson_solver, eps=1.0e-16,
                use_charge_center=True, use_charged_periodic_corrections=True)

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
        # Scaling factor:
        self.beta_c = scaling_factor[0]
        self.beta_x = scaling_factor[1]

        self.n_kps = wfs.kd.nibzkpts
        self.store_potentials = store_potentials
        self.grad = {}
        self.total_sic = 0.0

        if store_potentials:
            self.old_pot = {}
            for kpt in wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                n_occ = get_n_occ(kpt)[0]
                self.old_pot[k] = self.cgd.zeros(n_occ, dtype=float)

    def get_orbdens_compcharge_dm_kpt(self, wfs, kpt, n):

        if wfs.mode == 'pw':
            nt_G = wfs.pd.gd.zeros(global_array=True)
            psit_G = wfs.pd.alltoall1(kpt.psit.array[n: n + 1], kpt.q)
            if psit_G is not None:
                psit_R = wfs.pd.ifft(psit_G, kpt.q, local=True, safe=False)
                cgpaw.add_to_density(1.0, psit_R, nt_G)
            wfs.pd.gd.comm.sum(nt_G)
            nt_G = wfs.pd.gd.distribute(nt_G)
        else:
            nt_G = np.absolute(kpt.psit_nG[n]**2)

        # paw
        Q_aL = {}
        D_ap = {}
        for a, P_ni in kpt.P_ani.items():
            P_i = P_ni[n]
            D_ii = np.outer(P_i, P_i.conj()).real
            D_ap[a] = D_p = pack_density(D_ii)
            Q_aL[a] = np.dot(D_p, self.setups[a].Delta_pL)

        return nt_G, Q_aL, D_ap

    def get_energy_and_gradients_kpt(
            self, wfs, kpt, grad_knG=None, U_k=None, add_grad=False,
            ham=None, scalewithocc=True, exstate=False):

        k = self.n_kps * kpt.s + kpt.q
        n_occ = get_n_occ(kpt)[0]
        self.grad[k] = np.zeros_like(kpt.psit_nG) if exstate \
            else np.zeros_like(kpt.psit_nG[:n_occ])

        if exstate:
            self.get_gradient_ks_kpt(wfs, kpt, ham=ham)
        esic = self.get_esic_add_sic_gradient_kpt(
            wfs, kpt, grad_knG, U_k, add_grad, scalewithocc,
            exstate=exstate)

        return esic

    def get_gradient_ks_kpt(self, wfs, kpt, ham=None):

        k = self.n_kps * kpt.s + kpt.q
        wfs.timer.start('KS e/g grid calculations')
        wfs.apply_pseudo_hamiltonian(kpt, ham, kpt.psit_nG, self.grad[k])

        c_axi = {}
        for a, P_xi in kpt.P_ani.items():
            dH_ii = unpack_hermitian(ham.dH_asp[a][kpt.s])
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

        wfs.timer.stop('KS e/g grid calculations')

        return 0.0

    def get_esic_add_sic_gradient_kpt(
            self, wfs, kpt, grad_knG=None, U_k=None, add_grad=False,
            scalewithocc=True, exstate=False):

        wfs.timer.start('SIC e/g grid calculations')
        k = self.n_kps * kpt.s + kpt.q
        n_occ = get_n_occ(kpt)[0]
        e_total_sic = np.array([])

        for i in range(n_occ):
            if wfs.mode == 'pw':
                e_sic, vt_G, dH_ap = self.get_si_pot_dh_pw(
                    wfs, kpt, i, exstate=exstate)
            else:
                nt_G, Q_aL, D_ap = \
                    self.get_orbdens_compcharge_dm_kpt(wfs, kpt, i)
                e_sic, vt_G, dH_ap = \
                    self.get_pz_sic_ith_kpt(
                        nt_G, Q_aL, D_ap, i, k, wfs.timer)

            e_total_sic = np.append(e_total_sic,
                                    kpt.f_n[i] * e_sic, axis=0)
            if wfs.mode == 'pw':
                vt_G = wfs.pd.gd.collect(vt_G, broadcast=True)
                Q_G = wfs.pd.Q_qG[kpt.q]
                psit_G = wfs.pd.alltoall1(kpt.psit_nG[i: i + 1], kpt.q)
                if psit_G is not None:
                    psit_R = wfs.pd.ifft(
                        psit_G, kpt.q, local=True, safe=False)
                    psit_R *= vt_G
                    wfs.pd.fftplan.execute()
                    vtpsit_G = wfs.pd.tmp_Q.ravel()[Q_G]
                else:
                    vtpsit_G = wfs.pd.tmp_G
                tmp = np.zeros_like(self.grad[k][i: i + 1])
                wfs.pd.alltoall2(vtpsit_G, kpt.q, tmp)
                self.grad[k][i] += tmp[0]
                if scalewithocc:
                    self.grad[k][i] *= kpt.f_n[i]
            else:
                if scalewithocc:
                    self.grad[k][i] += kpt.psit_nG[i] * vt_G * kpt.f_n[i]
                else:
                    self.grad[k][i] += kpt.psit_nG[i] * vt_G
            c_axi = {}
            for a in kpt.P_ani.keys():
                dH_ii = unpack_hermitian(dH_ap[a])
                c_xi = np.dot(kpt.P_ani[a][i], dH_ii)
                c_axi[a] = c_xi * kpt.f_n[i]
            # add projectors to
            wfs.pt.add(self.grad[k][i], c_axi, kpt.q)

        if exstate:
            if U_k is not None:
                grad_knG[k][:] += np.tensordot(
                    U_k[k].conj(), self.grad[k], axes=1)
            if add_grad:
                grad_knG[k][:] += self.grad[k]
        else:
            if add_grad:
                if U_k is not None:
                    grad_knG[k][:n_occ] += np.tensordot(
                        U_k[k][:n_occ, :n_occ].conj(), self.grad[k][:n_occ],
                        axes=1)
                else:
                    grad_knG[k][:n_occ] += self.grad[k][:n_occ]
            else:
                if U_k is not None:
                    self.grad[k][:] = np.tensordot(
                        U_k[k][:n_occ, :n_occ].conj(), self.grad[k][:n_occ],
                        axes=1)

        self.e_sic_by_orbitals[k] = \
            e_total_sic.reshape(e_total_sic.shape[0] // 2, 2)

        wfs.timer.stop('SIC e/g grid calculations')

        return e_total_sic.sum()

    def get_pseudo_pot(self, nt, Q_aL, i, kpoint=None):

        if self.sic_coarse_grid is False:
            # fine grid
            vt_sg = self.finegd.zeros(2)
            v_ht_g = self.finegd.zeros()
            nt_sg = self.finegd.zeros(2)
        else:
            # coarse grid
            vt_sg = self.cgd.zeros(2)
            v_ht_g = self.cgd.zeros()
            nt_sg = self.cgd.zeros(2)

        if self.sic_coarse_grid is False:
            self.interpolator.apply(nt, nt_sg[0])
            nt_sg[0] *= self.cgd.integrate(nt) / \
                self.finegd.integrate(nt_sg[0])
            e_xc = self.xc.calculate(self.finegd, nt_sg, vt_sg)
        else:
            nt_sg[0] = nt
            e_xc = self.xc.calculate(self.cgd, nt_sg, vt_sg)

        vt_sg[0] *= -self.beta_x

        self.ghat.add(nt_sg[0], Q_aL)

        if self.store_potentials:
            if self.sic_coarse_grid:
                v_ht_g = self.old_pot[kpoint][i]
            else:
                self.interpolator.apply(self.old_pot[kpoint][i],
                                        v_ht_g)

        self.poiss.solve(v_ht_g, nt_sg[0],
                         zero_initial_phi=False)

        if self.store_potentials:
            if self.sic_coarse_grid is True:
                self.old_pot[kpoint][i] = v_ht_g.copy()
            else:
                self.restrictor.apply(v_ht_g, self.old_pot[kpoint][i])

        if self.sic_coarse_grid is False:
            ec = 0.5 * self.finegd.integrate(nt_sg[0] * v_ht_g)
        else:
            ec = 0.5 * self.cgd.integrate(nt_sg[0] * v_ht_g)

        vt_sg[0] -= v_ht_g * self.beta_c

        if self.sic_coarse_grid is False:
            vt_G = self.cgd.zeros()
            self.restrictor.apply(vt_sg[0], vt_G)
        else:
            vt_G = vt_sg[0]

        return np.array([-ec * self.beta_c, -e_xc * self.beta_x]), \
            vt_G, v_ht_g

    def get_paw_corrections(self, D_ap, vHt_g):

        # XC-PAW
        dH_ap = {}

        exc = 0.0
        for a, D_p in D_ap.items():
            setup = self.setups[a]
            dH_sp = np.zeros((2, len(D_p)))
            D_sp = np.array([D_p, np.zeros_like(D_p)])
            exc += self.xc.calculate_paw_correction(
                setup, D_sp, dH_sp, addcoredensity=False)
            dH_ap[a] = -dH_sp[0] * self.beta_x

        # Hartree-PAW
        ec = 0.0
        W_aL = self.ghat.dict()
        self.ghat.integrate(vHt_g, W_aL)

        for a, D_p in D_ap.items():
            setup = self.setups[a]
            M_p = np.dot(setup.M_pp, D_p)
            ec += np.dot(D_p, M_p)

            dH_ap[a] += -(2.0 * M_p + np.dot(setup.Delta_pL,
                                             W_aL[a])) * self.beta_c

        if self.sic_coarse_grid is False:
            ec = self.finegd.comm.sum_scalar(ec)
            exc = self.finegd.comm.sum_scalar(exc)
        else:
            ec = self.cgd.comm.sum_scalar(ec)
            exc = self.cgd.comm.sum_scalar(exc)

        return np.array([-ec * self.beta_c, -exc * self.beta_x]), dH_ap

    def get_energy_and_gradients_inner_loop(
            self, wfs, kpt, a_mat, evals, evec, ham=None,
            exstate=False):

        if exstate:
            ndim = wfs.bd.nbands
        else:
            ndim = 0
            for f in kpt.f_n:
                if f > 1.0e-10:
                    ndim += 1

        k = self.n_kps * kpt.s + kpt.q
        self.grad[k] = np.zeros_like(kpt.psit_nG[:ndim])
        e_sic = self.get_energy_and_gradients_kpt(
            wfs, kpt, ham=ham, exstate=exstate)
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

    def get_odd_corrections_to_forces(self, F_av, wfs, kpt, exstate=False):

        n_occ = get_n_occ(kpt)[0]
        n_kps = self.n_kps

        dP_amiv = wfs.pt.dict(n_occ, derivative=True)
        wfs.pt.derivative(kpt.psit_nG[:n_occ], dP_amiv)
        k = n_kps * kpt.s + kpt.q
        for m in range(n_occ):
            # calculate Hartree pot, compans. charge and PAW corrects
            if wfs.mode == 'fd':
                nt_G, Q_aL, D_ap = \
                    self.get_orbdens_compcharge_dm_kpt(wfs, kpt, m)
                e_sic, vt_G, v_ht_g = \
                    self.get_pseudo_pot(nt_G, Q_aL, m, kpoint=k)
                e_sic_paw_m, dH_ap = \
                    self.get_paw_corrections(D_ap, v_ht_g)
            elif wfs.mode == 'pw':
                e_sic, vt_G, dH_ap, Q_aL, v_ht_g = \
                    self.get_si_pot_dh_pw(
                        wfs, kpt, m, returnQalandVhq=True, exstate=exstate)
            else:
                raise NotImplementedError

            # Force from compensation charges:
            dF_aLv = self.ghat.dict(derivative=True)
            self.ghat.derivative(v_ht_g, dF_aLv)
            for a, dF_Lv in dF_aLv.items():
                F_av[a] -= kpt.f_n[m] * self.beta_c * \
                    np.dot(Q_aL[a], dF_Lv)

            # Force from projectors
            for a, dP_miv in dP_amiv.items():
                dP_vi = dP_miv[m].T.conj()
                dH_ii = unpack_hermitian(dH_ap[a])
                P_i = kpt.P_ani[a][m]
                F_v = np.dot(np.dot(dP_vi, dH_ii), P_i)
                F_av[a] += kpt.f_n[m] * 2.0 * F_v.real

    def get_pz_sic_ith_kpt(self, nt_G, Q_aL, D_ap, i, k, timer):

        """
        :param nt_G: one-electron orbital density
        :param Q_aL: its compensation charge
        :param D_ap: its density matrix
        :param i: number of orbital
        :param k: k-point number, k = n_kperspin * kpt.s + kpt.q
        :param timer:
        :return: E, v and dH
            E = -(beta_c * E_Hartree[n_i] + beta_x * E_xc[n_i])
            v = dE / dn_i
            dH - paw corrections

        """

        # calculate sic energy,
        # sic pseudo-potential and Hartree
        timer.start('Get Pseudo Potential')
        # calculate sic energy, sic pseudo-potential and Hartree
        e_pz, vt_G, v_ht_g = \
            self.get_pseudo_pot(nt_G, Q_aL, i, kpoint=k)
        timer.stop('Get Pseudo Potential')

        # calculate PAW corrections
        timer.start('PAW')
        # calculate PAW corrections
        e_pz_paw_m, dH_ap = self.get_paw_corrections(D_ap, v_ht_g)
        timer.stop('PAW')

        # total sic:
        e_pz += e_pz_paw_m

        return e_pz, vt_G, dH_ap

    def get_si_pot_dh_pw(
            self, wfs, kpt, n, returnQalandVhq=False, exstate=False):

        wfs.timer.start("IFFT: Get density on real grid")
        nt_G = wfs.pd.gd.zeros(global_array=True)
        psit_G = wfs.pd.alltoall1(kpt.psit.array[n:n + 1], kpt.q)
        if psit_G is not None:
            # real space wfs:
            psit_R = wfs.pd.ifft(psit_G, kpt.q,
                                 local=True, safe=False)
            cgpaw.add_to_density(1.0, psit_R, nt_G)
        wfs.pd.gd.comm.sum(nt_G)
        nt_G = wfs.pd.gd.distribute(nt_G)  # this is real space grid
        wfs.timer.stop("IFFT: Get density on real grid")

        # multipole moments and atomic dm
        wfs.timer.start("Multipole Moments and Atomic Dens. Mat.")
        Q_aL = {}
        D_ap = {}
        for a, P_ni in kpt.P_ani.items():
            P_i = P_ni[n]
            D_ii = np.outer(P_i, P_i.conj()).real
            D_ap[a] = D_p = pack_density(D_ii)
            Q_aL[a] = np.dot(D_p, self.setups[a].Delta_pL)
        wfs.timer.stop("Multipole Moments and Atomic Dens. Mat.")

        # xc
        wfs.timer.start("Calc. XC on pseudo density")
        nt_sg = wfs.gd.zeros(2)
        nt_sg[0] = nt_G
        vt_sg = wfs.gd.zeros(2)
        exc = self.xc.calculate(wfs.gd, nt_sg, vt_sg)
        vt_G = - vt_sg[0] * self.beta_x
        dH_ap = {}
        wfs.timer.stop("Calc. XC on pseudo density")

        wfs.timer.start("Calc. PAW-XC")
        excpaw = 0.0
        for a, D_p in D_ap.items():
            setup = self.setups[a]
            dH_sp = np.zeros((2, len(D_p)))
            D_sp = np.array([D_p, np.zeros_like(D_p)])
            excpaw += self.xc.calculate_paw_correction(
                setup, D_sp, dH_sp, addcoredensity=False)
            dH_ap[a] = -dH_sp[0] * self.beta_x
        excpaw = wfs.gd.comm.sum_scalar(excpaw)
        exc += excpaw
        wfs.timer.stop("Calc. PAW-XC")

        wfs.timer.start("FFT density")
        nt_Q = self.pd2.fft(nt_G)
        self.ghat.add(nt_Q, Q_aL)
        wfs.timer.stop("FFT density")

        wfs.timer.start("Calc. Hartree on pseudo density")
        if exstate:
            nt_G = self.pd2.ifft(nt_Q)
            vHt = np.zeros_like(nt_G)
            self.poiss.solve(vHt, nt_G, zero_initial_phi=False)
            ehart = 0.5 * self.cgd.integrate(nt_G, vHt)
            vHt_q = self.pd2.fft(vHt)
        else:
            if self.pd2.gd.comm.rank == 0:
                nt_Q[0] = 0.0
            vHt_q = 4 * np.pi * nt_Q / self.G2
            ehart = 0.5 * self.pd2.integrate(vHt_q, nt_Q)
            # correct for uniform background
            if self.pd2.gd.comm.rank == 0:
                vHt_q[0] += self.corr_q
            vHt = self.pd2.ifft(vHt_q)
            ehart += self.corr / 2.0
        wfs.timer.stop("Calc. Hartree on pseudo density")

        # PAW to Hartree
        ehartpaw = 0.0
        W_aL = self.ghat.dict()
        self.ghat.integrate(vHt_q, W_aL)

        wfs.timer.start("Calc. PAW-Hartree")
        for a, D_p in D_ap.items():
            setup = self.setups[a]
            M_p = np.dot(setup.M_pp, D_p)
            ehartpaw += np.dot(D_p, M_p)
            dH_ap[a] += -(2.0 * M_p + np.dot(setup.Delta_pL,
                                             W_aL[a])) * self.beta_c
        ehartpaw = wfs.gd.comm.sum_scalar(ehartpaw)
        ehart += ehartpaw
        wfs.timer.stop("Calc. PAW-Hartree")

        vt_G -= vHt * self.beta_c

        if returnQalandVhq:
            return np.array([-ehart * self.beta_c, -exc * self.beta_x]), \
                vt_G, dH_ap, Q_aL, vHt_q
        else:
            return np.array([-ehart * self.beta_c, -exc * self.beta_x]), \
                vt_G, dH_ap
