"""
Potentials for orbital density dependent energy functionals
"""
import numpy as np

from gpaw.directmin.tools import d_matrix
from gpaw.lfc import LFC
from gpaw.poisson import PoissonSolver
from gpaw.transformers import Transformer
from gpaw.utilities import pack_density, unpack_hermitian


class PZSICLCAO:
    """
    Perdew-Zunger self-interaction corrections

    """
    def __init__(self, wfs, dens, ham, scaling_factor=(1.0, 1.0),
                 sic_coarse_grid=True, store_potentials=False,
                 poisson_solver='FPS'):

        self.name = 'PZ-SIC'
        # what we need from wfs
        self.setups = wfs.setups
        self.bfs = wfs.basis_functions
        spos_ac = wfs.spos_ac
        self.cgd = wfs.gd

        # what we need from dens
        self.finegd = dens.finegd
        self.sic_coarse_grid = sic_coarse_grid

        if self.sic_coarse_grid:
            self.ghat_cg = LFC(self.cgd,
                               [setup.ghat_l for setup
                                in self.setups],
                               integral=np.sqrt(4 * np.pi),
                               forces=True)
            self.ghat_cg.set_positions(spos_ac)
            self.ghat = None
        else:
            self.ghat = dens.ghat  # we usually solve poiss. on finegd
            self.ghat_cg = None

        # what we need from ham
        self.xc = ham.xc

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
        # Scaling factor:
        self.beta_c = scaling_factor[0]
        self.beta_x = scaling_factor[1]

        self.n_kps = wfs.kd.nibzkpts
        self.store_potentials = store_potentials
        if store_potentials:
            self.old_pot = {}
            for kpt in wfs.kpt_u:
                k = self.n_kps * kpt.s + kpt.q
                n_occ = 0
                nbands = len(kpt.f_n)
                while n_occ < nbands and kpt.f_n[n_occ] > 1e-10:
                    n_occ += 1
                self.old_pot[k] = self.cgd.zeros(n_occ, dtype=float)

    def get_gradients(self, h_mm, c_nm, f_n, evec, evals, kpt,
                      wfs, timer, matrix_exp, repr_name,
                      ind_up, constraints, occupied_only=False):
        """
        :param C_nM: coefficients of orbitals
        :return: matrix G - gradients, and orbital SI energies

        which is G_{ij} = (1-delta_{ij}/2)*(int_0^1 e^{tA} L e^{-tA} dt )_{ji}

        Lambda_ij = (C_i, F_j C_j )

        L_{ij} = Lambda_ji^{cc} - Lambda_ij

        """

        # 0.
        n_occ = 0
        nbands = len(f_n)
        while n_occ < nbands and f_n[n_occ] > 1e-10:
            n_occ += 1

        if occupied_only is True:
            nbs = n_occ
        else:
            nbs = c_nm.shape[0]
        n_set = c_nm.shape[1]

        timer.start('Construct Gradient Matrix')
        hc_mn = np.dot(h_mm.conj(), c_nm[:nbs].T)
        h_mm = np.dot(c_nm[:nbs].conj(), hc_mn)
        # odd part
        b_mn = np.zeros(shape=(n_set, nbs), dtype=self.dtype)
        e_total_sic = np.array([])
        for n in range(n_occ):
            F_MM, sic_energy_n =\
                self.get_orbital_potential_matrix(f_n, c_nm, kpt,
                                                  wfs, wfs.setups,
                                                  n, timer)

            b_mn[:, n] = np.dot(c_nm[n], F_MM.conj()).T
            e_total_sic = np.append(e_total_sic, sic_energy_n, axis=0)
        l_odd = np.dot(c_nm[:nbs].conj(), b_mn)

        f = f_n[:nbs]
        grad = f * (h_mm[:nbs, :nbs] + l_odd) - \
            f[:, np.newaxis] * (h_mm[:nbs, :nbs] + l_odd.T.conj())

        if matrix_exp in ['pade-approx', 'egdecomp2']:
            grad = np.ascontiguousarray(grad)
        elif matrix_exp == 'egdecomp':
            timer.start('Use Eigendecomposition')
            grad = np.dot(evec.T.conj(), np.dot(grad, evec))
            grad = grad * d_matrix(evals)
            grad = np.dot(evec, np.dot(grad, evec.T.conj()))
            for i in range(grad.shape[0]):
                grad[i][i] *= 0.5
            timer.stop('Use Eigendecomposition')
        else:
            raise NotImplementedError('Check the keyword '
                                      'for matrix_exp. \n'
                                      'Must be '
                                      '\'pade-approx\' or '
                                      '\'egdecomp\'')
        if self.dtype == float:
            grad = grad.real
        if repr_name in ['sparse', 'u-invar']:
            grad = grad[ind_up]

        timer.stop('Construct Gradient Matrix')

        u = kpt.s * self.n_kps + kpt.q
        self.e_sic_by_orbitals[u] = \
            e_total_sic.reshape(e_total_sic.shape[0] // 2, 2)

        timer.start('Residual')
        hc_mn += b_mn
        h_mm += l_odd

        rhs2 = kpt.S_MM.conj() @ c_nm[:n_occ].T @ h_mm[:n_occ, :n_occ]
        hc_mn = hc_mn[:, :n_occ] - rhs2[:, :n_occ]

        if constraints:
            # Zero out the components of the residual that are constrained,
            # so that the constrained degrees of freedom are always considered
            # converged
            for con in constraints:
                con1 = con[0]
                hc_mn[:, con1] = 0.0

        norm = []
        for i in range(n_occ):
            norm.append(np.dot(hc_mn[:, i].conj(),
                               hc_mn[:, i]).real * kpt.f_n[i])

        error = sum(norm)
        del rhs2, hc_mn, norm
        timer.stop('Residual')

        if self.counter == 0:
            h_mm = 0.5 * (h_mm + h_mm.T.conj())
            kpt.eps_n[:nbs] = np.linalg.eigh(h_mm)[0]
        self.counter += 1

        if constraints:
            constrain_grad(grad, constraints, ind_up)

        return 2.0 * grad, error

    def get_orbital_potential_matrix(self, f_n, C_nM, kpt,
                                     wfs, setup, m, timer):
        """
        :param f_n:
        :param C_nM:
        :param kpt:
        :param wfs:
        :param setup:
        :return: F_i = <m|v_i + PAW_i|n > for occupied
                 F_i = 0 for unoccupied,
                 SI energies

        To calculate this, we need to calculate
        orbital-depended potential and PAW corrections to it.

        Fot this, we need firstly to get orbitals densities.

        """
        kpoint = self.n_kps * kpt.s + kpt.q
        n_set = C_nM.shape[1]
        F_MM = np.zeros(shape=(n_set, n_set),
                        dtype=self.dtype)
        # get orbital-density
        timer.start('Construct Density, Charge, and DM')
        nt_G, Q_aL, D_ap = \
            self.get_density(f_n,
                             C_nM, kpt,
                             wfs, setup, m)
        timer.stop('Construct Density, Charge, and DM')

        # calculate sic energy,
        # sic pseudo-potential and Hartree
        timer.start('Get Pseudo Potential')
        e_sic_m, vt_mG, vHt_g = \
            self.get_pseudo_pot(nt_G, Q_aL, m, kpoint, timer)
        timer.stop('Get Pseudo Potential')

        # calculate PAW corrections
        timer.start('PAW')
        e_sic_paw_m, dH_ap = \
            self.get_paw_corrections(D_ap, vHt_g, timer)
        timer.stop('PAW')

        # total sic:
        e_sic_m += e_sic_paw_m

        timer.start('ODD Potential Matrices')
        Vt_MM = np.zeros_like(F_MM)
        self.bfs.calculate_potential_matrix(vt_mG, Vt_MM, kpt.q)
        # make matrix hermitian
        ind_l = np.tril_indices(Vt_MM.shape[0], -1)
        Vt_MM[(ind_l[1], ind_l[0])] = Vt_MM[ind_l].conj()
        timer.stop('ODD Potential Matrices')

        # PAW:
        timer.start('Potential matrix - PAW')
        for a, dH_p in dH_ap.items():
            P_Mj = wfs.P_aqMi[a][kpt.q]
            dH_ij = unpack_hermitian(dH_p)

            if self.dtype == complex:
                F_MM += P_Mj @ dH_ij @ P_Mj.T.conj()
            else:
                F_MM += P_Mj @ dH_ij @ P_Mj.T

        if self.dtype == complex:
            F_MM += Vt_MM.astype(complex)
        else:
            F_MM += Vt_MM
        if self.sic_coarse_grid:
            self.cgd.comm.sum(F_MM)
        else:
            self.finegd.comm.sum(F_MM)
        timer.stop('Potential matrix - PAW')

        return F_MM, e_sic_m * f_n[m]

    def get_density(self, f_n, C_nM, kpt,
                    wfs, setup, m):

        # construct orbital density matrix
        if f_n[m] > 1.0 + 1.0e-4:
            occup_factor = f_n[m] / (3.0 - wfs.nspins)
        else:
            occup_factor = f_n[m]
        rho_MM = occup_factor * np.outer(C_nM[m].conj(), C_nM[m])

        nt_G = self.cgd.zeros()
        self.bfs.construct_density(rho_MM, nt_G, kpt.q)

        # calculate  atomic density matrix and
        # compensation charges
        D_ap = {}
        Q_aL = {}

        for a in wfs.P_aqMi.keys():
            P_Mi = wfs.P_aqMi[a][kpt.q]
            D_ii = np.zeros((wfs.P_aqMi[a].shape[2],
                             wfs.P_aqMi[a].shape[2]),
                            dtype=self.dtype)
            rhoP_Mi = rho_MM @ P_Mi
            D_ii = P_Mi.T.conj() @ rhoP_Mi
            if self.dtype == complex:
                D_ap[a] = D_p = pack_density(D_ii.real)
            else:
                D_ap[a] = D_p = pack_density(D_ii)

            Q_aL[a] = np.dot(D_p, setup[a].Delta_pL)

        return nt_G, Q_aL, D_ap

    def get_pseudo_pot(self, nt, Q_aL, i, kpoint, timer):

        if self.sic_coarse_grid is False:
            # change to fine grid
            vt_sg = self.finegd.zeros(2)
            vHt_g = self.finegd.zeros()
            nt_sg = self.finegd.zeros(2)
        else:
            vt_sg = self.cgd.zeros(2)
            vHt_g = self.cgd.zeros()
            nt_sg = self.cgd.zeros(2)

        if self.sic_coarse_grid is False:
            self.interpolator.apply(nt, nt_sg[0])
            nt_sg[0] *= self.cgd.integrate(nt) / \
                self.finegd.integrate(nt_sg[0])
        else:
            nt_sg[0] = nt

        timer.start('ODD XC 3D grid')
        if self.sic_coarse_grid is False:
            e_xc = self.xc.calculate(self.finegd, nt_sg, vt_sg)
        else:
            e_xc = self.xc.calculate(self.cgd, nt_sg, vt_sg)
        timer.stop('ODD XC 3D grid')
        vt_sg[0] *= -self.beta_x

        # Hartree
        if self.sic_coarse_grid is False:
            self.ghat.add(nt_sg[0], Q_aL)
        else:
            self.ghat_cg.add(nt_sg[0], Q_aL)

        timer.start('ODD Poisson')
        if self.store_potentials:
            if self.sic_coarse_grid:
                vHt_g = self.old_pot[kpoint][i]
            else:
                self.interpolator.apply(self.old_pot[kpoint][i],
                                        vHt_g)
        self.poiss.solve(vHt_g, nt_sg[0],
                         zero_initial_phi=self.store_potentials,
                         timer=timer)
        if self.store_potentials:
            if self.sic_coarse_grid:
                self.old_pot[kpoint][i] = vHt_g.copy()
            else:
                self.restrictor.apply(vHt_g, self.old_pot[kpoint][i])

        timer.stop('ODD Poisson')

        timer.start('ODD Hartree integrate')
        if self.sic_coarse_grid is False:
            ec = 0.5 * self.finegd.integrate(nt_sg[0] * vHt_g)
        else:
            ec = 0.5 * self.cgd.integrate(nt_sg[0] * vHt_g)

        timer.stop('ODD Hartree integrate')
        vt_sg[0] -= vHt_g * self.beta_c
        if self.sic_coarse_grid is False:
            vt_G = self.cgd.zeros()
            self.restrictor.apply(vt_sg[0], vt_G)
        else:
            vt_G = vt_sg[0]

        return np.array([-ec * self.beta_c,
                         -e_xc * self.beta_x]), vt_G, vHt_g

    def get_paw_corrections(self, D_ap, vHt_g, timer):

        # XC-PAW
        timer.start('xc-PAW')
        dH_ap = {}
        exc = 0.0
        for a, D_p in D_ap.items():
            setup = self.setups[a]
            dH_sp = np.zeros((2, len(D_p)))
            D_sp = np.array([D_p, np.zeros_like(D_p)])
            exc += self.xc.calculate_paw_correction(setup, D_sp,
                                                    dH_sp,
                                                    addcoredensity=False,
                                                    a=a)
            dH_ap[a] = -dH_sp[0] * self.beta_x
        timer.stop('xc-PAW')

        # Hartree-PAW
        timer.start('Hartree-PAW')
        ec = 0.0
        timer.start('ghat-PAW')
        if self.sic_coarse_grid is False:
            W_aL = self.ghat.dict()
            self.ghat.integrate(vHt_g, W_aL)
        else:
            W_aL = self.ghat_cg.dict()
            self.ghat_cg.integrate(vHt_g, W_aL)
        timer.stop('ghat-PAW')

        for a, D_p in D_ap.items():
            setup = self.setups[a]
            M_p = np.dot(setup.M_pp, D_p)
            ec += np.dot(D_p, M_p)
            dH_ap[a] += -(2.0 * M_p + np.dot(setup.Delta_pL,
                                             W_aL[a])) * self.beta_c
        timer.stop('Hartree-PAW')

        timer.start('Wait for sum')
        if self.sic_coarse_grid is False:
            ec = self.finegd.comm.sum_scalar(ec)
            exc = self.finegd.comm.sum_scalar(exc)
        else:
            ec = self.cgd.comm.sum_scalar(ec)
            exc = self.cgd.comm.sum_scalar(exc)
        timer.stop('Wait for sum')

        return np.array([-ec * self.beta_c, -exc * self.beta_x]), dH_ap

    def get_odd_corrections_to_forces(self, wfs, dens):

        timer = wfs.timer

        timer.start('ODD corrections')

        natoms = len(wfs.setups)
        F_av = np.zeros((natoms, 3))
        Ftheta_av = np.zeros_like(F_av)
        Frho_av = np.zeros_like(F_av)
        Fatom_av = np.zeros_like(F_av)
        Fpot_av = np.zeros_like(F_av)
        Fhart_av = np.zeros_like(F_av)

        ksl = wfs.ksl
        nao = ksl.nao
        mynao = ksl.mynao
        dtype = wfs.dtype
        manytci = wfs.manytci
        gd = wfs.gd

        Mstart = ksl.Mstart
        Mstop = ksl.Mstop
        n_kps = wfs.kd.nibzkpts

        timer.start('TCI derivative')
        dThetadR_qvMM, dTdR_qvMM = manytci.O_qMM_T_qMM(
            gd.comm, Mstart, Mstop, False, derivative=True)

        dPdR_aqvMi = manytci.P_aqMi(
            self.bfs.my_atom_indices, derivative=True)
        gd.comm.sum(dThetadR_qvMM)
        gd.comm.sum(dTdR_qvMM)
        timer.stop('TCI derivative')

        my_atom_indices = self.bfs.my_atom_indices
        atom_indices = self.bfs.atom_indices

        def _slices(indices):
            for a in indices:
                M1 = self.bfs.M_a[a] - Mstart
                M2 = M1 + self.setups[a].nao
                if M2 > 0:
                    yield a, max(0, M1), M2

        def slices():
            return _slices(atom_indices)

        def my_slices():
            return _slices(my_atom_indices)

        #
        #         -----                    -----
        #          \    -1                  \    *
        # E      =  )  S     H    rho     =  )  c     eps  f  c
        #  mu nu   /    mu x  x z    z nu   /    n mu    n  n  n nu
        #         -----                    -----
        #          x z                       n
        #
        # We use the transpose of that matrix.  The first form is used
        # if rho is given, otherwise the coefficients are used.

        for kpt in wfs.kpt_u:
            u = kpt.s * n_kps + kpt.q
            f_n = kpt.f_n
            n_occ = 0
            for f in f_n:
                if f > 1.0e-10:
                    n_occ += 1

            for m in range(n_occ):

                # calculate orbital-density matrix
                rho_xMM = kpt.f_n[m] * np.outer(
                    kpt.C_nM[m].conj(), kpt.C_nM[m]) / (3.0 - wfs.nspins)

                F_MM = self.get_orbital_potential_matrix(
                    f_n, kpt.C_nM, kpt, wfs, self.setups, m, timer)[0]

                sfrhoT_MM = np.linalg.solve(
                    wfs.S_qMM[kpt.q], F_MM @ rho_xMM).T.copy()

                del F_MM

                # Density matrix contribution due to basis overlap
                #
                #            ----- d Theta
                #  a          \           mu nu
                # F  += -2 Re  )   ------------  E
                #             /        d R        nu mu
                #            -----        mu nu
                #         mu in a; nu
                #

                dThetadRE_vMM = (dThetadR_qvMM[kpt.q] *
                                 sfrhoT_MM[np.newaxis]).real
                for a, M1, M2 in my_slices():
                    Ftheta_av[a, :] += \
                        -2.0 * dThetadRE_vMM[:, M1:M2].sum(-1).sum(-1)
                del dThetadRE_vMM

                # Density matrix contribution from PAW correction
                #
                #           -----                        -----
                #  a         \      a                     \     b
                # F +=  2 Re  )    Z      E        - 2 Re  )   Z      E
                #            /      mu nu  nu mu          /     mu nu  nu mu
                #           -----                        -----
                #           mu nu                    b; mu in a; nu
                #
                # with
                #                  b*
                #         -----  dP
                #   b      \       i mu    b   b
                #  Z     =  )   -------- dS   P
                #   mu nu  /     dR        ij  j nu
                #         -----    b mu
                #           ij
                #
                work_MM = np.zeros((mynao, nao), dtype)
                ZE_MM = None
                for b in my_atom_indices:
                    setup = self.setups[b]
                    dO_ii = np.asarray(setup.dO_ii, dtype)
                    dOP_iM = dO_ii @ wfs.P_aqMi[b][kpt.q].T.conj()

                    for v in range(3):
                        work_MM = \
                            dPdR_aqvMi[b][kpt.q][v][Mstart:Mstop] @ dOP_iM
                        ZE_MM = (work_MM * sfrhoT_MM).real
                        for a, M1, M2 in slices():
                            dE = 2 * ZE_MM[M1:M2].sum()
                            Frho_av[a, v] -= dE  # the "b; mu in a; nu" term
                            Frho_av[b, v] += dE  # the "mu nu" term
                del work_MM, ZE_MM

                # Potential contribution
                #
                #           -----      /  d Phi  (r)
                #  a         \        |        mu    ~
                # F += -2 Re  )       |   ---------- v (r)  Phi  (r) dr rho
                #            /        |     d R                nu         nu mu
                #           -----    /         a
                #        mu in a; nu
                #

                nt_G, Q_aL, D_ap = \
                    self.get_density(f_n, kpt.C_nM, kpt, wfs, self.setups, m)

                e_sic_m, vt_mG, vHt_g = \
                    self.get_pseudo_pot(nt_G, Q_aL, m, u, timer)
                e_sic_paw_m, dH_ap = \
                    self.get_paw_corrections(D_ap, vHt_g, timer)

                Fpot_av += \
                    self.bfs.calculate_force_contribution(
                        vt_mG, rho_xMM, kpt.q)

                # Atomic density contribution
                #            -----                         -----
                #  a          \     a                       \   b
                # F  += -2 Re  )   A      rho       + 2 Re   ) A rho
                #             /     mu nu    nu mu          / mununumu
                #            -----                         -----
                #            mu nu                     b; mu in a; nu
                #
                #                  b*
                #         ----- d P
                #  b       \       i mu   b   b
                # A     =   )   ------- dH   P
                #  mu nu   /    d R       ij  j nu
                #         -----    b mu
                #           ij
                #
                for b in my_atom_indices:
                    H_ii = np.asarray(unpack_hermitian(dH_ap[b]),
                                      dtype)
                    HP_iM = H_ii @ wfs.P_aqMi[b][kpt.q].T.conj()
                    for v in range(3):
                        dPdR_Mi = \
                            dPdR_aqvMi[b][kpt.q][v][Mstart:Mstop]
                        ArhoT_MM = \
                            ((dPdR_Mi @ HP_iM) * rho_xMM.T).real
                        for a, M1, M2 in slices():
                            dE = 2 * ArhoT_MM[M1:M2].sum()
                            Fatom_av[a, v] += dE  # the "b; mu in a; nu" term
                            Fatom_av[b, v] -= dE  # the "mu nu" term

                # contribution from hartree
                if self.sic_coarse_grid is False:
                    ghat_aLv = dens.ghat.dict(derivative=True)

                    dens.ghat.derivative(vHt_g, ghat_aLv)
                    for a, dF_Lv in ghat_aLv.items():
                        Fhart_av[a] -= self.beta_c * np.dot(Q_aL[a], dF_Lv)
                else:
                    ghat_aLv = self.ghat_cg.dict(derivative=True)

                    self.ghat_cg.derivative(vHt_g, ghat_aLv)
                    for a, dF_Lv in ghat_aLv.items():
                        Fhart_av[a] -= self.beta_c * np.dot(Q_aL[a], dF_Lv)

        F_av += Fpot_av + Ftheta_av + Frho_av + Fatom_av + Fhart_av

        wfs.gd.comm.sum(F_av, 0)

        timer.start('Wait for sum')
        ksl.orbital_comm.sum(F_av)
        if wfs.bd.comm.rank == 0:
            wfs.kd.comm.sum(F_av, 0)

        wfs.world.broadcast(F_av, 0)
        timer.stop('Wait for sum')

        timer.stop('ODD corrections')

        return F_av * (3.0 - wfs.nspins)

    def get_lagrange_matrices(self, h_mm, c_nm, f_n, kpt,
                              wfs, occupied_only=False,
                              update_eigenvalues=False):
        n_occ = 0
        nbands = len(f_n)
        while n_occ < nbands and f_n[n_occ] > 1e-10:
            n_occ += 1

        if occupied_only is True:
            nbs = n_occ
        else:
            nbs = c_nm.shape[0]
        n_set = c_nm.shape[1]

        hc_mn = np.dot(h_mm.conj(), c_nm[:nbs].T)
        h_mm = np.dot(c_nm[:nbs].conj(), hc_mn)
        # odd part
        b_mn = np.zeros(shape=(n_set, nbs), dtype=self.dtype)
        e_total_sic = np.array([])
        for n in range(n_occ):
            F_MM, sic_energy_n =\
                self.get_orbital_potential_matrix(f_n, c_nm, kpt,
                                                  wfs, wfs.setups,
                                                  n, wfs.timer)

            b_mn[:, n] = np.dot(c_nm[n], F_MM.conj()).T
            e_total_sic = np.append(e_total_sic, sic_energy_n, axis=0)
        l_odd = np.dot(c_nm[:nbs].conj(), b_mn)

        k = wfs.eigensolver.kpointval(kpt)

        fullham = h_mm + 0.5 * (l_odd + l_odd.T.conj())
        fullham[:n_occ, n_occ:] = 0.0
        fullham[n_occ:, :n_occ] = 0.0

        self.lagr_diag_s[k] = np.diagonal(fullham).real

        if update_eigenvalues:
            eigval, eigvec = np.linalg.eigh(fullham[0:n_occ, 0:n_occ])
            kpt.eps_n[0:n_occ] = eigval
            eigval, eigvec = np.linalg.eigh(fullham[n_occ:nbs, n_occ:nbs])
            kpt.eps_n[n_occ:nbs] = eigval

        return h_mm, l_odd


def constrain_grad(grad, constraints, ind_up):
    """
    Zero out the components of the gradient that are constrained, so that no
    optimization step is taken along the constrained degrees of freedom (It
    would be better not to evaluate these components of the gradient to begin
    with.).
    """

    for con in constraints:
        num = -1
        for ind1, ind2 in zip(ind_up[0], ind_up[1]):
            num += 1
            if con == [ind1, ind2]:
                grad[num] = 0.0
    return grad
