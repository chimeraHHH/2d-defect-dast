"""
Optimization of orbitals
among occupied and a few virtual states
represented on a grid or with plane waves
in order to calculate and excited state

arXiv:2102.06542 [physics.comp-ph]
"""

from gpaw.directmin.tools import get_n_occ, get_indices, expm_ed, \
    sort_orbitals_according_to_occ
from gpaw.directmin.sd_etdm import LSR1P
from gpaw.directmin.ls_etdm import MaxStep
from gpaw.directmin.derivatives import get_approx_analytical_hessian
from ase.units import Hartree
import numpy as np
import time


class ETDMInnerLoop:

    def __init__(self, odd_pot, wfs, nstates='all', maxiter=100,
                 maxstepxst=0.2, g_tol=5.0e-4, useprec=False, momevery=10):

        self.odd_pot = odd_pot
        self.n_kps = wfs.kd.nibzkpts
        self.g_tol = g_tol / Hartree
        self.dtype = wfs.dtype
        self.get_en_and_grad_iters = 0
        self.precond = {}
        self.max_iter_line_search = 6
        self.n_counter = maxiter
        self.maxstep = maxstepxst
        self.eg_count = 0
        self.total_eg_count = 0
        self.run_count = 0
        self.U_k = {}
        self.Unew_k = {}
        self.e_total = 0.0
        self.n_occ = {}
        self.useprec = useprec
        for kpt in wfs.kpt_u:
            k = self.kpointval(kpt)
            if nstates == 'all':
                self.n_occ[k] = wfs.bd.nbands
            elif nstates == 'occupied':
                self.n_occ[k] = get_n_occ(kpt)[0]
            else:
                raise NotImplementedError
            self.U_k[k] = np.eye(self.n_occ[k], dtype=self.dtype)
            self.Unew_k[k] = np.eye(self.n_occ[k], dtype=self.dtype)
        self.momcounter = 0
        self.momevery = momevery
        self.restart = False
        self.eks = 0.0
        self.esic = 0.0
        self.kappa = 0.0

    def update_ks_energy(self, wfs, dens, ham):
        wfs.timer.start('Update Kohn-Sham energy')
        # calc projectors
        for kpt in wfs.kpt_u:
            wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

        dens.update(wfs)
        ham.update(dens, wfs, False)
        wfs.timer.stop('Update Kohn-Sham energy')
        return ham.get_energy(0.0, wfs, False)

    def get_energy_and_gradients(self, a_k, wfs, dens, ham):
        """
        Energy E = E[A]. Gradients G_ij[A] = dE/dA_ij
        Returns E[A] and G[A] at psi = exp(A).T kpt.psi
        :param a_k: A
        :return:
        """

        g_k = {}
        self.e_total = 0.0
        self.esic = 0.0
        self.kappa = 0.0
        for k, kpt in enumerate(wfs.kpt_u):
            n_occ = self.n_occ[k]
            if n_occ == 0:
                g_k[k] = np.zeros_like(a_k[k])

        evecs, evals = self.rotate_wavefunctions(wfs, a_k)

        self.eks = self.update_ks_energy(wfs, dens, ham)

        for k, kpt in enumerate(wfs.kpt_u):
            wfs.timer.start('Energy and gradients')
            g_k[k], esic, kappa1 = \
                self.odd_pot.get_energy_and_gradients_inner_loop(
                    wfs, kpt, a_k[k], evals[k], evecs[k], ham=ham,
                    exstate=True)
            wfs.timer.stop('Energy and gradients')
            if kappa1 > self.kappa:
                self.kappa = kappa1
            self.esic += esic

        self.check_mom(wfs, dens)
        self.e_total = self.eks + self.esic

        self.kappa = wfs.kd.comm.max_scalar(self.kappa)
        self.eg_count += 1
        self.total_eg_count += 1

        return self.e_total, g_k

    def rotate_wavefunctions(self, wfs, a_k):
        evecs = {}
        evals = {}
        for k, kpt in enumerate(wfs.kpt_u):
            n_occ = self.n_occ[k]
            if n_occ == 0:
                continue
            wfs.timer.start('Unitary matrix')
            u_mat, evecs[k], evals[k] = expm_ed(a_k[k], evalevec=True)
            wfs.timer.stop('Unitary matrix')
            self.Unew_k[k] = u_mat.copy()
            kpt.psit_nG[:n_occ] = \
                np.tensordot(u_mat.T, self.psit_knG[k][:n_occ], axes=1)
            # calc projectors
            wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

        return evecs, evals

    def evaluate_phi_and_der_phi(self, a_k, p_k, alpha,
                                 wfs, dens, ham,
                                 phi=None, g_k=None):
        """
        phi = f(x_k + alpha_k*p_k)
        der_phi = grad f(x_k + alpha_k*p_k) cdot p_k
        :return:  phi, der_phi, grad f(x_k + alpha_k*p_k)
        """
        if phi is None or g_k is None:
            x_k = {k: a_k[k] + alpha * p_k[k] for k in a_k.keys()}
            phi, g_k = \
                self.get_energy_and_gradients(x_k, wfs, dens, ham)
            del x_k
        else:
            pass

        der_phi = 0.0
        for k in p_k.keys():
            il1 = get_indices(p_k[k].shape[0])
            der_phi += np.dot(g_k[k][il1].conj(), p_k[k][il1]).real

        der_phi = wfs.kd.comm.sum_scalar(der_phi)

        return phi, der_phi, g_k

    def get_search_direction(self, a_k, g_k, wfs):

        # structure of vector is
        # (x_1_up, x_2_up,..,y_1_up, y_2_up,..,
        #  x_1_down, x_2_down,..,y_1_down, y_2_down,.. )

        a = {}
        g = {}

        for k in a_k.keys():
            il1 = get_indices(a_k[k].shape[0])
            a[k] = a_k[k][il1]
            g[k] = g_k[k][il1]

        p = self.sd.update_data(wfs, a, g, precond=self.precond, mode='lcao')
        del a, g

        p_k = {}
        for k in p.keys():
            p_k[k] = np.zeros_like(a_k[k])
            il1 = get_indices(a_k[k].shape[0])
            p_k[k][il1] = p[k]
            # make it skew-hermitian
            ind_l = np.tril_indices(p_k[k].shape[0], -1)
            p_k[k][(ind_l[1], ind_l[0])] = -p_k[k][ind_l].conj()
        del p

        return p_k

    def run(self, wfs, dens, log, outer_counter=0, ham=None):

        log = log
        self.run_count += 1
        self.counter = 0
        self.eg_count = 0
        self.momcounter = 1
        self.converged = False
        # initial things
        self.psit_knG = {}
        for kpt in wfs.kpt_u:
            k = self.kpointval(kpt)
            n_occ = self.n_occ[k]
            self.psit_knG[k] = np.tensordot(
                self.U_k[k].T, kpt.psit_nG[:n_occ], axes=1)

        a_k = {}
        for kpt in wfs.kpt_u:
            k = self.kpointval(kpt)
            d = self.n_occ[k]
            a_k[k] = np.zeros(shape=(d, d), dtype=self.dtype)

        self.sd = LSR1P(memory=50)
        self.ls = MaxStep(self.evaluate_phi_and_der_phi, max_step=self.maxstep)

        threelasten = []
        # get initial energy and gradients
        self.e_total, g_k = self.get_energy_and_gradients(a_k, wfs, dens, ham)
        threelasten.append(self.e_total)
        g_max = g_max_norm(g_k, wfs)
        if g_max < self.g_tol:
            self.converged = True
            for kpt in wfs.kpt_u:
                k = self.kpointval(kpt)
                n_occ = self.n_occ[k]
                kpt.psit_nG[:n_occ] = np.tensordot(
                    self.U_k[k].conj(), self.psit_knG[k], axes=1)
                # calc projectors
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

                self.U_k[k] = self.U_k[k] @ self.Unew_k[k]
            if outer_counter is None:
                return self.e_total, self.counter
            else:
                return self.e_total, outer_counter

        if self.restart:
            del self.psit_knG
            return 0.0, 0

        # stuff which are needed for minim.
        phi_0 = self.e_total
        phi_old = None
        der_phi_old = None
        phi_old_2 = None
        der_phi_old_2 = None

        outer_counter += 1
        if log is not None:
            log_f(log, self.counter, self.kappa, self.eks, self.esic,
                  outer_counter, g_max)

        alpha = 1.0
        not_converged = True
        while not_converged:
            self.precond = self.update_preconditioning(wfs, self.useprec)

            # calculate search direction fot current As and Gs
            p_k = self.get_search_direction(a_k, g_k, wfs)

            # calculate derivative along the search direction
            phi_0, der_phi_0, g_k = \
                self.evaluate_phi_and_der_phi(
                    a_k, p_k, 0.0, wfs, dens, ham=ham, phi=phi_0, g_k=g_k)
            if self.counter > 1:
                phi_old = phi_0
                der_phi_old = der_phi_0

            # choose optimal step length along the search direction
            # also get energy and gradients for optimal step
            alpha, phi_0, der_phi_0, g_k = \
                self.ls.step_length_update(
                    a_k, p_k, wfs, dens, ham, mode='lcao',
                    phi_0=phi_0, der_phi_0=der_phi_0,
                    phi_old=phi_old_2, der_phi_old=der_phi_old_2,
                    alpha_max=3.0, alpha_old=alpha, kpdescr=wfs.kd)

            # broadcast data is gd.comm > 1
            if wfs.gd.comm.size > 1:
                alpha_phi_der_phi = np.array([alpha, phi_0, der_phi_0])
                wfs.gd.comm.broadcast(alpha_phi_der_phi, 0)
                alpha = alpha_phi_der_phi[0]
                phi_0 = alpha_phi_der_phi[1]
                der_phi_0 = alpha_phi_der_phi[2]
                for kpt in wfs.kpt_u:
                    k = self.kpointval(kpt)
                    if self.n_occ[k] == 0:
                        continue
                    wfs.gd.comm.broadcast(g_k[k], 0)

            phi_old_2 = phi_old
            der_phi_old_2 = der_phi_old

            if self.restart:
                if log is not None:
                    log('MOM has detected variational collapse, '
                        'occupied orbitals have changed')
                break

            if alpha > 1.0e-10:
                # calculate new matrices at optimal step lenght
                a_k = {k: a_k[k] + alpha * p_k[k] for k in a_k.keys()}
                g_max = g_max_norm(g_k, wfs)

                # output
                self.counter += 1
                if outer_counter is not None:
                    outer_counter += 1
                if log is not None:
                    log_f(
                        log, self.counter, self.kappa, self.eks, self.esic,
                        outer_counter, g_max)

                not_converged = \
                    g_max > self.g_tol and \
                    self.counter < self.n_counter
                if g_max <= self.g_tol:
                    self.converged = True
            else:
                break

        if log is not None:
            log('INNER LOOP FINISHED.\n')
            log('Total number of e/g calls:' + str(self.eg_count))

        if not self.restart:
            for kpt in wfs.kpt_u:
                k = self.kpointval(kpt)
                n_occ = self.n_occ[k]
                kpt.psit_nG[:n_occ] = np.tensordot(self.U_k[k].conj(),
                                                   self.psit_knG[k],
                                                   axes=1)
                # calc projectors
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
                self.U_k[k] = self.U_k[k] @ self.Unew_k[k]

        if outer_counter is None:
            return self.e_total, self.counter
        else:
            return self.e_total, outer_counter

    def update_preconditioning(self, wfs, use_prec):
        counter = 30
        if use_prec:
            if self.counter % counter == 0:
                for kpt in wfs.kpt_u:
                    k = self.kpointval(kpt)
                    hess = get_approx_analytical_hessian(kpt, self.dtype)
                    if self.dtype == float:
                        self.precond[k] = np.zeros_like(hess)
                        for i in range(hess.shape[0]):
                            if abs(hess[i]) < 1.0e-4:
                                self.precond[k][i] = 1.0
                            else:
                                self.precond[k][i] = 1.0 / hess[i].real
                    else:
                        self.precond[k] = np.zeros_like(hess)
                        for i in range(hess.shape[0]):
                            if abs(hess[i]) < 1.0e-4:
                                self.precond[k][i] = 1.0 + 1.0j
                            else:
                                self.precond[k][i] = \
                                    1.0 / hess[i].real + 1.0j / hess[i].imag
                return self.precond
            else:
                return self.precond
        else:
            return None

    def check_mom(self, wfs, dens):
        if self.momcounter % self.momevery == 0:
            occ_name = getattr(wfs.occupations, "name", None)
            if occ_name == 'mom':
                wfs.calculate_occupation_numbers(dens.fixed)
                self.restart = sort_orbitals_according_to_occ(
                    wfs, update_mom=True, update_eps=False)
        self.momcounter += 1

    def kpointval(self, kpt):
        return self.n_kps * kpt.s + kpt.q


def log_f(log, niter, kappa, eks, esic, outer_counter=None, g_max=np.inf):

    t = time.localtime()

    if niter == 0:
        header0 = '\nINNER LOOP:\n'
        header = '                      Kohn-Sham          SIC' \
                 '        Total             \n' \
                 '           time         energy:      energy:' \
                 '      energy:       Error:       G_max:'
        log(header0 + header)

    if outer_counter is not None:
        niter = outer_counter

    log('iter: %3d  %02d:%02d:%02d ' %
        (niter,
         t[3], t[4], t[5]
         ), end='')
    log('%11.6f  %11.6f  %11.6f  %11.1e  %11.1e' %
        (Hartree * eks,
         Hartree * esic,
         Hartree * (eks + esic),
         kappa,
         Hartree * g_max), end='')
    log(flush=True)


def g_max_norm(g_k, wfs):
    # get maximum of gradients
    n_kps = wfs.kd.nibzkpts

    max_norm = []
    for kpt in wfs.kpt_u:
        k = n_kps * kpt.s + kpt.q
        dim = g_k[k].shape[0]
        if dim == 0:
            max_norm.append(0.0)
        else:
            max_norm.append(np.max(np.absolute(g_k[k])))
    max_norm = np.max(np.asarray(max_norm))
    g_max = wfs.world.max_scalar(max_norm)

    return g_max
