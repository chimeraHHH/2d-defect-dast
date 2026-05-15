"""
Search directions in space of skew-hermitian matrices

LSR1 algorithm and application to excited states:
arXiv:2006.15922 [physics.chem-ph]
J. Chem. Theory Comput. 16, 6968 (2020).
https://doi.org/10.1021/acs.jctc.0c00597
"""

import numpy as np
import copy
from gpaw.directmin.tools import array_to_dict, dict_to_array


class SearchDirectionBase:
    """
    Base class for search direction algorithms
    """

    def __init__(self):
        self.iters = 0
        self.kp = None
        self.p = None
        self.k = None
        super().__init__()

    def __str__(self):
        raise NotImplementedError('Search direction class needs string '
                                  'representation.')

    def todict(self):
        raise NotImplementedError('Search direction class needs \'todict\' '
                                  'method.')

    def update_data(
            self, wfs, x_k1, g_k1, dimensions=None, precond=None, mode=None,
            subspace=False):
        raise NotImplementedError('Search direction class needs '
                                  '\'update_data\' method.')

    def reset(self):
        self.iters = 0
        self.kp = {}
        self.p = 0
        self.k = 0


class ModeFollowingBase:
    """
    Base gradient partitioning and negation implementation for minimum mode
    following
    """

    def __init__(self, partial_diagonalizer, convex_step_length=0.1,
                 reset_on_convex=False):
        self.eigv = None
        self.eigvec = None
        self.eigvec_old = None
        self.partial_diagonalizer = partial_diagonalizer
        self.fixed_sp_order = None
        self.was_convex = False
        self.convex_step_length = convex_step_length
        self.reset_on_convex = reset_on_convex

    def update_eigenpairs(self, g_k1, wfs, ham, dens):
        """
        Performs a partial Hessian diagonalization to obtain the eigenvectors
        with negative eigenvalues.

        :param g_k1: Gradient.
        :param wfs:
        :param ham:
        :param dens:
        :return:
        """

        self.partial_diagonalizer.grad = g_k1
        use_prev = False if self.eigv is None or (
            self.was_convex and self.reset_on_convex) else True
        self.partial_diagonalizer.run(wfs, ham, dens, use_prev)
        self.eigv = copy.deepcopy(self.partial_diagonalizer.lambda_w)
        self.eigvec_old = copy.deepcopy(self.eigvec)
        self.eigvec = copy.deepcopy(self.partial_diagonalizer.x_w.T)
        if wfs.dtype == complex:
            dimtot = int(len(self.eigvec[0]) / 2)
            eigvec = np.zeros(shape=(len(self.eigvec), dimtot),
                              dtype=complex)
            for i in range(len(self.eigvec)):
                eigvec[i] += self.eigvec[i][: dimtot] \
                    + 1.0j * self.eigvec[i][dimtot:]
            self.eigvec = eigvec
        self.fixed_sp_order = self.partial_diagonalizer.sp_order

    def invert_parallel_grad(self, g_k1):
        """
        Uses the stored eigenpairs and inverts the projections of the gradient
        parallel to the eigenvectors with negative eigenvalues.

        :param g_k1: Gradient.
        :return: Modified gradient.
        """

        grad, dim, dimtot = dict_to_array(g_k1)
        get_dots = 0
        if self.fixed_sp_order is None:
            for i in range(len(self.eigv)):
                if self.eigv[i] <= -1e-4:
                    get_dots += 1
                else:
                    break
        else:
            neg_temp = 0
            for i in range(len(self.eigv)):
                if self.eigv[i] <= -1e-4:
                    neg_temp += 1
                else:
                    break
            get_dots = self.fixed_sp_order
        grad_par = np.zeros_like(grad)
        if self.fixed_sp_order is not None:
            for i in range(get_dots):
                grad_par += self.eigvec[i] * np.dot(self.eigvec[i].conj(),
                                                    grad.T).real
            grad_mod = grad - 2.0 * grad_par
        else:
            for i in range(get_dots):
                grad_par += self.eigvec[i] \
                    * np.dot(self.eigvec[i].conj(), grad.T).real
            if get_dots == 0:
                grad_mod = -self.convex_step_length * grad_par \
                    / np.linalg.norm(grad_par)
                self.partial_diagonalizer.etdm.searchdir_algo.reset()
                self.was_convex = True
            else:
                grad_mod = grad - 2.0 * grad_par
                if self.was_convex:
                    self.partial_diagonalizer.etdm.searchdir_algo.reset()
                    self.was_convex = False
        return array_to_dict(grad_mod, dim)


class ModeFollowing(ModeFollowingBase, SearchDirectionBase):
    """
    Minimum mode following class handling the GMF tag of the search direction
    class for ETDM and negation of the gradient projection.
    """

    def __init__(self, partial_diagonalizer, search_direction,
                 convex_step_length=0.1):
        self.sd = search_direction
        self.name = self.sd.name + '_gmf'
        self.type = self.sd.type + '_gmf'
        super().__init__(partial_diagonalizer, convex_step_length)

    @property
    def beta_0(self):
        return self.sd.beta_0

    def __str__(self):
        return self.sd.__str__() + ' with minimum mode following'

    def todict(self):
        res = self.sd.todict()
        res['name'] += '_gmf'                    # tag will be removed in etdm
        return res

    def update_data(
            self, wfs, x_k1, g_k1, dimensions=None, precond=None, mode=None,
            subspace=False):
        if not subspace:
            g_k1 = self.invert_parallel_grad(g_k1)
        return self.sd.update_data(wfs, x_k1, g_k1, dimensions=dimensions,
                                   precond=precond, mode=mode)

    def reset(self):
        self.sd.reset()


class SteepestDescent(SearchDirectionBase):
    """
    Steepest descent algorithm
    """

    def __init__(self):
        super().__init__()

        self.name = 'sd'
        self.type = 'steepest-descent'

    def __str__(self):
        return 'Steepest Descent algorithm'

    def todict(self):
        return {'name': self.name}

    def update_data(
            self, wfs, x_k1, g_k1, dimensions=None, precond=None, mode=None,
            subspace=False):

        if precond is None:
            p_k = minus(g_k1)
        else:
            p_k = apply_prec(precond, g_k1, -1.0, wfs, mode)
        self.iters += 1
        return p_k


class FRcg(SteepestDescent):
    """
    The Fletcher-Reeves conj. grad. method
    See Jorge Nocedal and Stephen J. Wright 'Numerical
    Optimization' Second Edition, 2006 (p. 121)
    """

    def __init__(self):
        super().__init__()
        self.name = 'fr-cg'
        self.type = 'conjugate-gradients'

    def __str__(self):
        return 'Fletcher-Reeves conjugate gradient method'

    def todict(self):
        return {'name': self.name}

    def update_data(
            self, wfs, x_k1, g_k1, dimensions=None, precond=None, mode=None,
            subspace=False):

        if precond is not None:
            g_k1 = apply_prec(precond, g_k1, 1.0, wfs, mode)

        if self.iters == 0:
            self.p_k = minus(g_k1)
        else:
            dot_g_k1_g_k1 = dot_all_k_and_b(g_k1, g_k1, wfs, dimensions, mode)
            dot_g_g = dot_all_k_and_b(
                self.g_k, self.g_k, wfs, dimensions, mode)
            beta_k = dot_g_k1_g_k1 / dot_g_g
            self.p_k = calc_diff(self.p_k, g_k1, beta_k)

        # save this step
        self.g_k = copy.deepcopy(g_k1)

        self.iters += 1
        return self.p_k


class LBFGS(SearchDirectionBase):

    def __init__(self, memory=3):
        """
        :param memory: amount of previous steps to use
        """
        super().__init__()

        self.s_k = {i: None for i in range(memory)}
        self.y_k = {i: None for i in range(memory)}

        self.rho_k = np.zeros(shape=memory)

        self.kp = {}
        self.p = 0
        self.k = 0

        self.memory = memory
        self.stable = True
        self.name = 'l-bfgs'
        self.type = 'quasi-newton'

    def __str__(self):

        return 'L-BFGS'

    def todict(self):
        return {'name': self.name,
                'memory': self.memory}

    def update_data(
            self, wfs, x_k1, g_k1, dimensions=None, precond=None, mode=None,
            subspace=False):

        self.iters += 1

        if precond is not None:
            g_k1 = apply_prec(precond, g_k1, 1.0, wfs, mode)

        if self.k == 0:

            self.kp[self.k] = self.p
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = g_k1
            self.s_k[self.kp[self.k]] = zeros(g_k1)
            self.y_k[self.kp[self.k]] = zeros(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p

            return minus(g_k1)

        else:

            if self.p == self.memory:
                self.p = 0
                self.kp[self.k] = self.p

            s_k = self.s_k
            x_k = self.x_k
            y_k = self.y_k
            g_k = self.g_k

            x_k1 = copy.deepcopy(x_k1)

            rho_k = self.rho_k

            kp = self.kp
            k = self.k
            m = self.memory

            s_k[kp[k]] = calc_diff(x_k1, x_k)
            y_k[kp[k]] = calc_diff(g_k1, g_k)
            dot_ys = dot_all_k_and_b(
                y_k[kp[k]], s_k[kp[k]], wfs, dimensions, mode)

            if abs(dot_ys) > 1.0e-15:
                rho_k[kp[k]] = 1.0 / dot_ys
            else:
                rho_k[kp[k]] = 1.0e15

            if dot_ys < 0.0:
                self.stable = False

            q = copy.deepcopy(g_k1)

            alpha = np.zeros(np.minimum(k + 1, m))
            j = np.maximum(-1, k - m)

            for i in range(k, j, -1):
                dot_sq = dot_all_k_and_b(s_k[kp[i]], q, wfs, dimensions, mode)

                alpha[kp[i]] = rho_k[kp[i]] * dot_sq

                q = calc_diff(q, y_k[kp[i]], const=alpha[kp[i]])

            t = k
            dot_yy = dot_all_k_and_b(
                y_k[kp[t]], y_k[kp[t]], wfs, dimensions, mode)

            if abs(dot_yy) > 1.0e-15:
                r = multiply(q, 1.0 / (rho_k[kp[t]] * dot_yy))
            else:
                r = multiply(q, 1.0e15)

            for i in range(np.maximum(0, k - m + 1), k + 1):
                dot_yr = dot_all_k_and_b(
                    y_k[kp[i]], r, wfs, dimensions, mode)

                beta = rho_k[kp[i]] * dot_yr

                r = calc_diff(r, s_k[kp[i]], const=(beta - alpha[kp[i]]))

            # save this step:
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = copy.deepcopy(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p

            return multiply(r, -1.0)


class LBFGS_P(SearchDirectionBase):

    def __init__(self, memory=3, beta_0=1.0):
        """
        :param memory: amount of previous steps to use
        """
        super().__init__()
        self.s_k = {i: None for i in range(memory)}
        self.y_k = {i: None for i in range(memory)}
        self.rho_k = np.zeros(shape=memory)
        self.kp = {}
        self.p = 0
        self.k = 0
        self.memory = memory
        self.stable = True
        self.beta_0 = beta_0
        self.name = 'l-bfgs-p'
        self.type = 'quasi-newton'

    def __str__(self):

        return 'L-BFGS-P'

    def todict(self):

        return {'name': self.name,
                'memory': self.memory,
                'beta_0': self.beta_0}

    def update_data(
            self, wfs, x_k1, g_k1, dimensions=None, precond=None, mode=None,
            subspace=False):
        # For L-BFGS-P, the preconditioner passed here has to be differentiated
        # from the preconditioner passed in ETDM. To keep the UI of this member
        # function consistent, the term precond is still used in the signature
        hess_1 = precond
        self.iters += 1
        if self.k == 0:
            self.kp[self.k] = self.p
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = copy.deepcopy(g_k1)
            self.s_k[self.kp[self.k]] = zeros(g_k1)
            self.y_k[self.kp[self.k]] = zeros(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p
            if hess_1 is None:
                p = minus(g_k1)
            else:
                p = apply_prec(hess_1, g_k1, -1.0, wfs, mode)
            self.beta_0 = 1.0
            return p

        else:
            if self.p == self.memory:
                self.p = 0
                self.kp[self.k] = self.p

            s_k = self.s_k
            x_k = self.x_k
            y_k = self.y_k
            g_k = self.g_k
            rho_k = self.rho_k
            kp = self.kp
            k = self.k
            m = self.memory

            s_k[kp[k]] = calc_diff(x_k1, x_k)
            y_k[kp[k]] = calc_diff(g_k1, g_k)

            dot_ys = dot_all_k_and_b(
                y_k[kp[k]], s_k[kp[k]], wfs, dimensions, mode)

            if abs(dot_ys) > 1.0e-20:
                rho_k[kp[k]] = 1.0 / dot_ys
            else:
                rho_k[kp[k]] = 1.0e20

            if rho_k[kp[k]] < 0.0:
                self.stable = False
                self.__init__(memory=self.memory)
                return self.update_data(
                    wfs, x_k1, g_k1, precond=hess_1, dimensions=dimensions,
                    mode=mode)

            q = copy.deepcopy(g_k1)

            alpha = np.zeros(np.minimum(k + 1, m))
            j = np.maximum(-1, k - m)

            for i in range(k, j, -1):
                dot_sq = dot_all_k_and_b(s_k[kp[i]], q, wfs, dimensions, mode)
                alpha[kp[i]] = rho_k[kp[i]] * dot_sq
                q = calc_diff(q, y_k[kp[i]], const=alpha[kp[i]])

            t = k
            dot_yy = dot_all_k_and_b(
                y_k[kp[t]], y_k[kp[t]], wfs, dimensions, mode)
            rhoyy = rho_k[kp[t]] * dot_yy
            if abs(rhoyy) > 1.0e-20:
                self.beta_0 = 1.0 / rhoyy
            else:
                self.beta_0 = 1.0e20

            if hess_1 is not None:
                r = apply_prec(hess_1, q, wfs=wfs, mode=mode)
            else:
                r = multiply(q, self.beta_0)

            for i in range(np.maximum(0, k - m + 1), k + 1):
                dot_yr = dot_all_k_and_b(y_k[kp[i]], r, wfs, dimensions, mode)
                beta = rho_k[kp[i]] * dot_yr
                r = calc_diff(r, s_k[kp[i]], const=(beta - alpha[kp[i]]))

            # save this step:
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = copy.deepcopy(g_k1)

            self.k += 1
            self.p += 1

            self.kp[self.k] = self.p

            return multiply(r, -1.0)


class LSR1P(SearchDirectionBase):
    """
    This class describes limited memory versions of
    SR-1, Powell and their combinations (such as Bofill).
    """

    def __init__(self, memory=20, method='LSR1', phi=None):
        """
        :param memory: amount of previous steps to use
        """
        super().__init__()

        self.u_k = {i: None for i in range(memory)}
        self.j_k = {i: None for i in range(memory)}
        self.yj_k = np.zeros(shape=memory)
        self.method = method
        self.phi = phi

        self.phi_k = np.zeros(shape=memory)
        if self.phi is None:
            assert self.method in ['LSR1', 'LP', 'LBofill',
                                   'Linverse_Bofill'], 'Value Error'
            if self.method == 'LP':
                self.phi_k.fill(1.0)
        else:
            self.phi_k.fill(self.phi)

        self.kp = {}
        self.p = 0
        self.k = 0

        self.memory = memory
        self.name = 'l-sr1p'
        self.type = 'quasi-newton'

    def __str__(self):

        return 'LSR1P'

    def todict(self):

        return {'name': self.name,
                'memory': self.memory,
                'method': self.method}

    def update_data(
            self, wfs, x_k1, g_k1, dimensions=None, precond=None, mode=None,
            subspace=False):

        if precond is not None:
            bg_k1 = apply_prec(precond, g_k1, 1.0, wfs, mode)
        else:
            bg_k1 = g_k1.copy()

        if self.k == 0:
            self.kp[self.k] = self.p
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = copy.deepcopy(g_k1)
            self.u_k[self.kp[self.k]] = zeros(g_k1)
            self.j_k[self.kp[self.k]] = zeros(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p
        else:
            if self.p == self.memory:
                self.p = 0
                self.kp[self.k] = self.p

            x_k = self.x_k
            g_k = self.g_k
            u_k = self.u_k
            j_k = self.j_k
            yj_k = self.yj_k
            phi_k = self.phi_k

            x_k1 = copy.deepcopy(x_k1)

            kp = self.kp
            k = self.k
            m = self.memory

            s_k = calc_diff(x_k1, x_k)
            y_k = calc_diff(g_k1, g_k)
            if precond is not None:
                by_k = apply_prec(precond, y_k, 1.0, wfs, mode)
            else:
                by_k = y_k.copy()

            by_k = self.update_bv(wfs, by_k, y_k, u_k, j_k, yj_k, phi_k,
                                  np.maximum(1, k - m), k, dimensions, mode)

            j_k[kp[k]] = calc_diff(s_k, by_k)
            yj_k[kp[k]] = dot_all_k_and_b(
                y_k, j_k[kp[k]], wfs, dimensions, mode)

            if self.method == 'LSR1':
                if abs(yj_k[kp[k]]) < 1e-12:
                    yj_k[kp[k]] = 1e-12

            dot_yy = dot_all_k_and_b(y_k, y_k, wfs, dimensions, mode)
            if abs(dot_yy) > 1.0e-15:
                u_k[kp[k]] = multiply(y_k, 1.0 / dot_yy)
            else:
                u_k[kp[k]] = multiply(y_k, 1.0e15)

            if self.method == 'LBofill' and self.phi is None:
                jj_k = dot_all_k_and_b(
                    j_k[kp[k]], j_k[kp[k]], wfs, dimensions, mode)
                phi_k[kp[k]] = 1 - yj_k[kp[k]]**2 / (dot_yy * jj_k)
            elif self.method == 'Linverse_Bofill' and self.phi is None:
                jj_k = dot_all_k_and_b(
                    j_k[kp[k]], j_k[kp[k]], wfs, dimensions, mode)
                phi_k[kp[k]] = yj_k[kp[k]] ** 2 / (dot_yy * jj_k)

            bg_k1 = self.update_bv(wfs, bg_k1, g_k1, u_k, j_k, yj_k, phi_k,
                                   np.maximum(1, k - m + 1), k + 1, dimensions,
                                   mode)

            # save this step:
            self.x_k = copy.deepcopy(x_k1)
            self.g_k = copy.deepcopy(g_k1)
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p

        self.iters += 1
        return multiply(bg_k1, -1.0)

    def update_bv(
            self, wfs, bv, v, u_k, j_k, yj_k, phi_k, i_0, i_m, dimensions=None,
            mode=None):
        if mode is None:
            mode = wfs.mode

        kp = self.kp
        for i in range(i_0, i_m):
            dot_uv = dot_all_k_and_b(u_k[kp[i]], v, wfs, dimensions, mode)
            dot_jv = dot_all_k_and_b(j_k[kp[i]], v, wfs, dimensions, mode)
            alpha = dot_jv - yj_k[kp[i]] * dot_uv
            beta_p = calc_diff(j_k[kp[i]], u_k[kp[i]], dot_uv, -alpha)
            beta_ms = multiply(j_k[kp[i]], dot_jv / yj_k[kp[i]])
            beta = calc_diff(beta_ms, beta_p, 1 - phi_k[kp[i]], -phi_k[kp[i]])
            bv = calc_diff(bv, beta, const=-1.0)

        return bv


def multiply(x, const=1.0):
    """
    it must not change x!
    :param x:
    :param const:
    :return: new dictionary y = cons*x
    """
    y = {}
    for k in x.keys():
        y[k] = const * x[k]
    return y


def zeros(x):
    y = {}
    for k in x.keys():
        y[k] = np.zeros_like(x[k])
    return y


def minus(x):
    return multiply(x, -1.0)


def calc_diff(x1, x2, const_0=1.0, const=1.0):
    y_k = {}
    for k in x1.keys():
        y_k[k] = const_0 * x1[k] - const * x2[k]
    return y_k


def dot_all_k_and_b(x1, x2, wfs, dimensions=None, mode=None):
    if mode is None:
        mode = wfs.mode
    dot_pr_x1x2 = 0.0
    if mode == 'lcao':
        for k in x1.keys():
            dot_pr_x1x2 += np.dot(x1[k].conj(), x2[k]).real
    else:
        dot_pr_x1x2 = 0.0j if wfs.dtype == complex else 0.0
        for k, kpt in enumerate(wfs.kpt_u):
            for i in range(dimensions[k]):
                dot_prod = wfs.integrate(x1[k][i], x2[k][i], False)
                dot_prod = wfs.gd.comm.sum_scalar(dot_prod)
                dot_pr_x1x2 += dot_prod
        dot_pr_x1x2 = wfs.kd.comm.sum_scalar(dot_pr_x1x2)
        dot_pr_x1x2 = 2.0 * dot_pr_x1x2.real

    return dot_pr_x1x2


def apply_prec(prec, x, const=1.0, wfs=None, mode=None):
    if mode is None:
        mode = wfs.mode
    y = {}
    if mode == 'lcao':
        for k in x.keys():
            if prec[k].dtype == complex:
                y[k] = const * (prec[k].real * x[k].real
                                + 1.0j * prec[k].imag * x[k].imag)
            else:
                y[k] = const * prec[k] * x[k]
        return y
    elif mode == 'pw':
        deg = (3.0 - wfs.kd.nspins)
        deg *= 2.0
        for k, kpt in enumerate(wfs.kpt_u):
            y[k] = x[k].copy()
            for i, z in enumerate(x[k]):
                psit_G = kpt.psit.array[i]
                ekin = prec.calculate_kinetic_energy(psit_G, kpt)
                y[k][i] = - const * prec(z, kpt, ekin) / deg
    else:
        deg = (3.0 - wfs.kd.nspins)
        for k, kpt in enumerate(wfs.kpt_u):
            y[k] = x[k].copy()
            for i, z in enumerate(x[k]):
                y[k][i] = - const * prec(z, kpt, None) / deg
    return y
