"""
Class for finding an optimal
step alpha length in the optimization problem:
x = x + alpha * p.
It is for lcao
"""

import numpy as np
from gpaw.directmin.tools import minimum_cubic_interpol


def is_descent(phi_0, phi_j, eps=1.0e-2):
    return phi_j <= phi_0 + eps * abs(phi_0)


def is_descent_and_approximate_wolfe_conditions(
        der_phi_0, phi_0, der_phi_j, phi_j, eps=1.0e-6, delta=0.1, sigma=0.9):
    """
     William W. Hager and Hongchao Zhang
     SIAM J. optim., 16(1), 170-192.
     """

    descent = is_descent(phi_0, phi_j, eps)
    awc = (2.0 * delta - 1.0) * der_phi_0 >= der_phi_j >= sigma * der_phi_0

    return descent and awc


def get_slength(p, wfs, mode=None):
    if mode is None:
        mode = wfs.mode
    if mode == 'lcao':
        p_all_kpts = np.hstack([p[k] for k in p])
        return np.linalg.norm(p_all_kpts)
    else:
        ret = 0.0
        for k in p:
            for val in p[k]:
                ret += np.real(wfs.integrate(val, val, global_integral=False))
        ret = wfs.world.sum_scalar(ret)
        return np.sqrt(ret)


class MaxStep:

    def __init__(self, evaluate_phi_and_der_phi, max_step=0.2):
        """

        :param evaluate_phi_and_der_phi:
        """

        self.evaluate_phi_and_der_phi = evaluate_phi_and_der_phi
        self.max_step = max_step
        self.name = 'max-step'

    def todict(self):
        return {'name': self.name,
                'max_step': self.max_step}

    def step_length_update(self, x, p, wfs, *args, mode=None, **kwargs):

        kd = kwargs['kpdescr']

        slength = get_slength(p, wfs, mode)
        slength = kd.comm.max_scalar(slength)

        a_star = self.max_step / slength if slength > self.max_step else 1.0

        phi_star, der_phi_star, g_star = self.evaluate_phi_and_der_phi(
            x, p, a_star, wfs, *args)

        return a_star, phi_star, der_phi_star, g_star


class StrongWolfeConditions(MaxStep):
    """
    From a book of Jorge Nocedal and Stephen J. Wright 'Numerical
    Optimization' Second Edition, 2006 (p. 56)

    This call should return a_star, phi_star, der_phi_star, g_star,
    where a_star is step length satisfied the strong Wolfe condts:

    f(x_k + a_k p_k) <= f(x_k) + c_1 a_k grad f_k cdot p_k,

    |grad f(x_k + a_k p_k) cdot p_k | <= c_2 |grad f_k cdot p_k|,

    or descent conditions and approximate Wolfe conditions from

    William W. Hager and Hongchao Zhang
    SIAM J. optim., 16(1), 170-192.

    phi = f (x_k + a_k*p_k)
    der_phi = grad f(x_k + a_k p_k) cdot p_k
    g = grad f(x_k + a_k p_k)
    """

    def __init__(self, evaluate_phi_and_der_phi,
                 c1=1.0e-4, c2=0.9,
                 searchdirtype=None, max_iter=3, eps_dx=1.0e-10,
                 eps_df=1.0e-10, use_descent_and_awc=True):
        """
        :param evaluate_phi_and_der_phi: function which calculate
        phi, der_phi and g for given A_s, P_s, n_dim and alpha
        A_s[s] is skew-hermitian matrix, P_s[s] is matrix direction
        :param searchdirtype: used only in initial guess for alpha
        :param max_iter: maximum number of iterations
        :param eps_dx: length of minimal interval where alpha can
        be found
        :param eps_df: minimal change of function
        :param c1: see above
        :param c2: see above
        :param use_descent_and_awc: check descent and approximate Wolfe
        conditions

        this class works fine, but these parameters eps_dx, eps_df
        might not be needed
        """

        super().__init__(
            evaluate_phi_and_der_phi)

        self.max_iter = max_iter
        self.searchdirtype = searchdirtype
        self.eps_dx = eps_dx
        self.eps_df = eps_df
        self.c1 = c1
        self.c2 = c2
        self.use_descent_and_awc = use_descent_and_awc
        self.name = 'swc-awc'

    def todict(self):
        return {'name': self.name,
                'max_iter': self.max_iter,
                'searchdirtype': self.searchdirtype,
                'eps_dx': self.eps_dx,
                'eps_df': self.eps_df,
                'c1': self.c1,
                'c2': self.c2}

    def step_length_update(self, x, p, wfs, *args, **kwargs):
        c1 = self.c1
        c2 = self.c2
        phi_0 = kwargs['phi_0']
        der_phi_0 = kwargs['der_phi_0']
        phi_old = kwargs['phi_old']
        der_phi_old = kwargs['der_phi_old']
        alpha_old = kwargs['alpha_old']
        alpha_max = kwargs['alpha_max']
        alpha_1 = self.init_guess(phi_0=phi_0, der_phi_0=der_phi_0,
                                  phi_old=phi_old,
                                  der_phi_old=der_phi_old,
                                  alpha_old=alpha_old)
        i = 1

        if phi_0 is None or der_phi_0 is None:
            phi_0, der_phi_0, g_0 = \
                self.evaluate_phi_and_der_phi(x, p, 0.0, wfs, *args)

        alpha = [0.0, alpha_1]
        phi_i_1 = phi_0
        der_phi_i_1 = der_phi_0
        max_iter = self.max_iter
        phi_max = None
        der_phi_max = None
        g_max = None

        while True:
            if np.abs(alpha[-1] - alpha_max) < 1.0e-6 and phi_max is not None:
                phi_i, der_phi_i, g_i = phi_max, der_phi_max, g_max
                phi_max = None
                der_phi_max = None
            else:
                phi_i, der_phi_i, g_i = \
                    self.evaluate_phi_and_der_phi(x, p, alpha[i], wfs, *args)

            if self.use_descent_and_awc:
                if is_descent_and_approximate_wolfe_conditions(
                        der_phi_0, phi_0, der_phi_i, phi_i):
                    a_star = alpha[i]
                    phi_star = phi_i
                    der_phi_star = der_phi_i
                    g_star = g_i
                    break

            if phi_i > phi_0 + c1 * alpha[i] * der_phi_0 or \
                    (phi_i >= phi_i_1 and i > 1):
                a_star, phi_star, der_phi_star, g_star = \
                    self.zoom(alpha[i - 1], alpha[i],
                              phi_i_1, der_phi_i_1,
                              phi_i, der_phi_i, x, p,
                              phi_0, der_phi_0, c1, c2, wfs, *args)
                break

            if np.fabs(der_phi_i) <= -c2 * der_phi_0:
                a_star = alpha[i]
                phi_star = phi_i
                der_phi_star = der_phi_i
                g_star = g_i
                break

            if der_phi_i >= 0.0:
                a_star, phi_star, der_phi_star, g_star = \
                    self.zoom(alpha[i], alpha[i - 1],
                              phi_i, der_phi_i,
                              phi_i_1, der_phi_i_1,
                              x, p, phi_0, der_phi_0, c1, c2, wfs, *args)
                break

            if i == max_iter:
                a_star = alpha[i]
                phi_star = phi_i
                der_phi_star = der_phi_i
                g_star = g_i
                break

            if np.abs(alpha_max - alpha[i]) < 1.0e-6:
                alpha_max = 1.5 * alpha[i]

            if phi_max is None or der_phi_max is None:
                phi_max, der_phi_max, g_max = \
                    self.evaluate_phi_and_der_phi(x, p, alpha_max, wfs, *args)

                if self.use_descent_and_awc:
                    if is_descent_and_approximate_wolfe_conditions(
                            der_phi_0, phi_0, der_phi_max, phi_max):
                        a_star = alpha_max
                        phi_star = phi_max
                        der_phi_star = der_phi_max
                        g_star = g_max
                        break

            a_new = minimum_cubic_interpol(
                alpha[i], alpha_max, phi_i, phi_max,
                der_phi_i, der_phi_max)

            alpha.append(a_new)
            phi_i_1 = phi_i
            der_phi_i_1 = der_phi_i

            if np.abs(a_new - alpha_max) < 1.0e-6:
                phi_i = phi_max
                der_phi_i = der_phi_max
                g_i = g_max

            if abs(alpha[-1] - alpha[-2]) < 1.0e-5:
                a_star = alpha[i]
                phi_star = phi_i
                der_phi_star = der_phi_i
                g_star = g_i
                break

            i += 1

        return a_star, phi_star, der_phi_star, g_star

    def zoom(self, a_lo, a_hi, f_lo, df_lo, f_hi, df_hi, x, p, phi_0,
             der_phi_0, c1, c2, wfs, *args):

        max_iter = self.max_iter
        i = 0

        while True:
            a_j = minimum_cubic_interpol(
                a_lo, a_hi, f_lo, f_hi, df_lo, df_hi)

            if a_j < 0.01:
                a_j = 0.1
                phi_j, der_phi_j, g_j = \
                    self.evaluate_phi_and_der_phi(x, p, a_j, wfs, *args)
                return a_j, phi_j, der_phi_j, g_j

            phi_j, der_phi_j, g_j = \
                self.evaluate_phi_and_der_phi(x, p, a_j, wfs, *args)

            if self.use_descent_and_awc:
                if is_descent_and_approximate_wolfe_conditions(
                        der_phi_0, phi_0, der_phi_j, phi_j):
                    a_star = a_j
                    phi_star = phi_j
                    der_phi_star = der_phi_j
                    g_star = g_j
                    break

            if phi_j > phi_0 + c1 * a_j * der_phi_0 or phi_j >= f_lo:
                a_hi = a_j
                f_hi = phi_j
                df_hi = der_phi_j

            else:
                if abs(der_phi_j) <= -c2 * der_phi_0:
                    a_star = a_j
                    phi_star = phi_j
                    der_phi_star = der_phi_j
                    g_star = g_j
                    break

                if der_phi_j * (a_hi - a_lo) >= 0.0:
                    a_hi = a_lo
                    f_hi = f_lo
                    df_hi = df_lo

                a_lo = a_j
                f_lo = phi_j
                df_lo = der_phi_j

            i += 1

            if np.fabs(a_lo - a_hi) < self.eps_dx and a_lo < \
                    self.eps_dx:
                a_star = a_lo
                phi_star, der_phi_star, g_star = \
                    self.evaluate_phi_and_der_phi(x, p, a_star, wfs, *args)
                break

            elif np.fabs(a_lo - a_hi) < self.eps_dx:
                a_star = a_lo
                phi_star, der_phi_star, g_star = \
                    self.evaluate_phi_and_der_phi(x, p, a_star, wfs, *args)
                break

            if i == max_iter:
                if a_lo > self.eps_dx:
                    a_star = a_lo
                    phi_star, der_phi_star, g_star = \
                        self.evaluate_phi_and_der_phi(x, p, a_star, wfs, *args)
                else:
                    a_star = a_hi
                    phi_star, der_phi_star, g_star = \
                        self.evaluate_phi_and_der_phi(x, p, a_star, wfs, *args)
                break

        return a_star, phi_star, der_phi_star, g_star

    def init_guess(self, phi_0, der_phi_0, phi_old, der_phi_old,
                   alpha_old=1.0):

        # chose initial alpha
        if self.searchdirtype in ['quasi-newton', 'newton']:
            alpha_1 = 1.0
        else:
            if phi_old is not None and der_phi_old is not None:
                try:
                    alpha_1 = 2.0 * (phi_0 - phi_old) / der_phi_old
                    if alpha_1 < 0.1:
                        if alpha_old < 0.1:
                            alpha_1 = 10.0
                        else:
                            alpha_1 = alpha_old
                except ZeroDivisionError:
                    alpha_1 = 1.0
            else:
                alpha_1 = 1.0

        return alpha_1
