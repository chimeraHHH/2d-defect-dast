"""
A class for finding optimal orbitals corresponding to a stationary point of
the energy functional using direct optimization and exponential transformation
in FD and PW modes.

It can be used with Kohn-Sham and hybrid (exact exchange) functionals, and
can include Perdew-Zunger self-interaction correction (PZ-SIC) in the
calculations.

Ground state as well as variational excited state calculations can be
performed. Ground state calculations involve minimization of the energy in a
direction tangent to the orbitals without the exponential transformation
(direct minimization). For excited state calculations, the energy is optimized
by converging on a saddle point, which involves an inner loop using the
exponential transformation (direct optimization). PZ-SIC requires an
additional inner loop to minimize the energy with respect to unitary
transformation of the occupied orbitals (inner loop localization).

GPAW Implementation of direct optimization with preconditioned quasi-Newton
algorithms and maximum overlap method (DO-MOM) for excited state calculations
FD and PW modes:

    J. Chem. Theory Comput. 17, 5034–5049 (2021) :doi:10.1021/acs.jctc.1c00157
    arXiv:2102.06542 [physics.comp-ph]
"""

import time

from ase.parallel import parprint
from ase.units import Hartree
from ase.utils import basestring
import numpy as np

from gpaw.directmin import search_direction, line_search_algorithm
from gpaw.directmin.fdpw.etdm_inner_loop import ETDMInnerLoop
from gpaw.directmin.fdpw.pz_localization import PZLocalization
from gpaw.directmin.functional.fdpw import get_functional
from gpaw.directmin.locfunc.localize_orbitals import localize_orbitals
from gpaw.directmin.tools import get_n_occ, sort_orbitals_according_to_occ
from gpaw.utilities import unpack_hermitian
from gpaw.xc import xc_string_to_dict
from gpaw.xc.hybrid import HybridXC


class FDPWETDM:

    def __init__(self,
                 excited_state=False,
                 searchdir_algo=None,
                 linesearch_algo='max-step',
                 use_prec=True,
                 functional='ks',
                 need_init_orbs=None,
                 need_localization=True,
                 localizationtype=None,
                 localizationseed=None,
                 localization_tol=None,
                 maxiter_pz_localization=50,
                 maxiter_inner_loop=100,
                 max_step_inner_loop=0.2,
                 grad_tol_pz_localization=5.0e-4,
                 grad_tol_inner_loop=5.0e-4,
                 restartevery_iloop_notconverged=3,
                 restart_canonical=True,
                 momevery=10,
                 printinnerloop=False,
                 blocksize=1,
                 converge_unocc=False,
                 maxiter_unocc=333
                 ):
        """Class for direct orbital optimization in FD and PW modes.

        Parameters
        ----------
        excited_state: bool
            If False (default), perform a minimization in the tangent space of
            orbitals (ground state calculation), and set need_init_orbs to
            True. Otherwise, perform outer loop minimization in the tangent
            space and inner loop optimization with exponential transformation
            (excited state calculation), and set need_init_orbs to False.
        searchdir_algo: str, dict or instance
            Search direction algorithm for the outer loop minimization. Can be
            one of the algorithms available in sd_etdm.py:
                'sd': Steepest descent
                'fr-cg': Fletcher-Reeves conjugate gradient
                'l-bfgs': Limited-memory BFGS (default)
                'l-bfgs-p': Limited-memory BFGS with preconditioner presented
                    in :doi:`10.1016/j.cpc.2021.108047`
                'l-sr1p': Limited-memory SR1 algorithm presented in
                    :doi:`10.1021/acs.jctc.0c00597`
            The default memory for 'l-bfgs'/'l-bfgs-p' and 'l-sr1p' is 3 and
            20, respectively, and can be changed by supplying a dictionary:
            {'name': name, 'memory': memory}, where name should be 'l-bfgs',
            'l-bfgs-p' or 'l-sr1p' and memory should be an int.
        linesearch_algo: str, dict or instance
            Line search algorithm for the outer loop minimization. Can be one
            of the algorithms available in ls_etdm.py:
                'max-step': The quasi-Newton step is scaled if it exceeds a
                    maximum step length (default). The default maximum step
                    length is 0.20, and can be changed by supplying a
                    dictionary: {'name': 'max-step', 'max_step': max_step},
                    where max_step should be a float.
                'swc-awc': Line search with Wolfe conditions
        use_prec: bool
            If True (default) use a preconditioner. For the outer loop
            minimization, the preconditioner is the inverse kinetic energy
            operator. For the inner loop optimization, the preconditioner is
            the inverse of a diagonal approximation of the Hessian (see
            :doi:`10.1021/j100322a012`) apart for 'l-bfgs-p', which uses the
            composite preconditioner presented in
            :doi:`10.1016/j.cpc.2021.108047`.
        functional: str, dict or instance
            Type of functional. Can be one of:
                'ks': The functional as specified in the GPAW calculator is
                    used (default)
                'pz-sic': Apply the Perdew-Zunger self-interaction correction
                    on top of the functional as specified in the GPAW
                    calculator. Dy default full SIC is applied. A scaling
                    factor for SIC can be given by supplying a dictionary:
                    functional={'name': 'pz-sic', 'scaling_factor': (a, a)},
                    where a is the scaling factor (float).
        need_init_orbs: bool
            If True (default when excited_state is False), obtain initial
            orbitals from eigendecomposition of the Hamiltonian matrix. If
            False (default when excited_state is True), use orbitals stored in
            wfs object as initial guess.
        need_localization: bool
            If True (default), localize initial guess orbitals. Requires a
            specification of localizationtype, otherwise it is set to False.
            Recommended for calculations with PZ-SIC.
        localizationtype: str
            Method for localizing the initial guess orbitals. Can be one of:
                'pz': Unitary optimization among occupied orbitals (subspace
                    optimization) with PZ-SIC
                'ks':
                'er': Edmiston-Ruedenberg localization
                'pm': Pipek-Mezey localization (recommended for PZ-SIC)
                'fb' Foster-Boys localization
            Default is None, meaning that no localization is performed.
        localizationseed: int
            Seed for Edmiston-Ruedenberg, Pipek-Mezey or Foster-Boys
            localization. Default is None (no seed is used).
        localization_tol: float
            Tolerance for convergence of the localization. If not specified,
            the following default values will be used:
            'pz': 5.0e-4
            'ks': 5.0e-4
            'er': 5.0e-5
            'pm': 1.0e-10
            'fb': 1.0e-10
        maxiter_pz_localization: int
            Maximum number of iterations for PZ-SIC inner loop localization.
        maxiter_inner_loop: int
            Maximum number of iterations of inner loop optimization for
            excited state calculations. If the maximum number of inner loop
            iterations is exceeded, the optimization moves on with the outer
            loop step.
        max_step_inner_loop: float
            Maximum step length of inner loop optimization for excited state
            calculations. The inner loop optimization uses the 'l-sr1p' search
            direction. Default is 0.20.
        grad_tol_pz_localization: float
            Tolerance on the norm of the gradient for convergence of the
            PZ-SIC inner loop localization.
        grad_tol_inner_loop: float
            Tolerance on the norm of the gradient for convergence of the
            inner loop optimization for excited state calculations.
        restartevery_iloop_notconverged: int
            Number of iterations of the outer loop after which the calculation
            is restarted if the inner loop optimization for excited states is
            not converged.
        restart_canonical: bool
            If True (default) restart the calculations using orbitals from the
            eigedecomposition of the Hamiltonian matrix if the inner loop
            optimization does not converge or MOM detects variational
            collapse. Otherwise, the optimal orbitals are used.
        momevery: int
            MOM is applied every 'momevery' iterations of the inner loop
            optimization for excited states.
        printinnerloop: bool
            If True, print the iterations of the inner loop optimization for
            excited states to standard output. Default is False.
        blocksize: int
            Blocksize for base eigensolver class.
        converge_unocc: bool
            If True, converge also the unoccupied orbitals after convergence
            of the occupied orbitals. Default is False.
        maxiter_unocc: int
            Maximum number of iterations for convergence of the unoccupied
            orbitals.
        """

        self.error = np.inf
        self.blocksize = blocksize

        self.name = 'etdm-fdpw'
        self.sda = searchdir_algo
        self.lsa = linesearch_algo
        self.use_prec = use_prec
        self.func_settings = functional
        self.need_init_orbs = need_init_orbs
        self.localizationtype = localizationtype
        self.localizationseed = localizationseed
        self.need_localization = need_localization
        self.localization_tol = localization_tol
        self.maxiter_pz_localization = maxiter_pz_localization
        self.maxiter_inner_loop = maxiter_inner_loop
        self.max_step_inner_loop = max_step_inner_loop
        self.grad_tol_pz_localization = grad_tol_pz_localization
        self.grad_tol_inner_loop = grad_tol_inner_loop
        self.printinnerloop = printinnerloop
        self.converge_unocc = converge_unocc
        self.maxiter_unocc = maxiter_unocc
        self.restartevery_iloop_notconverged = restartevery_iloop_notconverged
        self.restart_canonical = restart_canonical
        self.momevery = momevery
        self.excited_state = excited_state
        self.check_inputs_and_init_search_algo()

        self.eg_count_iloop = 0
        self.total_eg_count_iloop = 0
        self.eg_count_outer_iloop = 0
        self.total_eg_count_outer_iloop = 0
        self.initial_random = True

        # for mom
        self.initial_occupation_numbers = None

        self.eg_count = 0
        self.etotal = 0.0
        self.globaliters = 0
        self.need_init_odd = True
        self.initialized = False

        self.gpaw_new = False

    def check_inputs_and_init_search_algo(self):
        defaults = self.set_defaults()
        if self.need_init_orbs is None:
            self.need_init_orbs = defaults['need_init_orbs']
        if self.localizationtype is None:
            self.need_localization = False

        if self.sda is None:
            self.sda = 'LBFGS'
        self.searchdir_algo = search_direction(self.sda)

        self.line_search = line_search_algorithm(
            self.lsa, self.evaluate_phi_and_der_phi,
            self.searchdir_algo)

        if isinstance(self.func_settings, basestring):
            self.func_settings = xc_string_to_dict(self.func_settings)

    def set_defaults(self):
        if self.excited_state:
            return {'need_init_orbs': False}
        else:
            return {'need_init_orbs': True}

    def __repr__(self):

        sda_name = self.searchdir_algo.name
        lsa_name = self.line_search.name
        if isinstance(self.func_settings, basestring):
            func_name = self.func_settings
        else:
            func_name = self.func_settings['name']

        sds = {'sd': 'Steepest Descent',
               'fr-cg': 'Fletcher-Reeves conj. grad. method',
               'l-bfgs': 'L-BFGS algorithm',
               'l-bfgs-p': 'L-BFGS algorithm with preconditioning',
               'l-sr1p': 'Limited-memory SR1P algorithm'}

        lss = {'max-step': 'step size equals one',
               'swc-awc': 'Inexact line search based on cubic interpolation,\n'
                          '                    strong and approximate Wolfe '
                          'conditions'}

        repr_string = 'Direct minimisation using exponential ' \
                      'transformation.\n'
        repr_string += '       ' \
                       'Search ' \
                       'direction: {}\n'.format(sds[sda_name])
        repr_string += '       ' \
                       'Line ' \
                       'search: {}\n'.format(lss[lsa_name])
        repr_string += '       ' \
                       'Preconditioning: {}\n'.format(self.use_prec)
        repr_string += '       ' \
                       'Orbital-density self-interaction ' \
                       'corrections: {}\n'.format(func_name)
        repr_string += '       ' \
                       'WARNING: do not use it for metals as ' \
                       'occupation numbers are\n' \
                       '                ' \
                       'not found variationally\n'

        return repr_string

    def reset(self, need_init_odd=True):
        self.initialized = False
        self.need_init_odd = need_init_odd
        self.searchdir_algo.reset()

    def todict(self):
        """
        Convert to dictionary, needs for saving and loading gpw
        :return:
        """
        return {'name': 'etdm-fdpw',
                'searchdir_algo': self.searchdir_algo.todict(),
                'linesearch_algo': self.line_search.todict(),
                'use_prec': self.use_prec,
                'functional': self.func_settings,
                'need_init_orbs': self.need_init_orbs,
                'localizationtype': self.localizationtype,
                'localizationseed': self.localizationseed,
                'need_localization': self.need_localization,
                'localization_tol': self.localization_tol,
                'maxiter_pz_localization': self.maxiter_pz_localization,
                'maxiter_inner_loop': self.maxiter_inner_loop,
                'max_step_inner_loop': self.max_step_inner_loop,
                'momevery': self.momevery,
                'grad_tol_pz_localization': self.grad_tol_pz_localization,
                'grad_tol_inner_loop': self.grad_tol_inner_loop,
                'restartevery_iloop_notconverged':
                    self.restartevery_iloop_notconverged,
                'restart_canonical': self.restart_canonical,
                'printinnerloop': self.printinnerloop,
                'blocksize': self.blocksize,
                'converge_unocc': self.converge_unocc,
                'maxiter_unocc': self.maxiter_unocc,
                'excited_state': self.excited_state
                }

    def initialize_dm_helper(self, wfs, ham, dens, log):
        self.initialize_eigensolver(wfs, ham)
        self.initialize_orbitals(wfs, ham)

        if not wfs.read_from_file_init_wfs_dm or self.excited_state:
            wfs.calculate_occupation_numbers(dens.fixed)

        self.initial_sort_orbitals(wfs)

        # localize orbitals?
        self.localize(wfs, dens, ham, log)

        # MOM
        self.initialize_mom_reference_orbitals(wfs, dens)

        # initialize search direction, line search and inner loops
        self.initialize_dm(wfs, dens, ham)

    def initialize_eigensolver(self, wfs, ham):
        """
        Initialize base eigensolver class

        :param wfs:
        :param ham:
        :return:
        """
        if isinstance(ham.xc, HybridXC):
            self.blocksize = wfs.bd.mynbands

        if self.blocksize is None:
            if wfs.mode == 'pw':
                S = wfs.pd.comm.size
                # Use a multiple of S for maximum efficiency
                self.blocksize = int(np.ceil(10 / S)) * S
            else:
                self.blocksize = 10

        from gpaw.eigensolvers.eigensolver import Eigensolver
        self.eigensolver = Eigensolver(keep_htpsit=False,
                                       blocksize=self.blocksize)
        self.eigensolver.initialize(wfs)

    def initialize_dm(
            self, wfs, dens, ham, converge_unocc=False):

        """
        initialize search direction algorithm,
        line search method, SIC corrections

        :param wfs:
        :param dens:
        :param ham:
        :param converge_unocc:
        :return:
        """

        self.searchdir_algo.reset()

        self.dtype = wfs.dtype
        self.n_kps = wfs.kd.nibzkpts
        # dimensionality, number of state to be converged
        self.dimensions = {}
        for kpt in wfs.kpt_u:
            bd = self.eigensolver.bd
            nocc = get_n_occ(kpt)[0]
            if converge_unocc:
                assert nocc < bd.nbands, \
                    'Please add empty bands in order to converge the' \
                    ' unoccupied orbitals'
                dim = bd.nbands - nocc
            elif self.excited_state:
                dim = bd.nbands
            else:
                dim = nocc

            k = self.n_kps * kpt.s + kpt.q
            self.dimensions[k] = dim

        if self.use_prec:
            self.prec = wfs.make_preconditioner(1)
        else:
            self.prec = None

        self.iters = 0
        self.alpha = 1.0  # step length
        self.phi_2i = [None, None]  # energy at last two iterations
        self.der_phi_2i = [None, None]  # energy gradient w.r.t. alpha
        self.grad_knG = None

        # odd corrections
        if self.need_init_odd:
            self.odd = get_functional(self.func_settings, wfs, dens, ham)
            self.e_sic = 0.0

            if 'SIC' in self.odd.name:
                self.iloop = PZLocalization(
                    self.odd, wfs, self.maxiter_pz_localization,
                    g_tol=self.grad_tol_pz_localization)
            else:
                self.iloop = None

            if self.excited_state:
                self.outer_iloop = ETDMInnerLoop(
                    self.odd, wfs, 'all', self.maxiter_inner_loop,
                    self.max_step_inner_loop, g_tol=self.grad_tol_inner_loop,
                    useprec=True, momevery=self.momevery)
                # if you have inner-outer loop then need to have
                # U matrix of the same dimensionality in both loops
                if 'SIC' in self.odd.name:
                    for kpt in wfs.kpt_u:
                        k = self.n_kps * kpt.s + kpt.q
                        self.iloop.U_k[k] = self.outer_iloop.U_k[k].copy()
            else:
                self.outer_iloop = None

        self.total_eg_count_iloop = 0
        self.total_eg_count_outer_iloop = 0

        self.initialized = True
        wfs.read_from_file_init_wfs_dm = False

    def initial_sort_orbitals(self, wfs):
        occ_name = getattr(wfs.occupations, "name", None)
        if occ_name == 'mom' and self.globaliters == 0:
            update_mom = True
            self.initial_occupation_numbers = \
                wfs.occupations.numbers.copy()
        else:
            update_mom = False
        sort_orbitals_according_to_occ(wfs, update_mom=update_mom)

    def iterate(self, ham, wfs, dens, log, converge_unocc=False):
        """
        One iteration of outer loop direct minimization

        :param ham:
        :param wfs:
        :param dens:
        :param log:
        :return:
        """

        n_kps = self.n_kps
        psi_copy = {}
        alpha = self.alpha
        phi_2i = self.phi_2i
        der_phi_2i = self.der_phi_2i

        wfs.timer.start('Direct Minimisation step')

        if self.iters == 0:
            # calculate gradients
            if not converge_unocc:
                phi_2i[0], grad_knG = \
                    self.get_energy_and_tangent_gradients(ham, wfs, dens)
            else:
                phi_2i[0], grad_knG = \
                    self.get_energy_and_tangent_gradients_unocc(ham, wfs)
        else:
            grad_knG = self.grad_knG

        wfs.timer.start('Get Search Direction')
        for kpt in wfs.kpt_u:
            k = n_kps * kpt.s + kpt.q
            if not converge_unocc:
                psi_copy[k] = kpt.psit_nG.copy()
            else:
                n_occ = get_n_occ(kpt)[0]
                dim = self.dimensions[k]
                psi_copy[k] = kpt.psit_nG[n_occ:n_occ + dim].copy()

        p_knG = self.searchdir_algo.update_data(
            wfs, psi_copy, grad_knG, precond=self.prec,
            dimensions=self.dimensions)
        self.project_search_direction(wfs, p_knG)
        wfs.timer.stop('Get Search Direction')

        # recalculate derivative with new search direction
        # as we used preconditiner before
        # here we project search direction on prec. gradients,
        # but should be just grad. But, it seems also works fine
        der_phi_2i[0] = 0.0
        for kpt in wfs.kpt_u:
            k = n_kps * kpt.s + kpt.q
            for i, g in enumerate(grad_knG[k]):
                if kpt.f_n[i] > 1.0e-10:
                    der_phi_2i[0] += self.dot(
                        wfs, g, p_knG[k][i], kpt, addpaw=False).item().real
        der_phi_2i[0] = wfs.kd.comm.sum_scalar(der_phi_2i[0])

        alpha, phi_alpha, der_phi_alpha, grad_knG = \
            self.line_search.step_length_update(
                psi_copy, p_knG, wfs, ham, dens, converge_unocc,
                phi_0=phi_2i[0], der_phi_0=der_phi_2i[0],
                phi_old=phi_2i[1], der_phi_old=der_phi_2i[1],
                alpha_max=3.0, alpha_old=alpha, kpdescr=wfs.kd)
        self.alpha = alpha
        self.grad_knG = grad_knG

        # and 'shift' phi, der_phi for the next iteration
        phi_2i[1], der_phi_2i[1] = phi_2i[0], der_phi_2i[0]
        phi_2i[0], der_phi_2i[0] = phi_alpha, der_phi_alpha,

        self.iters += 1
        if not converge_unocc:
            self.globaliters += 1
        wfs.timer.stop('Direct Minimisation step')
        return phi_2i[0], self.error

    def update_ks_energy(self, ham, wfs, dens, updateproj=True):
        """Update Kohn-Sham energy.

        It assumes the temperature is zero K.
        """

        if updateproj:
            # calc projectors
            with wfs.timer('projections'):
                for kpt in wfs.kpt_u:
                    wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

        dens.update(wfs)
        ham.update(dens, wfs, False)

        return ham.get_energy(0.0, wfs, False)

    def evaluate_phi_and_der_phi(self, psit_k, search_dir, alpha, wfs, ham,
                                 dens, converge_unocc, phi=None, grad_k=None):
        """
        phi = E(x_k + alpha_k*p_k)
        der_phi = grad_alpha E(x_k + alpha_k*p_k) cdot p_k
        :return:  phi, der_phi # floats
        """

        if phi is None or grad_k is None:
            alpha1 = np.array([alpha])
            wfs.world.broadcast(alpha1, 0)
            alpha = alpha1[0]

            x_knG = \
                {k: psit_k[k] +
                    alpha * search_dir[k] for k in psit_k.keys()}
            if not converge_unocc:
                phi, grad_k = self.get_energy_and_tangent_gradients(
                    ham, wfs, dens, psit_knG=x_knG)
            else:
                phi, grad_k = self.get_energy_and_tangent_gradients_unocc(
                    ham, wfs, x_knG)

        der_phi = 0.0
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            for i, g in enumerate(grad_k[k]):
                if not converge_unocc and kpt.f_n[i] > 1.0e-10:
                    der_phi += self.dot(
                        wfs, g, search_dir[k][i], kpt,
                        addpaw=False).item().real
                else:
                    der_phi += self.dot(
                        wfs, g, search_dir[k][i], kpt,
                        addpaw=False).item().real
        der_phi = wfs.kd.comm.sum_scalar(der_phi)

        return phi, der_phi, grad_k

    def get_energy_and_tangent_gradients(
            self, ham, wfs, dens, psit_knG=None, updateproj=True):

        """
        calculate energy for a given wfs, gradient dE/dpsi
        and then project gradient on tangent space to psi

        :param ham:
        :param wfs:
        :param dens:
        :param psit_knG:
        :return:
        """

        n_kps = self.n_kps
        if psit_knG is not None:
            for kpt in wfs.kpt_u:
                k = n_kps * kpt.s + kpt.q
                kpt.psit_nG[:] = psit_knG[k].copy()
                wfs.orthonormalize(kpt)
        elif not wfs.orthonormalized:
            wfs.orthonormalize()
        if not self.excited_state:
            energy = self.update_ks_energy(
                ham, wfs, dens, updateproj=updateproj)
            grad = self.get_gradients_2(ham, wfs)

            if 'SIC' in self.odd.name:
                e_sic = 0.0
                if self.iters > 0:
                    self.run_inner_loop(ham, wfs, dens, grad_knG=grad)
                else:
                    for kpt in wfs.kpt_u:
                        e_sic += self.odd.get_energy_and_gradients_kpt(
                            wfs, kpt, grad, self.iloop.U_k, add_grad=True)
                    self.e_sic = wfs.kd.comm.sum_scalar(e_sic)
                    ham.get_energy(0.0, wfs, kin_en_using_band=False,
                                   e_sic=self.e_sic)
                energy += self.e_sic
        else:
            grad = {}
            n_kps = self.n_kps
            for kpt in wfs.kpt_u:
                grad[n_kps * kpt.s + kpt.q] = np.zeros_like(kpt.psit_nG[:])
            self.run_inner_loop(ham, wfs, dens, grad_knG=grad)
            energy = self.etotal

        self.project_gradient(wfs, grad)
        self.error = self.error_eigv(wfs, grad)
        self.eg_count += 1
        return energy, grad

    def get_gradients_2(self, ham, wfs, scalewithocc=True):

        """
        calculate gradient dE/dpsi
        :return: H |psi_i>
        """

        grad_knG = {}
        n_kps = self.n_kps

        for kpt in wfs.kpt_u:
            grad_knG[n_kps * kpt.s + kpt.q] = \
                self.get_gradients_from_one_k_point_2(
                    ham, wfs, kpt, scalewithocc)

        return grad_knG

    def get_gradients_from_one_k_point_2(
            self, ham, wfs, kpt, scalewithocc=True):
        """
        calculate gradient dE/dpsi for one k-point
        :return: H |psi_i>
        """
        nbands = wfs.bd.mynbands
        Hpsi_nG = wfs.empty(nbands, q=kpt.q)
        wfs.apply_pseudo_hamiltonian(kpt, ham, kpt.psit_nG, Hpsi_nG)

        c_axi = {}
        if self.gpaw_new:
            dH_asii = ham.potential.dH_asii
            for a, P_xi in kpt.P_ani.items():
                dH_ii = dH_asii[a][kpt.s]
                c_xi = np.dot(P_xi, dH_ii)
                c_axi[a] = c_xi
        else:
            for a, P_xi in kpt.P_ani.items():
                dH_ii = unpack_hermitian(ham.dH_asp[a][kpt.s])
                c_xi = np.dot(P_xi, dH_ii)
                c_axi[a] = c_xi

        # not sure about this:
        ham.xc.add_correction(
            kpt, kpt.psit_nG, Hpsi_nG, kpt.P_ani, c_axi, n_x=None,
            calculate_change=False)
        # add projectors to the H|psi_i>
        wfs.pt.add(Hpsi_nG, c_axi, kpt.q)
        # scale with occupation numbers
        if scalewithocc:
            for i, f in enumerate(kpt.f_n):
                Hpsi_nG[i] *= f
        return Hpsi_nG

    def project_gradient(self, wfs, p_knG):
        """
        project gradient dE/dpsi on tangent space at psi
        See Eq.(22) and minimization algorithm p. 12 in
        arXiv:2102.06542v1 [physics.comp-ph]
        :return: H |psi_i>
        """

        n_kps = self.n_kps
        for kpt in wfs.kpt_u:
            kpoint = n_kps * kpt.s + kpt.q
            self.project_gradient_for_one_k_point(wfs, p_knG[kpoint], kpt)

    def project_gradient_for_one_k_point(self, wfs, p_nG, kpt):
        """
        project gradient dE/dpsi on tangent space at psi
        for one k-point.
        See Eq.(22) and minimization algorithm p. 12 in
        arXiv:2102.06542v1 [physics.comp-ph]
        :return: H |psi_i>
        """

        k = self.n_kps * kpt.s + kpt.q
        n_occ = self.dimensions[k]
        psc = wfs.integrate(p_nG[:n_occ], kpt.psit_nG[:n_occ], True)
        psc = 0.5 * (psc.conj() + psc.T)
        s_psit_nG = self.apply_S(wfs, kpt.psit_nG, kpt, kpt.P_ani)
        p_nG[:n_occ] -= np.tensordot(psc, s_psit_nG[:n_occ], axes=1)

    def project_search_direction(self, wfs, p_knG):

        """
        Project search direction on tangent space at psi
        it is slighlt different from project grad
        as it doesn't apply overlap matrix because of S^{-1}

        :param wfs:
        :param p_knG:
        :return:
        """

        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            n_occ = self.dimensions[k]
            psc = self.dot(
                wfs, p_knG[k][:n_occ], kpt.psit_nG[:n_occ], kpt, addpaw=False)
            psc = 0.5 * (psc.conj() + psc.T)
            p_knG[k][:n_occ] -= np.tensordot(psc, kpt.psit_nG[:n_occ], axes=1)

    def apply_S(self, wfs, psit_nG, kpt, proj_psi=None):
        """
        apply overlap matrix

        :param wfs:
        :param psit_nG:
        :param kpt:
        :param proj_psi:
        :return:
        """

        if proj_psi is None:
            proj_psi = wfs.pt.dict(shape=wfs.bd.mynbands)
            wfs.pt.integrate(psit_nG, proj_psi, kpt.q)

        s_axi = {}
        for a, P_xi in proj_psi.items():
            dO_ii = wfs.setups[a].dO_ii
            s_xi = np.dot(P_xi, dO_ii)
            s_axi[a] = s_xi

        new_psi_nG = psit_nG.copy()
        wfs.pt.add(new_psi_nG, s_axi, kpt.q)

        return new_psi_nG

    def dot(self, wfs, psi_1, psi_2, kpt, addpaw=True):
        """
        dor product between two arrays psi_1 and psi_2

        :param wfs:
        :param psi_1:
        :param psi_2:
        :param kpt:
        :param addpaw:
        :return:
        """

        dot_prod = wfs.integrate(psi_1, psi_2, global_integral=True)
        if not addpaw:
            if len(psi_1.shape) == 4 or len(psi_1.shape) == 2:
                sum_dot = dot_prod
            else:
                sum_dot = np.asarray([[dot_prod]])

            return sum_dot

        def dS(a, P_ni):
            """
            apply PAW
            :param a:
            :param P_ni:
            :return:
            """
            return np.dot(P_ni, wfs.setups[a].dO_ii)

        if len(psi_1.shape) == 3 or len(psi_1.shape) == 1:
            ndim = 1
        else:
            ndim = psi_1.shape[0]

        P1_ai = wfs.pt.dict(shape=ndim)
        P2_ai = wfs.pt.dict(shape=ndim)
        wfs.pt.integrate(psi_1, P1_ai, kpt.q)
        wfs.pt.integrate(psi_2, P2_ai, kpt.q)
        if ndim == 1:
            if self.dtype == complex:
                paw_dot_prod = np.array([[0.0 + 0.0j]])
            else:
                paw_dot_prod = np.array([[0.0]])

            for a in P1_ai.keys():
                paw_dot_prod += np.dot(dS(a, P2_ai[a]), P1_ai[a].T.conj())
        else:
            paw_dot_prod = np.zeros_like(dot_prod)
            for a in P1_ai.keys():
                paw_dot_prod += np.dot(dS(a, P2_ai[a]), P1_ai[a].T.conj()).T
        paw_dot_prod = np.ascontiguousarray(paw_dot_prod)
        wfs.gd.comm.sum(paw_dot_prod)
        if len(psi_1.shape) == 4 or len(psi_1.shape) == 2:
            sum_dot = dot_prod + paw_dot_prod
        else:
            sum_dot = [[dot_prod]] + paw_dot_prod

        return sum_dot

    def error_eigv(self, wfs, grad_knG):
        """
        calcualte norm of the gradient vector
        (residual)

        :param wfs:
        :param grad_knG:
        :return:
        """

        n_kps = wfs.kd.nibzkpts
        norm = [0.0]
        for kpt in wfs.kpt_u:
            k = n_kps * kpt.s + kpt.q
            for i, f in enumerate(kpt.f_n):
                if f > 1.0e-10:
                    a = self.dot(
                        wfs, grad_knG[k][i] / f, grad_knG[k][i] / f, kpt,
                        addpaw=False).item() * f
                    a = a.real
                    norm.append(a)
        error = sum(norm)
        error = wfs.kd.comm.sum_scalar(error)

        return error.real

    def get_canonical_representation(self, ham, wfs, rewrite_psi=True):
        """
        choose orbitals which diagonalize the hamiltonain matrix

        <psi_i| H |psi_j>

        For SIC, one diagonalizes L_{ij} = <psi_i| H + V_{j} |psi_j>
        for occupied subspace and
         <psi_i| H |psi_j> for unoccupied.

        :param ham:
        :param wfs:
        :param rewrite_psi:
        :return:
        """
        self.choose_optimal_orbitals(wfs)

        scalewithocc = not self.excited_state

        grad_knG = self.get_gradients_2(ham, wfs, scalewithocc=scalewithocc)
        if 'SIC' in self.odd.name:
            for kpt in wfs.kpt_u:
                self.odd.get_energy_and_gradients_kpt(
                    wfs, kpt, grad_knG, self.iloop.U_k,
                    add_grad=True, scalewithocc=scalewithocc)

        for kpt in wfs.kpt_u:
            # Separate diagonalization for occupied
            # and unoccupied subspaces
            bd = self.eigensolver.bd
            k = self.n_kps * kpt.s + kpt.q
            n_occ = get_n_occ(kpt)[0]
            dim = bd.nbands - n_occ
            if scalewithocc:
                scale = kpt.f_n[:n_occ]
            else:
                scale = 1.0

            if scalewithocc:
                grad_knG[k][n_occ:n_occ + dim] = \
                    self.get_gradients_unocc_kpt(ham, wfs, kpt)
            lamb = wfs.integrate(kpt.psit_nG[:], grad_knG[k][:], True)
            lamb1 = (lamb[:n_occ, :n_occ] +
                     lamb[:n_occ, :n_occ].T.conj()) / 2.0
            lumo = (lamb[n_occ:, n_occ:] +
                    lamb[n_occ:, n_occ:].T.conj()) / 2.0

            # Diagonal elements Lagrangian matrix
            lo_nn = np.diagonal(lamb1).real / scale
            lu_nn = np.diagonal(lumo).real / 1.0

            # Diagonalize occupied subspace
            if n_occ != 0:
                evals, lamb1 = np.linalg.eigh(lamb1)
                wfs.gd.comm.broadcast(evals, 0)
                wfs.gd.comm.broadcast(lamb1, 0)
                lamb1 = lamb1.T
                kpt.eps_n[:n_occ] = evals[:n_occ] / scale

            # Diagonalize unoccupied subspace
            evals_lumo, lumo = np.linalg.eigh(lumo)
            wfs.gd.comm.broadcast(evals_lumo, 0)
            wfs.gd.comm.broadcast(lumo, 0)
            lumo = lumo.T
            kpt.eps_n[n_occ:n_occ + dim] = evals_lumo.real

            if rewrite_psi:  # Only for SIC
                kpt.psit_nG[:n_occ] = np.tensordot(
                    lamb1.conj(), kpt.psit_nG[:n_occ], axes=1)
                kpt.psit_nG[n_occ:n_occ + dim] = np.tensordot(
                    lumo.conj(), kpt.psit_nG[n_occ:n_occ + dim], axes=1)

            wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

            if 'SIC' in self.odd.name:
                self.odd.lagr_diag_s[k] = np.append(lo_nn, lu_nn)

        del grad_knG

    def get_gradients_unocc_kpt(self, ham, wfs, kpt):
        """
        calculate gradient vector for unoccupied orbitals

        :param ham:
        :param wfs:
        :param kpt:
        :return:
        """
        orbital_dependent = ham.xc.orbital_dependent

        n_occ = get_n_occ(kpt)[0]
        bd = self.eigensolver.bd
        if not orbital_dependent:
            dim = bd.nbands - n_occ
            n0 = n_occ
        else:
            dim = bd.nbands
            n0 = 0

        # calculate gradients:
        psi = kpt.psit_nG[n0:n0 + dim].copy()
        P1_ai = wfs.pt.dict(shape=dim)
        wfs.pt.integrate(psi, P1_ai, kpt.q)
        Hpsi_nG = wfs.empty(dim, q=kpt.q)
        wfs.apply_pseudo_hamiltonian(kpt, ham, psi, Hpsi_nG)
        c_axi = {}
        for a, P_xi in P1_ai.items():
            if self.gpaw_new:
                dH_ii = ham.potential.dH_asii[a][kpt.s]
            else:
                dH_ii = unpack_hermitian(ham.dH_asp[a][kpt.s])
            c_xi = np.dot(P_xi, dH_ii)
            c_axi[a] = c_xi
        # not sure about this:
        ham.xc.add_correction(kpt, psi, Hpsi_nG,
                              P1_ai, c_axi, n_x=None,
                              calculate_change=False)
        # add projectors to the H|psi_i>
        wfs.pt.add(Hpsi_nG, c_axi, kpt.q)

        if orbital_dependent:
            Hpsi_nG = Hpsi_nG[n_occ:n_occ + dim]

        return Hpsi_nG

    def get_energy_and_tangent_gradients_unocc(self, ham, wfs, psit_knG=None):
        """
        calculate energy and trangent gradients of
        unooccupied orbitals

        :param ham:
        :param wfs:
        :param psit_knG:
        :return:
        """
        wfs.timer.start('Gradient unoccupied orbitals')
        n_kps = self.n_kps
        if psit_knG is not None:
            for kpt in wfs.kpt_u:
                k = n_kps * kpt.s + kpt.q
                # find lumo
                n_occ = get_n_occ(kpt)[0]
                dim = self.dimensions[k]
                kpt.psit_nG[n_occ:n_occ + dim] = psit_knG[k].copy()
                wfs.orthonormalize(kpt)
        elif not wfs.orthonormalized:
            wfs.orthonormalize()

        grad = {}
        energy_t = 0.0
        error_t = 0.0

        for kpt in wfs.kpt_u:
            n_occ = get_n_occ(kpt)[0]
            k = n_kps * kpt.s + kpt.q
            dim = self.dimensions[k]
            Hpsi_nG = self.get_gradients_unocc_kpt(ham, wfs, kpt)
            grad[k] = Hpsi_nG.copy()

            # calculate energy
            psi = kpt.psit_nG[:n_occ + dim].copy()
            wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
            lamb = wfs.integrate(psi, Hpsi_nG, global_integral=True)
            s_axi = {}
            for a, P_xi in kpt.P_ani.items():
                dO_ii = wfs.setups[a].dO_ii
                s_xi = np.dot(P_xi, dO_ii)
                s_axi[a] = s_xi
            wfs.pt.add(psi, s_axi, kpt.q)

            grad[k] -= np.tensordot(lamb.T, psi, axes=1)

            minstate = np.argmin(np.diagonal(lamb, offset=-n_occ).real)
            energy = np.diagonal(lamb, offset=-n_occ)[minstate].real
            norm = []
            for i in [minstate]:
                norm.append(self.dot(
                    wfs, grad[k][i], grad[k][i], kpt, addpaw=False).item())
            error = sum(norm).real * Hartree ** 2 / len(norm)
            error_t += error
            energy_t += energy

        error_t = wfs.kd.comm.sum_scalar(error_t)
        energy_t = wfs.kd.comm.sum_scalar(energy_t)
        self.error = error_t

        wfs.timer.stop('Gradient unoccupied orbitals')

        return energy_t, grad

    def run_unocc(self, ham, wfs, dens, max_err, log):

        """
        Converge unoccupied orbitals

        :param ham:
        :param wfs:
        :param dens:
        :param max_err:
        :param log:
        :return:
        """

        self.need_init_odd = False
        self.initialize_dm(wfs, dens, ham, converge_unocc=True)

        while self.iters < self.maxiter_unocc:
            en, er = self.iterate(ham, wfs, dens, log, converge_unocc=True)
            log_f(self.iters, en, er, log)
            # it is quite difficult to converge unoccupied orbitals
            # with the same accuracy as occupied orbitals
            if er < max(max_err, 5.0e-4):
                log('\nUnoccupied orbitals converged after'
                    ' {:d} iterations'.format(self.iters))
                break
            if self.iters >= self.maxiter_unocc:
                log('\nUnoccupied orbitals did not converge after'
                    ' {:d} iterations'.format(self.iters))

    def run_inner_loop(self, ham, wfs, dens, grad_knG, niter=0):

        """
        calculate optimal orbitals among occupied subspace
        which minimizes SIC.
        """

        if self.iloop is None and self.outer_iloop is None:
            return niter, False

        wfs.timer.start('Inner loop')

        if self.printinnerloop:
            log = parprint
        else:
            log = None

        if self.iloop is not None:
            if self.excited_state and self.iters == 0:
                eks = self.update_ks_energy(ham, wfs, dens)
            else:
                etotal = ham.get_energy(
                    0.0, wfs, kin_en_using_band=False, e_sic=self.e_sic)
                eks = etotal - self.e_sic
            if self.initial_random and self.iters == 1:
                small_random = True
            else:
                small_random = False
            self.e_sic, counter = self.iloop.run(
                eks, wfs, dens, log, niter, small_random=small_random,
                seed=self.localizationseed)
            self.eg_count_iloop = self.iloop.eg_count
            self.total_eg_count_iloop += self.iloop.eg_count

            if self.outer_iloop is None:
                if grad_knG is not None:
                    for kpt in wfs.kpt_u:
                        k = self.n_kps * kpt.s + kpt.q
                        n_occ = get_n_occ(kpt)[0]
                        grad_knG[k][:n_occ] += np.tensordot(
                            self.iloop.U_k[k].conj(),
                            self.iloop.odd_pot.grad[k], axes=1)
                wfs.timer.stop('Inner loop')

                ham.get_energy(0.0, wfs, kin_en_using_band=False,
                               e_sic=self.e_sic)
                return counter, True

            for kpt in wfs.kpt_u:
                k = self.iloop.n_kps * kpt.s + kpt.q
                U = self.iloop.U_k[k]
                n_occ = U.shape[0]
                kpt.psit_nG[:n_occ] = np.tensordot(
                    U.T, kpt.psit_nG[:n_occ], axes=1)
                # calc projectors
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

        self.etotal, counter = self.outer_iloop.run(
            wfs, dens, log, niter, ham=ham)
        self.eg_count_outer_iloop = self.outer_iloop.eg_count
        self.total_eg_count_outer_iloop += self.outer_iloop.eg_count
        self.e_sic = self.outer_iloop.esic
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            grad_knG[k] += np.tensordot(self.outer_iloop.U_k[k].conj(),
                                        self.outer_iloop.odd_pot.grad[k],
                                        axes=1)
            if self.iloop is not None:
                U = self.iloop.U_k[k]
                n_occ = U.shape[0]
                kpt.psit_nG[:n_occ] = \
                    np.tensordot(U.conj(),
                                 kpt.psit_nG[:n_occ], axes=1)
                # calc projectors
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
                grad_knG[k][:n_occ] = \
                    np.tensordot(U.conj(),
                                 grad_knG[k][:n_occ], axes=1)
                self.iloop.U_k[k] = \
                    self.iloop.U_k[k] @ self.outer_iloop.U_k[k]
                self.outer_iloop.U_k[k] = np.eye(n_occ, dtype=self.dtype)

        wfs.timer.stop('Inner loop')

        ham.get_energy(0.0, wfs, kin_en_using_band=False,
                       e_sic=self.e_sic)

        return counter, True

    def initialize_orbitals(self, wfs, ham):
        if self.need_init_orbs and not wfs.read_from_file_init_wfs_dm:
            if self.gpaw_new:
                def Ht(psit_nG, out, spin):
                    return wfs.hamiltonian.apply(
                        wfs.potential.vt_sR,
                        wfs.potential.dedtaut_sR,
                        wfs.ibzwfs,
                        wfs.density.D_asii,
                        psit_nG,
                        out,
                        spin)

                for w in wfs.ibzwfs:
                    w.subspace_diagonalize(Ht, wfs.potential.dH)
            else:
                for kpt in wfs.kpt_u:
                    wfs.orthonormalize(kpt)
                    self.eigensolver.subspace_diagonalize(
                        ham, wfs, kpt, True)
                wfs.gd.comm.broadcast(kpt.eps_n, 0)
            self.need_init_orbs = False
        if wfs.read_from_file_init_wfs_dm:
            self.initial_random = False

    def localize(self, wfs, dens, ham, log):
        if not self.need_localization:
            return
        localize_orbitals(wfs, dens, ham, log, self.localizationtype,
                          tol=self.localization_tol,
                          seed=self.localizationseed,
                          func_settings=self.func_settings)
        self.need_localization = False

    def choose_optimal_orbitals(self, wfs):
        """
        choose optimal orbitals and store them in wfs.kpt_u.
        Optimal orbitals are those which minimize the energy
        functional and might not coincide with canonical orbitals

        :param wfs:
        :return:
        """
        for kpt in wfs.kpt_u:
            k = self.n_kps * kpt.s + kpt.q
            if self.iloop is not None:
                dim = self.iloop.U_k[k].shape[0]
                kpt.psit_nG[:dim] = \
                    np.tensordot(
                        self.iloop.U_k[k].T, kpt.psit_nG[:dim],
                        axes=1)
                self.iloop.U_k[k] = np.eye(self.iloop.U_k[k].shape[0])
                self.iloop.Unew_k[k] = np.eye(
                    self.iloop.Unew_k[k].shape[0])
            if self.outer_iloop is not None:
                dim = self.outer_iloop.U_k[k].shape[0]
                kpt.psit_nG[:dim] = \
                    np.tensordot(
                        self.outer_iloop.U_k[k].T,
                        kpt.psit_nG[:dim], axes=1)
                self.outer_iloop.U_k[k] = np.eye(
                    self.outer_iloop.U_k[k].shape[0])
                self.outer_iloop.Unew_k[k] = np.eye(
                    self.outer_iloop.Unew_k[k].shape[0])
            if self.iloop is not None or \
                    self.outer_iloop is not None:
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

    def check_assertions(self, wfs, dens):

        assert dens.mixer.driver.basemixerclass.name == 'no-mixing', \
            'Please, use: mixer={\'backend\': \'no-mixing\'}'
        assert wfs.bd.comm.size == 1, \
            'Band parallelization is not supported'
        if wfs.occupations.name != 'mom':
            errormsg = \
                'Please, use occupations={\'name\': \'fixed-uniform\'}'
            assert wfs.occupations.name == 'fixed-uniform', errormsg

    def check_restart(self, wfs):
        occ_name = getattr(wfs.occupations, 'name', None)
        if occ_name != 'mom':
            return

        sic_calc = 'SIC' in self.func_settings['name']
        if self.outer_iloop is not None:
            # mom restart ?
            astmnt = self.outer_iloop.restart
            # iloop not converged?
            bstmnt = \
                (self.iters + 1) % self.restartevery_iloop_notconverged == 0 \
                and not self.outer_iloop.converged
            if astmnt or bstmnt:
                self.choose_optimal_orbitals(wfs)
                if not sic_calc and self.restart_canonical:
                    # Will restart using canonical orbitals
                    self.need_init_orbs = True
                    wfs.read_from_file_init_wfs_dm = False
                self.iters = 0
                self.initialized = False
                self.need_init_odd = True

        return self.outer_iloop.restart

    def initialize_mom_reference_orbitals(self, wfs, dens):
        # Reinitialize the MOM reference orbitals
        # after orthogonalization/localization
        occ_name = getattr(wfs.occupations, 'name', None)
        if occ_name == 'mom' and self.globaliters == 0:
            for kpt in wfs.kpt_u:
                wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
            wfs.orthonormalize()
            wfs.occupations.initialize_reference_orbitals()
            wfs.calculate_occupation_numbers(dens.fixed)


def log_f(niter, e_total, eig_error, log):
    """
    log function for convergence of unoccupied states.

    :param niter:
    :param e_total:
    :param eig_error:
    :param log:
    :return:
    """

    T = time.localtime()
    if niter == 1:
        header = '                              ' \
                 '     wfs    \n' \
                 '           time        Energy:' \
                 '     error(ev^2):'
        log(header)

    log('iter: %3d  %02d:%02d:%02d ' %
        (niter,
         T[3], T[4], T[5]
         ), end='')

    log('%11.6f %11.1e' %
        (Hartree * e_total, eig_error), end='')

    log(flush=True)
