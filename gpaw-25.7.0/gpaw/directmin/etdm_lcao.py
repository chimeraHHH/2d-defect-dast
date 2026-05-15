"""
A class for finding optimal orbitals corresponding to a stationary point of
the energy functional using direct optimization and exponential transformation
in LCAO mode.

It can be used with Kohn-Sham functionals and can include Perdew-Zunger
self-interaction correction (PZ-SIC) in the calculations.

Ground state as well as variational excited state calculations can be
performed. Ground state calculations involve minimization of the energy
(direct minimization), while excited state calculations involve convergence
on a saddle point (direct optimization).

Implementation of exponential transformation direct minimization (ETDM) for
ground state calculations LCAO mode:

    Comput. Phys. Commun. 267, 108047 (2021) :doi:10.1016/j.cpc.2021.108047
    arXiv:2101.12597 [physics.comp-ph]

GPAW implementations of direct optimization (DO) for variational excited state
calculations LCAO mode:

    DO with preconditioned quasi-Newton algorithms and maximum overlap method
    (DO-MOM)
    J. Chem. Theory Comput. 16, 6968 (2020) :doi:10.1021/acs.jctc.0c00597
    arXiv:2006.15922 [physics.chem-ph]

    DO with generalized mode following (DO-GMF)
    J. Chem. Theory Comput. 19, 3634 (2023) :doi:10.1021/acs.jctc.3c00178
    arXiv:2302.05912 [physics.chem-ph]
"""


import numpy as np
import warnings
from ase.utils import basestring
from gpaw.directmin.tools import expm_ed, expm_ed_unit_inv, random_a, \
    sort_orbitals_according_to_occ, sort_orbitals_according_to_energies
from gpaw.directmin.lcao.etdm_helper_lcao import ETDMHelperLCAO
from gpaw.directmin.locfunc.localize_orbitals import localize_orbitals
from scipy.linalg import expm
from gpaw.directmin import search_direction, line_search_algorithm
from gpaw.directmin.tools import get_n_occ
from gpaw.directmin.derivatives import get_approx_analytical_hessian
from gpaw import BadParallelization
from copy import deepcopy


class LCAOETDM:

    def __init__(self,
                 excited_state=False,
                 searchdir_algo=None,
                 linesearch_algo=None,
                 partial_diagonalizer='Davidson',
                 update_ref_orbs_counter=20,
                 update_ref_orbs_canonical=False,
                 update_precond_counter=1000,
                 use_prec=True,
                 matrix_exp='pade-approx',
                 representation='sparse',
                 functional='ks',
                 need_init_orbs=None,
                 orthonormalization='gramschmidt',
                 randomizeorbitals=None,
                 checkgraderror=False,
                 need_localization=True,
                 localizationtype=None,
                 localizationseed=None,
                 constraints=None,
                 subspace_convergence=5e-4
                 ):
        """Class for direct orbital optimization in LCAO mode.

        Parameters
        ----------
        excited_state: bool
            If False (default), use search direction and line search
            algorithms for ground state calculation ('l-bfgs-p' and 'swc-awc')
            if not specified by searchdir_algo linesearch_algo, and set
            need_init_orbs to True. Otherwise, use search direction and line
            search algorithms for excited state calculation ('l-sr1p' and
            'max-step'), and set need_init_orbs to False.
        searchdir_algo: str, dict or instance
            Search direction algorithm. Can be one of the algorithms available
            in sd_etdm.py:
                'sd': Steepest descent
                'fr-cg': Fletcher-Reeves conjugate gradient
                'l-bfgs': Limited-memory BFGS
                'l-bfgs-p': Limited-memory BFGS with preconditioner presented
                    in :doi:`10.1016/j.cpc.2021.108047` (default when
                    excited_state is False)
                'l-sr1p': limited-memory SR1 algorithm presented in
                    :doi:`10.1021/acs.jctc.0c00597` (default when excited_state
                    is True)
            The default memory for 'l-bfgs'/'l-bfgs-p' and 'l-sr1p' is 3 and
            20, respectively, and can be changed by supplying a dictionary:
            {'name': name, 'memory': memory}, where name should be 'l-bfgs',
            'l-bfgs-p' or 'l-sr1p' and memory should be an int.
            To use the generalized mode following (GMF) method for excited
            states, append '_gmf' to the search direction algorithm name. E.g.
            'l-bfgs-p_gmf'.
        linesearch_algo: str, dict or instance
            Line search algorithm. Can be one of the algorithms available
            in ls_etdm.py:
                'max-step': The quasi-Newton step is scaled if it exceeds a
                    maximum step length (default when excited_state is True).
                    The default maximum step length is 0.20, and can be changed
                    by supplying a dictionary:
                    {'name': 'max-step', 'max_step': max_step}, where max_step
                    should be a float
                'swc-awc': Line search with Wolfe conditions (default when
                    excited_state is False)
        partial_diagonalizer: 'str' or dict
            Algorithm for partial diagonalization of the electronic Hessian if
            GMF is used. Default is 'Davidson'.
        update_ref_orbs_counter: int
            When to update the coefficients of the reference orbitals. Default
            is 20 iterations.
        update_ref_orbs_canonical: bool
            If True, the coefficients of the reference orbitals are updated to
            canonical orbital coefficients, otherwise use the optimal orbital
            coefficients (default).
        use_prec: bool
            If True (default) use a preconditioner. The preconditioner is
            calculated as the inverse of a diagonal approximation of the
            Hessian (see :doi:`10.1021/j100322a012`) apart for 'l-bfgs-p',
            which uses the composite preconditioner presented in
            :doi:`10.1016/j.cpc.2021.108047`.
        update_precond_counter: int
            When to update the preconditioner. Default is 1000 iterations.
        representation: 'str'
            How to store the elements of the anti-Hermitian matrix for the
            matrix exponential. Can be one of 'sparse' (default), 'full',
            'u-invar'. The latter can be used only for unitary invariant
            functionals, such as Kohn-Sham functionals when the occupied
            orbitals have the same occupation number, but not for orbital
            density dependent functionals, such as when using PZ-SIC.
        matrix_exp: 'str'
            Algorithm for calculating the matrix exponential and the gradient
            with respect to the elements of the anti-Hermitian matrix for the
            exponential transformation. Can be one of 'pade-approx' (default),
            'egdecomp', 'egdecomp-u-invar' (the latter can be used only with
            'u-invar' representation).
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
        orthonormalization: str
            Method to orthonormalize the orbitals. Can be one of 'gramschmidt'
            (Gram-Schmidt orthonormalization, default), 'loewdin' (Loewdin
            orthonormalization) or 'diag' (eigendecomposition of the
            Hamiltonian matrix).
        randomizeorbitals: numpy.random.Generator
            Optional RNG to add random noise to the initial guess orbitals,
            default is to not add noise to them
        checkgraderror: bool
            If True, can be used to check the error in the estimation of the
            gradient (only with representation 'full'). Default is False.
        need_localization: bool
            If True (default), localize initial guess orbitals. Requires a
            specification of localizationtype, otherwise it is set to False.
            Recommended for calculations with PZ-SIC.
        localizationtype: str
            Method for localizing the initial guess orbitals. Can be one of:
                'pz': Unitary optimization among occupied orbitals (subspace
                    optimization) with PZ-SIC
                'pm': Pipek-Mezey localization (recommended for PZ-SIC)
                'fb' Foster-Boys localization
            Default is None, meaning that no localization is performed.
        localizationseed: int
            Seed for Pipek-Mezey or Foster-Boys localization. Default is
            None (no seed is used).
        constraints: list of lists of int
            List of constraints on the orbital rotation for each k-point. If
            a constraint is given as a pair of orbital indices, the rotation
            between these two orbitals is constrained. If a constraint is
            given as a single index, the orbital with this index is frozen.
            E.g.: calc.set(eigensolver=LCAOETDM(constraints=[[[i]], [[j]]]))
            will freeze orbital i in the first k-point and orbital j in the
            second k-point.
        subspace_convergence: float
            Tolerance on the norm of the gradient for convergence of the
            subspace optimization with PZ-SIC.
        """

        assert representation in ['sparse', 'u-invar', 'full'], 'Value Error'
        assert matrix_exp in ['egdecomp', 'egdecomp-u-invar', 'pade-approx'], \
            'Value Error'
        if matrix_exp == 'egdecomp-u-invar':
            assert representation == 'u-invar', 'Use u-invar representation ' \
                                                'with egdecomp-u-invar'
        assert orthonormalization in ['gramschmidt', 'loewdin', 'diag'], \
            'Value Error'

        self.sda = searchdir_algo
        self.lsa = linesearch_algo
        self.partial_diagonalizer = partial_diagonalizer
        self.localizationseed = localizationseed
        self.update_ref_orbs_counter = update_ref_orbs_counter
        self.update_ref_orbs_canonical = update_ref_orbs_canonical
        self.update_precond_counter = update_precond_counter
        self.use_prec = use_prec
        self.matrix_exp = matrix_exp
        self.localizationtype = localizationtype
        self.need_localization = need_localization
        self.need_init_orbs = need_init_orbs
        self.randomizeorbitals = randomizeorbitals
        self.representation = representation
        self.orthonormalization = orthonormalization
        self.constraints = constraints
        self.subspace_convergence = subspace_convergence
        self.released_subspace = False
        self.excited_state = excited_state
        self.name = 'etdm-lcao'
        self.iters = 0
        self.eg_count = 0
        self.subspace_iters = 0
        self.restart = False
        self.gmf = False
        self.check_inputs_and_init_search_algo()

        self.checkgraderror = checkgraderror
        self._norm_commutator, self._norm_grad = 0., 0.
        self.error = 0
        self.e_sic = 0.0
        self.subspace_optimization = False

        # these are things we cannot initialize now
        self.func_settings = functional
        self.dtype = None
        self.nkpts = None
        self.gd = None
        self.nbands = None

        # values: vectors of the elements of matrices, keys: kpt number
        self.a_vec_u = {}  # for the elements of the skew-Hermitian matrix A
        self.a_vec_oo_u = {}
        self.a_vec_ov_u = {}
        self.a_vec_all_u = {}
        self.g_vec_u = {}  # for the elements of the gradient matrix G
        self.g_vec_oo_u = {}
        self.g_vec_ov_u = {}
        self.g_vec_all_u = {}
        self.evecs = {}   # eigenvectors for i*a_vec_u
        self.evals = {}   # eigenvalues for i*a_vec_u
        self.ind_up = {}
        self.ind_oo_up = {}
        self.ind_ov_up = {}
        self.ind_sparse_up = {}
        self.ind_all_up = {}
        self.n_dim = {}
        self.n_dim_oo = {}
        self.n_dim_all = {}
        self.alpha = 1.0  # step length
        self.phi_2i = [None, None]  # energy at last two iterations
        self.der_phi_2i = [None, None]  # energy gradient w.r.t. alpha
        self.hess = {}  # approximate Hessian

        # for mom
        self.initial_occupation_numbers = None

        # in this attribute we store the object specific to each mode
        self.dm_helper = None

        self.initialized = False

    def check_inputs_and_init_search_algo(self):
        defaults = self.set_defaults()
        if self.sda is None:
            self.sda = defaults['searchdir_algo']
        if self.lsa is None:
            self.lsa = defaults['linesearch_algo']
        if self.need_init_orbs is None:
            self.need_init_orbs = defaults['need_init_orbs']
        if self.localizationtype is None:
            self.need_localization = False

        self.searchdir_algo = search_direction(
            self.sda, self, self.partial_diagonalizer)
        sd_name = self.searchdir_algo.name.split('_')
        if sd_name[0] == 'l-bfgs-p' and not self.use_prec:
            raise ValueError('Use l-bfgs-p with use_prec=True')
        if len(sd_name) == 2:
            if sd_name[1] == 'gmf':
                self.searchdir_algo.name = sd_name[0]
                self.gmf = True
                self.g_vec_u_original = None
                self.pd = self.partial_diagonalizer

        self.line_search = line_search_algorithm(self.lsa,
                                                 self.evaluate_phi_and_der_phi,
                                                 self.searchdir_algo)

    def set_defaults(self):
        if self.excited_state:
            return {'searchdir_algo': 'l-sr1p',
                    'linesearch_algo': 'max-step',
                    'need_init_orbs': False}
        else:
            return {'searchdir_algo': 'l-bfgs-p',
                    'linesearch_algo': 'swc-awc',
                    'need_init_orbs': True}

    def __repr__(self):

        sda_name = self.searchdir_algo.name
        lsa_name = self.line_search.name
        if isinstance(self.func_settings, basestring):
            func_name = self.func_settings
        else:
            func_name = self.func_settings['name']
        if self.gmf:
            if isinstance(self.pd, basestring):
                pd_name = self.pd
            else:
                pd_name = self.pd['name']

        add = ''
        pd_add = ''
        if self.gmf:
            add = ' with minimum mode following'
            pardi = {'Davidson': 'Finite difference generalized Davidson '
                     'algorithm'}
            pd_add = '       ' \
                     'Partial diagonalizer: {}\n'.format(
                         pardi[pd_name])

        sds = {'sd': 'Steepest Descent',
               'fr-cg': 'Fletcher-Reeves conj. grad. method',
               'l-bfgs': 'L-BFGS algorithm',
               'l-bfgs-p': 'L-BFGS algorithm with preconditioning',
               'l-sr1p': 'Limited-memory SR1P algorithm'}

        lss = {'max-step': 'step size equals one',
               'swc-awc': 'Inexact line search based on cubic interpolation,\n'
                          '                    strong and approximate Wolfe '
                          'conditions'}

        repr_string = 'Direct minimisation' + add + ' using exponential ' \
                      'transformation.\n'
        repr_string += '       ' \
                       'Search ' \
                       'direction: {}\n'.format(sds[sda_name] + add)
        repr_string += pd_add
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

    def initialize(self, gd, dtype, nbands, nkpts, nao, using_blacs,
                   bd_comm_size, kpt_u):

        assert nbands == nao, \
            'Please, use: nbands=\'nao\''
        if not bd_comm_size == 1:
            raise BadParallelization(
                'Band parallelization is not supported')
        if using_blacs:
            raise BadParallelization(
                'ScaLapack parallelization is not supported')

        self.gd = gd
        self.dtype = dtype
        self.nbands = nbands
        self.nkpts = nkpts

        for kpt in kpt_u:
            u = self.kpointval(kpt)
            # dimensionality of the problem.
            # this implementation rotates among all bands
            self.n_dim_all[u] = self.nbands

        self.initialized = True

        # need reference orbitals
        self.dm_helper = None

    def initialize_dm_helper(self, wfs, ham, dens, log):

        self.dm_helper = ETDMHelperLCAO(
            wfs, dens, ham, self.nkpts, self.func_settings,
            diagonalizer=None,
            orthonormalization=self.orthonormalization,
            need_init_orbs=self.need_init_orbs)
        self.need_init_orbs = self.dm_helper.need_init_orbs

        # randomize orbitals?
        if self.randomizeorbitals is not None:
            for kpt in wfs.kpt_u:
                self.randomize_orbitals_kpt(wfs, kpt)
            self.randomizeorbitals = None

        if not wfs.coefficients_read_from_file:
            wfs.calculate_occupation_numbers(dens.fixed)

        self.initial_sort_orbitals(wfs)

        # initialize matrices
        self.set_variable_matrices(wfs.kpt_u)

        # localize orbitals?
        self.localize(wfs, dens, ham, log)

        # MOM
        self.initialize_mom_reference_orbitals(wfs, dens)

        for kpt in wfs.kpt_u:
            f_unique = np.unique(kpt.f_n)
            if len(f_unique) > 2 and self.representation == 'u-invar':
                warnings.warn("Use representation == 'sparse' when "
                              "there are unequally occupied orbitals "
                              "as the functional is not unitary invariant")

        # set reference orbitals
        self.dm_helper.set_reference_orbitals(wfs, self.n_dim)

    def set_variable_matrices(self, kpt_u):

        # Matrices are sparse and Skew-Hermitian.
        # They have this structure:
        #  A_BigMatrix =
        #
        # (  A_1          A_2 )
        # ( -A_2.T.conj() 0   )
        #
        # where 0 is a zero-matrix of size of (M-N) * (M-N)
        #
        # A_1 i skew-hermitian matrix of N * N,
        # N-number of occupied states
        # A_2 is matrix of size of (M-N) * N,
        # M - number of basis functions
        #
        # if the energy functional is unitary invariant
        # then A_1 = 0
        # (see Hutter J et. al, J. Chem. Phys. 101, 3862 (1994))
        #
        # We will keep A_1 as we would like to work with metals,
        # SIC, and molecules with different occupation numbers.
        # this corresponds to 'sparse' representation
        #
        # Thus, for the 'sparse' we need to store upper
        # triangular part of A_1, and matrix A_2, so in total
        # (M-N) * N + N * (N - 1)/2 = N * (M - (N + 1)/2) elements
        #
        # we will store these elements as a vector and
        # also will store indices of the A_BigMatrix
        # which correspond to these elements.
        #
        # 'u-invar' corresponds to the case when we want to
        # store only A_2, that is this representation is sparser

        for kpt in kpt_u:
            n_occ = get_n_occ(kpt)[0]
            u = self.kpointval(kpt)
            self.n_dim[u] = deepcopy(self.n_dim_all[u])
            self.n_dim_oo[u] = n_occ
            # M - one dimension of the A_BigMatrix
            M = self.n_dim_all[u]
            i1_oo, i2_oo = [], []
            for i in range(n_occ):
                for j in range(i + 1, n_occ):
                    i1_oo.append(i)
                    i2_oo.append(j)
            self.ind_oo_up[u] = (np.asarray(i1_oo), np.asarray(i2_oo))
            i1_ov, i2_ov = [], []
            for i in range(n_occ):
                for j in range(n_occ, M):
                    i1_ov.append(i)
                    i2_ov.append(j)
            self.ind_ov_up[u] = (np.asarray(i1_ov), np.asarray(i2_ov))
            i1_sparse, i2_sparse = [], []
            for i in range(n_occ):
                for j in range(i + 1, M):
                    i1_sparse.append(i)
                    i2_sparse.append(j)
            self.ind_sparse_up[u] = (np.asarray(i1_sparse),
                                     np.asarray(i2_sparse))
            if self.representation == 'u-invar':
                self.ind_all_up[u] = self.ind_ov_up[u]
            if self.representation == 'sparse':
                self.ind_all_up[u] = self.ind_sparse_up[u]
            elif self.representation == 'full' and self.dtype == complex:
                # Take indices of all upper triangular and diagonal
                # elements of A_BigMatrix
                self.ind_all_up[u] = np.triu_indices(self.n_dim[u])

            shape_of_oo = len(self.ind_oo_up[u][0])
            shape_of_ov = len(self.ind_ov_up[u][0])
            shape_of_all = len(self.ind_all_up[u][0])

            self.a_vec_oo_u[u] = np.zeros(shape=shape_of_oo, dtype=self.dtype)
            self.a_vec_ov_u[u] = np.zeros(shape=shape_of_ov, dtype=self.dtype)
            self.a_vec_all_u[u] = np.zeros(
                shape=shape_of_all, dtype=self.dtype)
            self.g_vec_oo_u[u] = np.zeros(shape=shape_of_oo, dtype=self.dtype)
            self.g_vec_ov_u[u] = np.zeros(shape=shape_of_ov, dtype=self.dtype)
            self.g_vec_all_u[u] = np.zeros(
                shape=shape_of_all, dtype=self.dtype)
            self.evecs[u] = None
            self.evals[u] = None

            # All constraints passed as a list of a single orbital index are
            # converted to all lists of orbital pairs involving that orbital
            # index
            if self.constraints:
                self.constraints[u] = convert_constraints(
                    self.constraints[u], self.n_dim[u],
                    len(kpt.f_n[kpt.f_n > 1e-10]), self.representation)

            # If there are no degrees of freedom no need to optimize.
            # Indicated by setting n_dim to 0, as n_dim can never be 0
            # otherwise.
            for k in self.ind_all_up:
                if not self.ind_all_up[k][0].size \
                        or not self.ind_all_up[k][1].size:
                    self.n_dim_all[k] = 0  # Skip full space optimization
                if not self.ind_oo_up[k][0].size \
                        or not self.ind_oo_up[k][1].size:
                    self.n_dim_oo[k] = 0  # Skip PZ localization if requested

        self.ind_up = deepcopy(self.ind_all_up)
        self.a_vec_u = deepcopy(self.a_vec_all_u)
        self.g_vec_u = deepcopy(self.g_vec_all_u)

        # This conversion makes it so that constraint-related functions can
        # iterate through a list of no constraints rather than checking for
        # None every time
        if self.constraints is None:
            self.constraints = [[] for _ in range(len(kpt_u))]

        self.iters = 1

    def localize(self, wfs, dens, ham, log):
        if self.need_localization:
            localize_orbitals(
                wfs, dens, ham, log, self.localizationtype,
                seed=self.localizationseed)
            self.need_localization = False

    def lock_subspace(self, subspace='oo'):
        self.subspace_optimization = True
        self.subspace_iters = 1
        if subspace == 'oo':
            self.n_dim = deepcopy(self.n_dim_oo)
            self.ind_up = deepcopy(self.ind_oo_up)
            self.a_vec_u = deepcopy(self.a_vec_oo_u)
            self.g_vec_u = deepcopy(self.g_vec_oo_u)
        elif subspace == 'ov':
            self.n_dim = deepcopy(self.n_dim_all)
            self.ind_up = deepcopy(self.ind_ov_up)
            self.a_vec_u = deepcopy(self.a_vec_ov_u)
            self.g_vec_u = deepcopy(self.g_vec_ov_u)
        self.alpha = 1.0
        self.phi_2i = [None, None]
        self.der_phi_2i = [None, None]

    def release_subspace(self):
        self.subspace_optimization = False
        self.released_subspace = True
        self.n_dim = deepcopy(self.n_dim_all)
        self.ind_up = deepcopy(self.ind_all_up)
        self.a_vec_u = deepcopy(self.a_vec_all_u)
        self.g_vec_u = deepcopy(self.g_vec_all_u)
        self.alpha = 1.0
        self.phi_2i = [None, None]
        self.der_phi_2i = [None, None]

    def iterate(self, ham, wfs, dens):
        """
        One iteration of direct optimization
        for occupied orbitals

        :param ham:
        :param wfs:
        :param dens:
        :return:
        """
        with wfs.timer('Direct Minimisation step'):
            self.update_ref_orbitals(wfs, ham, dens)

            a_vec_u = self.a_vec_u
            alpha = self.alpha
            phi_2i = self.phi_2i
            der_phi_2i = self.der_phi_2i
            c_ref = self.dm_helper.reference_orbitals

            if self.iters == 1 or self.released_subspace or \
                    (self.subspace_optimization and self.subspace_iters == 1):
                phi_2i[0], g_vec_u = \
                    self.get_energy_and_gradients(
                        a_vec_u, ham, wfs, dens, c_ref)
            else:
                g_vec_u = self.g_vec_u_original if self.gmf \
                    and not self.subspace_optimization else self.g_vec_u

            make_pd = False
            if self.gmf and not self.subspace_optimization:
                with wfs.timer('Partial Hessian diagonalization'):
                    self.searchdir_algo.update_eigenpairs(
                        g_vec_u, wfs, ham, dens)
                # The diagonal Hessian approximation must be positive-definite
                make_pd = True

            with wfs.timer('Preconditioning:'):
                precond = self.get_preconditioning(
                    wfs, self.use_prec, make_pd=make_pd)

            with wfs.timer('Get Search Direction'):
                # calculate search direction according to chosen
                # optimization algorithm (e.g. L-BFGS)
                p_vec_u = self.searchdir_algo.update_data(
                    wfs, a_vec_u, g_vec_u, precond=precond,
                    subspace=self.subspace_optimization)

            # recalculate derivative with new search direction
            der_phi_2i[0] = 0.0
            for k in g_vec_u:
                der_phi_2i[0] += g_vec_u[k].conj() @ p_vec_u[k]
            der_phi_2i[0] = der_phi_2i[0].real
            der_phi_2i[0] = wfs.kd.comm.sum_scalar(der_phi_2i[0])

            alpha, phi_alpha, der_phi_alpha, g_vec_u = \
                self.line_search.step_length_update(
                    a_vec_u, p_vec_u, wfs, ham, dens, c_ref, phi_0=phi_2i[0],
                    der_phi_0=der_phi_2i[0], phi_old=phi_2i[1],
                    der_phi_old=der_phi_2i[1], alpha_max=5.0, alpha_old=alpha,
                    kpdescr=wfs.kd)

            if wfs.gd.comm.size > 1:
                with wfs.timer('Broadcast gradients'):
                    alpha_phi_der_phi = np.array([alpha, phi_2i[0],
                                                  der_phi_2i[0]])
                    wfs.gd.comm.broadcast(alpha_phi_der_phi, 0)
                    alpha = alpha_phi_der_phi[0]
                    phi_2i[0] = alpha_phi_der_phi[1]
                    der_phi_2i[0] = alpha_phi_der_phi[2]
                    for kpt in wfs.kpt_u:
                        k = self.kpointval(kpt)
                        wfs.gd.comm.broadcast(g_vec_u[k], 0)

            # calculate new matrices for optimal step length
            for k in a_vec_u:
                a_vec_u[k] += alpha * p_vec_u[k]
            self.alpha = alpha
            self.g_vec_u = g_vec_u
            if self.subspace_optimization:
                self.subspace_iters += 1
            else:
                self.iters += 1

            # and 'shift' phi, der_phi for the next iteration
            phi_2i[1], der_phi_2i[1] = phi_2i[0], der_phi_2i[0]
            phi_2i[0], der_phi_2i[0] = phi_alpha, der_phi_alpha,

            if self.subspace_optimization:
                self.error = np.inf  # Do not consider this converged!

    def get_grad_norm(self):
        norm = 0.0
        for k in self.g_vec_u.keys():
            norm += np.linalg.norm(self.g_vec_u[k])
        return norm

    def get_energy_and_gradients(self, a_vec_u, ham, wfs, dens,
                                 c_ref):

        """
        Energy E = E[C_ref exp(A)]. Gradients G_ij[C, A] = dE/dA_ij

        :param wfs:
        :param ham:
        :param dens:
        :param a_vec_u: A
        :param c_ref: C_ref
        :return:
        """

        self.rotate_wavefunctions(wfs, a_vec_u, c_ref)

        e_total = self.update_ks_energy(ham, wfs, dens)

        with wfs.timer('Calculate gradients'):
            g_vec_u = {}
            self.error = 0.0
            self.e_sic = 0.0  # this is odd energy
            for kpt in wfs.kpt_u:
                k = self.kpointval(kpt)
                if self.n_dim[k] == 0:
                    g_vec_u[k] = np.zeros_like(a_vec_u[k])
                    continue
                g_vec_u[k], error = self.dm_helper.calc_grad(
                    wfs, ham, kpt, self.evecs[k], self.evals[k],
                    self.matrix_exp, self.representation, self.ind_up[k],
                    self.constraints[k])

                if hasattr(self.dm_helper.func, 'e_sic_by_orbitals'):
                    self.e_sic \
                        += self.dm_helper.func.e_sic_by_orbitals[k].sum()

                self.error += error
            self.error = wfs.kd.comm.sum_scalar(self.error)
            self.e_sic = wfs.kd.comm.sum_scalar(self.e_sic)

        self.eg_count += 1

        if self.representation == 'full' and self.checkgraderror:
            self._norm_commutator = 0.0
            for kpt in wfs.kpt_u:
                u = self.kpointval(kpt)
                a_mat = vec2skewmat(a_vec_u[u], self.n_dim[u],
                                    self.ind_up[u], wfs.dtype)
                g_mat = vec2skewmat(g_vec_u[u], self.n_dim[u],
                                    self.ind_up[u], wfs.dtype)

                tmp = np.linalg.norm(g_mat @ a_mat - a_mat @ g_mat)
                if self._norm_commutator < tmp:
                    self._norm_commutator = tmp

                tmp = np.linalg.norm(g_mat)
                if self._norm_grad < tmp:
                    self._norm_grad = tmp

        return e_total + self.e_sic, g_vec_u

    def update_ks_energy(self, ham, wfs, dens):
        """
        Update Kohn-Sham energy
        It assumes the temperature is zero K.
        """

        dens.update(wfs)
        ham.update(dens, wfs, False)

        return ham.get_energy(0.0, wfs, False)

    def evaluate_phi_and_der_phi(self, a_vec_u, p_mat_u, alpha,
                                 wfs, ham, dens, c_ref,
                                 phi=None, g_vec_u=None):
        """
        phi = f(x_k + alpha_k*p_k)
        der_phi = \\grad f(x_k + alpha_k*p_k) \\cdot p_k
        :return:  phi, der_phi # floats
        """
        if phi is None or g_vec_u is None:
            x_mat_u = {k: a_vec_u[k] + alpha * p_mat_u[k] for k in a_vec_u}
            phi, g_vec_u = self.get_energy_and_gradients(
                x_mat_u, ham, wfs, dens, c_ref)

            # If GMF is used save the original gradient and invert the parallel
            # projection onto the eigenvectors with negative eigenvalues
            if self.gmf and not self.subspace_optimization:
                self.g_vec_u_original = deepcopy(g_vec_u)
                g_vec_u = self.searchdir_algo.invert_parallel_grad(g_vec_u)

        der_phi = 0.0
        for k in p_mat_u:
            der_phi += g_vec_u[k].conj() @ p_mat_u[k]

        der_phi = der_phi.real
        der_phi = wfs.kd.comm.sum_scalar(der_phi)

        return phi, der_phi, g_vec_u

    def update_ref_orbitals(self, wfs, ham, dens):
        """
        Update reference orbitals

        :param wfs:
        :param ham:
        :return:
        """

        if self.representation == 'full':
            badgrad = self._norm_commutator > self._norm_grad / 3. and \
                self.checkgraderror
        else:
            badgrad = False
        counter = self.update_ref_orbs_counter
        if (self.iters % counter == 0 and self.iters > 1) or \
                (self.restart and self.iters > 1) or badgrad:
            self.iters = 1
            if self.update_ref_orbs_canonical or self.restart:
                self.get_canonical_representation(ham, wfs, dens)
            else:
                self.set_ref_orbitals_and_a_vec(wfs)

            # Erase memory of search direction algorithm
            self.searchdir_algo.reset()

    def get_preconditioning(self, wfs, use_prec, make_pd=False):

        if not use_prec:
            return None

        if self.searchdir_algo.name == 'l-bfgs-p':
            beta0 = self.searchdir_algo.beta_0
            gamma = 0.25
        else:
            beta0 = 1.0
            gamma = 0.0

        counter = self.update_precond_counter
        precond = {}
        for kpt in wfs.kpt_u:
            k = self.kpointval(kpt)
            w = kpt.weight / (3.0 - wfs.nspins)
            if self.iters % counter == 0 or self.iters == 1:
                self.hess[k] = get_approx_analytical_hessian(
                    kpt, self.dtype, ind_up=self.ind_up[k])
                if make_pd:
                    if self.dtype == float:
                        self.hess[k] = np.abs(self.hess[k])
                    else:
                        self.hess[k] = np.abs(self.hess[k].real) \
                            + 1.0j * np.abs(self.hess[k].imag)
            hess = self.hess[k]
            precond[k] = np.zeros_like(hess)
            correction = w * gamma * beta0 ** (-1)
            if self.searchdir_algo.name != 'l-bfgs-p':
                correction = np.zeros_like(hess)
                zeros = abs(hess) < 1.0e-4
                correction[zeros] = 1.0
            precond[k] += 1. / ((1 - gamma) * hess.real + correction)
            if self.dtype == complex:
                precond[k] += 1.j / ((1 - gamma) * hess.imag + correction)

        return precond

    def get_canonical_representation(self, ham, wfs, dens,
                                     sort_eigenvalues=False):
        """
        Choose canonical orbitals as the orbitals that diagonalize the
        Lagrange matrix. It is probably necessary to do a subspace rotation
        with equally occupied orbitals as the total energy is unitary invariant
        within equally occupied subspaces.
        """

        with ((wfs.timer('Get canonical representation'))):
            for kpt in wfs.kpt_u:
                self.dm_helper.update_to_canonical_orbitals(
                    wfs, ham, kpt, self.update_ref_orbs_canonical,
                    self.restart)

            if self.update_ref_orbs_canonical or self.restart:
                wfs.calculate_occupation_numbers(dens.fixed)
                sort_orbitals_according_to_occ(
                    wfs, self.constraints, update_mom=True)

            if sort_eigenvalues:
                sort_orbitals_according_to_energies(
                    ham, wfs, self.constraints)

            self.set_ref_orbitals_and_a_vec(wfs)

    def set_ref_orbitals_and_a_vec(self, wfs):
        self.dm_helper.set_reference_orbitals(wfs, self.n_dim)
        for kpt in wfs.kpt_u:
            u = self.kpointval(kpt)
            self.a_vec_u[u] = np.zeros_like(self.a_vec_u[u])

    def reset(self):
        self.dm_helper = None
        self.error = np.inf
        self.initialized = False
        self.searchdir_algo.reset()

    def todict(self):
        ret = {'name': self.name,
               'searchdir_algo': self.searchdir_algo.todict(),
               'linesearch_algo': self.line_search.todict(),
               'update_ref_orbs_counter': self.update_ref_orbs_counter,
               'update_ref_orbs_canonical': self.update_ref_orbs_canonical,
               'update_precond_counter': self.update_precond_counter,
               'use_prec': self.use_prec,
               'matrix_exp': self.matrix_exp,
               'representation': self.representation,
               'functional': self.func_settings,
               'orthonormalization': self.orthonormalization,
               'randomizeorbitals': self.randomizeorbitals,
               'checkgraderror': self.checkgraderror,
               'localizationtype': self.localizationtype,
               'localizationseed': self.localizationseed,
               'need_localization': self.need_localization,
               'need_init_orbs': self.need_init_orbs,
               'constraints': self.constraints,
               'subspace_convergence': self.subspace_convergence,
               'excited_state': self.excited_state
               }
        if self.gmf:
            ret['partial_diagonalizer'] = \
                self.searchdir_algo.partial_diagonalizer.todict()
        return ret

    def rotate_wavefunctions(self, wfs, a_vec_u, c_ref):

        """
        Apply unitary transformation U = exp(A) to
        the orbitals c_ref

        :param wfs:
        :param a_vec_u:
        :param c_ref:
        :return:
        """

        with wfs.timer('Unitary rotation'):
            for kpt in wfs.kpt_u:
                k = self.kpointval(kpt)
                if self.n_dim[k] == 0:
                    continue

                u_nn = self.get_exponential_matrix_kpt(wfs, kpt, a_vec_u)

                self.dm_helper.appy_transformation_kpt(
                    wfs, u_nn.T, kpt, c_ref[k], False, False)

                with wfs.timer('Calculate projections'):
                    self.dm_helper.update_projections(wfs, kpt)

    def get_exponential_matrix_kpt(self, wfs, kpt, a_vec_u):
        """
        Get unitary matrix U as the exponential of a skew-Hermitian
        matrix A (U = exp(A))
        """

        k = self.kpointval(kpt)

        if self.gd.comm.rank == 0:
            if self.matrix_exp == 'egdecomp-u-invar' and \
                    self.representation == 'u-invar':
                n_occ = get_n_occ(kpt)[0]
                n_v = self.n_dim[k] - n_occ
                a_mat = a_vec_u[k].reshape(n_occ, n_v)
            else:
                a_mat = vec2skewmat(a_vec_u[k], self.n_dim[k],
                                    self.ind_up[k], self.dtype)

            if self.matrix_exp == 'pade-approx':
                # this function takes a lot of memory
                # for large matrices... what can we do?
                with wfs.timer('Pade Approximants'):
                    u_nn = np.ascontiguousarray(expm(a_mat))
            elif self.matrix_exp == 'egdecomp':
                # this method is based on diagonalization
                with wfs.timer('Eigendecomposition'):
                    u_nn, evecs, evals = expm_ed(a_mat, evalevec=True)
            elif self.matrix_exp == 'egdecomp-u-invar':
                with wfs.timer('Eigendecomposition'):
                    u_nn = expm_ed_unit_inv(a_mat, oo_vo_blockonly=False)

        with wfs.timer('Broadcast u_nn'):
            if self.gd.comm.rank != 0:
                u_nn = np.zeros(shape=(self.n_dim[k], self.n_dim[k]),
                                dtype=wfs.dtype)
            self.gd.comm.broadcast(u_nn, 0)

        if self.matrix_exp == 'egdecomp':
            with wfs.timer('Broadcast evecs and evals'):
                if self.gd.comm.rank != 0:
                    evecs = np.zeros(shape=(self.n_dim[k], self.n_dim[k]),
                                     dtype=complex)
                    evals = np.zeros(shape=self.n_dim[k],
                                     dtype=float)
                self.gd.comm.broadcast(evecs, 0)
                self.gd.comm.broadcast(evals, 0)
                self.evecs[k], self.evals[k] = evecs, evals

        return u_nn

    def check_assertions(self, wfs, dens):

        assert dens.mixer.driver.basemixerclass.name == 'no-mixing', \
            'Please, use: mixer={\'backend\': \'no-mixing\'}'
        if wfs.occupations.name != 'mom':
            errormsg = \
                'Please, use occupations={\'name\': \'fixed-uniform\'}'
            assert wfs.occupations.name == 'fixed-uniform', errormsg

    def initialize_mom_reference_orbitals(self, wfs, dens):
        # Reinitialize the MOM reference orbitals
        # after orthogonalization/localization
        occ_name = getattr(wfs.occupations, 'name', None)
        if occ_name == 'mom':
            wfs.occupations.initialize_reference_orbitals()
            wfs.calculate_occupation_numbers(dens.fixed)

    def initial_sort_orbitals(self, wfs):
        occ_name = getattr(wfs.occupations, "name", None)
        if occ_name == 'mom':
            update_mom = True
            self.initial_occupation_numbers = wfs.occupations.numbers.copy()
        else:
            update_mom = False
        sort_orbitals_according_to_occ(wfs,
                                       self.constraints,
                                       update_mom=update_mom)

    def check_mom(self, wfs, dens):
        occ_name = getattr(wfs.occupations, "name", None)
        if occ_name == 'mom':
            wfs.calculate_occupation_numbers(dens.fixed)
            self.restart = sort_orbitals_according_to_occ(
                wfs, self.constraints, update_mom=True)
            return self.restart
        else:
            return False

    def randomize_orbitals_kpt(self, wfs, kpt):
        """
        Add random noise to orbitals but keep them orthonormal
        """
        nst = self.nbands
        wt = kpt.weight * 0.01
        arand = wt * random_a((nst, nst),
                              wfs.dtype,
                              rng=self.randomizeorbitals)
        arand = arand - arand.T.conj()
        wfs.gd.comm.broadcast(arand, 0)
        self.dm_helper.appy_transformation_kpt(wfs, expm(arand), kpt)

    def calculate_hamiltonian_matrix(self, hamiltonian, wfs, kpt):
        H_MM = self.dm_helper.calculate_hamiltonian_matrix(
            hamiltonian, wfs, kpt)
        return H_MM

    def kpointval(self, kpt):
        return self.nkpts * kpt.s + kpt.q

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, e):
        self._error = e


def vec2skewmat(a_vec, dim, ind_up, dtype):

    a_mat = np.zeros(shape=(dim, dim), dtype=dtype)
    a_mat[ind_up] = a_vec
    a_mat -= a_mat.T.conj()
    np.fill_diagonal(a_mat, a_mat.diagonal() * 0.5)
    return a_mat


def convert_constraints(constraints, n_dim, n_occ, representation):
    """
    Parses and checks the user input of constraints. If constraints are passed
    as a list of a single orbital index all pairs of orbitals involving this
    index are added to the constraints.

    :param constraints: List of constraints for one K-point
    :param n_dim:
    :param n_occ:
    :param representation: Unitary invariant, sparse or full representation
                           determining the electronic degrees of freedom that
                           need to be constrained

    :return: Converted list of constraints
    """

    new = constraints.copy()
    for con in constraints:
        assert isinstance(con, list) or isinstance(con, int), \
            'Check constraints.'
        if isinstance(con, list):
            assert len(con) < 3, 'Check constraints.'
            if len(con) == 1:
                con = con[0]  # List of single index
            else:
                # Make first index always smaller than second index
                if representation != 'full' and con[1] < con[0]:
                    temp = deepcopy(con[0])
                    con[0] = deepcopy(con[1])
                    con[1] = temp
                check_indices(
                    con[0], con[1], n_dim, n_occ, representation)
                continue
        # Add all pairs containing the single index
        if isinstance(con, int):
            new += find_all_pairs(con, n_dim, n_occ, representation)

    # Delete all list containing a single index
    done = False
    while not done:
        done = True
        for i in range(len(new)):
            if len(new[i]) < 2:
                del new[i]
                done = False
                break
    return new


def check_indices(ind1, ind2, n_dim, n_occ, representation):
    """
    Makes sure the user input for the constraints makes sense.
    """

    assert ind1 != ind2, 'Check constraints.'
    if representation == 'full':
        assert ind1 < n_dim and ind2 < n_dim, 'Check constraints.'
    elif representation == 'sparse':
        assert ind1 < n_occ and ind2 < n_dim, 'Check constraints.'
    elif representation == 'u-invar':
        assert ind1 < n_occ and ind2 >= n_occ and ind2 < n_dim, \
            'Check constraints.'


def find_all_pairs(ind, n_dim, n_occ, representation):
    """
    Creates a list of all orbital pairs corresponding to degrees of freedom of
    the system containing an orbital index.

    :param ind: The orbital index
    :param n_dim:
    :param n_occ:
    :param representation: Unitary invariant, sparse or full, defining what
                           index pairs correspond to degrees of freedom of the
                           system
    """

    pairs = []
    if representation == 'u-invar':
        # Only ov rotations are degrees of freedom
        if ind < n_occ:
            for i in range(n_occ, n_dim):
                pairs.append([ind, i])
        else:
            for i in range(n_occ):
                pairs.append([i, ind])
    else:
        if (ind < n_occ and representation == 'sparse') \
                or representation == 'full':
            # oo and ov rotations are degrees of freedom
            for i in range(n_dim):
                if i == ind:
                    continue
                pairs.append([i, ind] if i < ind else [ind, i])
                if representation == 'full':
                    # The orbital rotation matrix is not assumed to be
                    # antihermitian, so the reverse order of indices must be
                    # added as a second constraint
                    pairs.append([ind, i] if i < ind else [i, ind])
        else:
            for i in range(n_occ):
                pairs.append([i, ind])
    return pairs
