import time
from ase.units import Hartree
from gpaw.directmin.derivatives import get_approx_analytical_hessian


class LCAOETDMLocalize:
    """Class for performing PZ-SIC localization for LCAO-ETDM."""

    def __init__(self, etdm_solver, wfs, log, tol=5.0e-4, maxiter=200):
        self.solver = etdm_solver
        self.wfs = wfs
        self.log = log
        self.tol = tol
        self.maxiter = maxiter
        self.ham = None
        self.dens = None

        assert self.solver.dm_helper.func.name == 'PZ-SIC', (
            'PZ-SIC localization requested, but functional '
            'settings do not use PZ-SIC.')

    def run(self, ham, dens):
        """Run the localization."""
        self.ham = ham
        self.dens = dens

        self.log('Perdew-Zunger localization started', flush=True)
        self.log_header()

        self.solver.lock_subspace('oo')
        self.solver.dm_helper.set_reference_orbitals(
            self.wfs, self.solver.n_dim)

        original_eg_count = self.solver.eg_count

        counter = 0
        converged = False
        while not converged:
            self.solver.iterate(self.ham, self.wfs, self.dens)
            counter += 1

            e_total = self.solver.phi_2i[0]
            g_max = self.solver.get_grad_norm()

            self.log_iteration(counter, e_total, g_max)

            if g_max < self.tol or counter >= self.maxiter:
                converged = True

        final_eg_count = self.solver.eg_count
        eg_calls_in_loop = final_eg_count - original_eg_count

        self.solver.release_subspace()
        self.solver.dm_helper.set_reference_orbitals(
            self.wfs, self.solver.n_dim
        )
        self.solver.searchdir_algo.reset()
        for k, kpt in enumerate(self.wfs.kpt_u):
            self.solver.hess[k] = get_approx_analytical_hessian(
                kpt, self.solver.dtype, ind_up=self.solver.ind_up[k])
            self.wfs.atomic_correction.calculate_projections(self.wfs, kpt)

        self.log('Perdew-Zunger localization finished', flush=True)
        self.log('Total number of e/g calls: %d' % eg_calls_in_loop)
        self.solver.subspace_iters = 0

    def log_header(self):
        self.log('\nINNER LOOP:')
        self.log(
            '                      Kohn-Sham'
            '          SIC        Total             '
        )
        self.log(
            '           time         energy:'
            '      energy:      energy:       G_max:'
        )

    def log_iteration(self, niter, e_total, g_max):
        t = time.localtime()
        e_ks = e_total - self.solver.e_sic

        self.log('iter: %3d  %02d:%02d:%02d ' %
                 (niter, t[3], t[4], t[5]), end='')

        log_items = [Hartree * e_ks,
                     Hartree * self.solver.e_sic,
                     Hartree * e_total,
                     Hartree * g_max]

        self.log('%11.6f  %11.6f  %11.6f  %11.1e' % tuple(log_items), end='')
        self.log(flush=True)
