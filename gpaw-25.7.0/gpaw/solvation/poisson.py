import warnings

import numpy as np
from scipy.special import erf

from gpaw.poisson import BasePoissonSolver, PoissonSolver, FDPoissonSolver
from gpaw.fd_operators import Laplace, Gradient
from gpaw.wfd_operators import WeightedFDOperator
from gpaw.utilities.gauss import Gaussian
from gpaw import PoissonConvergenceError
from gpaw.utilities.timing import NullTimer


class SolvationPoissonSolver(FDPoissonSolver):
    """Base class for Poisson solvers with spatially varying dielectric.

    The Poisson equation
    div(epsilon(r) grad phi(r)) = -4 pi rho(r)
    is solved.
    """

    def __init__(self, nn=3, relax='J', eps=2e-10, maxiter=1000,
                 remove_moment=None, use_charge_center=False):
        if remove_moment is not None:
            raise NotImplementedError(
                'Removing arbitrary multipole moments '
                'is not implemented for SolvationPoissonSolver!')
        if nn == 'M':
            raise NotImplementedError(
                'Mehrstellen stencil is not implemented '
                'for SolvationPoissonSolver!')
        FDPoissonSolver.__init__(self, nn, relax, eps, maxiter, remove_moment,
                                 use_charge_center=use_charge_center)

    def set_dielectric(self, dielectric):
        """Set the dielectric.

        Arguments:
        dielectric -- A Dielectric instance.
        """
        self.dielectric = dielectric

    def load_gauss(self, center=None):
        """Load compensating charge distribution for charged systems.

        See Appendix B of
        A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).
        """
        # XXX Check if update is needed (dielectric changed)?
        epsr, dx_epsr, dy_epsr, dz_epsr = self.dielectric.eps_gradeps
        gauss = Gaussian(self.gd, center=center)
        rho_g = gauss.get_gauss(0)
        phi_g = gauss.get_gauss_pot(0)
        x, y, z = gauss.xyz
        fac = 2. * np.sqrt(gauss.a) * np.exp(-gauss.a * gauss.r2)
        fac /= np.sqrt(np.pi) * gauss.r2
        fac -= erf(np.sqrt(gauss.a) * gauss.r) / (gauss.r2 * gauss.r)
        fac *= 2.0 * 1.7724538509055159
        dx_phi_g = fac * x
        dy_phi_g = fac * y
        dz_phi_g = fac * z
        sp = dx_phi_g * dx_epsr + dy_phi_g * dy_epsr + dz_phi_g * dz_epsr
        rho = epsr * rho_g - 1. / (4. * np.pi) * sp
        invnorm = np.sqrt(4. * np.pi) / self.gd.integrate(rho)
        self.phi_gauss = phi_g * invnorm
        self.rho_gauss = rho * invnorm


class WeightedFDPoissonSolver(SolvationPoissonSolver):
    """Weighted finite difference Poisson solver with dielectric.

    Following V. M. Sanchez, M. Sued and D. A. Scherlis,
    J. Chem. Phys. 131, 174108 (2009).
    """

    def create_laplace(self, gd, scale=1.0, n=1, dtype=float):
        operators = [Laplace(gd, scale, n, dtype)]
        operators += [Gradient(gd, j, scale, n, dtype) for j in (0, 1, 2)]
        return WeightedFDOperator(operators)

    def solve(self, phi, rho, charge=None,
              maxcharge=1e-6,
              zero_initial_phi=False, timer=None):
        self._init()
        if self.gd.pbc_c.all():
            actual_charge = self.gd.integrate(rho)
            if abs(actual_charge) > maxcharge:
                raise NotImplementedError(
                    'charged periodic systems are not implemented')
        self.restrict_op_weights()
        ret = FDPoissonSolver.solve(self, phi, rho, charge, maxcharge,
                                    zero_initial_phi)
        return ret

    def restrict_op_weights(self):
        """Restric operator weights to coarse grids."""
        self._init()
        weights = [self.dielectric.eps_gradeps] + self.op_coarse_weights
        for i, res in enumerate(self.restrictors):
            for j in range(4):
                res.apply(weights[i][j], weights[i + 1][j])
        self.step = 0.66666666 / self.operators[0].get_diagonal_element()

    def get_description(self):
        description = SolvationPoissonSolver.get_description(self)
        return description.replace(
            'solver with',
            'weighted FD solver with dielectric and')

    def _init(self):
        if self._initialized:
            return
        self.operators[0].set_weights(self.dielectric.eps_gradeps)
        self.op_coarse_weights = []
        for operator in self.operators[1:]:
            weights = [gd.empty() for gd in (operator.gd, ) * 4]
            self.op_coarse_weights.append(weights)
            operator.set_weights(weights)
        return SolvationPoissonSolver._init(self)


class PolarizationPoissonSolver(BasePoissonSolver):
    """Poisson solver with dielectric.

    Calculates the polarization charges first using only the
    vacuum poisson equation, then solves the vacuum equation
    with polarization charges.
    """

    def __init__(self, nn=3, relax='J', eps=2e-10, maxiter=1000,
                 remove_moment=None, use_charge_center=False,
                 gas_phase_poisson='fast'):
        self.nn = nn
        self.eps = eps
        self.maxiter = maxiter
        self.phi_tilde = None

        self.gas_phase_poisson = PoissonSolver(
            name=gas_phase_poisson, nn=nn, eps=eps,
            remove_moment=remove_moment,
            use_charge_center=use_charge_center)

    def set_dielectric(self, dielectric):
        """Set the dielectric.

        Arguments:
        dielectric -- A Dielectric instance.
        """
        self.dielectric = dielectric

    def get_description(self):
        return ('PolarizationPoissonSolver based on '
                + self.gas_phase_poisson.get_description())

    def set_grid_descriptor(self, gd):
        self.gd = gd
        self.gas_phase_poisson.set_grid_descriptor(gd)

    def solve(self, phi, rho, charge=None,
              maxcharge=1e-6,
              zero_initial_phi=False, timer=NullTimer()):
        # get initial meaningful phi -> only do this if necessary
        if zero_initial_phi:
            niter = self.gas_phase_poisson.solve(
                phi, rho, charge=None, maxcharge=maxcharge,
                zero_initial_phi=zero_initial_phi, timer=timer)
        else:
            niter = 0

        phi_old = phi.copy()
        while niter < self.maxiter:
            rho_mod = self.rho_with_polarization_charge(phi, rho)
            niter += self.gas_phase_poisson.solve(
                phi, rho_mod, charge=None, maxcharge=maxcharge,
                zero_initial_phi=zero_initial_phi, timer=timer)
            residual = phi - phi_old
            error = self.gd.comm.sum_scalar(
                np.dot(residual.ravel(), residual.ravel())) * self.gd.dv
            if error < self.eps:
                return niter
            phi_old = phi.copy()

        raise PoissonConvergenceError(
            'PolarizationPoisson solver did not converge in '
            + f'{niter} iterations!')

    def rho_with_polarization_charge(self, phi, rho):
        epsr, dx_epsr, dy_epsr, dz_epsr = self.dielectric.eps_gradeps
        dx_phi = self.gd.empty()
        dy_phi = self.gd.empty()
        dz_phi = self.gd.empty()
        Gradient(self.gd, 0, 1.0, self.nn).apply(phi, dx_phi)
        Gradient(self.gd, 1, 1.0, self.nn).apply(phi, dy_phi)
        Gradient(self.gd, 2, 1.0, self.nn).apply(phi, dz_phi)

        scalar_product = (
            dx_epsr * dx_phi +
            dy_epsr * dy_phi +
            dz_epsr * dz_phi)

        return (rho + scalar_product / (4. * np.pi)) / epsr

    def estimate_memory(self, mem):
        # XXX estimate your own contribution
        return self.gas_phase_poisson.estimate_memory(mem)

    def todict(self):
        my_dict = self.gas_phase_poisson.todict()
        my_dict['name'] = f"polarization-{my_dict['name']}"
        return my_dict


class ADM12PoissonSolver(SolvationPoissonSolver):
    """Poisson solver with dielectric.

    Following O. Andreussi, I. Dabo, and N. Marzari,
    J. Chem. Phys. 136, 064102 (2012).

    Warning: Not intended for production use, as it is not tested
             thouroughly!

    XXX TODO : * Correction for charged systems???
               * Check: Can the polarization charge introduce a monopole?
               * Convergence problems depending on eta. Apparently this
                 method works best with FFT as in the original Paper.
               * Optimize numerics.
    """

    def __init__(self, nn=3, relax='J', eps=2e-10, maxiter=1000,
                 remove_moment=None, eta=.6, use_charge_center=False):
        """Constructor for ADM12PoissonSolver.

        Additional arguments not present in SolvationPoissonSolver:
        eta -- linear mixing parameter
        """
        adm12_warning = UserWarning(
            'ADM12PoissonSolver is not tested thoroughly'
            ' and therefore not recommended for production code!')
        warnings.warn(adm12_warning)
        self.eta = eta
        SolvationPoissonSolver.__init__(
            self, nn, relax, eps, maxiter, remove_moment,
            use_charge_center=use_charge_center)

    def set_grid_descriptor(self, gd):
        SolvationPoissonSolver.set_grid_descriptor(self, gd)
        self.gradx = Gradient(gd, 0, 1.0, self.nn)
        self.grady = Gradient(gd, 1, 1.0, self.nn)
        self.gradz = Gradient(gd, 2, 1.0, self.nn)

    def get_description(self):
        if len(self.operators) == 0:
            return 'uninitialized ADM12PoissonSolver'
        else:
            description = SolvationPoissonSolver.get_description(self)
            return description.replace(
                'solver with',
                'ADM12 solver with dielectric and')

    def _init(self):
        if self._initialized:
            return
        self.rho_iter = self.gd.zeros()
        self.d_phi = self.gd.empty()
        return SolvationPoissonSolver._init(self)

    def solve(self, phi, rho, charge=None,
              maxcharge=1e-6,
              zero_initial_phi=False, timer=None):
        self._init()
        if self.gd.pbc_c.all():
            actual_charge = self.gd.integrate(rho)
            if abs(actual_charge) > maxcharge:
                raise NotImplementedError(
                    'charged periodic systems are not implemented')
        return FDPoissonSolver.solve(
            self, phi, rho, charge, maxcharge, zero_initial_phi,
            timer=timer)

    def solve_neutral(self, phi, rho, timer=None):
        self._init()
        self.rho = rho
        return SolvationPoissonSolver.solve_neutral(self, phi, rho)

    def iterate2(self, step, level=0):
        self._init()
        if level == 0:
            epsr, dx_epsr, dy_epsr, dz_epsr = self.dielectric.eps_gradeps
            self.gradx.apply(self.phis[0], self.d_phi)
            sp = dx_epsr * self.d_phi
            self.grady.apply(self.phis[0], self.d_phi)
            sp += dy_epsr * self.d_phi
            self.gradz.apply(self.phis[0], self.d_phi)
            sp += dz_epsr * self.d_phi
            self.rho_iter = self.eta / (4. * np.pi) * sp + \
                (1. - self.eta) * self.rho_iter
            self.rhos[0][:] = (self.rho_iter + self.rho) / epsr
        return SolvationPoissonSolver.iterate2(self, step, level)
