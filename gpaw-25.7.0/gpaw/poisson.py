# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import warnings
from math import pi

import numpy as np
from numpy.fft import fftn, ifftn, fft2, ifft2, rfft2, irfft2, fft, ifft
from scipy.fftpack import dst as scipydst

from gpaw import PoissonConvergenceError
from gpaw.dipole_correction import DipoleCorrection, dipole_correction
from gpaw.domain import decompose_domain
from gpaw.fd_operators import Laplace, LaplaceA, LaplaceB
from gpaw.transformers import Transformer
from gpaw.utilities.gauss import Gaussian
from gpaw.utilities.grid import grid2grid
from gpaw.utilities.ewald import madelung
from gpaw.utilities.tools import construct_reciprocal
from gpaw.utilities.timing import NullTimer

POISSON_GRID_WARNING = """Grid unsuitable for FDPoissonSolver!

Consider using FastPoissonSolver instead.

The FDPoissonSolver does not have sufficient multigrid levels for good
performance and will converge inefficiently if at all, or yield wrong
results.

You may need to manually specify a grid such that the number of points
along each direction is divisible by a high power of 2, such as 8, 16,
or 32 depending on system size; examples:

  GPAW(gpts=(32, 32, 288))

or

  from gpaw.utilities import h2gpts
  GPAW(gpts=h2gpts(0.2, atoms.get_cell(), idiv=16))

Parallelizing over very small domains can also undesirably limit the
number of multigrid levels even if the total number of grid points
is divisible by a high power of 2."""


def create_poisson_solver(name='fast', **kwargs):
    if isinstance(name, _PoissonSolver):
        return name
    elif isinstance(name, dict):
        kwargs.update(name)
        return create_poisson_solver(**kwargs)
    elif name == 'fft':
        return FFTPoissonSolver(**kwargs)
    elif name == 'fdtd':
        from gpaw.fdtd.poisson_fdtd import FDTDPoissonSolver
        return FDTDPoissonSolver(**kwargs)
    elif name == 'fd':
        return FDPoissonSolverWrapper(**kwargs)
    elif name == 'fast':
        return FastPoissonSolver(**kwargs)
    elif name == 'ExtraVacuumPoissonSolver':
        from gpaw.poisson_extravacuum import ExtraVacuumPoissonSolver
        return ExtraVacuumPoissonSolver(**kwargs)
    elif name == 'MomentCorrectionPoissonSolver':
        from gpaw.poisson_moment import MomentCorrectionPoissonSolver
        return MomentCorrectionPoissonSolver(**kwargs)
    elif name == 'nointeraction':
        return NoInteractionPoissonSolver()
    else:
        raise ValueError('Unknown poisson solver: %s' % name)


def PoissonSolver(name='fast', dipolelayer=None, zero_vacuum=False, **kwargs):
    p = create_poisson_solver(name=name, **kwargs)
    if dipolelayer is not None:
        p = DipoleCorrection(p, dipolelayer, zero_vacuum=zero_vacuum)
    return p


def FDPoissonSolverWrapper(dipolelayer=None, zero_vacuum=False, **kwargs):
    if dipolelayer is not None:
        return DipoleCorrection(FDPoissonSolver(**kwargs), dipolelayer,
                                zero_vacuum=zero_vacuum)
    return FDPoissonSolver(**kwargs)


class _PoissonSolver:
    """Abstract PoissonSolver class

       This class defines an interface and a common ancestor
       for various PoissonSolver implementations (including wrappers)."""
    def __init__(self):
        object.__init__(self)

    def set_grid_descriptor(self, gd):
        raise NotImplementedError()

    def solve(self):
        raise NotImplementedError()

    def todict(self):
        raise NotImplementedError(self.__class__.__name__)

    def get_description(self):
        return self.__class__.__name__

    def estimate_memory(self, mem):
        raise NotImplementedError()

    def build(self, grid, xp):
        from gpaw.new.poisson import PoissonSolverWrapper
        self.xp = xp
        self.set_grid_descriptor(grid._gd)
        return PoissonSolverWrapper(self)


class BasePoissonSolver(_PoissonSolver):
    def __init__(self, *, remove_moment=None,
                 use_charge_center=False,
                 metallic_electrodes=False,
                 eps=None,
                 use_charged_periodic_corrections=False,
                 xp=np):

        self.xp = xp

        if eps is not None:
            warnings.warn(
                "Please do not specify the eps parameter "
                f"for {self.__class__.__name__}. "
                "The parameter doesn't do anything for this solver "
                "and defining it will throw an error in the future.",
                FutureWarning)

        if remove_moment is not None:
            warnings.warn(
                "Please do not specify the remove_moment parameter "
                f"for {self.__class__.__name__}. "
                "The remove moment functionality is deprecated in this solver "
                "and will throw an error in the future. Instead "
                "use the MomentCorrectionPoissonSolver as a wrapper to "
                f"{self.__class__.__name__}.",
                FutureWarning)

        # metallic electrodes: mirror image method to allow calculation of
        # charged, partly periodic systems
        self.gd = None
        self.remove_moment = remove_moment
        self.use_charge_center = use_charge_center
        self.use_charged_periodic_corrections = \
            use_charged_periodic_corrections
        self.charged_periodic_correction = None
        self.eps = eps
        self.metallic_electrodes = metallic_electrodes
        assert self.metallic_electrodes in [False, None, 'single', 'both']

    def todict(self):
        d = {'name': 'basepoisson'}
        if self.remove_moment:
            d['remove_moment'] = self.remove_moment
        if self.use_charge_center:
            d['use_charge_center'] = self.use_charge_center
        if self.use_charged_periodic_corrections:
            d['use_charged_periodic_corrections'] = \
                self.use_charged_periodic_corrections
        if self.metallic_electrodes:
            d['metallic_electrodes'] = self.metallic_electrodes

        return d

    def get_description(self):
        # The idea is that the subclass writes a header and main parameters,
        # then adds the below string.
        lines = []
        if self.remove_moment is not None:
            lines.append('    Remove moments up to L=%d' % self.remove_moment)
        if self.use_charge_center:
            lines.append('    Compensate for charged system using center of '
                         'majority charge')
        if self.use_charged_periodic_corrections:
            lines.append('    Subtract potential of homogeneous background')

        return '\n'.join(lines)

    def solve(self, phi, rho, charge=None, maxcharge=1e-6,
              zero_initial_phi=False, timer=NullTimer()):
        self._init()
        assert np.all(phi.shape == self.gd.n_c)
        assert np.all(rho.shape == self.gd.n_c)

        actual_charge = self.gd.integrate(rho)
        background = (actual_charge / self.gd.dv /
                      self.gd.get_size_of_global_array().prod())

        if self.remove_moment:
            assert not self.gd.pbc_c.any()
            if not hasattr(self, 'gauss'):
                self.gauss = Gaussian(self.gd)
            rho_neutral = rho.copy()
            phi_cor_L = []
            for L in range(self.remove_moment):
                phi_cor_L.append(self.gauss.remove_moment(rho_neutral, L))
            # Remove multipoles for better initial guess
            for phi_cor in phi_cor_L:
                phi -= phi_cor

            niter = self.solve_neutral(phi, rho_neutral, timer=timer)
            # correct error introduced by removing multipoles
            for phi_cor in phi_cor_L:
                phi += phi_cor

            return niter
        if charge is None:
            charge = actual_charge
        if abs(charge) <= maxcharge:
            return self.solve_neutral(phi, rho - background, timer=timer)

        elif abs(charge) > maxcharge and self.gd.pbc_c.all():
            # System is charged and periodic. Subtract a homogeneous
            # background charge

            # Set initial guess for potential
            if zero_initial_phi:
                phi[:] = 0.0

            iters = self.solve_neutral(phi, rho - background, timer=timer)

            if self.use_charged_periodic_corrections:
                if self.charged_periodic_correction is None:
                    self.charged_periodic_correction = madelung(
                        self.gd.cell_cv)
                phi += actual_charge * self.charged_periodic_correction

            return iters

        elif abs(charge) > maxcharge and not self.gd.pbc_c.any():
            # The system is charged and in a non-periodic unit cell.
            # Determine the potential by 1) subtract a gaussian from the
            # density, 2) determine potential from the neutralized density
            # and 3) add the potential from the gaussian density.

            # Load necessary attributes

            # use_charge_center: The monopole will be removed at the
            # center of the majority charge, which prevents artificial
            # dipoles.
            # Due to the shape of the Gaussian and it's Fourier-Transform,
            # the Gaussian representing the charge should stay at least
            # 7 gpts from the borders - see:
            # listserv.fysik.dtu.dk/pipermail/gpaw-developers/2015-July/005806.html
            if self.use_charge_center:
                charge_sign = actual_charge / abs(actual_charge)
                rho_sign = rho * charge_sign
                rho_sign[np.where(rho_sign < 0)] = 0
                absolute_charge = self.gd.integrate(rho_sign)
                center = - (self.gd.calculate_dipole_moment(rho_sign) /
                            absolute_charge)
                border_offset = np.inner(self.gd.h_cv, np.array((7, 7, 7)))
                borders = np.inner(self.gd.h_cv, self.gd.N_c)
                borders -= border_offset
                if np.any(center > borders) or np.any(center < border_offset):
                    raise RuntimeError('Poisson solver: '
                                       'center of charge outside borders '
                                       '- please increase box')
                    center[np.where(center > borders)] = borders
                self.load_gauss(center=center)
            else:
                self.load_gauss()

            # Remove monopole moment
            q = actual_charge / np.sqrt(4 * pi)  # Monopole moment
            rho_neutral = rho - q * self.rho_gauss  # neutralized density

            # Set initial guess for potential
            if zero_initial_phi:
                phi[:] = 0.0
            else:
                phi -= q * self.phi_gauss

            # Determine potential from neutral density using standard solver
            niter = self.solve_neutral(phi, rho_neutral, timer=timer)

            # correct error introduced by removing monopole
            phi += q * self.phi_gauss

            return niter
        else:
            # System is charged with mixed boundaryconditions
            if self.metallic_electrodes == 'single':
                self.c = 2
                origin_c = [0, 0, 0]
                origin_c[self.c] = self.gd.N_c[self.c]
                drhot_g, dvHt_g, self.correction = dipole_correction(
                    self.c,
                    self.gd,
                    rho,
                    origin_c=origin_c)
                # self.correction *=-1.
                phi -= dvHt_g
                iters = self.solve_neutral(phi, rho + drhot_g, timer=timer)
                phi += dvHt_g
                phi -= self.correction
                self.correction = 0.0

                return iters

            elif self.metallic_electrodes == 'both':
                iters = self.solve_neutral(phi, rho, timer=timer)
                return iters

            else:
                # System is charged with mixed boundaryconditions
                msg = ('Charged systems with mixed periodic/zero'
                       ' boundary conditions')
                raise NotImplementedError(msg)

    def load_gauss(self, center=None):
        if not hasattr(self, 'rho_gauss') or center is not None:
            gauss = Gaussian(self.gd, center=center)
            self.rho_gauss = self.xp.asarray(gauss.get_gauss(0))
            self.phi_gauss = self.xp.asarray(gauss.get_gauss_pot(0))


class FDPoissonSolver(BasePoissonSolver):
    def __init__(self, nn=3, relax='J', eps=2e-10, maxiter=1000,
                 remove_moment=None, use_charge_center=False,
                 metallic_electrodes=False,
                 use_charged_periodic_corrections=False, **kwargs):
        super(FDPoissonSolver, self).__init__(
            remove_moment=remove_moment,
            use_charge_center=use_charge_center,
            metallic_electrodes=metallic_electrodes,
            use_charged_periodic_corrections=use_charged_periodic_corrections,
            **kwargs)
        self.eps = eps
        self.relax = relax
        self.nn = nn
        self.maxiter = maxiter

        # Relaxation method
        if relax == 'GS':
            # Gauss-Seidel
            self.relax_method = 1
        elif relax == 'J':
            # Jacobi
            self.relax_method = 2
        else:
            raise NotImplementedError('Relaxation method %s' % relax)

        self.description = None
        self._initialized = False

    def todict(self):
        d = super().todict()
        d.update({'name': 'fd', 'nn': self.nn, 'relax': self.relax,
                  'eps': self.eps})
        return d

    def get_stencil(self):
        return self.nn

    def create_laplace(self, gd, scale=1.0, n=1, dtype=float):
        """Instantiate and return a Laplace operator

        Allows subclasses to change the Laplace operator
        """
        return Laplace(gd, scale, n, dtype, xp=self.xp)

    def set_grid_descriptor(self, gd):
        # Should probably be renamed initialize
        self.gd = gd
        scale = -0.25 / pi

        if self.nn == 'M':
            if not gd.orthogonal:
                raise RuntimeError('Cannot use Mehrstellen stencil with '
                                   'non orthogonal cell.')

            self.operators = [LaplaceA(gd, -scale, xp=self.xp)]
            self.B = LaplaceB(gd)
        else:
            self.operators = [self.create_laplace(gd, scale, self.nn)]
            self.B = None

        self.interpolators = []
        self.restrictors = []

        level = 0
        self.presmooths = [2]
        self.postsmooths = [1]

        # Weights for the relaxation,
        # only used if 'J' (Jacobi) is chosen as method
        self.weights = [2.0 / 3.0]

        while level < 8:
            try:
                gd2 = gd.coarsen()
            except ValueError:
                break
            self.operators.append(self.create_laplace(gd2, scale, 1))
            self.interpolators.append(Transformer(gd2, gd, xp=self.xp))
            self.restrictors.append(Transformer(gd, gd2, xp=self.xp))
            self.presmooths.append(4)
            self.postsmooths.append(4)
            self.weights.append(1.0)
            level += 1
            gd = gd2

        self.levels = level

        if self.operators[-1].gd.N_c.max() > 36:
            # Try to warn exactly once no matter how one uses the solver.
            if gd.comm.parent is None:
                warn = (gd.comm.rank == 0)
            else:
                warn = (gd.comm.parent.rank == 0)

            if warn:
                warntxt = '\n'.join([POISSON_GRID_WARNING, '',
                                     self.get_description()])
            else:
                warntxt = ('Poisson warning from domain rank %d'
                           % self.gd.comm.rank)

            # Warn from all ranks to avoid deadlocks.
            warnings.warn(warntxt, stacklevel=2)

        self._initialized = False
        # The Gaussians depend on the grid as well so we have to 'unload' them
        if hasattr(self, 'rho_gauss'):
            del self.rho_gauss
            del self.phi_gauss

    def get_description(self):
        name = {1: 'Gauss-Seidel', 2: 'Jacobi'}[self.relax_method]
        coarsest_grid = self.operators[-1].gd.N_c
        coarsest_grid_string = ' x '.join([str(N) for N in coarsest_grid])
        assert self.levels + 1 == len(self.operators)
        lines = ['%s solver with %d multi-grid levels'
                 % (name, self.levels + 1),
                 '    Coarsest grid: %s points' % coarsest_grid_string]
        if coarsest_grid.max() > 24:
            # This friendly warning has lower threshold than the big long
            # one that we print when things are really bad.
            lines.extend(['    Warning: Coarse grid has more than 24 points.',
                          '             More multi-grid levels recommended.'])
        lines.extend(['    Stencil: %s' % self.operators[0].description,
                      '    Max iterations: %d' % self.maxiter])
        lines.extend(['    Tolerance: %e' % self.eps])
        lines.append(super().get_description())
        return '\n'.join(lines)

    def _init(self):
        if self._initialized:
            return
        # Should probably be renamed allocate
        gd = self.gd
        self.rhos = [gd.empty(xp=self.xp)]
        self.phis = [None]
        self.residuals = [gd.empty(xp=self.xp)]
        for level in range(self.levels):
            gd2 = gd.coarsen()
            self.phis.append(gd2.empty(xp=self.xp))
            self.rhos.append(gd2.empty(xp=self.xp))
            self.residuals.append(gd2.empty(xp=self.xp))
            gd = gd2
        assert len(self.phis) == len(self.rhos)
        level += 1
        assert level == self.levels

        self.step = 0.66666666 / self.operators[0].get_diagonal_element()
        self.presmooths[level] = 8
        self.postsmooths[level] = 8
        self._initialized = True

    def solve_neutral(self, phi, rho, timer=None):
        self._init()
        self.phis[0] = phi
        eps = self.eps
        if self.B is None:
            self.rhos[0][:] = rho
        else:
            self.B.apply(rho, self.rhos[0])

        niter = 1
        maxiter = self.maxiter
        while self.iterate2(self.step) > eps and niter < maxiter:
            niter += 1
        if niter == maxiter:
            msg = 'Poisson solver did not converge in %d iterations!' % maxiter
            raise PoissonConvergenceError(msg)

        # Set the average potential to zero in periodic systems
        if (self.gd.pbc_c).all():
            phi_ave = self.gd.comm.sum_scalar(float(np.sum(phi.ravel())))
            N_c = self.gd.get_size_of_global_array()
            phi_ave /= np.prod(N_c)
            phi -= phi_ave

        return niter

    def iterate2(self, step, level=0):
        """Smooths the solution in every multigrid level"""
        self._init()

        residual = self.residuals[level]

        if level < self.levels:
            self.operators[level].relax(self.relax_method,
                                        self.phis[level],
                                        self.rhos[level],
                                        self.presmooths[level],
                                        self.weights[level])

            self.operators[level].apply(self.phis[level], residual)
            residual -= self.rhos[level]
            self.restrictors[level].apply(residual,
                                          self.rhos[level + 1])
            self.phis[level + 1][:] = 0.0
            self.iterate2(4.0 * step, level + 1)
            self.interpolators[level].apply(self.phis[level + 1], residual)
            self.phis[level] -= residual

        self.operators[level].relax(self.relax_method,
                                    self.phis[level],
                                    self.rhos[level],
                                    self.postsmooths[level],
                                    self.weights[level])
        if level == 0:
            self.operators[level].apply(self.phis[level], residual)
            residual -= self.rhos[level]
            error = self.gd.comm.sum_scalar(
                float(self.xp.dot(residual.ravel(),
                                  residual.ravel()))) * self.gd.dv

            # How about this instead:
            # error = self.gd.comm.max(abs(residual).max())

            return error

    def estimate_memory(self, mem):
        # XXX Memory estimate works only for J and GS, not FFT solver
        # Poisson solver appears to use same amount of memory regardless
        # of whether it's J or GS, which is a bit strange

        gdbytes = self.gd.bytecount()
        nbytes = -gdbytes  # No phi on finest grid, compensate ahead
        for level in range(self.levels):
            nbytes += 3 * gdbytes  # Arrays: rho, phi, residual
            gdbytes //= 8
        mem.subnode('rho, phi, residual [%d levels]' % self.levels, nbytes)

    def __repr__(self):
        template = 'FDPoissonSolver(relax=\'%s\', nn=%s, eps=%e)'
        representation = template % (self.relax, repr(self.nn), self.eps)
        return representation


class NoInteractionPoissonSolver(_PoissonSolver):
    relax_method = 0
    nn = 1

    def get_description(self):
        return 'No interaction'

    def get_stencil(self):
        return 1

    def solve(self, phi, rho, charge, timer=None):
        return 0

    def set_grid_descriptor(self, gd):
        pass

    def todict(self):
        return {'name': 'nointeraction'}

    def estimate_memory(self, mem):
        pass


class FFTPoissonSolver(BasePoissonSolver):
    """FFT Poisson solver for general unit cells."""

    relax_method = 0
    nn = 999

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialized = False

    def get_description(self):
        return 'Parallel FFT'

    def todict(self):
        return {'name': 'fft'}

    def set_grid_descriptor(self, gd):
        # We will probably want to use this on non-periodic grids too...
        assert gd.pbc_c.all()
        self.gd = gd

        self.grids = [gd]
        for c in range(3):
            N_c = gd.N_c.copy()
            N_c[c] = 1  # Will be serial in that direction
            parsize_c = decompose_domain(N_c, gd.comm.size)
            self.grids.append(gd.new_descriptor(parsize_c=parsize_c))
        self._initialized = False

    def _init(self):
        if self._initialized:
            return

        gd = self.grids[-1]
        k2_Q, N3 = construct_reciprocal(gd)
        self.poisson_factor_Q = 4.0 * np.pi / k2_Q
        self._initialized = True

    def solve_neutral(self, phi_g, rho_g, timer=None):
        self._init()
        # Will be a bit more efficient if reduced dimension is always
        # contiguous.  Probably more things can be improved...

        gd1 = self.gd
        work1_g = rho_g

        for c in range(3):
            gd2 = self.grids[c + 1]
            work2_g = gd2.empty(dtype=work1_g.dtype)
            grid2grid(gd1.comm, gd1, gd2, work1_g, work2_g)
            work1_g = fftn(work2_g, axes=[c])
            gd1 = gd2

        work1_g *= self.poisson_factor_Q

        for c in [2, 1, 0]:
            gd2 = self.grids[c]
            work2_g = ifftn(work1_g, axes=[c])
            work1_g = gd2.empty(dtype=work2_g.dtype)
            grid2grid(gd1.comm, gd1, gd2, work2_g, work1_g)
            gd1 = gd2

        phi_g[:] = work1_g.real
        return 1

    def estimate_memory(self, mem):
        mem.subnode('k squared', self.grids[-1].bytecount())


"""def rfst2(A_g, axes=[0,1]):
    assert axes[0] == 0
    assert axes[1] == 1
    x,y,z = A_g.shape
    temp_g = np.zeros((x*2+2, y*2+2, z))
    temp_g[1:x+1, 1:y+1,:] = A_g
    temp_g[x+2:, 1:y+1,:] = -A_g[::-1, :, :]
    temp_g[1:x+1, y+2:,:] = -A_g[:, ::-1, :]
    temp_g[x+2:, y+2:,:] = A_g[::-1, ::-1, :]
    X = -4*rfft2(temp_g, axes=axes)
    return X[1:x+1, 1:y+1, :].real

def irfst2(A_g, axes=[0,1]):
    assert axes[0] == 0
    assert axes[1] == 1
    x,y,z = A_g.shape
    temp_g = np.zeros((x*2+2, (y*2+2)//2+1, z))
    temp_g[1:x+1, 1:y+1,:] = A_g
    temp_g[x+2:, 1:y+1,:] = -A_g[::-1, :, :]
    return -0.25*irfft2(temp_g, axes=axes)[1:x+1, 1:y+1, :].real
"""


use_scipy_transforms = True


def rfst2(A_g, axes=[0, 1]):
    all = {0, 1, 2}
    third = [all.difference(set(axes)).pop()]

    if use_scipy_transforms:
        Y = A_g
        for axis in axes:
            Y = scipydst(Y, axis=axis, type=1)
        Y *= 2**len(axes)
        return Y

    A_g = np.transpose(A_g, axes + third)
    x, y, z = A_g.shape
    temp_g = np.zeros((x * 2 + 2, y * 2 + 2, z))
    temp_g[1:x + 1, 1:y + 1, :] = A_g
    temp_g[x + 2:, 1:y + 1, :] = -A_g[::-1, :, :]
    temp_g[1:x + 1, y + 2:, :] = -A_g[:, ::-1, :]
    temp_g[x + 2:, y + 2:, :] = A_g[::-1, ::-1, :]
    X = -4 * rfft2(temp_g, axes=[0, 1])[1:x + 1, 1:y + 1, :].real
    return np.transpose(X, np.argsort(axes + third))


def irfst2(A_g, axes=[0, 1]):
    if use_scipy_transforms:
        Y = A_g
        for axis in axes:
            Y = scipydst(Y, axis=axis, type=1)
        magic = 1.0 / (16 * np.prod([A_g.shape[axis] + 1 for axis in axes]))
        Y *= magic
        # Y /= 211200
        return Y

    all = {0, 1, 2}
    third = [all.difference(set(axes)).pop()]
    A_g = np.transpose(A_g, axes + third)
    x, y, z = A_g.shape
    temp_g = np.zeros((x * 2 + 2, (y * 2 + 2) // 2 + 1, z))
    temp_g[1:x + 1, 1:y + 1, :] = A_g.real
    temp_g[x + 2:, 1:y + 1, :] = -A_g[::-1, :, :].real
    X = -0.25 * irfft2(temp_g, axes=[0, 1])[1:x + 1, 1:y + 1, :]

    T = np.transpose(X, np.argsort(axes + third))
    return T


# This method needs to be taken from fftw / scipy to gain speedup of ~4x
def fst(A_g, axis):
    x, y, z = A_g.shape
    N_c = np.array([x, y, z])
    N_c[axis] = N_c[axis] * 2 + 2
    temp_g = np.zeros(N_c, dtype=A_g.dtype)
    if axis == 0:
        temp_g[1:x + 1, :, :] = A_g
        temp_g[x + 2:, :, :] = -A_g[::-1, :, :]
    elif axis == 1:
        temp_g[:, 1:y + 1, :] = A_g
        temp_g[:, y + 2:, :] = -A_g[:, ::-1, :]
    elif axis == 2:
        temp_g[:, :, 1:z + 1] = A_g
        temp_g[:, :, z + 2:] = -A_g[:, ::, ::-1]
    else:
        raise NotImplementedError()
    X = 0.5j * fft(temp_g, axis=axis)
    if axis == 0:
        return X[1:x + 1, :, :]
    elif axis == 1:
        return X[:, 1:y + 1, :]
    elif axis == 2:
        return X[:, :, 1:z + 1]


def ifst(A_g, axis):
    x, y, z = A_g.shape
    N_c = np.array([x, y, z])
    N_c[axis] = N_c[axis] * 2 + 2
    temp_g = np.zeros(N_c, dtype=A_g.dtype)

    if axis == 0:
        temp_g[1:x + 1, :, :] = A_g
        temp_g[x + 2:, :, :] = -A_g[::-1, :, :]
    elif axis == 1:
        temp_g[:, 1:y + 1, :] = A_g
        temp_g[:, y + 2:, :] = -A_g[:, ::-1, :]
    elif axis == 2:
        temp_g[:, :, 1:z + 1] = A_g
        temp_g[:, :, z + 2:] = -A_g[:, ::, ::-1]
    else:
        raise NotImplementedError()

    X_g = ifft(temp_g, axis=axis)
    if axis == 0:
        return -2j * X_g[1:x + 1, :, :]
    elif axis == 1:
        return -2j * X_g[:, 1:y + 1, :]
    elif axis == 2:
        return -2j * X_g[:, :, 1:z + 1]


def transform(A_g, axis=None, pbc=True):
    if pbc:
        if A_g.size == 0:
            return A_g.astype(complex)

        return fft(A_g, axis=axis)
    else:
        if A_g.size == 0:
            return A_g

        if not use_scipy_transforms:
            x = fst(A_g, axis)
            return x
        y = scipydst(A_g, axis=axis, type=1)
        y *= .5
        return y


def transform2(A_g, axes=None, pbc=[True, True]):
    if all(pbc):
        if A_g.size == 0:
            return A_g.astype(complex)

        return fft2(A_g, axes=axes)
    elif not any(pbc):
        if A_g.size == 0:
            return A_g

        return rfst2(A_g, axes=axes)
    else:
        return transform(transform(A_g, axis=axes[0], pbc=pbc[0]),
                         axis=axes[1], pbc=pbc[1])


def itransform(A_g, axis=None, pbc=True):
    if pbc:
        if A_g.size == 0:
            return A_g.astype(complex)

        return ifft(A_g, axis=axis)
    else:
        if A_g.size == 0:
            return A_g

        if not use_scipy_transforms:
            x = ifst(A_g, axis)
            return x
        y = scipydst(A_g, axis=axis, type=1)
        magic = 1.0 / (A_g.shape[axis] + 1)
        y *= magic
        return y


def itransform2(A_g, axes=None, pbc=[True, True]):
    if all(pbc):
        if A_g.size == 0:
            return A_g.astype(complex)

        return ifft2(A_g, axes=axes)
    elif not any(pbc):
        if A_g.size == 0:
            return A_g

        return irfst2(A_g, axes=axes)
    else:
        return itransform(itransform(A_g, axis=axes[0], pbc=pbc[0]),
                          axis=axes[1], pbc=pbc[1])


class BadAxesError(ValueError):
    pass


class FastPoissonSolver(BasePoissonSolver):
    def __init__(self, nn=3, **kwargs):
        BasePoissonSolver.__init__(self, **kwargs)
        self.nn = nn
        # We may later enable this to work with Cholesky, but not now:
        self.use_cholesky = False

    def _init(self):
        pass

    def set_grid_descriptor(self, gd):
        self.gd = gd
        axes = np.arange(3)
        pbc_c = np.array(gd.pbc_c, dtype=bool)
        periodic_axes = axes[pbc_c]
        non_periodic_axes = axes[np.logical_not(pbc_c)]

        # Find out which axes are orthogonal (0, 1 or 3)
        # Note that one expects that the axes are always rotated in
        # conventional form, thus for all axes to be
        # classified as orthogonal, the cell_cv needs to be diagonal.
        # This may always be achieved by rotating
        # the unit-cell along with the atoms. The classification is
        # inherited from grid_descriptor.orthogonal.
        dotprods = np.dot(gd.cell_cv, gd.cell_cv.T)
        # For each direction, check whether there is only one nonzero
        # element in that row (necessarily being the diagonal element,
        # since this is a cell vector length and must be > 0).
        orthogonal_c = (np.abs(dotprods) > 1e-10).sum(axis=0) == 1
        assert sum(orthogonal_c) in [0, 1, 3]

        non_orthogonal_axes = axes[np.logical_not(orthogonal_c)]

        if not all(pbc_c | orthogonal_c):
            raise BadAxesError('Each axis must be periodic or orthogonal '
                               'to other axes.  But we have pbc={} '
                               'and orthogonal={}'
                               .format(pbc_c.astype(int),
                                       orthogonal_c.astype(int)))

        # We sort them, and pick the longest non-periodic axes as the
        # cholesky axis.
        sorted_non_periodic_axes = sorted(non_periodic_axes,
                                          key=lambda c: gd.N_c[c])
        if self.use_cholesky:
            if len(sorted_non_periodic_axes) > 0:
                cholesky_axes = [sorted_non_periodic_axes[-1]]
                if cholesky_axes[0] in non_orthogonal_axes:
                    msg = ('Cholesky axis cannot be non-orthogonal. '
                           'Do you really want a non-orthogonal non-periodic '
                           'axis? If so, run with use_cholesky=False.')
                    raise NotImplementedError(msg)
                fst_axes = sorted_non_periodic_axes[0:-1]
            else:
                cholesky_axes = []
                fst_axes = []
        else:
            cholesky_axes = []
            fst_axes = sorted_non_periodic_axes
        fft_axes = list(periodic_axes)

        (self.cholesky_axes, self.fst_axes,
         self.fft_axes) = cholesky_axes, fst_axes, fft_axes

        fftfst_axes = self.fft_axes + self.fst_axes
        axes = self.fft_axes + self.fst_axes + self.cholesky_axes
        self.axes = axes

        # Create xy flat decomposition (where x=axes[0] and y=axes[1])
        parsize_c = [1, 1, 1]
        parsize_c[axes[2]] = gd.comm.size
        gd1d = gd.new_descriptor(parsize_c=parsize_c,
                                 allow_empty_domains=True)
        self.gd1d = gd1d

        # Create z flat decomposition
        domain = gd.N_c.copy()
        domain[axes[2]] = 1
        parsize_c = decompose_domain(domain, gd.comm.size)
        gd2d = gd.new_descriptor(parsize_c=parsize_c)
        self.gd2d = gd2d

        # Calculate eigenvalues in fst/fft decomposition for
        # non-cholesky axes in parallel
        xp = self.xp
        r_cx = xp.indices(gd2d.n_c)
        r_cx += xp.asarray(gd2d.beg_c[:, xp.newaxis, xp.newaxis, xp.newaxis])
        r_cx = r_cx.astype(complex)
        for c, axis in enumerate(fftfst_axes):
            r_cx[axis] *= 2j * xp.pi / gd2d.N_c[axis]
            if axis in fst_axes:
                r_cx[axis] /= 2
        for c, axis in enumerate(cholesky_axes):
            r_cx[axis] = 0.0
        xp.exp(r_cx, out=r_cx)
        fft_lambdas = xp.zeros_like(r_cx[0], dtype=complex)
        laplace = Laplace(self.gd, -0.25 / pi, self.nn)
        self.stencil_description = laplace.description

        for coeff, offset_c in zip(laplace.coef_p, laplace.offset_pc):
            offset_c = np.array(offset_c)
            if not any(offset_c):
                # The centerpoint is handled with (temp-1.0)
                continue
            non_zero_axes, = np.where(offset_c)
            if set(non_zero_axes).issubset(fftfst_axes):
                temp = xp.ones_like(fft_lambdas)
                for c, axis in enumerate(fftfst_axes):
                    temp *= r_cx[axis] ** offset_c[axis]
                fft_lambdas += coeff * (temp - 1.0)

        assert xp.linalg.norm(fft_lambdas.imag) < 1e-10
        fft_lambdas = fft_lambdas.real.copy()  # arr.real is not contiguous

        # If there is no Cholesky decomposition, the system is already
        # fully diagonal and we can directly invert the linear problem
        # by dividing with the eigenvalues.
        # TODO: Remove cholesky alltogether, since it is not used
        assert len(cholesky_axes) == 0
        # if len(cholesky_axes) == 0:
        with np.errstate(divide='ignore'):
            self.inv_fft_lambdas = xp.where(
                xp.abs(fft_lambdas) > 1e-10, 1.0 / fft_lambdas, 0)

    def solve_neutral(self, phi_g, rho_g, timer=None):
        if len(self.cholesky_axes) != 0:
            raise NotImplementedError

        gd = self.gd
        gd1d = self.gd1d
        gd2d = self.gd2d
        comm = self.gd.comm
        axes = self.axes

        with timer('Communicate to 1D'):
            work1d_g = gd1d.empty(dtype=rho_g.dtype, xp=self.xp)
            grid2grid(comm, gd, gd1d, rho_g, work1d_g, xp=self.xp)
        with timer('FFT 2D'):
            work1d_g = transform2(work1d_g, axes=axes[:2],
                                  pbc=gd.pbc_c[axes[:2]])
        with timer('Communicate to 2D'):
            work2d_g = gd2d.empty(dtype=work1d_g.dtype, xp=self.xp)
            grid2grid(comm, gd1d, gd2d, work1d_g, work2d_g, xp=self.xp)
        with timer('FFT 1D'):
            work2d_g = transform(work2d_g, axis=axes[2],
                                 pbc=gd.pbc_c[axes[2]])

        # The remaining problem is 0D dimensional, i.e the problem
        # has been fully diagonalized
        work2d_g *= self.inv_fft_lambdas

        with timer('FFT 1D'):
            work2d_g = itransform(work2d_g, axis=axes[2],
                                  pbc=gd.pbc_c[axes[2]])
        with timer('Communicate from 2D'):
            work1d_g = gd1d.empty(dtype=work2d_g.dtype, xp=self.xp)
            grid2grid(comm, gd2d, gd1d, work2d_g, work1d_g, xp=self.xp)
        with timer('FFT 2D'):
            work1d_g = itransform2(work1d_g, axes=axes[1::-1],
                                   pbc=gd.pbc_c[axes[1::-1]])
        with timer('Communicate from 1D'):
            work_g = gd.empty(dtype=work1d_g.dtype, xp=self.xp)
            grid2grid(comm, gd1d, gd, work1d_g, work_g, xp=self.xp)

        phi_g[:] = work_g.real
        return 1  # Non-iterative method, return 1 iteration

    def todict(self):
        d = super().todict()
        d.update({'name': 'fast', 'nn': self.nn})
        return d

    def estimate_memory(self, mem):
        pass

    def get_description(self):
        lines = [f'{self.__class__.__name__} using',
                 f'    Stencil: {self.stencil_description}',
                 f'    FFT axes: {self.fft_axes}',
                 f'    FST axes: {self.fst_axes}',
                 ]
        lines.append(BasePoissonSolver.get_description(self))
        return '\n'.join(lines)
