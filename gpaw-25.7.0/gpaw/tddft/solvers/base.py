from abc import ABC, abstractmethod


class BaseSolver(ABC):
    """Abstract base class for solvers.

    Implementations of this class solves a set of linear equations A.x = b.

    Parameters
    ----------
    tolerance: float
        tolerance for the norm of the residual ||b - A.x||^2
    max_iterations: integer
        maximum number of iterations
    eps: float
        if abs(rho) or (omega) < eps, it's regarded as zero
        and the method breaks down

    Note
    ----
    Tolerance should not be smaller than attainable accuracy, which is
    order of kappa(A) * eps, where kappa(A) is the (spectral) condition
    number of the matrix. The maximum number of iterations should be
    significantly less than matrix size, approximately
    .5 sqrt(kappa) ln(2/tolerance). A small number is treated as zero
    if it's magnitude is smaller than argument eps.
    """
    def __init__(self,
                 tolerance=1e-8,
                 max_iterations=1000,
                 eps=1e-15):
        self.tol = tolerance
        self.max_iter = max_iterations
        if (eps <= tolerance):
            self.eps = eps
        else:
            raise ValueError(
                "Invalid tolerance (tol = %le < eps = %le)."
                % (tolerance, eps))

        self.iterations = -1

    def todict(self):
        return {'name': self.__class__.__name__,
                'tolerance': self.tol,
                'max_iterations': self.max_iter,
                'eps': self.eps}

    def initialize(self, gd, timer):
        """Initialize propagator using runtime objects.

        Parameters
        ----------
        gd: GridDescriptor
            grid descriptor for coarse (pseudowavefunction) grid
        timer: Timer
            timer
        """
        self.gd = gd
        self.timer = timer

    @abstractmethod
    def solve(self, A, x, b):
        """Solve a set of linear equations A.x = b.

        Parameters:
        A           matrix A
        x           initial guess x_0 (on entry) and the result (on exit)
        b           right-hand side (multi)vector

        """
        raise NotImplementedError()
