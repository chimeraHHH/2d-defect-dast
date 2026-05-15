from collections import deque
from inspect import signature

import numpy as np
from ase.calculators.calculator import InputError
from ase.units import Bohr, Ha

from gpaw.mpi import broadcast_float


def get_criterion(name):
    """Returns one of the pre-specified criteria by it's .name attribute,
    and raises sensible error if missing."""
    # All built-in criteria should be in this list.
    criteria = [Energy, Density, Eigenstates, Eigenvalues, Forces,
                WorkFunction, MinIter, MaxIter]
    criteria = {c.name: c for c in criteria}
    try:
        return criteria[name]
    except KeyError:
        known = ', '.join(f'{key!r}' for key in criteria)
        msg = (
            f'The convergence keyword "{name}" was supplied, which we do not '
            'know how to handle. If this is a typo, please correct '
            f'(known keywords are {known}). If this'
            ' is a user-written convergence criterion, it cannot be '
            'imported with this function; please see the GPAW manual for '
            'details.')
        raise InputError(msg)


def dict2criterion(dictionary):
    """Converts a dictionary to a convergence criterion.

    The dictionary can either be that generated from 'todict'; that is like
    {'name': 'energy', 'tol': 0.005, 'n_old': 3}. Or from user-specified
    shortcut like {'energy': 0.005} or {'energy': (0.005, 3)}, or a
    combination like {'energy': {'name': 'energy', 'tol': 0.005, 'n_old': 3}.
    """
    d = dictionary.copy()
    if 'name' in d:  # from 'todict'
        name = d.pop('name')
        ThisCriterion = get_criterion(name)
        return ThisCriterion(**d)
    assert len(d) == 1
    name = list(d.keys())[0]
    if isinstance(d[name], dict) and 'name' in d[name]:
        return dict2criterion(d[name])
    ThisCriterion = get_criterion(name)
    return ThisCriterion(*[d[name]])


def check_convergence(criteria, ctx):
    entries = {}  # for log file, per criteria
    converged_items = {}  # True/False, per criteria
    override_others = False
    converged = True
    for name, criterion in criteria.items():
        if not criterion.calc_last:
            ok, entry = criterion(ctx)
            if criterion.override_others:
                if ok:
                    override_others = True
            else:
                converged = converged and ok
            converged_items[name] = ok
            entries[name] = entry

    for name, criterion in criteria.items():
        if criterion.calc_last:
            if converged:
                ok, entry = criterion(ctx)
                converged &= ok
                converged_items[name] = ok
                entries[name] = entry
            else:
                converged_items[name] = False
                entries[name] = ''

    # Converged?
    return converged or override_others, converged_items, entries


class Criterion:
    """Base class for convergence criteria.

    Automates the creation of the __repr__ and todict methods for generic
    classes. This will work for classes that save all arguments directly,
    like __init__(self, a, b):  --> self.a = a, self.b = b. The todict
    method requires the class have a self.name attribute. All criteria
    (subclasses of Criterion) must define self.name, self.tablename,
    self.description, self.__init__, and self.__call___. See the online
    documentation for details.
    """
    # If calc_last is True, will only be checked after all other (non-last)
    # criteria have been met.
    calc_last = False
    override_others = False
    description: str

    def __repr__(self):
        parameters = signature(self.__class__).parameters
        s = ', '.join([str(getattr(self, p)) for p in parameters])
        return self.__class__.__name__ + '(' + s + ')'

    def todict(self):
        d = {'name': self.name}
        parameters = signature(self.__class__).parameters
        for parameter in parameters:
            d[parameter] = getattr(self, parameter)
        return d

    def reset(self):
        pass


# Built-in criteria follow. Make sure that any new criteria added below
# are also added to to the list in get_criterion() so that it can import
# them correctly by name.


class Energy(Criterion):
    """A convergence criterion for the total energy.

    Parameters:

    tol: float
        Tolerance for conversion; that is the maximum variation among the
        last n_old values of the (extrapolated) total energy.
    n_old: int
        Number of energy values to compare. I.e., if n_old is 3, then this
        compares the peak-to-peak difference among the current total energy
        and the two previous.
    relative: bool
        Use total energy [eV] or total energy relative to number of
        valence electrons [eV/(valence electron)].
    """
    name = 'energy'
    tablename = 'energy'

    def __init__(self, tol: float, *, n_old: int = 3, relative: bool = True):
        self.tol = tol
        self.n_old = n_old
        self.relative = relative
        self.description = (
            f'Maximum [total energy] change in last {self.n_old} cyles: '
            f'{self.tol:g} eV')
        if relative:
            self.description += ' / valence electron'

    def reset(self):
        self._old = deque(maxlen=self.n_old)

    def __call__(self, context):
        """Should return (bool, entry), where bool is True if converged and
        False if not, and entry is a <=5 character string to be printed in
        the user log file."""
        # Note the previous code was calculating the peak-to-
        # peak energy difference on e_total_free, while reporting
        # e_total_extrapolated in the SCF table (logfile). I changed it to
        # use e_total_extrapolated for both. (Should be a miniscule
        # difference, but more consistent.)
        total_energy = context.ham.e_total_extrapolated * Ha
        if context.wfs.nvalence == 0 or not self.relative:
            energy = total_energy
        else:
            energy = total_energy / context.wfs.nvalence
        self._old.append(energy)  # Pops off >3!
        error = np.inf
        if len(self._old) == self._old.maxlen:
            error = np.ptp(self._old)
        converged = error < self.tol
        entry = ''
        if np.isfinite(energy):
            entry = f'{total_energy:11.6f}'
        return converged, entry


class Density(Criterion):
    """A convergence criterion for the electron density.

    Parameters:

    tol : float
        Tolerance for conversion; that is the maximum change in the electron
        density, calculated as the integrated absolute value of the density
        change, normalized per valence electron. [electrons/(valence electron)]
    """
    name = 'density'
    tablename = 'dens'

    def __init__(self, tol):
        self.tol = tol
        self.description = ('Maximum integral of absolute [dens]ity change: '
                            '{:g} electrons / valence electron'
                            .format(self.tol))

    def __call__(self, context):
        """Should return (bool, entry), where bool is True if converged and
        False if not, and entry is a <=5 character string to be printed in
        the user log file."""
        if context.dens.fixed:
            # Old GPAW needs this
            return True, ''
        nv = context.wfs.nvalence
        if nv == 0:
            return True, ''
        # Make sure all agree on the density error.
        error = broadcast_float(context.dens.error, context.wfs.world) / nv
        converged = (error < self.tol)
        if (error is None or np.isinf(error) or error == 0):
            entry = ''
        else:
            entry = f'{np.log10(error):+5.2f}'
        return converged, entry


class Eigenstates(Criterion):
    """A convergence criterion for the eigenstates.

    Parameters:

    tol : float
        Tolerance for conversion; that is the maximum change in the
        eigenstates, calculated as the integration of the square of the
        residuals of the Kohn--Sham equations, normalized per valence
        electron. [eV^2/(valence electron)]
    """
    name = 'eigenstates'
    tablename = 'eigst'

    def __init__(self, tol):
        self.tol = tol
        self.description = ('Maximum integral of absolute [eigenst]ate '
                            'change: {:g} eV^2 / valence electron'
                            .format(self.tol))

    def __call__(self, context):
        """Should return (bool, entry), where bool is True if converged and
        False if not, and entry is a <=5 character string to be printed in
        the user log file."""
        if context.wfs.nvalence == 0:
            return True, ''
        error = self.get_error(context)
        converged = (error < self.tol)
        if (context.wfs.nvalence == 0 or error == 0 or np.isinf(error)):
            entry = ''
        else:
            entry = f'{np.log10(error):+6.2f}'
        return converged, entry

    def get_error(self, context):
        """Returns the raw error."""
        return context.wfs.eigensolver.error * Ha**2 / context.wfs.nvalence


class Eigenvalues(Criterion):
    name = 'eigenvalues'
    tablename = 'eigs'
    calc_last = False

    def __init__(self, tol=1e-3):
        self.tol = tol
        self.description = 'Maximum absolute change in eigenvalues [eV].'

    def __call__(self, context):
        if context.wfs.nvalence == 0:
            return True, ''
        error = self.get_error(context)
        converged = (error < self.tol)
        if (context.wfs.nvalence == 0 or error == 0 or np.isinf(error)):
            entry = ''
        else:
            entry = f'{np.log10(error):+6.2f}'
        return converged, entry

    def get_error(self, context):
        return context.eig_error * Ha


class Forces(Criterion):
    """A convergence criterion for the forces.

    Parameters:

    atol : float
        Absolute tolerance for convergence; that is, the force on each atom
        is compared with its force from the previous iteration, and the change
        in each atom's force is calculated as an l2-norm
        (Euclidean distance). The atom with the largest norm must be less
        than tol. [eV/Angstrom]
    rtol : float
        Relative tolerance for convergence. The difference in the l2-norm of
        force on each atom is calculated, and convergence is achieved when
        the largest difference between two iterations is rtol * max force.
    calc_last : bool
        If True, calculates forces last; that is, it waits until all other
        convergence criteria are satisfied before checking to see if the
        forces have converged. (This is more computationally efficient.)
        If False, checks forces at each SCF step.
    """
    name = 'forces'
    tablename = 'force'

    def __init__(self, atol, rtol=np.inf, calc_last=True):
        self.atol = atol
        self.rtol = rtol
        self.description = ('Maximum change in the atomic [forces] across '
                            f'last 2 cycles: {self.atol} eV/Ang OR\n'
                            'Maximum error relative to the maximum '
                            f'force is below {self.rtol}')
        self.calc_last = calc_last
        self.reset()

    def __call__(self, context):
        """Should return (bool, entry), where bool is True if converged and
        False if not, and entry is a <=5 character string to be printed in
        the user log file."""

        # criterion is off; backwards compatibility
        if np.isinf(self.atol) and np.isinf(self.rtol):
            return True, ''
        F_av = context.calculate_forces() * (Ha / Bohr)
        error = np.inf
        max_force = np.max(np.linalg.norm(F_av, axis=1))
        if self.old_F_av is not None:
            error = ((F_av - self.old_F_av)**2).sum(1).max()**0.5
        self.old_F_av = F_av

        if np.isfinite(self.rtol):
            error_threshold = min(self.atol, self.rtol * max_force)
        else:
            # Avoid possible inf * 0.0:
            error_threshold = self.atol
        converged = error < error_threshold

        entry = ''
        if np.isfinite(error):
            if error:
                entry = f'{np.log10(error):+5.2f}'
            else:
                entry = '-inf'
        return converged, entry

    def reset(self):
        self.old_F_av = None


class WorkFunction(Criterion):
    """A convergence criterion for the work function.

    Parameters:

    tol : float
        Tolerance for conversion; that is the maximum variation among the
        last n_old values of either work function. [eV]
    n_old : int
        Number of work functions to compare. I.e., if n_old is 3, then this
        compares the peak-to-peak difference among the current work
        function and the two previous.
    """
    name = 'work function'
    tablename = 'wkfxn'

    def __init__(self, tol=0.005, n_old=3):
        self.tol = tol
        self.n_old = n_old
        self.description = ('Maximum change in the last {:d} '
                            'work functions [wkfxn]: {:g} eV'
                            .format(n_old, tol))

    def reset(self):
        self._old = deque(maxlen=self.n_old)

    def __call__(self, context):
        """Should return (bool, entry), where bool is True if converged and
        False if not, and entry is a <=5 character string to be printed in
        the user log file."""
        workfunctions = context.ham.get_workfunctions(context.wfs)
        workfunctions = Ha * np.array(workfunctions)
        self._old.append(workfunctions)  # Pops off >3!
        if len(self._old) == self._old.maxlen:
            error = max(np.ptp(self._old, axis=0))
        else:
            error = np.inf
        converged = (error < self.tol)
        if error < np.inf:
            entry = f'{np.log10(error):+5.2f}'
        else:
            entry = ''
        return converged, entry


class MinIter(Criterion):
    """A convergence criterion that enforces a minimum number of iterations.

    Parameters:

    n : int
        Minimum number of iterations that must be complete before
        the SCF cycle exits.
    """
    calc_last = False
    name = 'minimum iterations'
    tablename = 'minit'

    def __init__(self, n):
        self.n = n
        self.description = f'Minimum number of iterations [minit]: {n}'

    def __call__(self, context):
        converged = context.niter >= self.n
        entry = f'{context.niter:d}'
        return converged, entry


class MaxIter(Criterion):
    """A convergence criterion that enforces a maximum number of iterations.

    Parameters:

    n : int
        Maximum number of iterations that must be complete before
        the SCF cycle exits.
    """
    calc_last = False
    name = 'maximum iterations'
    tablename = 'maxit'
    override_others = True

    def __init__(self, n):
        self.n = n
        self.description = f'Maximum number of iterations [minit]: {n}'

    def __call__(self, context):
        converged = context.niter >= self.n
        entry = f'{context.niter:d}'
        return converged, entry
