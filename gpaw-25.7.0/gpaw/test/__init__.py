from functools import wraps
from typing import Tuple

import gpaw.mpi as mpi
import numpy as np
from gpaw.atom.configurations import parameters, tf_parameters
from gpaw.atom.generator import Generator
from gpaw.typing import Array1D


def print_reference(data_i, name='ref_i', fmt='%.12le'):
    if mpi.world.rank == 0:
        print('%s = [' % name, end='')
        for i, val in enumerate(data_i):
            if i > 0:
                print('', end='\n')
                print(' ' * (len(name) + 4), end='')
            print(fmt % val, end='')
            print(',', end='')
        print('\b]')


def findpeak(x: Array1D, y: Array1D) -> Tuple[float, float]:
    """Find peak.

    >>> x = np.linspace(1, 5, 10)
    >>> y = 1 - (x - np.pi)**2
    >>> x0, y0 = findpeak(x, y)
    >>> f'x0={x0:.6f}, y0={y0:.6f}'
    'x0=3.141593, y0=1.000000'
    """
    i = y.argmax()
    a, b, c = np.polyfit(x[i - 1:i + 2] - x[i], y[i - 1:i + 2], 2)
    assert a < 0
    dx = -0.5 * b / a
    x0 = x[i] + dx
    return x0, a * dx**2 + b * dx + c


def gen(symbol, exx=False, name=None, yukawa_gamma=None,
        write_xml=False, **kwargs):
    setup = None
    if mpi.rank == 0:
        if 'scalarrel' not in kwargs:
            kwargs['scalarrel'] = True
        g = Generator(symbol, **kwargs)
        if 'orbital_free' in kwargs:
            setup = g.run(exx=exx, name=name, yukawa_gamma=yukawa_gamma,
                          write_xml=write_xml,
                          **tf_parameters.get(symbol, {'rcut': 0.9}))
        else:
            setup = g.run(exx=exx, name=name, yukawa_gamma=yukawa_gamma,
                          write_xml=write_xml,
                          **parameters[symbol])
    setup = mpi.broadcast(setup, 0)
    return setup


def only_on_master(comm, broadcast=None):
    """Decorator for executing the function only on the rank 0.

    Parameters
    ----------
    comm
        communicator
    broadcast
        function for broadcasting the return value or
        `None` for no broadcasting
    """
    def wrap(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if comm.rank == 0:
                ret = func(*args, **kwargs)
            else:
                ret = None
            comm.barrier()
            if broadcast is not None:
                ret = broadcast(ret, comm=comm)
            return ret
        return wrapped_func
    return wrap


def calculate_numerical_forces(atoms, eps=1e-6, iatoms=None, icarts=None):
    try:
        from ase.calculators.fd import calculate_numerical_forces as cnf
    except ImportError:
        pass
    else:
        return cnf(atoms, eps, iatoms, icarts)
    from ase.calculators.test import numeric_force
    if iatoms is None:
        iatoms = range(len(atoms))
    if icarts is None:
        icarts = [0, 1, 2]
    return np.array(
        [[numeric_force(atoms, a, c, eps) for c in icarts] for a in iatoms])
