from .base import BaseSolver
from .bicgstab import BiCGStab
from .cscg import CSCG

__all__ = ['create_solver', 'BiCGStab', 'CSCG']


def create_solver(name, **kwargs):
    if isinstance(name, BaseSolver):
        return name
    elif isinstance(name, dict):
        kwargs.update(name)
        return create_solver(**kwargs)
    elif name == 'CSCG':
        return CSCG(**kwargs)
    elif name == 'BiCGStab':
        return BiCGStab(**kwargs)
    else:
        raise ValueError('Unknown solver: %s' % name)
