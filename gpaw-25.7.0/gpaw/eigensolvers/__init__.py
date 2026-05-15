"""Module defining  ``Eigensolver`` classes."""

from gpaw.eigensolvers.rmmdiis import RMMDIIS
from gpaw.eigensolvers.cg import CG
from gpaw.eigensolvers.davidson import Davidson
from gpaw.eigensolvers.direct import DirectPW
from gpaw.lcao.eigensolver import DirectLCAO
from gpaw.directmin.etdm_fdpw import FDPWETDM
from gpaw.directmin.etdm_lcao import LCAOETDM


def get_eigensolver(eigensolver, mode, convergence=None):
    """Create eigensolver object."""
    if eigensolver is None:
        if mode.name == 'lcao':
            eigensolver = 'lcao'
        else:
            eigensolver = 'davidson'

    if isinstance(eigensolver, str):
        eigensolver = {'name': eigensolver}

    if isinstance(eigensolver, dict):
        eigensolver = eigensolver.copy()
        name = eigensolver.pop('name')
        if name == 'etdm':
            # Compatibility with old versions
            name = 'etdm-lcao'
        eigensolver = {'rmm-diis': RMMDIIS,
                       'cg': CG,
                       'dav': Davidson,
                       'davidson': Davidson,
                       'lcao': DirectLCAO,
                       'direct': DirectPW,
                       'etdm-lcao': LCAOETDM,
                       'etdm-fdpw': FDPWETDM}[name](**eigensolver)

    if isinstance(eigensolver, CG):
        eigensolver.tolerance = convergence.get('eigenstates', 4.0e-8)

    assert isinstance(eigensolver, DirectLCAO) == (mode.name == 'lcao') or \
           isinstance(eigensolver, LCAOETDM) == (mode.name == 'lcao')

    return eigensolver
