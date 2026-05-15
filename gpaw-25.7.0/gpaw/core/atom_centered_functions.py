from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from gpaw.core.atom_arrays import (AtomArrays, AtomArraysLayout,
                                   AtomDistribution)
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.mpi import MPIComm, serial_comm
from gpaw.new import zips, trace
from gpaw.spline import Spline
from gpaw.typing import Array1D, ArrayLike2D
from gpaw.gpu import XP

if TYPE_CHECKING:
    from gpaw.core.uniform_grid import UGArray


def to_spline(l: int,
              rcut: float,
              f: Callable[[Array1D], Array1D]) -> Spline:
    """Convert to GPAW's Spline object."""
    r = np.linspace(0, rcut, 100)
    return Spline.from_data(l, rcut, f(r))


class AtomCenteredFunctions(XP):
    def __init__(self,
                 functions,
                 relpos_ac: ArrayLike2D,
                 atomdist: AtomDistribution | None = None,
                 xp=None):
        XP.__init__(self, xp or np)
        self.functions = [[to_spline(*f) if isinstance(f, tuple) else f
                           for f in funcs]
                          for funcs in functions]
        self.relpos_ac = np.array(relpos_ac)
        self._atomdist = atomdist

        self._layout = None
        self._lfc = None

    def __repr__(self):
        funcs = [['spdfgh'[f.l] for f in ff] for ff in self.functions[:4]]
        if len(self.functions) > 4:
            funcs.append(...)
        return (f'{self.__class__.__name__}'
                f'(functions={funcs}, atomdist={self.atomdist})')

    def new(self, desc, atomdist):
        raise NotImplementedError

    @property
    def layout(self):
        self._lazy_init()
        return self._layout

    @property
    def atomdist(self):
        self._lazy_init()
        return self._atomdist

    def _lazy_init(self):
        raise NotImplementedError

    def empty(self,
              dims: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm) -> AtomArrays:
        """Create AtomsArray for coefficients."""
        return self.layout.empty(dims, comm)

    def move(self,
             relpos_ac: np.ndarray,
             atomdist: AtomDistribution) -> AtomDistribution:
        """Move atoms to new positions."""
        self.relpos_ac = np.array(relpos_ac)
        self._atomdist = atomdist
        if self._lfc is not None:
            self._layout = self._layout.new(atomdist=atomdist)
            migration = self._lfc.set_positions(relpos_ac, atomdist)
            if migration:
                atomdist = AtomDistribution(
                    [sphere.rank for sphere in self._lfc.sphere_a],
                    atomdist.comm)
        return atomdist

    def add_to(self, functions, coefs=1.0):
        """Add atom-centered functions multiplied by *coefs* to *functions*."""
        self._lazy_init()
        if isinstance(coefs, float):
            self._lfc.add(functions.data, coefs)
        else:
            self._lfc.add(functions.data, coefs, q=0)

    @trace
    def integrate(self, functions, out=None, add_to=False):
        """Calculate integrals of atom-centered functions multiplied by
        *functions*.
        """
        self._lazy_init()
        if out is None:
            assert not add_to
            out = self.layout.empty(functions.dims, functions.comm)
        self._lfc.integrate(functions.data, out, q=0, add_to=add_to)
        return out

    def derivative(self, functions, out=None):
        """Calculate derivatives of integrals with respect to atom
        positions.
        """
        self._lazy_init()
        if out is None:
            out = self.layout.empty(functions.dims + (3,), functions.comm)
        coef_axiv = {a: self.xp.moveaxis(array_xvi, -2, -1)
                     for a, array_xvi in out._arrays.items()}
        self._lfc.derivative(functions.data, coef_axiv, q=0)
        return out

    def stress_contribution(self, a, c=1.0):
        self._lazy_init()
        return self._lfc.stress_tensor_contribution(a.data, c)


class UGAtomCenteredFunctions(AtomCenteredFunctions):
    def __init__(self,
                 functions,
                 relpos_ac,
                 grid,
                 *,
                 atomdist=None,
                 integrals=None,
                 cut=False,
                 xp=np):
        AtomCenteredFunctions.__init__(self,
                                       functions,
                                       relpos_ac,
                                       atomdist, xp=xp)
        self.grid = grid
        self.integrals = integrals
        self.cut = cut

    def new(self, grid, atomdist):
        return UGAtomCenteredFunctions(
            self.functions,
            self.relpos_ac,
            grid,
            atomdist=atomdist,
            integrals=self.integrals,
            cut=self.cut,
            xp=self.xp)

    def _lazy_init(self):
        if self._lfc is not None:
            return
        gd = self.grid._gd
        kd = KPointDescriptor(np.array([self.grid.kpt]))
        self._lfc = LFC(gd, self.functions, kd,
                        dtype=self.grid.dtype,
                        integral=self.integrals,
                        forces=True,
                        cut=self.cut,
                        xp=self.xp)
        self._lfc.set_positions(self.relpos_ac)

        if self._atomdist is None:
            self._atomdist = AtomDistribution(
                ranks=np.array([sphere.rank for sphere in self._lfc.sphere_a]),
                comm=self.grid.comm)
        else:
            for sphere, rank in zips(self._lfc.sphere_a,
                                     self._atomdist.rank_a):
                assert sphere.rank == rank
            assert self.grid.comm is self._atomdist.comm

        self._layout = AtomArraysLayout([sum(2 * f.l + 1 for f in funcs)
                                         for funcs in self.functions],
                                        self._atomdist,
                                        self.grid.dtype, xp=self.xp)

    def to_uniform_grid(self,
                        out: UGArray,
                        scale: float = 1.0) -> UGArray:
        out.data[:] = 0.0
        self.add_to(out, scale)
        return out
