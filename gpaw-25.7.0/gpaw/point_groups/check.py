"""Symmetry checking code."""
import sys
from typing import Any, Dict, List, Sequence, Union

import numpy as np
from ase import Atoms
from numpy.linalg import det, inv, solve
from scipy.ndimage import map_coordinates

from gpaw.typing import Array1D, Array2D, Array3D, ArrayLike

from . import PointGroup

Axis = Union[str, Sequence[float], Array1D, None]


class SymmetryChecker:
    def __init__(self,
                 group: Union[str, PointGroup],
                 center: ArrayLike,
                 radius: float = 2.0,
                 x: Axis = None,
                 y: Axis = None,
                 z: Axis = None,
                 grid_spacing: float = 0.2):
        """Check point-group symmetries.

        If a non-standard orientation is desired then two of
        *x*, *y*, *z* can be specified.
        """
        if isinstance(group, str):
            group = PointGroup(group)
        self.group = group
        self.normalized_table = group.get_normalized_table()
        self.points = sphere(radius, grid_spacing)
        self.center = center
        self.grid_spacing = grid_spacing
        self.rotation = rotation_matrix([x, y, z])

    def check_atoms(self, atoms: Atoms, tol: float = 1e-5) -> bool:
        """Check if atoms have all the symmetries.

        Unit of *tol* is Angstrom.
        """
        numbers = atoms.numbers
        positions = (atoms.positions - self.center).dot(self.rotation.T)
        icell = np.linalg.inv(atoms.cell.dot(self.rotation.T))
        for opname, op in self.group.operations.items():
            P = positions.dot(op.T)
            for i, pos in enumerate(P):
                sdiff = (pos - positions).dot(icell)
                sdiff -= sdiff.round() * atoms.pbc
                dist2 = (sdiff.dot(atoms.cell)**2).sum(1)
                j = dist2.argmin()
                if dist2[j] > tol**2 or numbers[j] != numbers[i]:
                    return False
        return True

    def check_function(self,
                       function: Array3D,
                       grid_vectors: Array2D = None) -> Dict[str, Any]:
        """Check function on uniform grid."""
        if grid_vectors is None:
            grid_vectors = np.eye(3)
        dv = abs(det(grid_vectors))
        norm1 = (function**2).sum() * dv
        M = inv(grid_vectors).T
        overlaps: List[float] = []
        for op in self.group.operations.values():
            op = self.rotation.T @ op @ self.rotation
            pts = (self.points @ op.T + self.center) @ M.T
            pts %= function.shape
            values = map_coordinates(function, pts.T, mode='wrap')
            if not overlaps:
                values1 = values
            overlaps.append(values.dot(values1) * self.grid_spacing**3)

        reduced_overlaps = []
        i1 = 0
        for n in self.group.nops:
            i2 = i1 + n
            reduced_overlaps.append(sum(overlaps[i1:i2]) / n / overlaps[0])
            i1 = i2

        characters = solve(self.normalized_table.T, reduced_overlaps)
        best = self.group.symmetries[characters.argmax()]

        return {'symmetry': best,
                'norm': norm1,
                'overlaps': overlaps,
                'characters': {symmetry: value
                               for symmetry, value
                               in zip(self.group.symmetries, characters)}}

    def check_band(self,
                   calc,
                   band: int,
                   spin: int = 0) -> Dict[str, Any]:
        """Check wave function from GPAW calculation."""
        wfs = calc.get_pseudo_wave_function(band, spin=spin)
        grid_vectors = (calc.atoms.cell.T / wfs.shape).T
        return self.check_function(wfs, grid_vectors)

    def check_calculation(self,
                          calc,
                          n1: int,
                          n2: int,
                          spin: int = 0,
                          output: str = '-') -> None:
        """Check several wave functions from GPAW calculation."""
        lines = ['band    energy     norm  normcut     best     ' +
                 ''.join(f'{sym:8}' for sym in self.group.symmetries)]
        n2 = n2 or calc.get_number_of_bands()
        for n in range(n1, n2):
            dct = self.check_band(calc, n, spin)
            best = dct['symmetry']
            norm = dct['norm']
            normcut = dct['overlaps'][0]
            eig = calc.get_eigenvalues(spin=spin)[n]
            lines.append(
                f'{n:4} {eig:9.3f} {norm:8.3f} {normcut:8.3f} {best:>8}' +
                ''.join(f'{x:8.3f}'
                        for x in dct['characters'].values()))

        fd = sys.stdout if output == '-' else open(output, 'w')
        fd.write('\n'.join(lines) + '\n')
        if output != '-':
            fd.close()


def sphere(radius: float, grid_spacing: float) -> Array2D:
    """Return sphere of grid-points.

    >>> points = sphere(1.1, 1.0)
    >>> points.shape
    (7, 3)
    """
    npts = int(radius / grid_spacing) + 1
    x = np.linspace(-npts, npts, 2 * npts + 1) * grid_spacing
    points = np.array(np.meshgrid(x, x, x, indexing='ij')).reshape((3, -1)).T
    points = points[(points**2).sum(1) <= radius**2]
    return points


def rotation_matrix(axes: Sequence[Axis]) -> Array3D:
    """Calculate rotation matrix.

    >>> rotation_matrix(['-y', 'x', None])
    array([[ 0, -1,  0],
           [ 1,  0,  0],
           [ 0,  0,  1]])
    """
    if all(axis is None for axis in axes):
        return np.eye(3)

    j = -1
    for i, axis in enumerate(axes):
        if axis is None:
            assert j == -1
            j = i
    assert j != -1

    axes = [normalize(axis) if axis is not None else None
            for axis in axes]
    axes[j] = np.cross(axes[j - 2], axes[j - 1])  # type: ignore

    return np.array(axes)


def normalize(vector: Union[str, Sequence[float], Array1D]) -> Array1D:
    """Normalize a vector.

    The *vector* must be a sequence of three numbers or one of the following
    strings: x, y, z, -x, -y, -z.
    """
    if isinstance(vector, str):
        if vector[0] == '-':
            return -np.array(normalize(vector[1:]))
        return {'x': np.array([1, 0, 0]),
                'y': np.array([0, 1, 0]),
                'z': np.array([0, 0, 1])}[vector]
    return np.array(vector) / np.linalg.norm(vector)
