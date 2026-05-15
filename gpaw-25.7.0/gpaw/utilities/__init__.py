# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Utility functions and classes."""

import io
import os
import re
import sys
import time
from contextlib import contextmanager
from math import sqrt
from pathlib import Path
from typing import Union

import gpaw.cgpaw as cgpaw
import gpaw.mpi as mpi
import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list
from gpaw import GPAW_NO_C_EXTENSION, debug
from gpaw.typing import DTypeLike

# Code will crash for setups without any projectors.  Setups that have
# no projectors therefore receive a dummy projector as a hacky
# workaround.  The projector is assigned a certain, small size.  If
# the grid is so coarse that no point falls within the projector's range,
# there'll also be an error.  So this limits allowed grid spacings.
min_locfun_radius = 0.85  # Bohr
smallest_safe_grid_spacing = 2 * min_locfun_radius / np.sqrt(3)  # ~0.52 Ang


class AtomsTooClose(ValueError):
    pass


def check_atoms_too_close(atoms: Atoms) -> None:
    radii = covalent_radii[atoms.numbers] * 0.01
    dists = neighbor_list('d', atoms, radii)
    if len(dists):
        raise AtomsTooClose(f'Atoms are too close, e.g. {dists[0]} Å')


def check_atoms_too_close_to_boundary(atoms: Atoms,
                                      dist: float = 0.2) -> None:
    """Check if any atoms are too close to the boundary of the box.

    >>> atoms = Atoms('H', cell=[1, 1, 1])
    >>> check_atoms_too_close_to_boundary(atoms)
    Traceback (most recent call last):
    ...
        raise AtomsTooClose('Atoms too close to boundary')
    gpaw.utilities.AtomsTooClose: Atoms too close to boundary
    >>> atoms.center()
    >>> check_atoms_too_close_to_boundary(atoms)
    >>> atoms = Atoms('H',
    ...               positions=[[0.5, 0.5, 0.0]],
    ...               cell=[1, 1, 0],  # no bounday in z-direction
    ...               pbc=(1, 1, 0))
    >>> check_atoms_too_close_to_boundary(atoms)
    """
    for axis_v, recip_v, pbc in zip(atoms.cell,
                                    atoms.cell.reciprocal(),
                                    atoms.pbc):
        if pbc:
            continue
        L = np.linalg.norm(axis_v)
        if L < 1e-12:  # L==0 means no boundary
            continue
        spos_a = atoms.positions @ recip_v
        eps = dist / L
        if (spos_a < eps).any() or (spos_a > 1 - eps).any():
            raise AtomsTooClose('Atoms too close to boundary')


def unpack_atomic_matrices(M_sP, setups, new=False, density=False):
    M_asp = {}
    P1 = 0
    for a, setup in enumerate(setups):
        ni = setup.ni
        P2 = P1 + ni * (ni + 1) // 2
        M_sp = M_sP[:, P1:P2]
        if new:
            M2_sp = np.empty_like(M_sp)
            pnew = 0
            for c in range(ni):
                for r in range(c + 1):
                    pold = c - r + r * (2 * ni - r + 1) // 2
                    if density and r < c:
                        M2_sp[:, pold] = 2 * M_sp[:, pnew]
                    else:
                        M2_sp[:, pold] = M_sp[:, pnew]
                    pnew += 1
            M_asp[a] = M2_sp
        else:
            M_asp[a] = M_sp.copy()
        P1 = P2
    return M_asp


def pack_atomic_matrices(M_asp):
    M2_asp = M_asp.copy()
    M2_asp.redistribute(M2_asp.partition.as_serial())
    return M2_asp.toarray(axis=1)


def h2gpts(h, cell_cv, idiv=4):
    """Convert grid spacing to number of grid points divisible by idiv.

    Note that units of h and cell_cv must match!

    h: float
        Desired grid spacing in.
    cell_cv: 3x3 ndarray
        Unit cell.
    """

    L_c = (np.linalg.inv(cell_cv)**2).sum(0)**-0.5
    return np.maximum(idiv, (L_c / h / idiv + 0.5).astype(int) * idiv)


def is_contiguous(array, dtype=None):
    """Check for contiguity and type."""
    if dtype is None:
        return array.flags.c_contiguous
    else:
        return array.flags.c_contiguous and array.dtype == dtype


# Radial-grid Hartree solver:
#
#                       l
#             __  __   r
#     1      \   4||    <   * ^    ^
#   ------ =  )  ---- ---- Y (r)Y (r'),
#    _ _     /__ 2l+1  l+1  lm   lm
#   |r-r'|    lm      r
#                      >
# where
#
#   r = min(r, r')
#    <
#
# and
#
#   r = max(r, r')
#    >
#
def hartree(l: int, nrdr: np.ndarray, r: np.ndarray, vr: np.ndarray) -> None:
    """Calculates radial Coulomb integral.

    The following integral is calculated::

                                   ^
                          n (r')Y (r')
              ^    / _     l     lm
      v (r)Y (r) = |dr' --------------,
       l    lm     /        _   _
                           |r - r'|

    where input and output arrays `nrdr` and `vr`::

              dr
      n (r) r --  and  v (r) r.
       l      dg        l
    """
    assert is_contiguous(nrdr, float)
    assert is_contiguous(r, float)
    assert is_contiguous(vr, float)
    assert nrdr.shape == vr.shape and len(vr.shape) == 1
    assert len(r.shape) == 1
    assert len(r) >= len(vr)
    return cgpaw.hartree(l, nrdr, r, vr)


def packed_index(i1, i2, ni):
    """Return a packed index"""
    if i1 > i2:
        return (i2 * (2 * ni - 1 - i2) // 2) + i1
    else:
        return (i1 * (2 * ni - 1 - i1) // 2) + i2


def unpacked_indices(p, ni):
    """Return unpacked indices corresponding to upper triangle"""
    assert 0 <= p < ni * (ni + 1) // 2
    i1 = int(ni + .5 - sqrt((ni - .5)**2 - 2 * (p - ni)))
    return i1, p - i1 * (2 * ni - 1 - i1) // 2


packing_conventions = """\n
The convention is that density matrices are constructed using (un)pack_density
and anything that should be multiplied onto such, e.g. corrections to the
Hamiltonian, are constructed according to (un)pack_hermitian.
"""


def pack_hermitian(M2, tolerance=1e-10):
    r"""Pack Hermitian

    This functions packs a Hermitian 2D array to a
    1D array, averaging off-diagonal terms with complex conjugation.

    The matrix::

           / a00 a01 a02 \
       A = | a10 a11 a12 |
           \ a20 a21 a22 /

    is transformed to the vector::

       (a00, [a01 + a10*]/2, [a02 + a20*]/2, a11, [a12 + a21*]/2, a22)
    """
    if M2.ndim == 3:
        return np.array([pack_hermitian(m2) for m2 in M2])
    n = len(M2)
    M = np.zeros(n * (n + 1) // 2, M2.dtype)
    p = 0
    for r in range(n):
        M[p] = M2[r, r]
        p += 1
        for c in range(r + 1, n):
            M[p] = (M2[r, c] + np.conjugate(M2[c, r])) / 2.  # note / 2.
            error = abs(M2[r, c] - np.conjugate(M2[c, r]))
            assert error < tolerance, 'Pack not symmetric by %s' % error + ' %'
            p += 1
    assert p == len(M)
    return M


def unpack_hermitian(M):
    """Unpack 1D array to Hermitian 2D array,
    assuming a packing as in ``pack_hermitian``."""

    if M.ndim == 2:
        return np.array([unpack_hermitian(m) for m in M])
    assert is_contiguous(M)
    assert M.ndim == 1
    n = int(sqrt(0.25 + 2.0 * len(M)))
    M2 = np.zeros((n, n), M.dtype.char)
    if M.dtype == complex:
        cgpaw.unpack_complex(M, M2)
    else:
        cgpaw.unpack(M, M2)
    return M2


def pack_density(A: np.ndarray) -> np.ndarray:
    r"""Pack off-diagonal sum

    This function packs a 2D Hermitian array to 1D, adding off-diagonal terms.

    The matrix::

           / a00 a01 a02 \
       A = | a10 a11 a12 |
           \ a20 a21 a22 /

    is transformed to the vector::

       (a00, a01 + a10, a02 + a20, a11, a12 + a21, a22)"""

    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    assert A.dtype in [float, complex]
    return cgpaw.pack(A)


# We cannot recover the complex part of the off-diag elements from a
# pack_density array since they are summed to zero (we only pack Hermitian
# arrays). We should consider if "unpack_density" even makes sense to have.


def unpack_density(M):
    """Unpack 1D array to 2D Hermitian array,
    assuming a packing as in ``pack_density``."""
    if M.ndim == 2:
        return np.array([unpack_density(m) for m in M])
    M2 = unpack_hermitian(M)
    M2 *= 0.5  # divide all by 2
    M2.flat[0::len(M2) + 1] *= 2  # rescale diagonal to original size
    return M2


for method in (pack_hermitian, unpack_hermitian, pack_density, pack_density):
    method.__doc__ += packing_conventions  # type: ignore


def element_from_packed(M, i, j):
    """Return a specific element from a packed array (by ``pack``)."""
    n = int(sqrt(2 * len(M) + .25))
    assert i < n and j < n
    p = packed_index(i, j, n)
    if i == j:
        return M[p]
    elif i > j:
        return .5 * M[p]
    else:
        return .5 * np.conjugate(M[p])


def logfile(name, rank=0):
    """Create file object from name.

    Use None for /dev/null and '-' for sys.stdout.  Ranks > 0 will
    get /dev/null."""

    if rank == 0:
        if name is None:
            fd = devnull
        elif name == '-':
            fd = sys.stdout
        elif isinstance(name, str):
            fd = open(name, 'w')
        else:
            fd = name
    else:
        fd = devnull
    return fd


def uncamelcase(name):
    """Convert a CamelCase name to a string of space-seperated words."""
    words = re.split('([A-Z]{1}[a-z]+)', name)
    return ' '.join([word for word in words if word != ''])


def divrl(a_g, l, r_g):
    """Return array divided by r to the l'th power."""
    b_g = a_g.copy()
    if l > 0:
        b_g[1:] /= r_g[1:]**l
        b1, b2 = b_g[1:3]
        r12, r22 = r_g[1:3]**2
        b_g[0] = (b1 * r22 - b2 * r12) / (r22 - r12)
    return b_g


def compiled_with_sl():
    return hasattr(cgpaw, 'new_blacs_context')


def compiled_with_libvdwxc():
    return hasattr(cgpaw, 'libvdwxc_create')


def load_balance(paw, atoms):
    try:
        paw.initialize(atoms)
    except SystemExit:
        pass
    atoms_r = np.zeros(paw.wfs.world.size)
    rnk_a = paw.wfs.gd.get_ranks_from_positions(paw.spos_ac)
    for rnk in rnk_a:
        atoms_r[rnk] += 1
    max_atoms = max(atoms_r)
    min_atoms = min(atoms_r)
    ave_atoms = atoms_r.sum() / paw.wfs.world.size
    stddev_atoms = sqrt((atoms_r**2).sum() / paw.wfs.world.size - ave_atoms**2)
    print("Information about load balancing")
    print("--------------------------------")
    print("Number of atoms:", len(paw.spos_ac))
    print("Number of CPUs:", paw.wfs.world.size)
    print("Max. number of atoms/CPU:   ", max_atoms)
    print("Min. number of atoms/CPU:   ", min_atoms)
    print("Average number of atoms/CPU:", ave_atoms)
    print("    standard deviation:     %5.1f" % stddev_atoms)


if not debug and not GPAW_NO_C_EXTENSION:
    hartree = cgpaw.hartree  # noqa
    pack_density = cgpaw.pack


def unlink(path: Union[str, Path], world=None):
    """Safely unlink path (delete file or symbolic link)."""

    if isinstance(path, str):
        path = Path(path)
    if world is None:
        world = mpi.world

    # Remove file:
    if world.rank == 0:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    else:
        while path.is_file():
            time.sleep(1.0)
    world.barrier()


@contextmanager
def file_barrier(path: Union[str, Path], world=None):
    """Context manager for writing a file.

    After the with-block all cores will be able to read the file.

    >>> with file_barrier('something.txt'):
    ...     result = 2 + 2
    ...     Path('something.txt').write_text(f'{result}')  # doctest: +SKIP

    This will remove the file, write the file and wait for the file.
    """

    if isinstance(path, str):
        path = Path(path)
    if world is None:
        world = mpi.world

    # Remove file:
    unlink(path, world)

    yield

    # Wait for file:
    while not path.is_file():
        time.sleep(1.0)
    world.barrier()


class _NullIO(io.BufferedIOBase):
    # Implement as few methods as possible in order to be the target of
    # TextIOWrapper.  Python docs are not very specific.
    def writable(self):
        return True

    def flush(self):
        pass


devnull = io.TextIOWrapper(_NullIO())  # type: ignore


def convert_string_to_fd(name, world=None):
    """Create a file-descriptor for text output.

    Will open a file for writing with given name.  Use None for no output and
    '-' for sys.stdout.
    """
    if world is None:
        from ase.parallel import world
    if name is None or world.rank != 0:
        return open(os.devnull, 'w')
    if name == '-':
        return sys.stdout
    if isinstance(name, (str, Path)):
        return open(name, 'w')
    return name  # we assume name is already a file-descriptor


_complex_float = {
    np.float32: np.complex64,
    np.float64: np.complex128,
    np.complex64: np.complex64,
    np.complex128: np.complex128,
    float: complex,
    complex: complex}

_real_float = {
    np.complex64: np.float32,
    np.complex128: np.float64,
    np.float32: np.float32,
    np.float64: np.float64,
    complex: float,
    float: float}


def as_complex_dtype(dtype: DTypeLike) -> np.dtype:
    """Convert dtype to complex dtype.

    >>> [as_complex_dtype(dt) for dt in
    ...  [np.float32, np.float64, complex]]
    [dtype('complex64'), dtype('complex128'), dtype('complex128')]
    """
    return np.dtype(_complex_float[np.dtype(dtype).type])


def as_real_dtype(dtype: DTypeLike) -> np.dtype:
    """Convert dtype to real dtype.

    >>> [as_real_dtype(dt) for dt in
    ...  [np.float32, np.float64, complex]]
    [dtype('float32'), dtype('float64'), dtype('float64')]
    """
    return np.dtype(_real_float[np.dtype(dtype).type])
