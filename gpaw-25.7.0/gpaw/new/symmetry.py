from __future__ import annotations

from collections import defaultdict
from functools import cached_property
from typing import Any, Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.units import Bohr
from gpaw import debug
from gpaw.core.domain import normalize_cell
from gpaw.new import zips
from gpaw.rotation import rotation
from gpaw.symmetry import Symmetry as OldSymmetry
from gpaw.symmetry import frac
from gpaw.typing import Array2D, Array3D, ArrayLike1D, ArrayLike2D, ArrayLike3D


class SymmetryBrokenError(Exception):
    """Broken-symmetry error."""


def create_symmetries_object(atoms: Atoms,
                             *,
                             setup_ids: Sequence | None = None,
                             magmoms: ArrayLike2D | None = None,
                             rotations: ArrayLike3D | None = None,
                             translations: ArrayLike2D | None = None,
                             atommaps: ArrayLike2D | None = None,
                             extra_ids: Sequence[int] | None = None,
                             tolerance: float | None = None,  # Å
                             point_group: bool = True,
                             symmorphic: bool = True,
                             _backwards_compatible=False) -> Symmetries:
    """Find symmetries from atoms object.

    >>> atoms = Atoms('H', cell=[1, 1, 1], pbc=True)
    >>> sym = create_symmetries_object(atoms)
    >>> len(sym)
    48
    >>> sym.rotation_scc.shape
    (48, 3, 3)
    """
    cell_cv = atoms.cell.complete()

    if tolerance is None:
        tolerance = 1e-7 if _backwards_compatible else 1e-5
    if _backwards_compatible:
        cell_cv *= 1 / Bohr

    # Create int atom-ids from setups, magmoms and user-supplied
    # (extra_ids) ids:
    if setup_ids is None:
        ids = atoms.numbers
    else:
        ids = integer_ids(setup_ids)
    if magmoms is not None:
        ids = integer_ids((id, m) for id, m in zips(ids, safe_id(magmoms)))
    if extra_ids is not None:
        ids = integer_ids((id, x) for id, x in zips(ids, extra_ids))

    if rotations is None:
        # Find symmetries from cell, ids and positions:
        if point_group:
            sym = Symmetries.from_cell(
                cell_cv,
                pbc=atoms.pbc,
                tolerance=tolerance,
                _backwards_compatible=_backwards_compatible)
        else:
            # No symmetries (identity only):
            sym = Symmetries(cell=cell_cv,
                             tolerance=tolerance,
                             _backwards_compatible=_backwards_compatible)

        sym = sym.analyze_positions(
            atoms.get_scaled_positions(),
            ids=ids,
            symmorphic=symmorphic)
    else:
        sym = Symmetries(cell=cell_cv,
                         rotations=rotations,
                         translations=translations,
                         atommaps=atommaps,
                         tolerance=tolerance,
                         _backwards_compatible=_backwards_compatible)
        if atommaps is None:
            sym = sym.with_atom_maps(atoms.get_scaled_positions(), ids=ids)

    # Legacy:
    sym._old_symmetry = OldSymmetry(
        ids, cell_cv, atoms.pbc, tolerance,
        point_group,
        time_reversal='?',
        symmorphic=symmorphic)
    sym._old_symmetry.op_scc = sym.rotation_scc
    sym._old_symmetry.ft_sc = sym.translation_sc
    sym._old_symmetry.a_sa = sym.atommap_sa
    sym._old_symmetry.has_inversion = sym.has_inversion
    sym._old_symmetry.gcd_c = sym.gcd_c

    return sym


class Symmetries:
    def __init__(self,
                 *,
                 cell: ArrayLike1D | ArrayLike2D,
                 rotations: ArrayLike3D | None = None,
                 translations: ArrayLike2D | None = None,
                 atommaps: ArrayLike2D | None = None,
                 tolerance: float | None = None,
                 _backwards_compatible=False):
        """Symmetries object.

        "Rotations" here means rotations, mirror and inversion operations.

        Units of "cell" and "tolerance" should match.

        >>> sym = Symmetries.from_cell([1, 2, 3])
        >>> sym.has_inversion
        True
        >>> len(sym)
        8
        >>> sym2 = sym.analyze_positions([[0, 0, 0], [0, 0, 0.4]], ids=[1, 2])
        >>> sym2.has_inversion
        False
        >>> len(sym2)
        4
        """
        self.cell_cv = normalize_cell(cell)
        if tolerance is None:
            tolerance = 1e-7 if _backwards_compatible else 1e-5
        self.tolerance = tolerance
        self._backwards_compatible = _backwards_compatible
        if rotations is None:
            rotations = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
        self.rotation_scc = np.array(rotations, dtype=int)
        assert (self.rotation_scc == rotations).all()
        if translations is None:
            self.translation_sc = np.zeros((len(self.rotation_scc), 3))
        else:
            self.translation_sc = np.array(translations)
        if atommaps is None:
            self.atommap_sa = np.empty((len(self.rotation_scc), 0), int)
        else:
            self.atommap_sa = np.array(atommaps)
            assert self.atommap_sa.dtype == int

        # Legacy stuff:
        self.op_scc = self.rotation_scc  # old name
        self._old_symmetry: OldSymmetry

    @cached_property
    def symmorphic(self):
        return not self.translation_sc.any()

    @cached_property
    def has_inversion(self):
        inv_cc = -np.eye(3, dtype=int)
        for r_cc, t_c in zip(self.rotation_scc, self.translation_sc):
            if (r_cc == inv_cc).all() and not t_c.any():
                return True
        return False

    @classmethod
    def from_cell(cls,
                  cell: ArrayLike1D | ArrayLike2D,
                  *,
                  pbc: ArrayLike1D = (True, True, True),
                  tolerance: float | None = None,
                  _backwards_compatible=False) -> Symmetries:
        if isinstance(pbc, int):
            pbc = (pbc,) * 3
        cell_cv = normalize_cell(cell)
        if tolerance is None:
            tolerance = 1e-7 if _backwards_compatible else 1e-5
        rotation_scc = find_lattice_symmetry(cell_cv, pbc, tolerance,
                                             _backwards_compatible)
        return cls(cell=cell_cv,
                   rotations=rotation_scc,
                   tolerance=tolerance,
                   _backwards_compatible=_backwards_compatible)

    def analyze_positions(self,
                          relative_positions: ArrayLike2D,
                          ids: Sequence[int],
                          *,
                          symmorphic: bool = True) -> Symmetries:
        return prune_symmetries(
            self, np.asarray(relative_positions), ids, symmorphic)

    def with_atom_maps(self,
                       relative_positions: Array2D,
                       ids: Sequence[int]) -> Symmetries:
        atommap_sa = np.empty((len(self), len(relative_positions)), int)
        a_ij = defaultdict(list)
        for a, id in enumerate(ids):
            a_ij[id].append(a)
        for U_cc, t_c, map_a in zip(self.rotation_scc,
                                    self.translation_sc,
                                    atommap_sa):
            map_a[:] = self.check_one_symmetry(relative_positions,
                                               U_cc, t_c, a_ij)
        return Symmetries(cell=self.cell_cv,
                          rotations=self.rotation_scc,
                          translations=self.translation_sc,
                          atommaps=atommap_sa,
                          tolerance=self.tolerance,
                          _backwards_compatible=self._backwards_compatible)

    @classmethod
    def from_atoms(cls,
                   atoms,
                   *,
                   ids: Sequence[int] | None = None,
                   symmorphic: bool = True,
                   tolerance: float | None = None):
        sym = cls.from_cell(atoms.cell,
                            pbc=atoms.pbc,
                            tolerance=tolerance)
        if ids is None:
            ids = atoms.numbers
        return sym.analyze_positions(atoms.positions,
                                     ids=ids,
                                     symmorphic=symmorphic)

    def __len__(self):
        return len(self.rotation_scc)

    def __str__(self):
        lines = ['symmetry:',
                 f'  number of symmetries: {len(self)}']
        if self.symmorphic:
            lines.append('  rotations: [')
            for rot_cc in self.rotation_scc:
                lines.append(f'    {mat(rot_cc)},')
        else:
            nt = self.translation_sc.any(1).sum()
            lines.append(f'  number of symmetries with translation: {nt}')
            lines.append('  rotations and translations: [')
            for rot_cc, t_c in zips(self.rotation_scc, self.translation_sc):
                a, b, c = t_c
                lines.append(f'    [{mat(rot_cc)}, '
                             f'[{a:6.3f}, {b:6.3f}, {c:6.3f}]],')
        lines[-1] = lines[-1][:-1] + ']\n'
        return '\n'.join(lines)

    def check_positions(self, fracpos_ac):
        for U_cc, t_c, b_a in zip(self.rotation_scc,
                                  self.translation_sc,
                                  self.atommap_sa):
            error_ac = fracpos_ac @ U_cc - t_c - fracpos_ac[b_a]
            error_ac -= error_ac.round()
            if self._backwards_compatible:
                if abs(error_ac).max() > self.tolerance:
                    raise SymmetryBrokenError
            else:
                error_av = error_ac @ self.cell_cv
                if (error_av**2).sum(1).max() > self.tolerance**2:
                    raise SymmetryBrokenError

    def symmetrize_forces(self, F0_av):
        """Symmetrize forces."""
        F_av = np.zeros_like(F0_av)
        for map_a, op_cc in zip(self.atommap_sa, self.rotation_scc):
            op_vv = np.linalg.inv(self.cell_cv) @ op_cc @ self.cell_cv
            for a1, a2 in enumerate(map_a):
                F_av[a2] += np.dot(F0_av[a1], op_vv)
        return F_av / len(self)

    def lcm(self) -> list[int]:
        """Find least common multiple compatible with translations."""
        return [np.lcm.reduce([frac(t, tol=1e-4)[1] for t in t_s])
                for t_s in self.translation_sc.T]

    @cached_property
    def gcd_c(self):
        # Needed for old gpaw.utilities.gpts.get_number_of_grid_points()
        # function ...
        return np.array(self.lcm())

    def check_grid(self, N_c) -> bool:
        """Check that symmetries are commensurate with grid."""
        for U_cc, t_c in zip(self.rotation_scc, self.translation_sc):
            t_c = t_c * N_c
            # Make sure all grid-points map onto another grid-point:
            if (((N_c * U_cc).T % N_c).any() or
                not np.allclose(t_c, t_c.round())):
                return False
        return True

    def check_one_symmetry(self,
                           spos_ac,
                           op_cc,
                           ft_c,
                           a_ia):
        """Checks whether atoms satisfy one given symmetry operation."""

        a_a = np.zeros(len(spos_ac), int)
        for b_a in a_ia.values():
            spos_jc = spos_ac[b_a]
            for b in b_a:
                spos_c = np.dot(spos_ac[b], op_cc)
                sdiff_jc = spos_c - spos_jc - ft_c
                sdiff_jc -= sdiff_jc.round()
                if self._backwards_compatible:
                    indices = np.where(
                        abs(sdiff_jc).max(1) < self.tolerance)[0]
                else:
                    sdiff_jv = sdiff_jc @ self.cell_cv
                    indices = np.where(
                        (sdiff_jv**2).sum(1) < self.tolerance**2)[0]
                if len(indices) == 1:
                    a = indices[0]
                    a_a[b] = b_a[a]
                else:
                    assert len(indices) == 0
                    return None

        return a_a


def find_lattice_symmetry(cell_cv, pbc_c, tol, _backwards_compatible=False):
    """Determine list of symmetry operations."""
    # Symmetry operations as matrices in 123 basis.
    # Operation is a 3x3 matrix, with possible elements -1, 0, 1, thus
    # there are 3**9 = 19683 possible matrices:
    combinations = 1 - np.indices([3] * 9)
    U_scc = combinations.reshape((3, 3, 3**9)).transpose((2, 0, 1))

    # The metric of the cell should be conserved after applying
    # the operation:
    metric_cc = cell_cv.dot(cell_cv.T)
    metric_scc = np.einsum('sij, jk, slk -> sil',
                           U_scc, metric_cc, U_scc,
                           optimize=True)
    if _backwards_compatible:
        mask_s = abs(metric_scc - metric_cc).sum(2).sum(1) <= tol
    else:
        mask_s = abs(metric_scc - metric_cc).sum(2).sum(1) <= tol**2
    U_scc = U_scc[mask_s]

    # Operation must not swap axes that don't have same PBC:
    pbc_cc = np.logical_xor.outer(pbc_c, pbc_c)
    mask_s = ~U_scc[:, pbc_cc].any(axis=1)
    U_scc = U_scc[mask_s]
    return U_scc


def prune_symmetries(sym: Symmetries,
                     relpos_ac: Array2D,
                     id_a: Sequence[int],
                     symmorphic: bool = True) -> Symmetries:
    """Remove symmetries that are not satisfied by the atoms."""

    if len(relpos_ac) == 0:
        return sym

    # Build lists of atom numbers for each type of atom - one
    # list for each combination of atomic number, setup type,
    # magnetic moment and basis set:
    a_ij = defaultdict(list)
    for a, id in enumerate(id_a):
        a_ij[id].append(a)

    a_j = a_ij[id_a[0]]  # just pick the first species

    def check(op_cc, ft_c):
        return sym.check_one_symmetry(relpos_ac, op_cc, ft_c, a_ij)

    # if supercell disable fractional translations:
    if not symmorphic:
        op_cc = np.identity(3, int)
        ftrans_sc = relpos_ac[a_j[1:]] - relpos_ac[a_j[0]]
        ftrans_sc -= np.rint(ftrans_sc)
        for ft_c in ftrans_sc:
            a_a = check(op_cc, ft_c)
            if a_a is not None:
                symmorphic = True
                break

    symmetries = []
    ftsymmetries = []

    # go through all possible symmetry operations
    for op_cc in sym.rotation_scc:
        # first ignore fractional translations
        a_a = check(op_cc, [0, 0, 0])
        if a_a is not None:
            symmetries.append((op_cc, [0, 0, 0], a_a))
        elif not symmorphic:
            # check fractional translations
            sposrot_ac = np.dot(relpos_ac, op_cc)
            ftrans_jc = sposrot_ac[a_j] - relpos_ac[a_j[0]]
            ftrans_jc -= np.rint(ftrans_jc)
            for ft_c in ftrans_jc:
                a_a = check(op_cc, ft_c)
                if a_a is not None:
                    ftsymmetries.append((op_cc, ft_c, a_a))

    # Add symmetry operations with fractional translations at the end:
    symmetries.extend(ftsymmetries)

    sym = Symmetries(cell=sym.cell_cv,
                     rotations=[s[0] for s in symmetries],
                     translations=[s[1] for s in symmetries],
                     atommaps=[s[2] for s in symmetries],
                     tolerance=sym.tolerance,
                     _backwards_compatible=sym._backwards_compatible)
    if debug:
        sym.check_positions(relpos_ac)
    return sym


class SymmetrizationPlan:
    def __init__(self,
                 symmetries: Symmetries,
                 l_aj):
        self.symmetries = symmetries
        self.l_aj = l_aj
        self.rotation_svv = np.einsum('vc, scd, dw -> svw',
                                      np.linalg.inv(symmetries.cell_cv),
                                      symmetries.rotation_scc,
                                      symmetries.cell_cv)
        lmax = max((max(l_j) for l_j in l_aj), default=-1)
        self.rotation_lsmm = [
            np.array([rotation(l, r_vv) for r_vv in self.rotation_svv])
            for l in range(lmax + 1)]
        self._rotations: dict[tuple[int, ...], Array3D] = {}

    def rotations(self, l_j, xp=np):
        ells = tuple(l_j)
        rotation_sii = self._rotations.get(ells)
        if rotation_sii is None:
            ni = sum(2 * l + 1 for l in l_j)
            rotation_sii = np.zeros((len(self.symmetries), ni, ni))
            i1 = 0
            for l in l_j:
                i2 = i1 + 2 * l + 1
                rotation_sii[:, i1:i2, i1:i2] = self.rotation_lsmm[l]
                i1 = i2
            rotation_sii = xp.asarray(rotation_sii)
            self._rotations[ells] = rotation_sii
        return rotation_sii

    def apply_distributed(self, D_asii, dist_D_asii):
        for a1, D_sii in dist_D_asii.items():
            D_sii[:] = 0.0
            rotation_sii = self.rotations(self.l_aj[a1])
            for a2, rotation_ii in zips(self.symmetries.atommap_sa[:, a1],
                                        rotation_sii):
                D_sii += np.einsum('ij, sjk, lk -> sil',
                                   rotation_ii, D_asii[a2], rotation_ii)
        dist_D_asii.data *= 1.0 / len(self.symmetries)


class GPUSymmetrizationPlan(SymmetrizationPlan):
    def __init__(self,
                 symmetries: Symmetries,
                 l_aj,
                 layout):
        super().__init__(symmetries, l_aj)

        xp = layout.xp
        a_sa = symmetries.atommap_sa

        ns = a_sa.shape[0]  # Number of symmetries
        na = a_sa.shape[1]  # Number of atoms

        if xp is np:
            import scipy
            sparse = scipy.sparse
        else:
            from gpaw.gpu import cupyx
            sparse = cupyx.scipy.sparse

        # Find orbits, i.e. point group action,
        # which also equals to set of all cosets.
        # In practical terms, these are just atoms which map
        # to each other via symmetry operations.
        # Mathematically {{as: s∈ S}: a∈ A}, where a is an atom.
        cosets = {frozenset(a_sa[:, a]) for a in range(na)}

        S_aZZ = {}
        work = []
        for coset in map(list, cosets):
            nA = len(coset)  # Number of atoms in this orbit
            a = coset[0]  # Representative atom for coset

            # The atomic density matrices transform as
            # ρ'_ii = R_sii ρ_ii R^T_sii
            # Which equals to vec(ρ'_ii) = (R^s_ii ⊗  R^s_ii) vec(ρ_ii)
            # Here we to the Kronecker product for each of the
            # symmetry transformations.
            R_sii = xp.asarray(self.rotations(l_aj[a], xp))
            i2 = R_sii.shape[1]**2
            R_sPP = xp.einsum('sab, scd -> sacbd', R_sii, R_sii)
            R_sPP = R_sPP.reshape((ns, i2, i2)) / ns

            S_ZZ = xp.zeros((nA * i2,) * 2)

            # For each orbit, the symetrization operation is represented by
            # a full matrix operating on a subset of indices to the full array.
            for loca1, a1 in enumerate(coset):
                Z1 = loca1 * i2
                Z2 = Z1 + i2
                for s, a2 in enumerate(a_sa[:, a1]):
                    loca2 = coset.index(a2)
                    Z3 = loca2 * i2
                    Z4 = Z3 + i2
                    S_ZZ[Z1:Z2, Z3:Z4] += R_sPP[s]
            # Utilize sparse matrices if sizes get out of hand
            # Limit is hard coded to 100MB per orbit
            if S_ZZ.nbytes > 100 * 1024**2:
                S_ZZ = sparse.csr_matrix(S_ZZ)
            S_aZZ[a] = S_ZZ
            indices = []
            for loca1, a1 in enumerate(coset):
                a1_, start, end = layout.myindices[a1]
                # When parallelization is done, this needs to be rewritten
                assert a1_ == a1
                for X in range(i2):
                    indices.append(start + X)
            work.append((a, xp.array(indices)))

        self.work = work
        self.S_aZZ = S_aZZ
        self.xp = xp

    def apply(self, source, target):
        total = 0
        for a, ind in self.work:
            for spin in range(len(source)):
                total += len(ind)
                target[spin, ind] = self.S_aZZ[a] @ source[spin, ind]
        assert total / len(source) == source.shape[1]


def mat(rot_cc) -> str:
    """Convert 3x3 matrix to str.

    >>> mat([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    '[[-1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]]'

    """
    return '[[' + '], ['.join(', '.join(f'{r:2}'
                                        for r in rot_c)
                              for rot_c in rot_cc) + ']]'


def integer_ids(ids: Iterable) -> list[int]:
    """Convert arbitrary ids to int ids.

    >>> integer_ids([(1, 'a'), (12, 'b'), (1, 'a')])
    [0, 1, 0]
    """
    dct: dict[Any, int] = {}
    iids = []
    for id in ids:
        iid = dct.get(id)
        if iid is None:
            iid = len(dct)
            dct[id] = iid
        iids.append(iid)
    return iids


def safe_id(magmom_av, tolerance=1e-3):
    """Convert magnetic moments to integer id's.

    While calculating id's for atoms, there may be rounding errors
    in magnetic moments supplied. This will create an unique integer
    identifier for each magnetic moment double, based on the range
    as set by the first occurence of each floating point number:
    [magmom_a - tolerance, magmom_a + tolerance].

    >>> safe_id([1.01, 0.99, 0.5], tolerance=0.025)
    [0, 0, 2]
    """
    id_a = []
    for a, magmom_v in enumerate(magmom_av):
        quantized = None
        for a2 in range(a):
            if np.linalg.norm(magmom_av[a2] - magmom_v) < tolerance:
                quantized = a2
                break
        if quantized is None:
            quantized = a
        id_a.append(quantized)
    return id_a
