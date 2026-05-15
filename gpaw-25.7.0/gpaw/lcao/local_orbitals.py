from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from ase.units import Bohr, Hartree
from gpaw.calculator import GPAW
from gpaw.lcao.tightbinding import TightBinding  # as LCAOTightBinding
from gpaw.lcao.tools import get_bfi
from gpaw.typing import Array1D, Array2D, Array4D
from gpaw.utilities.blas import r2k
from gpaw.utilities.tools import lowdin, tri2full
from scipy.linalg import eigh


def get_subspace(A_MM: Array2D, indices: Sequence[int]):
    """Get the subspace spanned by the basis function listed in index."""
    assert A_MM.ndim == 2 and A_MM.shape[0] == A_MM.shape[1]
    return A_MM.take(indices, 0).take(indices, 1)


def get_orthonormal_subspace(H_MM: Array2D,
                             S_MM: Array2D,
                             indices: Sequence[int] = None):
    """Get orthonormal eigenvalues and -vectors of subspace listed in index."""
    if indices is not None:
        h_ww = get_subspace(H_MM, indices)
        s_ww = get_subspace(S_MM, indices)
    else:
        h_ww = H_MM
        s_ww = S_MM
    eps, v = eigh(h_ww, s_ww)
    return eps, v


def subdiagonalize(H_MM: Array2D,
                   S_MM: Array2D,
                   blocks: Sequence[Sequence[int]]):
    """Subdiagonalize blocks."""
    nM = len(H_MM)
    v_MM = np.eye(nM)
    eps_M = np.zeros(nM)
    mask_M = np.ones(nM, dtype=int)
    for block in blocks:
        eps, v = get_orthonormal_subspace(H_MM, S_MM, block)
        v_MM[np.ix_(block, block)] = v
        eps_M[block] = eps
        mask_M[block] = 0
    epsx_M = np.ma.masked_array(eps_M, mask=mask_M)  # type: ignore
    return epsx_M, v_MM


def subdiagonalize_atoms(calc: GPAW,
                         H_MM: Array2D,
                         S_MM: Array2D,
                         atom_indices: int | Sequence[int] | None = None):
    """Subdiagonalize atomic sub-spaces."""
    if atom_indices is None:
        atom_indices = range(len(calc.atoms))
    if isinstance(atom_indices, int):
        atom_indices = [atom_indices]
    block_lists = []
    for a in atom_indices:
        M = calc.wfs.basis_functions.M_a[a]
        block = range(M, M + calc.wfs.setups[a].nao)
        block_lists.append(block)
    return subdiagonalize(H_MM, S_MM, block_lists)


def get_orbitals(calc: GPAW, U_Mw: Array2D, q: int = 0):
    """Get orbitals from AOs coefficients.

    Parameters
    ----------
    calc : GPAW
        LCAO calculator
    U_Mw : array_like
        LCAO expansion coefficients.
    """
    Nw = U_Mw.shape[1]
    C_wM = np.ascontiguousarray(U_Mw.T).astype(calc.wfs.dtype)
    w_wG = calc.wfs.gd.zeros(Nw, dtype=calc.wfs.dtype)
    calc.wfs.basis_functions.lcao_to_grid(C_wM, w_wG, q=q)
    return w_wG


def get_xc(calc: GPAW, v_wG: Array4D, P_awi=None):
    """Get exchange-correlation part of the Hamiltonian."""
    if calc.density.nt_sg is None:
        calc.density.interpolate_pseudo_density()
    nt_sg = calc.density.nt_sg
    vxct_sg = calc.density.finegd.zeros(calc.wfs.nspins)
    calc.hamiltonian.xc.calculate(calc.density.finegd, nt_sg, vxct_sg)
    vxct_G = calc.wfs.gd.empty()
    calc.hamiltonian.restrict_and_collect(vxct_sg[0], vxct_G)

    # Integrate pseudo part
    Nw = len(v_wG)
    xc_ww = np.empty((Nw, Nw))
    r2k(0.5 * calc.wfs.gd.dv, v_wG, vxct_G * v_wG, 0.0, xc_ww)
    tri2full(xc_ww, "L")

    # Atomic PAW corrections required? XXX
    if P_awi is not None:
        raise NotImplementedError(
            'Atomic PAW corrections not included. '
            'Have a look at pwf2::get_xc2 for inspiration.')

    return xc_ww * Hartree


def get_Fcore():
    pass


class BasisTransform:
    """Class to perform a basis transformation.

    Attributes
    ----------
    U_MM : array_like
        2-D rotation matrix between 2 basis.
    indices : array_like, optional
        1-D array of sub-indices of the new basis.
    U_Mw : array_like, optional
        Same as `U_MM` but includes only `indices` of the new basis.

    Methods
    -------
    rotate_matrx(A_MM, keep_rest=False)
        Rotate a matrix.
    rotate_projections(P_aMi, keep_rest=False)
        Rotate PAW atomic projects.
    rotate_function(P_aMi, keep_rest=False)
        Rotate PAW atomic projects.

    """

    def __init__(self, U_MM: Array2D, indices: Sequence[int] = None) -> None:
        """

        Parameters
        ----------
        See class docstring

        """
        self.U_MM = U_MM
        self.indices = indices
        self.U_Mw: Array2D | None
        if indices is not None:
            self.U_Mw = np.ascontiguousarray(U_MM[:, indices])
        else:
            self.U_Mw = None

    def get_rotation(self, keep_rest: bool = False):
        if keep_rest or self.U_Mw is None:
            return self.U_MM
        return self.U_Mw

    def rotate_matrix(self, A_MM: Array2D, keep_rest: bool = False):
        U_Mx = self.get_rotation(keep_rest)
        return U_Mx.T.conj() @ A_MM @ U_Mx

    def rotate_projections(self, P_aMi, keep_rest: bool = False):
        U_Mx = self.get_rotation(keep_rest)
        P_awi = {}
        for a, P_Mi in P_aMi.items():
            P_awi[a] = np.tensordot(U_Mx, P_Mi, axes=([0], [0]))
        return P_awi

    def rotate_function(self, Psi_MG: Array4D, keep_rest: bool = False):
        U_Mx = self.get_rotation(keep_rest)
        return np.tensordot(U_Mx, Psi_MG, axes=([0], [0]))


class EffectiveModel(BasisTransform):
    """Class for an effective model.

    See Also
    --------
    BasisTranform


    Methods
    -------
    get_static_correction(H_MM: npt.NDArray,
                          S_MM: npt.NDArray,
                          z: complex = 0. + 1e-5j)
        Hybridization of the effective model with the rest evaluated at `z`.

    """

    def __init__(self,
                 U_MM: Array2D,
                 indices: Sequence[int],
                 S_MM: Array2D = None) -> None:
        """

        See Also
        --------
        BasisTransform

        Parameters
        ----------
        S_MM : array_like, optional
            2-D LCAO overlap matrix. If provided, the resulting basis
            is orthogonalized.
        """
        if S_MM is not None:
            lowdin(self.U_Mw, self.rotate_matrix(S_MM))
            assert self.U_Mw is not None
            np.testing.assert_allclose(self.rotate_matrix(
                S_MM), np.eye(self.U_Mw.shape[1]))
            U_MM = U_MM[:]
            U_MM = U_MM[:, indices] = self.U_Mw

        super().__init__(U_MM, indices)

    def get_static_correction(self,
                              H_MM: Array2D,
                              S_MM: Array2D,
                              z: complex = 0. + 1e-5j):
        """Get static correction to model Hamiltonian.

        Parameters
        ----------
        H_MM, S_MM : array_like
            2-D LCAO Hamiltonian and overlap matrices.
        z : complex
            Energy with a small positive immaginary shift.

        """
        w = self.indices  # Alias
        assert w is not None

        Hp_MM = self.rotate_matrix(H_MM, keep_rest=True)
        Sp_MM = self.rotate_matrix(S_MM, keep_rest=True)
        Up_Mw = Sp_MM[:, w].dot(np.linalg.inv(Sp_MM[np.ix_(w, w)]))

        H_ww = self.rotate_matrix(H_MM)
        S_ww = self.rotate_matrix(S_MM)

        # Coupled
        G = np.linalg.inv(z * Sp_MM - Hp_MM)
        # G_inv = np.linalg.inv(rotate_matrix(G, Up_Mw))
        G_inv = np.linalg.inv(Up_Mw.T.conj() @ G @ Up_Mw)
        # Uncoupled
        G0_inv = z * S_ww - H_ww
        # Hybridization
        D0 = G0_inv - G_inv
        return D0.real

    def __len__(self):
        return len(self.indices)


class Subdiagonalization(BasisTransform):
    """Class to perform a subdiagonalization of the Hamiltonian.

    Attributes
    ----------
    blocks : list of array_like
        List of blocks to subdiagonalize.
    H_MM, S_MM : array_like
        2-D LCAO Hamiltonian and overlap matrices.
    U_MM : array_like
        2-D rotation matrix that subdiagonalizes the LCAO Hamiltonian.
    eps_M : array_like
        1-D array of local orbital energies.

    Methods
    -------
    group_energies(round=1)
        Group local orbitals based on energy
    group_symmetries(cutoff=0.9)
        Group local orbitals based on symmetries and energy.
    get_effective_model(indices, ortho=None)
        Builds and effective model from an array of indices.

    """

    def __init__(self,
                 H_MM: Array2D,
                 S_MM: Array2D,
                 blocks: Sequence[Sequence[int]]) -> None:
        """

        Parameters
        ----------
        See class docstring

        """
        self.blocks = blocks
        self.H_MM = H_MM
        self.S_MM = S_MM
        self.eps_M, U_MM = subdiagonalize(
            self.H_MM, self.S_MM, blocks)
        super().__init__(U_MM)
        # Groups of local orbitals with the same symmetry.
        self.groups: dict[float, list[int]] | None = None

    def group_energies(self, decimals: int = 1):
        """Group local orbitals with a similar energy.

        Parameters
        ----------
        decimals : int
            Round energies to the given number of decimals.

        """
        eps = self.eps_M.round(decimals)
        show = np.where(~eps.mask)[0]
        groups = defaultdict(list)
        for index in show:
            groups[eps[index]].append(index)

        self.groups = groups  # type: ignore[assignment]
        return self.groups

    def group_symmetries(self, decimals: int = 1, cutoff: float = 0.9):
        """Group local orbitals with a similar spatial symmetry and energy.

        Parameters
        ----------
        decimals : int
            Round energies to the given number of decimals.
        cutoff : float
            Sets minimum degree of overlap. Can be any value between 0 and 1.

        """
        col_1: list[int] = []  # Keyword.
        col_2: list[int] = []  # Value.
        groups = defaultdict(set)
        blocks = self.blocks
        # Loop over pair of blocks.
        for bb1, bb2 in zip(*np.triu_indices(len(blocks), k=1)):
            b1 = int(bb1)
            b2 = int(bb2)
            if len(blocks[b1]) != len(blocks[b2]):
                # Blocks with different dimensions not compatible.
                continue
            U1 = self.U_MM[np.ix_(blocks[b1], blocks[b1])]
            U2 = self.U_MM[np.ix_(blocks[b2], blocks[b2])]
            # Compute pair overlap between orbitals in the two blocks.
            for o1, o2 in np.ndindex(len(blocks[b1]), len(blocks[b1])):
                v1 = abs(U1[:, o1])
                v2 = abs(U2[:, o2])
                o12 = 2 * v1.dot(v2) / (v1.dot(v1) + v2.dot(v2))
                # Overlap larger than cutoff?
                if o12 >= cutoff:
                    # Yes.
                    i1 = blocks[b1][o1]
                    i2 = blocks[b2][o2]
                    # Use orbital with minimal index as keyword.
                    i1, i2 = min(i1, i2), max(i1, i2)
                    # Check if `i1` is already present in `col_2` and
                    # use corresponding keyword in col_1 instead.
                    present = False
                    for i, i3 in enumerate(col_2):
                        if i1 == i3:
                            present = True
                            break
                    if present:
                        a1 = col_1[i]
                    else:
                        a1 = i1
                    col_1.append(a1)
                    col_2.append(i2)
                    groups[a1].add(i2)
        # Try to further group by energy.
        new: dict[float, list[int]] = defaultdict(list)
        for k, v in groups.items():
            v.add(k)
            new[self.eps_M[k].round(decimals)] += groups[k]
        self.groups = {k: list(sorted(new[k])) for k in sorted(new)}  # groups
        return self.groups

    def get_model(self,
                  indices: Sequence[int],
                  ortho: bool = False) -> EffectiveModel:
        """Extract an effective model from the subdiagonalized space.

        Parameters
        ----------
        indices : array_like
            1-D array of indices to include in the model from
            the new basis.
        ortho : bool, default=False
            Whether to orthogonalize the model basis.
        """
        return EffectiveModel(self.U_MM, indices, self.S_MM if ortho else None)


class LocalOrbitals(TightBinding):
    """Local Orbitals.

    Attributes
    ----------
    TODO

    Methods:
    --------
    subdiagonalize(self, symbols=None, blocks=None, groupby='energy')
        Subdiagonalize the LCAO Hamiltonian.
    take_model(self, indices=None, minimal=True, cutoff=1e-3, ortho=False)
        Take an effective model of local orbitals.
    TODO

    """

    def __init__(self, calc: GPAW):
        self.calc = calc
        self.gamma = calc.wfs.kd.gamma  # Gamma point calculation
        self.subdiag: Subdiagonalization | None = None
        self.model: EffectiveModel | None = None

        if self.gamma:
            self.calc = calc
            h = self.calc.hamiltonian
            wfs = self.calc.wfs
            kpt = wfs.kpt_u[0]

            H_MM = wfs.eigensolver.calculate_hamiltonian_matrix(h, wfs, kpt)
            S_MM = wfs.S_qMM[kpt.q]
            # XXX Converting to full matrices here
            tri2full(H_MM)
            tri2full(S_MM)
            self.H_NMM = H_MM[None, ...] * Hartree  # eV
            self.S_NMM = S_MM[None, ...]
            self.N0 = 0
        else:
            super().__init__(calc.atoms, calc)
            # Bloch to real
            self.H_NMM, self.S_NMM = TightBinding.h_and_s(self)
            self.H_NMM *= Hartree  # eV
            try:
                self.N0 = int(np.argwhere(
                    self.R_cN.T.dot(self.R_cN) < 1e-13).flat[0])
            except Exception as exc:
                raise RuntimeError(
                    "Must include central unit cell, i.e. R=[0,0,0].") from exc

    def subdiagonalize(self,
                       symbols: Array1D = None,
                       blocks: Sequence[Sequence[int]] = None,
                       groupby: str = 'energy'):
        """Subdiagonalize Hamiltonian and overlap matrices.

        Parameters
        ----------
        symbols : array_like, optional
            Element or elements to subdiagonalize.
        blocks : list of array_like, optional
            List of blocks to subdiagonalize.
        groupby : {'energy,'symmetry'}, optional
            Group local orbitals based on energy or
            symmetry and energy. Default is 'energy'.

        """
        if symbols is not None:
            atoms = self.calc.atoms.symbols.search(symbols)
            blocks = [get_bfi(self.calc, [c]) for c in atoms]
        if blocks is None:
            raise RuntimeError("""User must provide either the element(s)
                               or a list of blocks to subdiagonalize.""")
        self.blocks = blocks
        self.subdiag = Subdiagonalization(
            self.H_NMM[self.N0], self.S_NMM[self.N0], blocks)

        self.groupby(groupby)

    def groupby(self,
                method: str = 'energy',
                decimals: int = 1,
                cutoff: float = 0.9):
        """Group local orbitals by symmetry.

        Parameters
        ----------
        method : {'energy,'symmetry'}, optional
            Group local orbitals based on energy or
            symmetry and energy. Default is 'energy'.
        decimals, cutoff : optional
            Parameters passed to the group methods.

        """
        assert self.subdiag is not None
        if method == 'energy':
            self.groups = self.subdiag.group_energies(decimals=decimals)
        elif method == 'symmetry':
            self.groups = self.subdiag.group_symmetries(
                decimals=decimals, cutoff=cutoff)
        else:
            raise RuntimeError(
                f"Invalid method type. {method} not in {'energy', 'symmetry'}")
        # Ensure previous model is invalid.
        self.model = None

    def take_model(self,
                   indices: list[int] = None,
                   minimal: bool = True,
                   cutoff: float = 1e-3,
                   ortho: bool = False):
        """Build an effective model.

        Parameters
        ----------
        indices : array_like
            1-D array of indices to include in the model
            from the new basis.
        minimal : bool, default=True
            Whether to add (minimal=False) or not (minimal=True)
            the orbitals with an overlap larger than `cuoff` with any of the
            orbital specified by `indices`.
        cutoff : float
            Cutoff value for the maximum matrix element connecting a group
            with the minimal model.
        ortho : bool, default=False
            Whether to orthogonalize the model.

        """
        if self.subdiag is None:
            raise RuntimeError("""Not yet subdiagonalized.""")

        eps = self.subdiag.eps_M.round(1)
        indices_from_input = indices is not None

        if indices is None:
            # Find active orbitals with energy closest to Fermi.

            fermi = round(self.calc.get_fermi_level(), 1)
            # diffs = [] # Min distance from fermi for each block
            indices = []  # Min distance index for each block
            for block in self.blocks:
                eb = eps[block]
                ib = np.abs(eb - fermi).argmin()
                indices.append(block[ib])
                # diffs.append(abs(eb[ib]))

        if not minimal:
            # Find orbitals that connect to active with a matrix
            # element larger than cutoff

            # Look at gamma of 1st neighbor
            H_MM = self.H_NMM[(self.N0 + 1) % len(self.H_NMM)]
            H_MM = self.subdiag.rotate_matrix(H_MM)
            # H_MM = dots(self.subdiag.U_MM.T.conj(), H_MM, self.subdiag.U_MM)

            extend = []
            for group in self.groups.values():
                if np.isin(group, indices).any():
                    continue
                if np.abs(H_MM[np.ix_(indices, group)]).max() > cutoff:
                    extend += group

            # Expand model
            indices += extend

        self.indices = indices
        self.model = self.subdiag.get_model(indices, ortho=ortho)

        if self.gamma:
            H_Nww = self.model.rotate_matrix(self.H_NMM[0])[None, ...]
            S_Nww = self.model.rotate_matrix(self.S_NMM[0])[None, ...]

        else:
            # Bypass parent's LCAO construction.
            shape = (self.R_cN.shape[1],) + 2 * (len(self.indices),)
            dtype = self.H_NMM.dtype
            H_Nww = np.empty(shape, dtype)
            S_Nww = np.empty(shape, dtype)

            for N, (H_MM, S_MM) in enumerate(zip(self.H_NMM, self.S_NMM)):
                H_Nww[N] = self.model.rotate_matrix(H_MM)
                S_Nww[N] = self.model.rotate_matrix(S_MM)
        self.H_Nww = H_Nww
        self.S_Nww = S_Nww

        if minimal and not indices_from_input:
            print("Add static correction.")
            # Add static correction of hybridization to minimal model.
            self.H_Nww[self.N0] += self.model.get_static_correction(
                self.H_NMM[self.N0], self.S_NMM[self.N0])

    def h_and_s(self):
        # Hartree units.
        # Bypass TightBinding method.
        eV = 1 / Hartree
        return self.H_Nww * eV, self.S_Nww

    def band_structure(self, path_kc, blochstates=False):
        # Broute force hack to restore matrices.
        H_NMM = self.H_NMM
        S_NMM = self.S_NMM
        ret = TightBinding.band_structure(self, path_kc, blochstates)
        self.H_NMM = H_NMM
        self.S_NMM = S_NMM
        return ret

    def get_hamiltonian(self):
        """Get the Hamiltonian in the home unit cell."""
        return self.H_Nww[self.N0]

    def get_overlap(self):
        """Get the overlap in the home unit cell."""
        return self.S_Nww[self.N0]

    def get_orbitals(self, indices):
        """Get orbitals on the real-space grid."""
        if self.model is None:
            basis = self.subdiag
        else:
            # Maybe model is orthogonal and subdiag does not know.
            basis = self.model
        return get_orbitals(self.calc, basis.U_MM[:, indices])

    def plot_group(self, group):
        return plot2D_orbitals(self, self.groups[group])

    def get_projections(self, q=0):
        P_aMi = {a: P_aqMi[q] for a, P_aqMi in self.calc.wfs.P_aqMi.items()}
        return self.model.rotate_projections(P_aMi)

    def get_xc(self):
        return get_xc(self.calc, self.get_orbitals(), self.get_projections())

    def get_Fcore(self):
        pass


# Plotting tools


def get_plane_dirs(plane):
    """Get normal and in-plane directions for a plane identified
    by any combination of {'x','y','z'}.

    Parameters
    ----------
    plane : str
        Pair of chars identifying the plane, e.g. 'xy'.

    Returns
    -------
    norm_dir : int
        Normal direction
    norm_dir : list
        In-plane directions
    """
    plane_dirs = ['xyz'.index(i) for i in sorted(plane)]
    norm_dir = [i for i in [0, 1, 2] if i not in plane_dirs]
    return norm_dir[0], plane_dirs


def get_atoms(calc, indices):
    """Get the list of atoms corresponding to the given indices.

    Parameters
    ----------
    calc : GPAW
        Calculator
    indices : array_like
        List of orbitals for which to retreive the atoms to which they belong.

    Returns
    -------
    a_list : list
        List of atoms for each index.
    unique : list
        Indices of first occurrences of an atom.
    """
    atoms = calc.atoms
    a_list = []
    unique = []
    for a in range(len(atoms)):
        matches = np.where(np.isin(get_bfi(calc, [a]), indices))[0]
        if len(matches) > 0:
            a_list += [a] * len(matches)
            unique.append(len(a_list) - len(matches))
    return a_list, unique


def plot2D_orbitals(los, indices, plane='yz'):
    """Plot a 2D slice of the orbitals.

    Parameters
    ----------
    los : LocalOrbitals
        Local orbital wrapper
    indices : array_like
        List of orbitals to display
    plane : str, optional
        Pair of chars identifying the plane, by default 'yz'.

    Returns
    -------
    _type_
        _description_
    """

    import matplotlib.pyplot as plt

    norm_dir, plane_dirs = get_plane_dirs(plane)

    calc = los.calc
    atoms = los.calc.atoms

    def get_coord(c):
        return calc.wfs.gd.coords(c, pad=False) * Bohr

    w_wG = los.get_orbitals(indices)

    radii = covalent_radii[calc.atoms.numbers]
    colors = jmol_colors[calc.atoms.numbers]
    pos = atoms.positions

    # Take planes at atomic positions.
    a_list, _ = get_atoms(calc, indices)
    slice_planes = np.searchsorted(get_coord(norm_dir), pos[a_list, norm_dir])

    # Take box limited by external atoms plus 4 Ang vacuum.
    box_lims = [(l - 2, u + 2) for l, u in zip(pos[:, plane_dirs].min(0),
                                               pos[:, plane_dirs].max(0))]
    box_widths = [lims[1] - lims[0] for lims in box_lims]
    ratio = box_widths[0] / box_widths[1]  # Cell ratio
    num_orbs = len(indices)
    max_cols = 6
    nrows = (num_orbs - 1) // max_cols + 1
    ncols = min(num_orbs, max_cols)

    figsize = 5
    fig, axs = plt.subplots(nrows, ncols, figsize=(
        ncols / nrows * ratio * figsize, figsize))

    X, Y = np.meshgrid(get_coord(plane_dirs[0]), get_coord(
        plane_dirs[1]), indexing='ij')
    take_plane = [slice(None)] * 3

    it = np.nditer(axs, flags=['refs_ok', 'c_index', 'multi_index'])
    for _ in it:
        ax = axs[it.multi_index]
        w = np.ravel_multi_index(it.multi_index, axs.shape)
        if w >= num_orbs:
            ax.axis('off')
            continue

        take_plane[norm_dir] = slice_planes[w]
        C = w_wG[w][tuple(take_plane)]

        ax.pcolormesh(X, Y, C, cmap='jet', shading='gouraud')
        ax.set_xlim(box_lims[0])
        ax.set_ylim(box_lims[1])
        ax.axis('off')

        ax.scatter(pos[:, plane_dirs[0]], pos[:, plane_dirs[1]],
                   c=colors, s=radii * 1e3 / (nrows * 2))
    return fig
