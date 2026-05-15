from typing import List, Tuple

import numpy as np
from ase.units import Ha

from gpaw.typing import Array1D, Array2D, Array3D, ArrayLike2D


def parse_hubbard_string(type: str) -> Tuple[str, 'HubbardU']:

    # Parse DFT+U parameters from type-string:
    # Examples: "type:l,U" or "type:l,U,scale"
    type, lus = type.split(':')
    if type == '':
        type = 'paw'

    l = []
    U = []
    scale = []

    for lu in lus.split(';'):  # Multiple U corrections
        l_, u_, scale_ = (lu + ',,').split(',')[:3]
        l.append('spdf'.find(l_))
        U.append(float(u_) / Ha)
        if scale_:
            scale.append(bool(int(scale_)))
        else:
            scale.append(True)
    return type, HubbardU(U, l, scale)


class HubbardU:
    def __init__(self, U, l, scale=1):
        self.scale = scale
        self.U = U
        self.l = l

    def _tuple(self):
        # Tests use this method to compare to expected values
        return (self.l, self.U, self.scale)

    def calculate(self, setup, D_sii):
        eU = 0.
        dHU_sii = np.zeros_like(D_sii)
        for l, U, scale in zip(self.l, self.U, self.scale):
            nl = np.argwhere(np.equal(setup.l_j, l))
            if not (len(nl) == 1 or len(nl) == 2):
                raise RuntimeError(f'Setup has {len(nl)} radial partial waves '
                                   f'with angular momentum quantum number {l}.'
                                   ' Must be 1 or 2 for DFT+U.')
            if (len(nl) == len(np.argwhere(np.equal(np.array(setup.n_j)[nl],
                                                    -1))) and scale == 1):
                raise RuntimeError('DFT+U correction cannot be scaled if '
                                   'there is no bounded partial waves.')

            eU1, dHU1_sii = hubbard(D_sii, U=U, l=l,
                                    l_j=setup.l_j, n_j=setup.n_j,
                                    N0_q=setup.N0_q, scale=scale)
            eU += eU1.real
            dHU_sii += dHU1_sii
        return eU, dHU_sii

    def descriptions(self):
        for U, l, scale in zip(self.U, self.l, self.scale):
            yield f'Hubbard: {{U: {U * Ha},  # eV\n'
            yield f'          l: {l},\n'
            yield f'          scale: {bool(scale)}}}'


def hubbard(D_sii: Array3D,
            U: float,
            l: int,
            l_j: List[int],
            n_j: List[int],
            N0_q: Array1D,
            scale: bool) -> Tuple[float, ArrayLike2D]:
    nspins = len(D_sii)

    # j-indices which have the correct angular momentum quantum number
    nl = np.where(np.equal(l_j, l))[0]

    nm_j = 2 * np.array(l_j) + 1
    nm = nm_j[nl[0]]

    # Get relevant entries of the density matrix
    i1 = slice(nm_j[:nl[0]].sum(), nm_j[:nl[0]].sum() + nm)

    eU = 0.
    dHU_sii = np.zeros_like(D_sii)

    for s, D_ii in enumerate(D_sii):
        N_mm, dHU_ii = aoom(D_ii, l, l_j, n_j, N0_q, scale)
        N_mm = N_mm / 2 * nspins

        if nspins == 4:
            N_mm = N_mm / 2.0
            if s == 0:
                eU1 = U / 2. * (N_mm - 0.5 * N_mm @ N_mm).trace()

                dHU_mm = U / 2. * (np.eye(nm) - N_mm.T)

            else:
                eU1 = -U / 2. * (0.5 * N_mm @ N_mm).trace()

                dHU_mm = -U / 2. * N_mm.T
        else:
            eU1 = U / 2. * (N_mm - N_mm @ N_mm).trace()

            dHU_mm = U / 2. * (np.eye(nm) - 2 * N_mm)

        eU += eU1
        if nspins == 1:
            # Add contribution from other spin manifold
            eU += eU1

        if len(nl) == 1:
            dHU_ii[i1, i1] *= dHU_mm
        elif len(nl) == 2:
            i2 = slice(nm_j[:nl[1]].sum(), nm_j[:nl[1]].sum() + nm)

            dHU_ii[i1, i1] *= dHU_mm
            dHU_ii[i1, i2] *= dHU_mm
            dHU_ii[i2, i1] *= dHU_mm
            dHU_ii[i2, i2] *= dHU_mm
        else:
            raise NotImplementedError

        dHU_sii[s] = dHU_ii

    return eU, dHU_sii


def aoom(D_ii: Array2D,
         l: int,
         l_j: List[int],
         n_j: List[int],
         N0_q: Array1D,
         scale: bool = True) -> Tuple[Array2D, Array2D]:
    """
    This function returns the atomic orbital occupation matrix (aoom) for a
    given l quantum number.

    The submatrix / submatrices of the density matrix (D_ii) for the
    selected l quantum number are determined and summed together which
    represents the orbital occupation matrix. For l=2 this is a 5x5 matrix.

    If scale = True, the inner products are scaled such that the inner product
    of the bounded partial waves is 1.
    """

    # j-indices which have the correct angular momentum quantum number
    nl = np.where(np.equal(l_j, l))[0]

    nm_j = 2 * np.array(l_j) + 1
    nm = nm_j[nl[0]]

    # Get relevant entries of the density matrix
    i1 = slice(nm_j[:nl[0]].sum(), nm_j[:nl[0]].sum() + nm)

    dHU_ii = np.zeros_like(D_ii)
    if len(nl) == 2:
        # First get q-indices for the radial inner products
        q1 = nl[0] * len(l_j) - (nl[0] - 1) * nl[0] // 2  # Bounded-bounded
        q2 = nl[1] * len(l_j) - (nl[1] - 1) * nl[1] // 2  # Unbounded-unbounded
        q12 = q1 + nl[1] - nl[0]  # Bounded-unbounded

        # If the Hubbard correction should be scaled, the three inner products
        # will be divided by the inner product of the bounded partial wave,
        # increasing the value of these inner products since 0 < N0_q[q1] < 1
        if scale:
            if n_j[nl[1]] == -1:
                N1 = 1
                N2 = N0_q[q2] / N0_q[q1]
                N12 = N0_q[q12] / N0_q[q1]
            else:
                N1 = 1
                N2 = 1
                N12 = N0_q[q12] / np.sqrt(N0_q[q1] * N0_q[q2])
        else:
            N1 = N0_q[q1]
            N2 = N0_q[q2]
            N12 = N0_q[q12]

        # Get the entries in the density matrix of the unbounded partial waves
        i2 = slice(nm_j[:nl[1]].sum(), nm_j[:nl[1]].sum() + nm)

        # Scale and add the four submatrices to get the occupation matrix
        N_mm = D_ii[i1, i1] * N1 + D_ii[i2, i2] * N2 + (D_ii[i1, i2]
                                                        + D_ii[i2, i1]) * N12

        dHU_ii[i1, i1] = N1
        dHU_ii[i1, i2] = N12
        dHU_ii[i2, i1] = N12
        dHU_ii[i2, i2] = N2

        return N_mm, dHU_ii
    elif len(nl) == 1:
        q1 = nl[0] * len(l_j) - (nl[0] - 1) * nl[0] // 2
        if scale:
            N1 = 1
        else:
            N1 = N0_q[q1]

        N_mm = D_ii[i1, i1] * N1
        dHU_ii[i1, i1] = N1
        return N_mm, dHU_ii
    else:
        raise NotImplementedError(f'Setup has {len(nl)} partial waves with '
                                  f'angular momentum quantum number {l}. '
                                  'Must be 1 or 2 for DFT+U correction.')
