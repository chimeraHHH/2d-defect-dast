from __future__ import annotations

from typing import Dict

import numpy as np

from gpaw.typing import Array3D, Array4D

_gaunt: Dict[int, np.ndarray] = {}
_nabla: Dict[int, np.ndarray] = {}
_super_gaunt: Dict[int, np.ndarray] = {}


def gaunt(lmax: int = 2) -> Array3D:
    r"""Gaunt coefficients

    :::

         ^      ^     -- L      ^
      Y (r)  Y (r) =  > G    Y (r)
       L      L       -- L L  L
        1      2      L   1 2
    """

    if lmax in _gaunt:
        return _gaunt[lmax]

    Lmax = (lmax + 1)**2
    L2max = (2 * lmax + 1)**2

    from gpaw.spherical_harmonics import YL, gam
    G_LLL = np.zeros((Lmax, L2max, L2max))
    for L1 in range(Lmax):
        for L2 in range(L2max):
            for L in range(L2max):
                r = 0.0
                for c1, n1 in YL[L1]:
                    for c2, n2 in YL[L2]:
                        for c, n in YL[L]:
                            nx = n1[0] + n2[0] + n[0]
                            ny = n1[1] + n2[1] + n[1]
                            nz = n1[2] + n2[2] + n[2]
                            r += c * c1 * c2 * gam(nx, ny, nz)
                G_LLL[L1, L2, L] = r
    _gaunt[lmax] = G_LLL
    return G_LLL


def nabla(lmax: int = 2) -> Array3D:
    """Create the array of derivative integrals.

    :::

      /  ^    ^   1-l' d   l'    ^
      | dr Y (r) r     --[r  Y  (r)]
      /     L           _     L'
                       dr
    """

    if lmax in _nabla:
        return _nabla[lmax]

    Lmax = (lmax + 1)**2
    from gpaw.spherical_harmonics import YL, gam
    Y_LLv = np.zeros((Lmax, Lmax, 3))
    # Insert new values
    for L1 in range(Lmax):
        for L2 in range(Lmax):
            for v in range(3):
                r = 0.0
                for c1, n1 in YL[L1]:
                    for c2, n2 in YL[L2]:
                        n = [0, 0, 0]
                        n[0] = n1[0] + n2[0]
                        n[1] = n1[1] + n2[1]
                        n[2] = n1[2] + n2[2]
                        if n2[v] > 0:
                            # apply derivative
                            n[v] -= 1
                            # add integral
                            r += n2[v] * c1 * c2 * gam(n[0], n[1], n[2])
                Y_LLv[L1, L2, v] = r
    _nabla[lmax] = Y_LLv
    return Y_LLv


def super_gaunt(lmax: int = 2) -> Array4D:
    r"""Product of two Gaunt coefficients.

    This gives the coefficients to contractions of three spherical harmonics:
                        __
       ˰     ˰     ˰    \   L       ˰
    Y (r) Y (r) Y (r) = /  G      Y (r)
     L     L     L      ‾‾  L L L  L
      1     2     3     L    1 2 3

    where:
              __
     L        \   L'    L
    G       = /  G     G
     L L L    ‾‾  L L   L' L
      1 2 3   L'   1 2      3
    """
    if lmax in _super_gaunt:
        return _super_gaunt[lmax]

    G1_LLL = gaunt(lmax)
    G2_LLL = gaunt(2 * lmax)

    Lmax1 = G1_LLL.shape[0]  # l=0 to l=lmax
    Lmax2 = G2_LLL.shape[0]  # l=0 to l=2*lmax
    G_LLLL = np.einsum('ijk,klm->ijlm',
                       G1_LLL[:, :Lmax1],
                       G2_LLL[:, :Lmax2])

    _super_gaunt[lmax] = G_LLLL
    return G_LLLL
