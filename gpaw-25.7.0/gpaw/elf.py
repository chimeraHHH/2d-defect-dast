"""This module defines an ELF function."""
from __future__ import annotations

import sys

import numpy as np
from gpaw.core import UGArray
from gpaw.fd_operators import Gradient
from gpaw.new.ase_interface import GPAW, ASECalculator
from gpaw.new.calculation import DFTCalculation


def elf(nt_sR: np.ndarray,
        nt_grad2_sR: np.ndarray,
        taut_sR: np.ndarray,
        ncut: float | None = None) -> np.ndarray:
    """Pseudo electron localisation function (ELF).

    See:

      Becke and Edgecombe, J. Chem. Phys., vol 92 (1990) 5397

    More comprehensive definition in
    M. Kohout and A. Savin, Int. J. Quantum Chem., vol 60 (1996) 875-882

    Parameters
    ==========
    nt_sR:
        Pseudo valence density.
    nt_grad2_sR:
        Squared norm of the density gradient.
    taut_sR:
        Kinetic energy density.
    ncut:
        Minimum density cutoff parameter.

    Returns
    =======
    np.ndarray:
        Array of ELF values.
    """

    # Fermi constant
    cF = 3.0 / 10 * (3 * np.pi**2)**(2 / 3)

    eps = 1e-11
    nt_sR = nt_sR.copy()
    nt_sR[nt_sR < eps] = eps

    if nt_sR.shape[0] == 2:
        # Kouhut eq. (9)
        D0 = 2**(2 / 3) * cF * (nt_sR[0]**(5 / 3) +
                                nt_sR[1]**(5 / 3))

        taut = taut_sR.sum(axis=0)
        D = taut - (nt_grad2_sR[0] / nt_sR[0] + nt_grad2_sR[1] / nt_sR[1]) / 8
    else:
        # Kouhut eq. (7)
        D0 = cF * nt_sR[0]**(5 / 3)
        taut = taut_sR[0]
        D = taut - nt_grad2_sR[0] / nt_sR[0] / 8

    elf_R = 1.0 / (1.0 + (D / D0)**2)

    if ncut is not None:
        nt = nt_sR.sum(axis=0)
        elf_R[nt < ncut] = 0.0

    return elf_R


def elf_from_dft_calculation(dft: DFTCalculation | ASECalculator,
                             ncut: float = 1e-6) -> UGArray:
    """Calculate the electronic localization function.

    Parameters
    ==========
    dft:
        DFT-calculation object.
    ncut:
        Density cutoff below which the ELF is zero.

    Returns
    =======
    UGArray:
        ELF values.
    """
    if isinstance(dft, ASECalculator):
        dft = dft.dft
    density = dft.density
    density.update_ked(dft.ibzwfs)
    taut_sR = density.taut_sR
    assert taut_sR is not None
    nt_sR = density.nt_sR
    grad_v = [Gradient(nt_sR.desc._gd, v, n=2) for v in range(3)]
    gradnt2_sR = nt_sR.new(zeroed=True)
    for gradnt2_R, nt_R in zip(gradnt2_sR, nt_sR):
        for grad in grad_v:
            gradnt_R = grad(nt_R)
            gradnt2_R.data += gradnt_R.data**2
    elf_R = nt_sR.desc.empty()
    elf_R.data[:] = elf(
        nt_sR.data, gradnt2_sR.data, taut_sR.data, ncut)
    return elf_R


if __name__ == '__main__':
    e_R = elf_from_dft_calculation(GPAW(sys.argv[1]).dft, 0.001)
    e_R.isosurface(isomin=0.8, isomax=0.8)
