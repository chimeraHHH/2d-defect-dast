from __future__ import annotations

import numpy as np
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.setup import Setups


def move_wave_functions(oldrelpos_ac: np.ndarray,
                        newrelpos_ac: np.ndarray,
                        P_ani: AtomArrays,
                        psit_nX: XArray,
                        setups: Setups) -> None:
    """Move wavefunctions with atoms according to PAW basis

    Wavefunctions are approximated as:::

      ~ _    -- ~a _   ~a  ~
      ψ(r) = >  φ (r) <p | ψ >,
       n     --  i      i   n
             ai

    where i runs over the bound partial-waves only.
    This quantity is then subtracted and re-added at the new
    positions.
    """
    desc = psit_nX.desc
    atomdist = P_ani.layout.atomdist

    # Create partial-wave ACF object (b denotes bound states):
    phit_abX = desc.atom_centered_functions(
        [setup.get_partial_waves_for_atomic_orbitals() for setup in setups],
        oldrelpos_ac,
        atomdist=atomdist,
        cut=True,
        xp=psit_nX.xp)

    P_anb = phit_abX.empty(psit_nX.dims, comm=psit_nX.comm)
    for a, P_nb in P_anb.items():
        P_nb[:] = -P_ani[a][:, :P_nb.shape[1]]

    # Subtract partial wave expansion:
    phit_abX.add_to(psit_nX, P_anb)

    if desc.dtype == complex:
        disp_ac = (newrelpos_ac - oldrelpos_ac).round()
        phase_a = np.exp(2j * np.pi * disp_ac @ desc.kpt_c)
        for a, P_nb in P_anb.items():
            P_nb *= -phase_a[a]
    else:
        P_anb.data *= -1.0

    # Add partial wave expansion at new positions:
    atomdist2 = phit_abX.move(newrelpos_ac, atomdist)
    if atomdist2 is not atomdist:
        P_anb = P_anb.moved(atomdist2)
    phit_abX.add_to(psit_nX, P_anb)
