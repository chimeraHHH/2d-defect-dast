from __future__ import annotations

import numpy as np
from gpaw.core import UGArray
from gpaw.core.arrays import DistributedArrays as XArray


class Hamiltonian:
    band_local = True
    # Used for knowing if wavefunctions can be sliced
    # along bands, when applying the hamiltonian.

    def apply(self,
              vt_sR: UGArray,
              dedtaut_sR: UGArray | None,
              ibzwfs,
              D_asii,
              psit_nG: XArray,
              out: XArray,
              spin: int,
              calculate_energy: bool = False) -> XArray:
        self.apply_local_potential(vt_sR[spin], psit_nG, out)
        if dedtaut_sR is not None:
            self.apply_mgga(dedtaut_sR[spin], psit_nG, out)
        self.apply_orbital_dependent(ibzwfs, D_asii, psit_nG, spin, out,
                                     calculate_energy)
        return out

    def apply_local_potential(self,
                              vt_R: UGArray,
                              psit_nG: XArray,
                              out: XArray) -> None:
        raise NotImplementedError

    def apply_mgga(self,
                   dedtaut_R: UGArray,
                   psit_nG: XArray,
                   vt_nG: XArray) -> None:
        raise NotImplementedError

    def apply_orbital_dependent(self,
                                ibzwfs,
                                D_asii,
                                psit_nG: XArray,
                                spin: int,
                                out: XArray,
                                calculate_energy: bool) -> None:
        pass

    def create_preconditioner(self, blocksize, xp=np):
        raise NotImplementedError

    def update_wave_functions(self, ibzwfs):
        return
