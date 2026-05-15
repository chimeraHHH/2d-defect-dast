from typing import Any, Dict

import numpy as np
from ase.units import Ha

from gpaw.density import Density
from gpaw.external import NoExternalPotential
from gpaw.pw.hamiltonian import ReciprocalSpaceHamiltonian
from gpaw.typing import Array1D, Array2D, ArrayLike1D


class BField(NoExternalPotential):
    def __init__(self, field: ArrayLike1D):
        """Constant magnetic field.

        field:
            B-field vector in units of eV/bohr-magnoton.
        """
        self.name = 'BField'
        self.field_v = np.array(field) / Ha
        assert self.field_v.shape == (3,)

    def get_potential(self, gd):
        raise NotImplementedError('BField can only be used in PW-mode!')

    def update_potential_pw(self,
                            ham: ReciprocalSpaceHamiltonian,
                            dens: Density) -> float:
        magmom_v, _ = dens.estimate_magnetic_moments()
        eext = -self.field_v.dot(magmom_v)
        if dens.collinear:
            ham.vt_sG[:] = ham.pd2.ifft(ham.vt_Q)
            ham.vt_sG[0] -= self.field_v[2]
            ham.vt_sG[1] += self.field_v[2]
        else:
            ham.vt_xG[0] = ham.pd2.ifft(ham.vt_Q)
            ham.vt_xG[1:] = -self.field_v.reshape((3, 1, 1, 1))
        return eext

    def paw_correction(self, Delta_p: Array1D, dH_sp: Array2D) -> None:
        if len(dH_sp) == 2:
            c = (4 * np.pi)**0.5 * self.field_v[2]
            dH_sp[0] -= c * Delta_p
            dH_sp[1] += c * Delta_p
        else:
            c_vp = (4 * np.pi)**0.5 * self.field_v[:, np.newaxis]
            dH_sp[1:] -= c_vp * Delta_p

    def todict(self) -> Dict[str, Any]:
        return {'name': self.name,
                'field': tuple(self.field_v * Ha)}
