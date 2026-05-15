from __future__ import annotations

import numpy as np
from ase.units import Ha
from gpaw.new.extensions import Extension
from gpaw.typing import Array1D, Array3D, Vector


class SpinDirectionConstraint(Extension):
    name = 'spin_direction_constraint'

    def __init__(self,
                 constraint: dict[int, Vector],
                 penalty: float = 0.8):
        """Spin-direction constraint.

        Parameters
        ==========
        constraint:
            Dictionary mapping atom numbers to directions.
            Example: ``{0: (0, 0, 1), 1: (1, 0, 0), ...}``.
        penalty:
            Strength of penalty term in eV.
        """
        self.constraint = {a: np.array(u_v) / np.linalg.norm(u_v)
                           for a, u_v in constraint.items()}
        self.penalty = penalty / Ha

    def todict(self):
        return dict(constraint=dict((a, u_v.tolist())
                                    for a, u_v in self.constraint.items()),
                    penalty=self.penalty * Ha)

    def update_non_local_hamiltonian(self,
                                     D_sii,
                                     setup,
                                     atom_index,
                                     dH_sii) -> float:
        eL, dHL_vii = self.calculate(D_sii[1:4].real, atom_index,
                                     setup.l_j, setup.N0_q)
        dH_sii[1:4] += dHL_vii
        return eL

    def calculate(self,
                  M_vii: Array3D,
                  a: int,
                  l_j: Array1D,
                  N0_q: Array1D,
                  return_energy: bool = False):
        dHL_vii = np.zeros_like(M_vii)

        if a not in self.constraint:
            return 0.0, dHL_vii
        u_v = self.constraint[a]

        smm_v = np.zeros(3)  # Spin magnetic moment

        nj = len(l_j)
        i1 = slice(0, 0)
        for j1, l1 in enumerate(l_j):
            i1 = slice(i1.stop, i1.stop + 2 * l1 + 1)
            i2 = slice(0, 0)
            for j2, l2 in enumerate(l_j):
                i2 = slice(i2.stop, i2.stop + 2 * l2 + 1)
                if l1 != l2:
                    continue
                N0 = N0_q[(j2 + j1 * nj - j1 * (j1 + 1) // 2
                           if j1 < j2 else
                           j1 + j2 * nj - j2 * (j2 + 1) // 2)]

                smm_v += np.sum(M_vii[:, i1, i2], axis=(1, 2)) * N0
                dHL_vii[:, i1, i2] += np.eye(2 * l1 + 1) * N0

        for v in range(3):
            dHL_vii[v] *= (1 - u_v[v]**2) * smm_v[v] - u_v[v] * (
                u_v[(v + 1) % 3] * smm_v[(v + 1) % 3]
                + u_v[(v + 2) % 3] * smm_v[(v + 2) % 3])
        dHL_vii *= 2 * self.penalty

        if not return_energy:
            return 0.0, dHL_vii
        else:
            return self.penalty * (smm_v @ smm_v - (u_v @ smm_v)**2), dHL_vii
