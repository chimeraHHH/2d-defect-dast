from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from gpaw.core.uniform_grid import UGArray
from gpaw.new import zips
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.new.potential import Potential
from gpaw.typing import Array2D, Array3D
from gpaw.utilities.blas import mmm

Derivatives = Tuple[Array3D,
                    Array3D,
                    Dict[int, Array3D]]


class TCIDerivatives:
    def __init__(self, manytci, atomdist, nao: int):
        self.manytci = manytci
        self.atomdist = atomdist
        self.nao = nao

        self._derivatives_q: dict[int, Derivatives] = {}

    def calculate_derivatives(self,
                              q: int) -> Derivatives:
        if not self._derivatives_q:
            dThetadR_qvMM, dTdR_qvMM = self.manytci.O_qMM_T_qMM(
                self.atomdist.comm,
                0, self.nao,
                False, derivative=True)

            self.atomdist.comm.sum(dThetadR_qvMM)
            self.atomdist.comm.sum(dTdR_qvMM)

            dPdR_aqvMi = self.manytci.P_aqMi(
                self.atomdist.indices,
                derivative=True)

            dPdR_qavMi = [{a: dPdR_qvMi[q]
                           for a, dPdR_qvMi in dPdR_aqvMi.items()}
                          for q in range(len(dThetadR_qvMM))]

            self._derivatives_q = {
                q: (dThetadR_vMM, dTdR_vMM, dPdR_avMi)
                for q, (dThetadR_vMM, dTdR_vMM, dPdR_avMi)
                in enumerate(zips(dThetadR_qvMM, dTdR_qvMM, dPdR_qavMi))}

        return self._derivatives_q[q]


def add_force_contributions(wfs: LCAOWaveFunctions,
                            potential: Potential,
                            F_av: Array2D) -> None:
    (dThetadR_vMM,
     dTdR_vMM,
     dPdR_avMi) = wfs.tci_derivatives.calculate_derivatives(wfs.q)

    indices = []
    M1 = 0
    for a, sphere in enumerate(wfs.basis.sphere_a):
        M2 = M1 + sphere.Mmax
        indices.append((a, M1, M2))
        M1 = M2

    setups = wfs.setups

    my_row_range = wfs.T_MM.dist.my_row_range()
    m1, m2 = my_row_range
    rhoT_MM = wfs.calculate_density_matrix(transposed=True)
    erhoT_MM = wfs.calculate_density_matrix(transposed=True, eigs=True)
    add_kinetic_term(rhoT_MM, dTdR_vMM[:, m1:m2], F_av,
                     indices, wfs.atomdist.indices, my_row_range)
    add_pot_term(potential.vt_sR[wfs.spin], wfs.basis, wfs.q, rhoT_MM, F_av)
    add_den_mat_term(erhoT_MM, dThetadR_vMM[:, m1:m2], F_av, indices,
                     wfs.atomdist.indices, my_row_range)

    for b in wfs.atomdist.indices:
        add_den_mat_paw_term(b,
                             setups[b].dO_ii,
                             wfs.P_aMi[b],
                             dPdR_avMi[b],
                             erhoT_MM,
                             indices,
                             F_av,
                             my_row_range)

        add_atomic_density_term(b,
                                potential.dH_asii[b][wfs.spin],
                                wfs.P_aMi[b],
                                dPdR_avMi[b],
                                rhoT_MM,
                                indices,
                                F_av,
                                my_row_range)


def add_kinetic_term(rhoT_MM, dTdR_vMM, F_av, indices, mya, my_row_range):
    """Calculate Kinetic energy term in LCAO

    :::

                      dT
     _a        --- --   μν
     F += 2 Re >   >  ---- ρ
               --- --  _    νμ
               μ=a ν  dR
                        μν
            """

    for a, M1, M2 in indices:
        if a in mya and M2 >= my_row_range[0] and M1 <= my_row_range[1]:
            m1 = max(M1, my_row_range[0]) - my_row_range[0]
            m2 = min(M2, my_row_range[1]) - my_row_range[0]
            F_av[a, :] += 2 * np.einsum('vmM, mM -> v',
                                        dTdR_vMM[:, m1:m2],
                                        rhoT_MM[m1:m2]).real


def add_pot_term(vt_R: UGArray,
                 basis,
                 q: int,
                 rhoT_MM,
                 F_av) -> None:
    """Calculate potential term"""
    # Potential contribution
    #
    #           -----      /  d Phi  (r)
    #  a         \        |        mu    ~
    # F += -2 Re  )       |   ---------- v (r)  Phi  (r) dr rho
    #            /        |     d R                nu          nu mu
    #           -----    /         a
    #        mu in a; nu
    #
    F_av += basis.calculate_force_contribution(vt_R.data,
                                               rhoT_MM,
                                               q)


def add_den_mat_term(erhoT_MM, dThetadR_vMM, F_av, indices, mya, my_row_range):
    """Calculate density matrix term in LCAO"""
    # Density matrix contribution due to basis overlap
    #
    #            ----- d Theta
    #  a          \           mu nu
    # F  += -2 Re  )   ------------  E
    #             /        d R        nu mu
    #            -----        mu nu
    #         mu in a; nu
    #
    for a, M1, M2 in indices:
        if a in mya and M2 >= my_row_range[0] and M1 <= my_row_range[1]:
            m1 = max(M1, my_row_range[0]) - my_row_range[0]
            m2 = min(M2, my_row_range[1]) - my_row_range[0]
            F_av[a, :] -= 2 * np.einsum('vmM, mM -> v',
                                        dThetadR_vMM[:, m1:m2],
                                        erhoT_MM[m1:m2]).real


def add_den_mat_paw_term(b, dO_ii, P_Mi, dPdR_vMi, erhoT_MM,
                         indices, F_av, my_row_range):
    """Calcualte PAW correction"""
    # TO DO: split this function into
    # _get_den_mat_paw_term (which calculate Frho_av) and
    # get_paw_correction (which calculate ZE_MM)
    # Density matrix contribution from PAW correction
    #
    #           -----                        -----
    #  a         \      a                     \     b
    # F +=  2 Re  )    Z      E        - 2 Re  )   Z      E
    #            /      mu nu  nu mu          /     mu nu  nu mu
    #           -----                        -----
    #           mu nu                    b; mu in a; nu
    #
    # with
    #                  b*
    #         -----  dP
    #   b      \       i mu    b   b
    #  Z     =  )   -------- dS   P
    #   mu nu  /     dR        ij  j nu
    #         -----    b mu
    #           ij
    #
    m1, m2 = my_row_range
    dtype = P_Mi.dtype
    Z_MM = np.zeros((m2 - m1, len(P_Mi)), dtype)
    dOP_iM = np.zeros((len(dO_ii), len(P_Mi)), dtype)
    mmm(1.0, dO_ii.astype(dtype), 'N', P_Mi, 'C', 0.0, dOP_iM)
    for v in range(3):
        mmm(1.0,
            dPdR_vMi[v, m1:m2], 'N',
            dOP_iM, 'N',
            0.0, Z_MM)
        ZE_M = np.einsum('MN, MN -> M', Z_MM, erhoT_MM).real
        for a, M1, M2 in indices:
            if M2 >= my_row_range[0] and M1 <= my_row_range[1]:
                am1 = max(M1, my_row_range[0]) - my_row_range[0]
                am2 = min(M2, my_row_range[1]) - my_row_range[0]
                dE = 2 * ZE_M[am1:am2].sum()
                F_av[a, v] -= dE  # the "b; mu in a; nu" term
                F_av[b, v] += dE  # the "mu nu" term


def add_atomic_density_term(b, dH_ii, P_Mi, dPdR_vMi, rhoT_MM,
                            indices, F_av, my_row_range):
    # Atomic density contribution
    #            -----                         -----
    #  a          \     a                       \     b
    # F  += -2 Re  )   A      rho       + 2 Re   )   A      rho
    #             /     mu nu    nu mu          /     mu nu    nu mu
    #            -----                         -----
    #            mu nu                     b; mu in a; nu
    #
    #                  b*
    #         ----- d P
    #  b       \       i mu   b   b
    # A     =   )   ------- dH   P
    #  mu nu   /    d R       ij  j nu
    #         -----    b mu
    #           ij
    #
    m1, m2 = my_row_range
    dtype = P_Mi.dtype
    A_MM = np.zeros((m2 - m1, len(P_Mi)), dtype)
    dHP_iM = np.zeros((len(dH_ii), len(P_Mi)), dtype)
    mmm(1.0, dH_ii.astype(dtype), 'N', P_Mi, 'C', 0.0, dHP_iM)
    for v in range(3):
        mmm(1.0,
            dPdR_vMi[v, m1:m2], 'N',
            dHP_iM, 'N',
            0.0, A_MM)
        AR_M = np.einsum('MN, MN -> M', A_MM, rhoT_MM).real
        for a, M1, M2 in indices:
            if M2 >= my_row_range[0] and M1 <= my_row_range[1]:
                am1 = max(M1, my_row_range[0]) - my_row_range[0]
                am2 = min(M2, my_row_range[1]) - my_row_range[0]
                dE = 2 * AR_M[am1:am2].sum()
                F_av[a, v] += dE  # the "b; mu in a; nu" term
                F_av[b, v] -= dE  # the "mu nu" term
