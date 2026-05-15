from __future__ import annotations

import numpy as np
from gpaw.core.matrix import Matrix
from gpaw.external import ExternalPotential
from gpaw.lfc import BasisFunctions
from gpaw.new import zips
from gpaw.new.calculation import DFTState
from gpaw.new.fd.pot_calc import FDPotentialCalculator
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.typing import Array2D, Array3D


class HamiltonianMatrixCalculator:
    def calculate_matrix(self,
                         wfs: LCAOWaveFunctions) -> Matrix:
        raise NotImplementedError


class CollinearHamiltonianMatrixCalculator(HamiltonianMatrixCalculator):
    def __init__(self,
                 V_sxMM: list[np.ndarray],
                 dH_saii: list[dict[int, Array2D]],
                 basis: BasisFunctions,
                 include_kinetic: bool = True):
        self.V_sxMM = V_sxMM
        self.dH_saii = dH_saii
        self.basis = basis
        self.include_kinetic = include_kinetic

    def calculate_matrix(self,
                         wfs: LCAOWaveFunctions) -> Matrix:
        if self.include_kinetic:
            return self._calculate_matrix_with_kinetic(wfs)
        return self._calculate_matrix_without_kinetic(wfs)

    def _calculate_potential_matrix(self,
                                    wfs: LCAOWaveFunctions,
                                    V_xMM: Array3D) -> Matrix:
        data = V_xMM[0]
        _, M = data.shape
        if wfs.dtype == complex:
            data = data.astype(complex)
        V_MM = Matrix(M, M, data=data, dist=(wfs.band_comm, -1, 1))
        if wfs.dtype == complex:
            phase_x = np.exp(-2j * np.pi *
                             self.basis.sdisp_xc[1:] @ wfs.kpt_c)
            V_MM.data += np.einsum('x, xMN -> MN',
                                   2 * phase_x, V_xMM[1:],
                                   optimize=True)
        return V_MM

    def _calculate_matrix_without_kinetic(self,
                                          wfs: LCAOWaveFunctions,
                                          V_xMM: Array3D = None,
                                          dH_aii: dict[int, Array2D] = None
                                          ) -> Matrix:
        if V_xMM is None:
            V_xMM = self.V_sxMM[wfs.spin]
        if dH_aii is None:
            dH_aii = self.dH_saii[wfs.spin]

        V_MM = self._calculate_potential_matrix(wfs, V_xMM)

        M1, M2 = V_MM.dist.my_row_range()
        for a, dH_ii in dH_aii.items():
            P_Mi = wfs.P_aMi[a]
            V_MM.data += (P_Mi[M1:M2] @ dH_ii).conj() @ P_Mi.T  # XXX use gemm

        return V_MM

    def _calculate_matrix_with_kinetic(self,
                                       wfs: LCAOWaveFunctions) -> Matrix:
        H_MM = self._calculate_matrix_without_kinetic(wfs)
        H_MM.data += wfs.T_MM.data

        wfs.domain_comm.sum(H_MM.data, 0)

        if wfs.domain_comm.rank == 0:
            if wfs.dtype == complex:
                H_MM.add_hermitian_conjugate(scale=0.5)
            else:
                H_MM.tril2full()
        return H_MM


class NonCollinearHamiltonianMatrixCalculator(HamiltonianMatrixCalculator):
    def __init__(self, matcalc: CollinearHamiltonianMatrixCalculator):
        self.matcalc = matcalc

    def calculate_matrix(self,
                         wfs: LCAOWaveFunctions) -> Matrix:
        V_sMM = [
            self.matcalc._calculate_matrix_without_kinetic(wfs, V_xMM, dH_aii)
            for V_xMM, dH_aii in zips(self.matcalc.V_sxMM,
                                      self.matcalc.dH_saii)]

        V_sMM[0] += wfs.T_MM

        assert wfs.domain_comm.size == 1

        for V_MM in V_sMM:
            wfs.domain_comm.sum(V_MM.data, 0)

            if wfs.domain_comm.rank == 0:
                if wfs.dtype == complex:
                    V_MM.add_hermitian_conjugate(scale=0.5)
                else:
                    V_MM.tril2full()

        _, M = V_MM.shape
        v_MM, x_MM, y_MM, z_MM = (V_MM.data for V_MM in V_sMM)
        H_sMsM = Matrix(2 * M, 2 * M, dtype=complex, dist=(wfs.band_comm,))
        H_sMsM.data[:M, :M] = v_MM + z_MM
        H_sMsM.data[:M, M:] = x_MM + 1j * y_MM
        H_sMsM.data[M:, :M] = x_MM - 1j * y_MM
        H_sMsM.data[M:, M:] = v_MM - z_MM
        return H_sMsM


class LCAOHamiltonian(Hamiltonian):
    def __init__(self,
                 basis: BasisFunctions):
        self.basis = basis

    def create_hamiltonian_matrix_calculator(self,
                                             potential,
                                             ) -> HamiltonianMatrixCalculator:
        V_sxMM = [self.basis.calculate_potential_matrices(vt_R.data)
                  for vt_R in potential.vt_sR.to_xp(np)]

        dH_saii = [{a: dH_sii[s]
                    for a, dH_sii
                    in potential.dH_asii.to_xp(np).items()}
                   for s in range(len(V_sxMM))]

        matcalc = CollinearHamiltonianMatrixCalculator(V_sxMM, dH_saii,
                                                       self.basis,
                                                       include_kinetic=True)
        if len(V_sxMM) < 4:
            return matcalc

        return NonCollinearHamiltonianMatrixCalculator(matcalc)

    def create_kick_matrix_calculator(self,
                                      state: DFTState,
                                      ext: ExternalPotential,
                                      pot_calc: FDPotentialCalculator
                                      ) -> HamiltonianMatrixCalculator:
        from gpaw.utilities import unpack_hermitian
        vext_r = pot_calc.vbar_r.new()
        finegd = vext_r.desc._gd

        vext_r.data = ext.get_potential(finegd)
        vext_R = pot_calc.restrict(vext_r)

        nspins = state.ibzwfs.nspins

        V_MM = self.basis.calculate_potential_matrices(vext_R.data)
        V_sxMM = [V_MM for s in range(nspins)]

        W_aL = pot_calc.ghat_aLr.integrate(vext_r)

        assert state.ibzwfs.ibz.bz.gamma_only
        setups_a = state.ibzwfs.wfs_qs[0][0].setups

        dH_saii = [{a: unpack_hermitian(setups_a[a].Delta_pL @ W_L)
                    for (a, W_L) in W_aL.items()}
                   for s in range(nspins)]

        return CollinearHamiltonianMatrixCalculator(V_sxMM, dH_saii,
                                                    self.basis,
                                                    include_kinetic=False)
