import numpy as np

from gpaw.new.eigensolver import Eigensolver, calculate_weights
from gpaw.new.lcao.hamiltonian import HamiltonianMatrixCalculator
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.new.energies import DFTEnergies
from gpaw.core.matrix import MatrixWithNoData


class LCAOEigensolver(Eigensolver):
    def __init__(self,
                 basis,
                 converge_bands='occupied'):
        self.basis = basis
        self.converge_bands = converge_bands

    def iterate(self,
                ibzwfs,
                density,
                potential,
                hamiltonian,
                pot_calc=None,
                energies=None) -> tuple[float, float, DFTEnergies]:
        matrix_calculator = hamiltonian.create_hamiltonian_matrix_calculator(
            potential)

        weight_un = calculate_weights(self.converge_bands, ibzwfs)
        eig_error = 0.0
        for wfs, weight_n in zip(ibzwfs, weight_un):
            _, temp_eig_error = \
                self.iterate_kpt(wfs, weight_n, self.iterate1,
                                 matrix_calculator=matrix_calculator)
            if eig_error < temp_eig_error:
                eig_error = temp_eig_error

        eig_error = ibzwfs.kpt_band_comm.max_scalar(eig_error)
        return eig_error, 0.0, energies

    def iterate1(self,
                 wfs: LCAOWaveFunctions,
                 weight_n: np.ndarray,  # XXX: Unused
                 matrix_calculator: HamiltonianMatrixCalculator):
        H_MM = matrix_calculator.calculate_matrix(wfs)
        eig_M = H_MM.eighg(wfs.L_MM, wfs.domain_comm)
        C_Mn = H_MM  # rename (H_MM now contains the eigenvectors)
        assert len(eig_M) >= wfs.nbands
        N = wfs.nbands
        wfs._eig_n = np.empty(wfs.nbands)
        wfs._eig_n[:] = eig_M[:N]
        comm = C_Mn.dist.comm
        if isinstance(wfs.C_nM, MatrixWithNoData):
            wfs.C_nM = wfs.C_nM.create()
        if comm.size == 1:
            wfs.C_nM.data[:] = C_Mn.data.T[:N]
        else:
            C_Mn = C_Mn.gather(broadcast=True)
            n1, n2 = wfs.C_nM.dist.my_row_range()
            wfs.C_nM.data[:] = C_Mn.data.T[n1:n2]

        # Make sure wfs.C_nM and (lazy) wfs.P_ani are in sync:
        wfs._P_ani = None
