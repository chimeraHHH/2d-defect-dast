from math import pi
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.density import Density
from gpaw.core.matrix import MatrixWithNoData


class LCAOIBZWaveFunctions(IBZWaveFunctions):
    def has_wave_functions(self):
        return not isinstance(self.wfs_qs[0][0].C_nM, MatrixWithNoData)

    def move(self, relpos_ac, atomdist):
        from gpaw.new.lcao.builder import tci_helper

        super().move(relpos_ac, atomdist)

        for wfs in self:
            basis = wfs.basis
            setups = wfs.setups
            break
        basis.set_positions(relpos_ac)
        myM = (basis.Mmax + self.band_comm.size - 1) // self.band_comm.size
        basis.set_matrix_distribution(
            min(self.band_comm.rank * myM, basis.Mmax),
            min((self.band_comm.rank + 1) * myM, basis.Mmax))

        S_qMM, T_qMM, P_qaMi, tciexpansions, tci_derivatives = tci_helper(
            basis, self.ibz, self.domain_comm, self.band_comm, self.kpt_comm,
            relpos_ac, atomdist,
            self.grid, self.dtype, setups)

        for wfs in self:
            wfs.tci_derivatives = tci_derivatives
            wfs.S_MM = S_qMM[wfs.q]
            wfs.T_MM = T_qMM[wfs.q]
            wfs.P_aMi = P_qaMi[wfs.q]

    def normalize_density(self, density: Density) -> None:
        """Normalize density.

        Basis functions may extend outside box!
        """
        pseudo_charge = density.nt_sR.integrate().sum()
        ccc_aL = density.calculate_compensation_charge_coefficients()
        comp_charge = (4 * pi)**0.5 * sum(float(ccc_L[0])
                                          for ccc_L in ccc_aL.values())
        comp_charge = ccc_aL.layout.atomdist.comm.sum_scalar(comp_charge)
        density.nt_sR.data *= -comp_charge / pseudo_charge
