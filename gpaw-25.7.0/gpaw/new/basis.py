import numpy as np
from gpaw import GPAW_NO_C_EXTENSION
from gpaw.core import PWDesc, UGDesc
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.lfc import BasisFunctions
from gpaw.mpi import serial_comm
from gpaw.new.brillouin import IBZ


def create_basis(ibz: IBZ,
                 nspins,
                 pbc_c,
                 grid,
                 setups,
                 dtype,
                 relpos_ac,
                 comm=serial_comm,
                 kpt_comm=serial_comm,
                 band_comm=serial_comm):
    kd = KPointDescriptor(ibz.bz.kpt_Kc, nspins)

    kd.ibzk_kc = ibz.kpt_kc
    kd.weight_k = ibz.weight_k
    kd.sym_k = ibz.s_K
    kd.time_reversal_k = ibz.time_reversal_K
    kd.bz2ibz_k = ibz.bz2ibz_K
    kd.ibz2bz_k = ibz.ibz2bz_k
    kd.bz2bz_ks = ibz.bz2bz_Ks
    kd.nibzkpts = len(ibz)
    kd.symmetry = ibz.symmetries._old_symmetry
    kd.set_communicator(kpt_comm)
    if GPAW_NO_C_EXTENSION:
        return SimpleBasis(grid, setups, relpos_ac)
    basis_dtype = complex if \
        np.issubdtype(dtype, np.complexfloating) else float
    basis = BasisFunctions(grid._gd,
                           [setup.basis_functions_J for setup in setups],
                           kd,
                           dtype=basis_dtype,
                           cut=True)
    basis.set_positions(relpos_ac)
    myM = (basis.Mmax + band_comm.size - 1) // band_comm.size
    basis.set_matrix_distribution(
        min(band_comm.rank * myM, basis.Mmax),
        min((band_comm.rank + 1) * myM, basis.Mmax))
    return basis


class SimpleBasis:
    def __init__(self,
                 grid: UGDesc,
                 setups,
                 relpos_ac):
        self.grid = grid
        self.pw = PWDesc(cell=grid.cell,
                         ecut=min(12.5, grid.ekin_max()))
        self.phit_aIG = self.pw.atom_centered_functions(
            [setup.basis_functions_J for setup in setups],
            relpos_ac)

    def add_to_density(self,
                       nt_sR: np.ndarray,
                       f_asi):
        nI = sum(f_si.shape[1] for f_si in f_asi.values())
        c_aiI = self.phit_aIG.empty(nI)
        c_aiI.data[:] = np.eye(nI)
        phit_IG = self.pw.zeros(nI)
        self.phit_aIG.add_to(phit_IG, c_aiI)
        I = 0
        for f_si in f_asi.values():
            for f_s in f_si.T:
                phit_R = phit_IG[I].ifft(grid=self.grid)
                nt_sR += f_s[:, np.newaxis, np.newaxis, np.newaxis] * (
                    phit_R.data**2)
                I += 1
