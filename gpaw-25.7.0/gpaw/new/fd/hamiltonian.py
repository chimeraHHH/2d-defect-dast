import numpy as np
from gpaw.core import UGArray
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.fd_operators import Gradient, Laplace
from gpaw.new import zips
from gpaw.new.hamiltonian import Hamiltonian


class FDHamiltonian(Hamiltonian):
    def __init__(self, grid, *, kin_stencil=3, xp=np):
        self.grid = grid
        self._gd = grid._gd
        self.kin = Laplace(self._gd, -0.5, kin_stencil, grid.dtype, xp=xp)

        # For MGGA:
        self.grad_v = []

    def apply_local_potential(self,
                              vt_R: UGArray,
                              psit_nR: XArray,
                              out: XArray,
                              ) -> None:
        assert isinstance(psit_nR, UGArray)
        assert isinstance(out, UGArray)
        self.kin(psit_nR, out)
        for p, o in zips(psit_nR.data, out.data):
            o += p * vt_R.data

    def apply_mgga(self,
                   dedtaut_R: UGArray,
                   psit_nR: XArray,
                   vt_nR: XArray) -> None:
        if len(self.grad_v) == 0:
            grid = psit_nR.desc
            self.grad_v = [
                Gradient(grid._gd, v, n=3, dtype=grid.dtype)
                for v in range(3)]

        tmp_R = psit_nR.desc.empty()
        for psit_R, out_R in zips(psit_nR, vt_nR):
            for grad in self.grad_v:
                grad(psit_R, tmp_R)
                tmp_R.data *= dedtaut_R.data
                grad(tmp_R, tmp_R)
                tmp_R.data *= 0.5
                out_R.data -= tmp_R.data

    def create_preconditioner(self, blocksize, xp=np):
        from types import SimpleNamespace

        from gpaw.preconditioner import Preconditioner as PC
        pc = PC(self._gd, self.kin, self.grid.dtype, blocksize, xp=xp)

        def apply(psit, residuals, out, ekin_n=None):
            kpt = SimpleNamespace(phase_cd=psit.desc.phase_factor_cd)
            pc(residuals.data, kpt, out=out.data)

        return apply

    def calculate_kinetic_energy(self, wfs, skip_sum=False):
        e_kin = 0.0
        for f, psit_R in zip(wfs.myocc_n, wfs.psit_nX):
            if f > 1.0e-10:
                e_kin += f * psit_R.integrate(self.kin(psit_R), skip_sum).real
        if not skip_sum:
            e_kin = psit_R.desc.comm.sum_scalar(e_kin)
            e_kin = wfs.band_comm.sum_scalar(e_kin)
        return e_kin * wfs.spin_degeneracy
