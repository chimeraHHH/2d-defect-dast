from __future__ import annotations

import numpy as np
from ase.units import Bohr, Ha

from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core.atom_arrays import AtomArrays, AtomDistribution
from gpaw.core.domain import Domain as XDesc
from gpaw.core import PWArray, UGArray, UGDesc
from gpaw.mpi import MPIComm, broadcast_float
from gpaw.new import zips


class Potential:
    def __init__(self,
                 vt_sR: UGArray,
                 dH_asii: AtomArrays,
                 dedtaut_sR: UGArray | None,
                 vHt_x: XArray | None = None,
                 e_stress: float = np.nan):
        self.vt_sR = vt_sR
        self.dH_asii = dH_asii
        self.dedtaut_sR = dedtaut_sR
        self.vHt_x = vHt_x  # initial guess for Hartree potential
        self.e_stress = e_stress  # idotropic contribution to stress tensor

    def __repr__(self):
        return (f'Potential({self.vt_sR}, {self.dH_asii}, '
                f'{self.dedtaut_sR})')

    def __str__(self) -> str:
        return (f'potential:\n'
                f'  grid points: {self.vt_sR.desc.size}\n')

    def dH(self, P_ani, out_ani, spin):
        if len(P_ani.dims) == 1:  # collinear wave functions
            P_ani.block_diag_multiply(self.dH_asii, out_ani, spin)
            return

        # Non-collinear wave functions:
        P_ansi = P_ani
        out_ansi = out_ani

        for (a, P_nsi), out_nsi in zips(P_ansi.items(), out_ansi.values()):
            v_ii, x_ii, y_ii, z_ii = (dh_ii.T for dh_ii in self.dH_asii[a])
            assert v_ii.dtype == complex
            out_nsi[:, 0] = (P_nsi[:, 0] @ (v_ii + z_ii) +
                             P_nsi[:, 1] @ (x_ii - 1j * y_ii))
            out_nsi[:, 1] = (P_nsi[:, 1] @ (v_ii - z_ii) +
                             P_nsi[:, 0] @ (x_ii + 1j * y_ii))
        return out_ansi

    def move(self, atomdist: AtomDistribution) -> None:
        """Move atoms inplace."""
        self.dH_asii = self.dH_asii.moved(atomdist)

    def redist(self,
               grid: UGDesc,
               desc: XDesc,
               atomdist: AtomDistribution,
               comm1: MPIComm,
               comm2: MPIComm) -> Potential:
        return Potential(
            self.vt_sR.redist(grid, comm1, comm2),
            self.dH_asii.redist(atomdist, comm1, comm2),
            None if self.dedtaut_sR is None else self.dedtaut_sR.redist(
                grid, comm1, comm2),
            None if self.vHt_x is None else self.vHt_x.redist(
                desc, comm1, comm2))

    def write_to_gpw(self, writer, flags):
        dH_asp = self.dH_asii.to_cpu().to_lower_triangle().gather()
        vt_sR = self.vt_sR.to_xp(np).gather()
        if self.dedtaut_sR is not None:
            dedtaut_sR = self.dedtaut_sR.to_xp(np).gather()
        if self.vHt_x is not None:
            vHt_x = self.vHt_x.to_xp(np).gather()
        if dH_asp is None:
            return

        writer.write(
            potential=flags.to_storage_dtype(vt_sR.data * Ha),
            atomic_hamiltonian_matrices=dH_asp.data * Ha)
        if self.vHt_x is not None:
            vHt_x_data = flags.to_storage_dtype(vHt_x.data * Ha)
            writer.write(electrostatic_potential=vHt_x_data)
        if self.dedtaut_sR is not None:
            dedtaut_sR_data = flags.to_storage_dtype(dedtaut_sR.data * Bohr**3)
            writer.write(mgga_potential=dedtaut_sR_data)

    def get_vacuum_level(self) -> float:
        grid = self.vt_sR.desc
        if grid.pbc_c.all():
            return np.nan
        if grid.zerobc_c.any():
            return 0.0
        if self.vHt_x is None:
            raise ValueError('No electrostatic potential')
        if isinstance(self.vHt_x, UGArray):
            vHt_r = self.vHt_x.gather()
        elif isinstance(self.vHt_x, PWArray):
            vHt_g = self.vHt_x.gather()
            if vHt_g is not None:
                vHt_r = vHt_g.ifft(grid=vHt_g.desc.minimal_uniform_grid())
            else:
                vHt_r = None
        else:
            return np.nan  # TB-mode
        vacuum_level = 0.0
        if vHt_r is not None:
            for c, periodic in enumerate(grid.pbc_c):
                if not periodic:
                    xp = vHt_r.xp
                    vacuum_level += float(xp.moveaxis(vHt_r.data,
                                                      c, 0)[0].mean())

            vacuum_level /= (3 - grid.pbc_c.sum())

        return broadcast_float(vacuum_level, grid.comm) * Ha
