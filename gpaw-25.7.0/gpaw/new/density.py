from __future__ import annotations

from math import pi, sqrt

import numpy as np
from ase.units import Bohr, Ha
from gpaw.core.atom_arrays import AtomArrays, AtomDistribution
from gpaw.core.atom_centered_functions import (AtomArraysLayout,
                                               AtomCenteredFunctions)
from gpaw.core.plane_waves import PWDesc
from gpaw.core.uniform_grid import UGArray, UGDesc
from gpaw.gpu import as_np
from gpaw.mpi import MPIComm
from gpaw.new import trace, zips
from gpaw.typing import Array3D, Vector
from gpaw.utilities import unpack_hermitian, unpack_density
from gpaw.new.symmetry import SymmetrizationPlan, GPUSymmetrizationPlan
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.setup import Setups


class Density:
    @classmethod
    def from_data_and_setups(cls,
                             nt_sR: UGArray,
                             taut_sR: UGArray,
                             D_asii: AtomArrays,
                             charge: float,
                             setups: Setups,
                             nct_aX: AtomCenteredFunctions,
                             tauct_aX: AtomCenteredFunctions) -> Density:
        xp = nt_sR.xp
        return cls(nt_sR,
                   taut_sR,
                   D_asii,
                   charge,
                   setups.nvalence + setups.core_charge,
                   [xp.asarray(setup.Delta_iiL) for setup in setups],
                   [setup.Delta0 for setup in setups],
                   [unpack_hermitian(setup.N0_p) for setup in setups],
                   [setup.n_j for setup in setups],
                   [setup.l_j for setup in setups],
                   nct_aX,
                   tauct_aX)

    @classmethod
    def from_superposition(cls,
                           *,
                           grid,
                           nct_aX,
                           tauct_aX,
                           atomdist,
                           setups,
                           basis_set,
                           magmom_av,
                           ncomponents,
                           charge=0.0,
                           hund=False,
                           mgga=False):
        nt_sR = grid.zeros(ncomponents)
        atom_array_layout = AtomArraysLayout(
            [(setup.ni, setup.ni) for setup in setups],
            atomdist=atomdist, dtype=float if ncomponents < 4 else complex)
        D_asii = atom_array_layout.empty(ncomponents)
        f_asi = {a: atomic_occupation_numbers(setup,
                                              magmom_v,
                                              ncomponents,
                                              hund,
                                              charge / len(setups))
                 for a, (setup, magmom_v)
                 in enumerate(zips(setups, magmom_av))}
        basis_set.add_to_density(nt_sR.data, f_asi)
        for a, D_sii in D_asii.items():
            D_sii[:] = unpack_density(
                setups[a].initialize_density_matrix(f_asi[a]))

        xp = nct_aX.xp
        nt_sR = nt_sR.to_xp(xp)
        density = cls.from_data_and_setups(nt_sR,
                                           None,
                                           D_asii.to_xp(xp),
                                           charge,
                                           setups,
                                           nct_aX,
                                           tauct_aX)
        ndensities = ncomponents % 3
        density.nt_sR.data[:ndensities] += density.nct_R.data
        if mgga:
            density.taut_sR = nt_sR.new()
            density.taut_sR.data[:] = density.tauct_R.data
        return density

    def __init__(self,
                 nt_sR: UGArray,
                 taut_sR: UGArray | None,
                 D_asii: AtomArrays,
                 charge: float,
                 nvalence: float,
                 delta_aiiL: list[Array3D],
                 delta0_a: list[float],
                 N0_aii,
                 n_aj: list[list[int]],
                 l_aj: list[list[int]],
                 nct_aX: AtomCenteredFunctions,
                 tauct_aX: AtomCenteredFunctions):
        self.nt_sR = nt_sR
        self.taut_sR = taut_sR
        self.D_asii = D_asii
        self.delta_aiiL = delta_aiiL
        self.delta0_a = delta0_a
        self.N0_aii = N0_aii
        self.n_aj = n_aj
        self.l_aj = l_aj
        self.charge = charge
        self.nvalence = nvalence
        self.nct_aX = nct_aX
        self.tauct_aX = tauct_aX

        self.grid = nt_sR.desc
        self.ncomponents = nt_sR.dims[0]
        self.ndensities = self.ncomponents % 3
        self.collinear = self.ncomponents != 4
        self.natoms = len(delta0_a)

        self._nct_R = None
        self._tauct_R = None

        self.symplan = None

    def __repr__(self):
        return f'Density({self.nt_sR}, {self.D_asii}, charge={self.charge})'

    def __str__(self) -> str:
        return (f'density:\n'
                f'  valence electrons: {self.nvalence}\n'
                f'  components: {self.ncomponents}\n'
                f'  grid points: {self.nt_sR.desc.size}\n'
                f'  charge: {self.charge}  # |e|\n')

    @property
    def nct_R(self):
        if self._nct_R is None:
            self._nct_R = self.grid.empty(xp=self.nt_sR.xp)
            self.nct_aX.to_uniform_grid(out=self._nct_R,
                                        scale=1.0 / (self.ncomponents % 3))
        return self._nct_R

    @property
    def tauct_R(self):
        if self._tauct_R is None:
            self._tauct_R = self.grid.empty(xp=self.nt_sR.xp)
            self.tauct_aX.to_uniform_grid(out=self._tauct_R,
                                          scale=1.0 / (self.ncomponents % 3))
        return self._tauct_R

    def new(self, new_grid, pw, relpos_ac, atomdist):
        self.move(relpos_ac, atomdist)
        new_pw = PWDesc(ecut=0.99 * new_grid.ekin_max(),
                        cell=new_grid.cell,
                        comm=new_grid.comm)
        old_grid = self.nt_sR.desc
        old_pw = PWDesc(ecut=0.99 * old_grid.ekin_max(),
                        cell=old_grid.cell,
                        comm=new_grid.comm)
        new_nt_sR = new_grid.empty(self.ncomponents, xp=self.nt_sR.xp)
        for new_nt_R, old_nt_R in zips(new_nt_sR, self.nt_sR):
            old_nt_R.fft(pw=old_pw).morph(new_pw).ifft(out=new_nt_R)

        self.nct_aX.change_cell(pw)
        self.tauct_aX.change_cell(pw)

        return Density(
            new_nt_sR,
            None if self.taut_sR is None else new_nt_sR.new(zeroed=True),
            self.D_asii,
            self.charge,
            self.nvalence,
            self.delta_aiiL,
            self.delta0_a,
            self.N0_aii,
            self.n_aj,
            self.l_aj,
            self.nct_aX,
            self.tauct_aX)

    @trace
    def calculate_compensation_charge_coefficients(self) -> AtomArrays:
        xp = self.D_asii.layout.xp
        ccc_aL = AtomArraysLayout(
            [delta_iiL.shape[2] for delta_iiL in self.delta_aiiL],
            atomdist=self.D_asii.layout.atomdist,
            xp=xp).empty()

        for a, D_sii in self.D_asii.items():
            Q_L = xp.einsum('sij, ijL -> L',
                            D_sii[:self.ndensities].real, self.delta_aiiL[a])
            Q_L[0] += self.delta0_a[a]
            ccc_aL[a] = Q_L

        return ccc_aL

    def normalize(self, background_charge: float) -> None:
        comp_charge = 0.0
        xp = self.D_asii.layout.xp
        for a, D_sii in self.D_asii.items():
            comp_charge += xp.einsum('sij, ij ->',
                                     D_sii[:self.ndensities].real,
                                     self.delta_aiiL[a][:, :, 0])
            comp_charge += self.delta0_a[a]
        # comp_charge could be cupy.ndarray:
        comp_charge = float(comp_charge) * sqrt(4 * pi)
        comp_charge = self.nt_sR.desc.comm.sum_scalar(comp_charge)
        charge = comp_charge + self.charge - background_charge
        pseudo_charge = self.nt_sR[:self.ndensities].integrate().sum()
        if pseudo_charge != 0.0:
            x = -charge / pseudo_charge
            self.nt_sR.data *= x

    @trace
    def update(self, ibzwfs: IBZWaveFunctions, ked=False):
        self.nt_sR.data[:] = 0.0
        self.D_asii.data[:] = 0.0
        ibzwfs.add_to_density(self.nt_sR, self.D_asii)
        self.nt_sR.data[:self.ndensities] += self.nct_R.data

        # LCAO ...:
        ibzwfs.normalize_density(self)

        if ked:
            self.update_ked(ibzwfs, symmetrize=False)

        self.symmetrize(ibzwfs.ibz.symmetries)

    def update_ked(self, ibzwfs, symmetrize=True):
        if self.taut_sR is None:
            self.taut_sR = self.nt_sR.new(zeroed=True)
        else:
            self.taut_sR.data[:] = 0.0
        ibzwfs.add_to_ked(self.taut_sR)
        self.taut_sR.data[:self.ndensities] += self.tauct_R.data
        if symmetrize:
            symmetries = ibzwfs.ibz.symmetries
            self.taut_sR.symmetrize(symmetries.rotation_scc,
                                    symmetries.translation_sc)

    @trace
    def symmetrize(self, symmetries):
        self.nt_sR.symmetrize(symmetries.rotation_scc,
                              symmetries.translation_sc)
        if self.taut_sR is not None:
            self.taut_sR.symmetrize(symmetries.rotation_scc,
                                    symmetries.translation_sc)

        xp = self.nt_sR.xp
        if xp is np:
            D_asii = self.D_asii.gather(broadcast=True, copy=True)
            if self.symplan is None:
                self.symplan = SymmetrizationPlan(symmetries, self.l_aj)
            self.symplan.apply_distributed(D_asii, self.D_asii)
        else:
            # GPU version does all the work in rank 0 for now
            D_asii = self.D_asii.gather(copy=True)
            if self.D_asii.layout.atomdist.comm.rank == 0:
                if self.symplan is None:
                    self.symplan = GPUSymmetrizationPlan(
                        symmetries, self.l_aj, D_asii.layout)
                self.symplan.apply(D_asii.data, D_asii.data)
            self.D_asii.scatter_from(D_asii)

    @trace
    def move(self, relpos_ac, atomdist):
        self.nt_sR.data[:self.ndensities] -= self.nct_R.data
        self.nct_aX.move(relpos_ac, atomdist)
        self.tauct_aX.move(relpos_ac, atomdist)
        self._nct_R = None
        self._tauct_R = None
        self.nt_sR.data[:self.ndensities] += self.nct_R.data
        self.D_asii = self.D_asii.moved(atomdist)

    @trace
    def redist(self,
               grid: UGDesc,
               xdesc,
               atomdist: AtomDistribution,
               comm1: MPIComm,
               comm2: MPIComm) -> Density:
        return Density(
            self.nt_sR.redist(grid, comm1, comm2),
            None
            if self.taut_sR is None else
            self.taut_sR.redist(grid, comm1, comm2),
            self.D_asii.redist(atomdist, comm1, comm2),
            self.charge,
            self.nvalence,
            self.delta_aiiL,
            self.delta0_a,
            self.N0_aii,
            self.n_aj,
            self.l_aj,
            nct_aX=self.nct_aX.new(xdesc, atomdist),
            tauct_aX=self.tauct_aX.new(xdesc, atomdist))

    def calculate_dipole_moment(self, relpos_ac):
        dip_v = np.zeros(3)
        ccc_aL = self.calculate_compensation_charge_coefficients()
        ccc_aL = ccc_aL.to_cpu()
        pos_av = relpos_ac @ self.nt_sR.desc.cell_cv
        for a, ccc_L in ccc_aL.items():
            c = ccc_L[0]
            dip_v -= c * (4 * pi)**0.5 * pos_av[a]
            if len(ccc_L) > 1:
                y, z, x = ccc_L[1:4]
                dip_v -= np.array([x, y, z]) * (4 * pi / 3)**0.5
        self.nt_sR.desc.comm.sum(dip_v)
        for nt_R in self.nt_sR[:self.ndensities]:
            dip_v -= as_np(nt_R.moment())
        return dip_v

    def calculate_orbital_magnetic_moments(self):
        if self.collinear:
            from gpaw.new.calculation import CalculationModeError
            raise CalculationModeError(
                'Calculator is in collinear mode. '
                'Collinear calculations require spin–orbit '
                'coupling for nonzero orbital magnetic moments.')

        D_asii = self.D_asii
        if D_asii.layout.size != D_asii.layout.mysize:
            raise ValueError(
                'Atomic density matrices should be collected on all '
                'ranks when calculating orbital magnetic moments.')

        from gpaw.new.orbmag import calculate_orbmag_from_density
        return calculate_orbmag_from_density(D_asii, self.n_aj, self.l_aj)

    def calculate_magnetic_moments(self):
        magmom_av = np.zeros((self.natoms, 3))
        magmom_v = np.zeros(3)
        domain_comm = self.nt_sR.desc.comm

        if self.ncomponents == 2:
            for a, D_sii in self.D_asii.items():
                M_ii = as_np(D_sii[0] - D_sii[1])
                magmom_av[a, 2] = np.einsum('ij, ij ->', M_ii, self.N0_aii[a])
                delta_ii = as_np(self.delta_aiiL[a][:, :, 0])
                magmom_v[2] += (np.einsum('ij, ij ->', M_ii, delta_ii) *
                                sqrt(4 * pi))
            domain_comm.sum(magmom_av)
            domain_comm.sum(magmom_v)

            M_s = self.nt_sR.integrate()
            magmom_v[2] += M_s[0] - M_s[1]

        elif self.ncomponents == 4:
            for a, D_sii in self.D_asii.items():
                M_vii = D_sii[1:4].real
                magmom_av[a] = np.einsum('vij, ij -> v',
                                         M_vii, self.N0_aii[a])
                magmom_v += (np.einsum('vij, ij -> v', M_vii,
                                       self.delta_aiiL[a][:, :, 0]) *
                             sqrt(4 * pi))
            domain_comm.sum(magmom_av)
            domain_comm.sum(magmom_v)
            magmom_v += self.nt_sR.integrate()[1:]

        return magmom_v, magmom_av

    @trace
    def write_to_gpw(self, writer, flags):
        D_asp = self.D_asii.to_cpu().to_lower_triangle().gather()
        nt_sR = self.nt_sR.to_xp(np).gather()
        if self.taut_sR is not None:
            taut_sR = self.taut_sR.to_xp(np).gather()
        if D_asp is None:
            return  # let master do the writing
        writer.write(
            density=flags.to_storage_dtype(nt_sR.data * Bohr**-3),
            atomic_density_matrices=D_asp.data)
        if self.taut_sR is not None:
            writer.write(ked=flags.to_storage_dtype(
                taut_sR.data * (Ha * Bohr**-3)))


def atomic_occupation_numbers(setup,
                              magmom_v: Vector,
                              ncomponents: int,
                              hund: bool = False,
                              charge: float = 0.0):
    M = np.linalg.norm(magmom_v)
    nspins = min(ncomponents, 2)
    f_si = setup.calculate_initial_occupation_numbers(
        M, hund, charge=charge, nspins=nspins)

    if ncomponents == 1:
        pass
    elif ncomponents == 2:
        if magmom_v[2] < 0:
            f_si = f_si[::-1].copy()
    else:
        f_i = f_si.sum(0)
        fm_i = f_si[0] - f_si[1]
        f_si = np.zeros((4, len(f_i)))
        f_si[0] = f_i
        if M > 0:
            f_si[1:] = np.asarray(magmom_v)[:, np.newaxis] / M * fm_i

    return f_si
