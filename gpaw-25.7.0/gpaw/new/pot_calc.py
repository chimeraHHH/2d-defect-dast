"""
==  ==========
R
r
G
g
h
x   r or h
==  ==========

"""

from __future__ import annotations

import functools
import operator
from collections import defaultdict
from typing import DefaultDict

import numpy as np
from ase.units import Ha
from gpaw.core.arrays import DistributedArrays
from gpaw.core.atom_arrays import AtomArrays
from gpaw.core.uniform_grid import UGArray
from gpaw.mpi import MPIComm, serial_comm
from gpaw.new import trace, zips
from gpaw.new.energies import DFTEnergies
from gpaw.new.environment import Environment
from gpaw.new.logger import indent
from gpaw.new.potential import Potential
from gpaw.new.xc import Functional
from gpaw.setup import Setup
from gpaw.spinorbit import soc as soc_terms
from gpaw.typing import Array1D, Array2D, Array3D
from gpaw.utilities import pack_density, pack_hermitian, unpack_hermitian


class PotentialCalculator:
    def __init__(self,
                 xc: Functional,
                 poisson_solver,
                 setups: list[Setup],
                 *,
                 relpos_ac: Array2D,
                 environment: Environment,
                 extensions: list | None = None,
                 soc: bool = False):
        self.poisson_solver = poisson_solver
        self.xc = xc
        self.setups = setups
        self.relpos_ac = relpos_ac
        self.soc = soc
        self.environment = environment or Environment(len(relpos_ac))
        self.extensions: list = extensions or []

    def __str__(self):
        return (f'{self.poisson_solver}\n'
                f'xc functional:\n{indent(self.xc)}\n')

    def calculate_pseudo_potential(self,
                                   density,
                                   ibzwfs,
                                   vHt_x: DistributedArrays | None
                                   ) -> tuple[dict[str, float],
                                              UGArray,
                                              UGArray,
                                              DistributedArrays,
                                              AtomArrays,
                                              float]:
        raise NotImplementedError

    def move(self, relpos_ac, atomdist):
        for ext in self.extensions:
            ext.move_atoms(relpos_ac)

    @property
    def extensions_force_av(self):
        if not self.extensions:
            return np.zeros((len(self.setups), 3))
        return functools.reduce(operator.add, [ext.force_contribution()
                                for ext in self.extensions])

    @property
    def extensions_stress_contribution(self):
        if not self.extensions:
            return np.zeros((3, 3))
        return functools.reduce(operator.add, [ext.stress_contribution()
                                for ext in self.extensions])

    def calculate_charges(self, vHt_x):
        raise NotImplementedError

    def restrict(self, a_r, a_R=None):
        raise NotImplementedError

    def calculate_without_orbitals(self,
                                   density,
                                   ibzwfs=None,
                                   vHt_x: DistributedArrays | None = None,
                                   kpt_band_comm: MPIComm | None = None
                                   ) -> tuple[Potential,
                                              DFTEnergies,
                                              AtomArrays]:
        xc = self.xc
        if xc.exx_fraction != 0.0:
            from gpaw.new.xc import create_functional
            self.xc = create_functional('PBE', xc.grid)
        potential, energies, V_al = self.calculate(
            density, ibzwfs, vHt_x, kpt_band_comm)
        if xc.exx_fraction != 0.0:
            self.xc = xc
        return potential, energies, V_al

    @trace
    def calculate(self,
                  density,
                  ibzwfs=None,
                  vHt_x: DistributedArrays | None = None,
                  kpt_band_comm: MPIComm | None = None
                  ) -> tuple[Potential, DFTEnergies, AtomArrays]:
        energies, vt_sR, dedtaut_sr, vHt_x, V_aL, e_stress = (
            self.calculate_pseudo_potential(density, ibzwfs, vHt_x))
        e_kinetic = 0.0
        for spin, (vt_R, nt_R) in enumerate(zips(vt_sR, density.nt_sR)):
            e_kinetic -= vt_R.integrate(nt_R)
            if spin < density.ndensities:
                e_kinetic += vt_R.integrate(density.nct_R)

        if dedtaut_sr is not None:
            dedtaut_sR = self.restrict(dedtaut_sr)
            for dedtaut_R, taut_R in zips(dedtaut_sR,
                                          density.taut_sR):
                e_kinetic -= dedtaut_R.integrate(taut_R)
                e_kinetic += dedtaut_R.integrate(density.tauct_R)
        else:
            dedtaut_sR = None

        energies['kinetic_correction'] = e_kinetic

        if kpt_band_comm is None:
            if ibzwfs is None:
                kpt_band_comm = serial_comm
            else:
                kpt_band_comm = ibzwfs.kpt_band_comm
        dH_asii, corrections = calculate_non_local_potential(
            self.setups,
            density,
            self.xc,
            V_aL,
            self.soc,
            self.extensions,
            kpt_band_comm)

        for ext in self.extensions:
            dct = ext.get_energy_contributions()
            for name, e in dct.items():
                assert name not in energies
                energies[name] = e

        energies['spinorbit'] = 0.0
        for key, e in corrections.items():
            if False:
                print(f'{key:10} {energies[key] * Ha:15.9f} {e * Ha:15.9f}')
            energies[key] += e

        return (Potential(vt_sR, dH_asii, dedtaut_sR, vHt_x, e_stress),
                DFTEnergies(**energies),
                V_aL)


@trace
def calculate_non_local_potential(setups,
                                  density,
                                  xc,
                                  V_aL,
                                  soc: bool,
                                  extensions,
                                  kpt_band_comm: MPIComm
                                  ) -> tuple[AtomArrays,
                                             dict[str, float]]:
    dtype = float if density.ncomponents < 4 else complex
    D_asii = density.D_asii.to_xp(np)
    dH_asii = D_asii.layout.new(dtype=dtype).empty(density.ncomponents)
    V_aL = V_aL.to_xp(np)
    energy_corrections: DefaultDict[str, float] = defaultdict(float)
    rank = 0
    for a, D_sii in D_asii.items():
        if rank % kpt_band_comm.size == kpt_band_comm.rank:
            V_L = V_aL[a]
            setup = setups[a]
            dH_sii, corrections = calculate_non_local_potential1(
                setup, xc, D_sii, V_L, soc, extensions, a)
            dH_asii[a][:] = dH_sii
            for key, e in corrections.items():
                energy_corrections[key] += e
        else:
            dH_asii[a][:] = 0.0
        rank += 1

    kpt_band_comm.sum(dH_asii.data)

    # Sum over domain:
    names = ['kinetic_correction', 'coulomb', 'zero', 'xc', 'external',
             'spinorbit']
    energies = np.array([energy_corrections[name] for name in names])
    density.D_asii.layout.atomdist.comm.sum(energies)
    kpt_band_comm.sum(energies)

    return (dH_asii.to_xp(density.D_asii.layout.xp),
            dict(zips(names, energies)))


def calculate_non_local_potential1(setup: Setup,
                                   xc: Functional,
                                   D_sii: Array3D,
                                   V_L: Array1D,
                                   soc: bool,
                                   extensions,
                                   atom_index) -> tuple[Array3D,
                                                        dict[str, float]]:
    ncomponents = len(D_sii)
    ndensities = 2 if ncomponents == 2 else 1
    D_sp = np.array([pack_density(D_ii.real) for D_ii in D_sii])

    D_p = D_sp[:ndensities].sum(0)

    dH_p = (setup.K_p + setup.M_p +
            setup.MB_p + 2.0 * setup.M_pp @ D_p +
            setup.Delta_pL @ V_L)
    e_kinetic = setup.K_p @ D_p + setup.Kc
    e_zero = setup.MB + setup.MB_p @ D_p
    e_coulomb = setup.M + D_p @ (setup.M_p + setup.M_pp @ D_p)

    dH_sp = np.zeros_like(D_sp, dtype=float if ncomponents < 4 else complex)

    e_soc = 0.
    if soc:
        dHsoc_sii = soc_terms(setup, xc.xc, D_sp)
        e_soc += (D_sii[1:4] * dHsoc_sii).sum().real
        dH_sp[1:4] = pack_hermitian(dHsoc_sii)

    dH_sp[:ndensities] = dH_p
    e_xc = xc.calculate_paw_correction(setup, D_sp, dH_sp)

    # e_external = ext_pot.add_paw_correction(setup.Delta_pL[:, 0], dH_sp)

    dH_sii = unpack_hermitian(dH_sp)

    if setup.hubbard_u is not None:
        eU, dHU_sii = setup.hubbard_u.calculate(setup, D_sii)
        e_xc += eU
        dH_sii += dHU_sii

    for extension in extensions:
        e_xc += extension.update_non_local_hamiltonian(
            D_sii, setup, atom_index, dH_sii)

    e_kinetic -= (D_sii * dH_sii).sum().real

    return dH_sii, {'kinetic_correction': e_kinetic,
                    'coulomb': e_coulomb,
                    'zero': e_zero,
                    'xc': e_xc,
                    'external': 0.0,  # e_external,
                    'spinorbit': e_soc}
