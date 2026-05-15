from __future__ import annotations
from dataclasses import dataclass
from typing import Union
from pathlib import Path
from functools import cached_property
from types import SimpleNamespace
from typing import TYPE_CHECKING
import numpy as np

from ase.units import Ha, Bohr

import gpaw.mpi as mpi
from gpaw.ibz2bz import IBZ2BZMaps
from gpaw.calculator import GPAW as OldGPAW
from gpaw.new.ase_interface import ASECalculator as NewGPAW
from gpaw.response.paw import LeanPAWDataset

if TYPE_CHECKING:
    from gpaw.setup import Setups, LeanSetup


class PAWDatasetCollection:
    def __init__(self, setups: Setups):
        by_species = {}
        by_atom = []
        id_by_atom = []

        for atom_id, setup in enumerate(setups):
            species_id = setups.id_a[atom_id]
            if species_id not in by_species:
                by_species[species_id] = ResponsePAWDataset(setup)
            by_atom.append(by_species[species_id])
            id_by_atom.append(species_id)

        self.by_species = by_species
        self.by_atom = by_atom
        self.id_by_atom = id_by_atom


GPAWCalculator = Union[OldGPAW, NewGPAW]
GPWFilename = Union[Path, str]
ResponseGroundStateAdaptable = Union['ResponseGroundStateAdapter',
                                     GPAWCalculator,
                                     GPWFilename]


class ResponseGroundStateAdapter:
    def __init__(self, calc: GPAWCalculator):
        wfs = calc.wfs  # wavefunction object from gpaw.wavefunctions

        self.atoms = calc.atoms
        self.kd = wfs.kd  # KPointDescriptor object from gpaw.kpt_descriptor.
        self.world = calc.world  # _Communicator object from gpaw.mpi

        # GridDescriptor from gpaw.grid_descriptor.
        # Describes a grid in real space
        self.gd = wfs.gd

        # Also a GridDescriptor, with a finer grid...
        self.finegd = calc.density.finegd
        self.bd = wfs.bd  # BandDescriptor from gpaw.band_descriptor
        self.nspins = wfs.nspins  # number of spins: int
        self.dtype = wfs.dtype  # data type of wavefunctions, real or complex

        self.spos_ac = calc.spos_ac  # scaled position vector: np.ndarray

        self.kpt_u = wfs.kpt_u  # kpoints: list of Kpoint from gpaw.kpoint
        self.kpt_qs = wfs.kpt_qs  # kpoints: list of Kpoint from gpaw.kpoint

        self.fermi_level = wfs.fermi_level  # float
        self.atoms = calc.atoms  # ASE Atoms object
        self.pawdatasets = PAWDatasetCollection(calc.setups)

        self.pbc = self.atoms.pbc
        self.volume = self.gd.volume

        self.nvalence = int(round(wfs.nvalence))
        assert self.nvalence == wfs.nvalence

        self.nocc1, self.nocc2 = self.count_occupied_bands()

        self.ibz2bz = IBZ2BZMaps.from_calculator(calc)

        self._wfs = wfs
        self._density = calc.density
        self._hamiltonian = calc.hamiltonian
        self._calc = calc

    @staticmethod
    def from_input(
            gs: ResponseGroundStateAdaptable) -> ResponseGroundStateAdapter:
        if isinstance(gs, ResponseGroundStateAdapter):
            return gs
        elif isinstance(gs, (OldGPAW, NewGPAW)):
            return ResponseGroundStateAdapter(calc=gs)
        elif isinstance(gs, (Path, str)):  # GPWFilename
            return ResponseGroundStateAdapter.from_gpw_file(gpw=gs)
        raise ValueError('Expected ResponseGroundStateAdaptable, got', gs)

    @classmethod
    def from_gpw_file(cls, gpw: GPWFilename) -> ResponseGroundStateAdapter:
        """Initiate the ground state adapter directly from a .gpw file."""
        from gpaw import GPAW, disable_dry_run
        assert Path(gpw).is_file()
        with disable_dry_run():
            calc = GPAW(gpw, txt=None, communicator=mpi.serial_comm)
        return cls(calc)

    @property
    def pd(self):
        # This is an attribute error in FD/LCAO mode.
        # We need to abstract away "calc" in all places used by response
        # code, and that includes places that are also compatible with FD.
        return self._wfs.pd

    def is_parallelized(self):
        """Are we dealing with a parallel calculator?"""
        return self.world.size > 1

    @cached_property
    def global_pd(self):
        """Get a PWDescriptor that includes all k-points.

        In particular, this is necessary to allow all cores to be able to work
        on all k-points in the case where calc is parallelized over k-points,
        see gpaw.response.kspair
        """
        from gpaw.pw.descriptor import PWDescriptor

        assert self.gd.comm.size == 1
        kd = self.kd.copy()  # global KPointDescriptor without a comm
        return PWDescriptor(self.pd.ecut, self.gd,
                            dtype=self.pd.dtype,
                            kd=kd, fftwflags=self.pd.fftwflags,
                            gammacentered=self.pd.gammacentered)

    def get_occupations_width(self):
        # Ugly hack only used by pair.intraband_pair_density I think.
        # Actually: was copy-pasted in chi0 also.
        # More duplication can probably be eliminated around those.

        # Only works with Fermi-Dirac distribution
        occs = self._wfs.occupations
        assert occs.name in {'fermi-dirac', 'zero-width'}

        # No carriers when T=0
        width = getattr(occs, '_width', 0.0) / Ha
        return width

    @cached_property
    def cd(self):
        return CellDescriptor(self.gd.cell_cv, self.pbc)

    @property
    def nt_sR(self):
        # Used by localft and fxc_kernels
        return self._density.nt_sG

    @property
    def nt_sr(self):
        # Used by localft
        if self._density.nt_sg is None:
            self._density.interpolate_pseudo_density()
        return self._density.nt_sg

    @cached_property
    def n_sR(self):
        return self._density.get_all_electron_density(
            atoms=self.atoms, gridrefinement=1)[0]

    @cached_property
    def n_sr(self):
        return self._density.get_all_electron_density(
            atoms=self.atoms, gridrefinement=2)[0]

    @property
    def D_asp(self):
        # Used by fxc_kernels
        return self._density.D_asp

    def get_pseudo_density(self, gridrefinement=2):
        # Used by localft
        if gridrefinement == 1:
            return self.nt_sR, self.gd
        elif gridrefinement == 2:
            return self.nt_sr, self.finegd
        else:
            raise ValueError(f'Invalid gridrefinement {gridrefinement}')

    def get_all_electron_density(self, gridrefinement=2):
        # Used by fxc, fxc_kernels and localft
        if gridrefinement == 1:
            return self.n_sR, self.gd
        elif gridrefinement == 2:
            return self.n_sr, self.finegd
        else:
            raise ValueError(f'Invalid gridrefinement {gridrefinement}')

    # Things used by EXX.  This is getting pretty involved.
    #
    # EXX naughtily accesses the density object in order to
    # interpolate_pseudo_density() which is in principle mutable.

    def hacky_all_electron_density(self, **kwargs):
        # fxc likes to get all electron densities.  It calls
        # calc.get_all_electron_density() and so we wrap that here.
        # But it also collects to serial (bad), and it also zeropads
        # nonperiodic directions (probably WRONG!).
        #
        # Also this one returns in user units, whereas the calling
        # code actually wants internal units.  Very silly then.
        #
        # ALso, the calling code often wants the gd, which is not
        # returned, so it is redundantly reconstructed in multiple
        # places by refining the "right" number of times.
        n_g = self._calc.get_all_electron_density(**kwargs)
        n_g *= Bohr**3
        return n_g

    # Used by EXX.
    @property
    def hamiltonian(self):
        return self._hamiltonian

    # Used by EXX.
    @property
    def density(self):
        return self._density

    # Ugh SOC
    def soc_eigenstates(self, **kwargs):
        from gpaw.spinorbit import soc_eigenstates
        return soc_eigenstates(self._calc, **kwargs)

    @property
    def xcname(self):
        return self.hamiltonian.xc.name

    def get_xc_difference(self, xc):
        # XXX used by gpaw/xc/tools.py
        return self._calc.get_xc_difference(xc)

    def get_wave_function_array(self, u, n):
        # XXX used by gpaw/xc/tools.py in a hacky way
        return self._wfs._get_wave_function_array(
            u, n, realspace=True)

    def pair_density_paw_corrections(self, qpd):
        from gpaw.response.paw import get_pair_density_paw_corrections
        return get_pair_density_paw_corrections(
            pawdatasets=self.pawdatasets, qpd=qpd, spos_ac=self.spos_ac,
            atomrotations=self.atomrotations)

    def matrix_element_paw_corrections(self, qpd, rshe_a):
        from gpaw.response.paw import get_matrix_element_paw_corrections
        return get_matrix_element_paw_corrections(
            qpd, self.pawdatasets, rshe_a, self.spos_ac)

    def get_pos_av(self):
        # gd.cell_cv must always be the same as pd.gd.cell_cv, right??
        return np.dot(self.spos_ac, self.gd.cell_cv)

    def count_occupied_bands(self, ftol: float = 1e-6) -> tuple[int, int]:
        """Count the number of filled (nocc1) and nonempty bands (nocc2).

        ftol : float
            Threshold determining whether a band is completely filled
            (f > 1 - ftol) or completely empty (f < ftol).
        """
        # Count the number of occupied bands for this rank
        nocc1, nocc2 = self._count_occupied_bands(ftol=ftol)
        # Minimize/maximize over k-points
        nocc1 = self.kd.comm.min_scalar(nocc1)  # bands filled for all k
        nocc2 = self.kd.comm.max_scalar(nocc2)  # bands filled for any k
        # Sum over band distribution
        nocc1 = self.bd.comm.sum_scalar(nocc1)  # number of filled bands
        nocc2 = self.bd.comm.sum_scalar(nocc2)  # number of nonempty bands
        return int(nocc1), int(nocc2)

    def _count_occupied_bands(self, *, ftol: float) -> tuple[int, int]:
        nocc1 = 9999999  # number of completely filled bands
        nocc2 = 0  # number of nonempty bands
        for kpt in self.kpt_u:
            f_n = kpt.f_n / kpt.weight
            nocc1 = min((f_n > 1 - ftol).sum(), nocc1)
            nocc2 = max((f_n > ftol).sum(), nocc2)
        return int(nocc1), int(nocc2)

    def get_band_transitions(self, nbands: int | slice | None = None):
        """Determine the indices the define the range of occupied bands
        n1, n2 and unoccupied bands m1, m2"""

        if nbands is None:
            n1 = 0
            m2 = self.nbands
        elif isinstance(nbands, int):
            n1 = 0
            m2 = nbands
            assert 1 <= m2 <= self.nbands
        elif isinstance(nbands, slice):
            n1 = nbands.start
            m2 = nbands.stop
            assert n1 >= 0 and m2 >= 0
            assert nbands.step in {None, 1}
            assert n1 < m2 <= self.nbands
            assert n1 <= self.nocc1
        else:
            raise ValueError(
                f"Invalid type for nbands: {type(nbands)}."
                "Expected None, int, or slice.")

        n2 = self.nocc2
        m1 = self.nocc1

        assert n1 < n2

        return n1, n2, m1, m2

    def get_eigenvalue_range(self, nbands: int | slice | None = None):
        """Get smallest and largest Kohn-Sham eigenvalues."""
        n1, n2, m1, m2 = self.get_band_transitions(nbands)
        epsmin = np.inf
        epsmax = -np.inf
        for kpt in self.kpt_u:
            epsmin = min(epsmin, kpt.eps_n[n1])  # the eigenvalues are ordered
            epsmax = max(epsmax, kpt.eps_n[m2 - 1])
        return epsmin, epsmax

    @property
    def nbands(self):
        return self.bd.nbands

    @property
    def metallic(self):
        # Does the number of filled bands equal the number of non-empty bands?
        return self.nocc1 != self.nocc2

    @cached_property
    def ibzq_qc(self):
        # For G0W0Kernel
        kd = self.kd
        bzq_qc = kd.get_bz_q_points(first=True)
        U_scc = kd.symmetry.op_scc
        ibzq_qc = kd.get_ibz_q_points(bzq_qc, U_scc)[0]

        return ibzq_qc

    def get_ibz_vertices(self):
        # For the tetrahedron method in Chi0
        from gpaw.bztools import get_bz
        # NB: We are ignoring the pbc_c keyword to get_bz() in order to mimic
        # find_high_symmetry_monkhorst_pack() in gpaw.bztools. XXX
        _, ibz_vertices_kc, _ = get_bz(self._calc)
        return ibz_vertices_kc

    def get_aug_radii(self):
        return np.array([max(pawdata.rcut_j)
                         for pawdata in self.pawdatasets.by_atom])

    @cached_property
    def micro_setups(self):
        from gpaw.response.localft import extract_micro_setup
        micro_setups = []
        for a, pawdata in enumerate(self.pawdatasets.by_atom):
            micro_setups.append(extract_micro_setup(pawdata, self.D_asp[a]))
        return micro_setups

    @property
    def atomrotations(self):
        return self._wfs.setups.atomrotations

    @cached_property
    def kpoints(self):
        from gpaw.response.kpoints import ResponseKPointGrid
        return ResponseKPointGrid(self.kd)


@dataclass
class CellDescriptor:
    cell_cv: np.ndarray
    pbc_c: np.ndarray

    @property
    def nonperiodic_hypervolume(self):
        """Get the hypervolume of the cell along nonperiodic directions.

        Returns the hypervolume Λ in units of Å, where

        Λ = 1        in 3D
        Λ = L        in 2D, where L is the out-of-plane cell vector length
        Λ = A        in 1D, where A is the transverse cell area
        Λ = V        in 0D, where V is the cell volume
        """
        cell_cv = self.cell_cv
        pbc_c = self.pbc_c
        if sum(pbc_c) > 0:
            # In 1D and 2D, we assume the cartesian representation of the unit
            # cell to be block diagonal, separating the periodic and
            # nonperiodic cell vectors in different blocks.
            assert np.allclose(cell_cv[~pbc_c][:, pbc_c], 0.) and \
                np.allclose(cell_cv[pbc_c][:, ~pbc_c], 0.), \
                "In 1D and 2D, please put the periodic/nonperiodic axis " \
                "along a cartesian component"
        L = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))
        return L * Bohr**sum(~pbc_c)  # Bohr -> Å


# Contains all the relevant information
# from Setups class for response calculators
class ResponsePAWDataset(LeanPAWDataset):
    def __init__(self, setup: LeanSetup, **kwargs):
        super().__init__(
            rgd=setup.rgd, l_j=setup.l_j, rcut_j=setup.rcut_j,
            phit_jg=setup.data.phit_jg, phi_jg=setup.data.phi_jg, **kwargs)
        assert setup.ni == self.ni

        self.n_j = setup.n_j
        self.N0_q = setup.N0_q
        self.nabla_iiv = setup.nabla_iiv
        self.xc_correction: SimpleNamespace | None
        if setup.xc_correction is not None:
            self.xc_correction = SimpleNamespace(
                rgd=setup.xc_correction.rgd, Y_nL=setup.xc_correction.Y_nL,
                n_qg=setup.xc_correction.n_qg, nt_qg=setup.xc_correction.nt_qg,
                nc_g=setup.xc_correction.nc_g, nct_g=setup.xc_correction.nct_g,
                nc_corehole_g=setup.xc_correction.nc_corehole_g,
                B_pqL=setup.xc_correction.B_pqL,
                e_xc0=setup.xc_correction.e_xc0)
        else:
            # If there is no `xc_correction` in the setup, we assume to be
            # using pseudo potentials.
            self.xc_correction = None
            # In this case, we set l_j to an empty list in order to bypass the
            # calculation of PAW corrections to pair densities etc.
            # This is quite an ugly hack...
            # If we want to support pseudo potential calculations for real, we
            # should skip the PAW corrections at the matrix element calculator
            # level, not by an odd hack.
            self.l_j = np.array([], dtype=float)
        self.hubbard_u = setup.hubbard_u
