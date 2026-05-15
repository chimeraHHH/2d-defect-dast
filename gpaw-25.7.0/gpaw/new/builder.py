from __future__ import annotations
from functools import cached_property
from types import ModuleType, SimpleNamespace
from typing import Any, TYPE_CHECKING
import warnings
import numpy as np
from gpaw import GPAW_USE_GPUS, GPAW_CPUPY
from ase import Atoms
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.geometry.cell import cell_to_cellpar
from ase.units import Bohr
from gpaw.core import UGDesc
from gpaw.core.atom_arrays import (AtomArrays, AtomArraysLayout,
                                   AtomDistribution)
from gpaw.core.domain import Domain
from gpaw.gpu import cpupy as fake_cupy
from gpaw.gpu.mpi import CuPyMPI
from gpaw.lfc import BasisFunctions
from gpaw.mixer import MixerWrapper, get_mixer_from_keywords
from gpaw.mpi import (MPIComm, Parallelization, broadcast, serial_comm,
                      synchronize_atoms, world)
from gpaw.new import prod
from gpaw.new.basis import create_basis
from gpaw.new.brillouin import BZPoints, MonkhorstPackKPoints
from gpaw.new.c import GPU_AWARE_MPI
from gpaw.new.density import Density
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.logger import Logger
from gpaw.new.potential import Potential
from gpaw.new.scf import SCFLoop
from gpaw.new.smearing import OccupationNumberCalculator
from gpaw.new.xc import create_functional
from gpaw.setup import Setups
from gpaw.typing import Array2D, ArrayLike1D, ArrayLike2D, DTypeLike
from gpaw.utilities.gpts import get_number_of_grid_points
if TYPE_CHECKING:
    from gpaw.dft import Parameters


class DFTComponentsBuilder:
    def __init__(self,
                 atoms: Atoms,
                 params: Parameters,
                 *,
                 log=None,
                 comm=None):

        self.atoms = atoms.copy()
        self.mode = params.mode.name
        self.params = params
        if not isinstance(log, Logger):
            log = Logger(log, comm)
        self.log = log
        comm = log.comm

        parallel = params.parallel

        synchronize_atoms(atoms, comm)
        self.check_cell(atoms.cell)

        self.initial_magmom_av, self.ncomponents = normalize_initial_magmoms(
            atoms, params.magmoms, params.spinpol or params.hund)

        self.soc = params.soc
        self.nspins = self.ncomponents % 3
        self.spin_degeneracy = self.ncomponents % 2 + 1

        xcfunc = params.xc.functional(collinear=(self.ncomponents < 4))

        if self.ncomponents == 4 and xcfunc.type != 'LDA':
            raise ValueError('Only LDA supported for '
                             'SC Non-collinear calculations')

        self._backwards_comatible = params.experimental.get(
            'backwards_compatible', True)

        self.setups = Setups(
            atoms.numbers,
            params.setups,
            params.basis,
            xcfunc,
            world=comm,
            backwards_compatible=self._backwards_comatible)
        if params.hund:
            c = params.charge / len(atoms)
            for a, setup in enumerate(self.setups):
                self.initial_magmom_av[a, 2] = setup.get_hunds_rule_moment(c)

        symmetries = params.symmetry.build(
            atoms,
            setup_ids=self.setups.id_a,
            magmoms=self.initial_magmom_av,
            _backwards_compatible=self._backwards_comatible)

        use_time_reversal = params.symmetry.time_reversal

        symmetries._old_symmetry.time_reversal = use_time_reversal  # legacy
        self.setups.set_symmetry(symmetries._old_symmetry)  # legacy

        if self.ncomponents == 4:
            assert (len(symmetries) == 1 and not use_time_reversal)

        bz = params.kpts.build(atoms)
        self.ibz = bz.reduce(
            symmetries,
            strict=False,
            comm=comm,
            use_time_reversal=use_time_reversal)

        d = parallel.get('domain', 1 if xcfunc.type == 'HYB' else None)
        k = parallel.get('kpt', None)
        b = parallel.get('band', None)
        self.communicators = create_communicators(comm, len(self.ibz),
                                                  d, k, b, self.xp)

        if self.mode == 'fd':
            pass  # filter = create_fourier_filter(grid)
            # setups = setups.filter(filter)

        self.nbands = calculate_number_of_bands(params.nbands,
                                                self.setups,
                                                params.charge,
                                                self.initial_magmom_av,
                                                self.mode == 'lcao')
        if self.ncomponents == 4:
            self.nbands *= 2

        self.dtype: DTypeLike
        if params.mode.dtype is None:
            if self.params.mode.force_complex_dtype:
                self.dtype = complex
            else:
                if self.ibz.bz.gamma_only and self.ncomponents < 4:
                    self.dtype = float
                else:
                    self.dtype = complex
        else:
            self.dtype = params.mode.dtype

        self.grid, self.fine_grid = self.create_uniform_grids()

        self.relpos_ac = self.atoms.get_scaled_positions()
        self.relpos_ac %= 1
        self.relpos_ac %= 1  # yes, we need to do this twice!

        self.xc = create_functional(xcfunc, self.fine_grid, self.xp)

        self.interpolation_desc: Domain
        self.electrostatic_potential_desc: Domain

    def __repr__(self):
        return f'{self.__class__.__name__}({self.atoms}, {self.params})'

    def get_extensions(self):
        return [ext.build(self.atoms,
                          self.communicators,
                          self.log) for ext in self.params.extensions]

    @cached_property
    def charge(self) -> float:
        return self.setups.core_charge + self.params.charge

    @cached_property
    def nelectrons(self) -> float:
        return self.setups.nvalence - self.charge

    @cached_property
    def atomdist(self) -> AtomDistribution:
        return AtomDistribution(
            self.grid.ranks_from_fractional_positions(self.relpos_ac),
            self.grid.comm)

    def create_uniform_grids(self):
        raise NotImplementedError

    def check_cell(self, cell):
        number_of_lattice_vectors = cell.rank
        if number_of_lattice_vectors < 3:
            raise ValueError(
                'GPAW requires 3 lattice vectors.  '
                f'Your system has {number_of_lattice_vectors}.')
        angles = cell_to_cellpar(cell)[3:]
        if not all(40.0 < a < 140.0 for a in angles):
            a, b, c = angles
            warnings.warn(
                'The angles between your unit-cell vectors are '
                f'{a:.1}, {b:.1} and {c:.1} degrees.  '
                'Results may be wrong!  '
                'Please Niggli-reduce your unit-cell so that the angle '
                'are closer to 90 degrees:\n\n'
                '  from ase.build import niggli_reduce\n'
                '  nigli_reduce(atoms)\n')

    @cached_property
    def wf_desc(self) -> Domain:
        return self.create_wf_description()

    @cached_property
    def gpu(self) -> bool:
        """Are we running on a GPU?

        If parallel dict does not specify 'gpu': True or False,
        GPAW_USE_GPUS environment variable will be used to
        determine whether we use GPUs or not.
        """
        if self.params.parallel.get('gpu', GPAW_USE_GPUS):
            from gpaw.gpu import cupy_is_fake
            if cupy_is_fake and not GPAW_CPUPY:
                parallel_source = ('the `parallel` parameter'
                                   if self.params.parallel.get('gpu') else
                                   'the environment variable `GPAW_USE_GPUS`')
                raise ValueError(
                    f'GPU calculation is requested via {parallel_source}, '
                    'but the requisite CuPy library is not found; '
                    'please set GPAW_CPUPY=1 if you really want to do "GPU" '
                    'calculations with GPAW\'s fake CuPy library '
                    '(gpaw.gpu.cpupy)')
            return True
        return False

    @cached_property
    def xp(self) -> ModuleType:
        """Array module: Numpy or Cupy."""
        if self.gpu:
            from gpaw.gpu import cupy
            if cupy is fake_cupy:
                self.log(fake_cupy.FAKE_CUPY_WARNING)
            return cupy
        return np

    def create_wf_description(self) -> Domain:
        raise NotImplementedError

    def get_pseudo_core_densities(self):
        raise NotImplementedError

    def get_pseudo_core_ked(self):
        raise NotImplementedError

    def create_basis_set(self):
        return create_basis(self.ibz,
                            self.ncomponents % 3,
                            self.atoms.pbc,
                            self.grid,
                            self.setups,
                            self.dtype,
                            self.relpos_ac,
                            self.communicators['w'],
                            self.communicators['k'],
                            self.communicators['b'])

    def density_from_superposition(self, basis_set):
        return Density.from_superposition(
            grid=self.grid,
            nct_aX=self.get_pseudo_core_densities(),
            tauct_aX=self.get_pseudo_core_ked(),
            atomdist=self.atomdist,
            setups=self.setups,
            basis_set=basis_set,
            magmom_av=self.initial_magmom_av,
            ncomponents=self.ncomponents,
            charge=self.charge,
            hund=self.params.hund,
            mgga=self.xc.type == 'MGGA')

    def create_occupation_number_calculator(self):
        return OccupationNumberCalculator(
            self.params.occupations.params,
            self.atoms.pbc,
            self.ibz,
            self.nbands,
            self.communicators,
            self.initial_magmom_av.sum(0),
            self.ncomponents,
            self.nelectrons,
            np.linalg.inv(self.atoms.cell.complete()).T)

    def create_ibz_wave_functions(self,
                                  basis: BasisFunctions,
                                  potential: Potential) -> IBZWaveFunctions:
        raise NotImplementedError

    def create_hamiltonian_operator(self):
        raise NotImplementedError

    def create_eigensolver(self, hamiltonian):
        raise NotImplementedError

    def create_scf_loop(self):
        hamiltonian = self.create_hamiltonian_operator()
        occ_calc = self.create_occupation_number_calculator()
        eigensolver = self.create_eigensolver(hamiltonian)

        mixer = MixerWrapper(
            get_mixer_from_keywords(self.atoms.pbc.any(),
                                    self.ncomponents,
                                    **self.params.mixer.params),
            self.ncomponents,
            self.grid._gd,
            world=self.communicators['w'])

        return SCFLoop(hamiltonian, occ_calc,
                       eigensolver, mixer, self.communicators['w'],
                       {key: value
                        for key, value in self.params.convergence.items()
                        if key != 'bands'},
                       self.params.maxiter)

    def read_ibz_wave_functions(self, reader):
        raise NotImplementedError

    def create_potential_calculator(self):
        raise NotImplementedError

    def read_wavefunction_values(self,
                                 reader,
                                 ibzwfs: IBZWaveFunctions) -> None:
        """Read eigenvalues, occuptions and projections and fermi levels.

        The values are read using reader and set as the appropriate properties
        of (the already instantiated) wavefunctions contained in ibzwfs
        """
        ha = reader.ha

        domain_comm = self.communicators['d']
        band_comm = self.communicators['b']

        eig_skn = reader.wave_functions.eigenvalues
        occ_skn = reader.wave_functions.occupations

        for wfs in ibzwfs:
            index: tuple[int, ...]
            if self.ncomponents < 4:
                dims = [self.nbands]
                index = (wfs.spin, wfs.k)
            else:
                dims = [self.nbands, 2]
                index = (wfs.k,)

            wfs._eig_n = eig_skn[index] / ha
            wfs._occ_n = occ_skn[index]
            layout = AtomArraysLayout([(setup.ni,) for setup in self.setups],
                                      atomdist=self.atomdist,
                                      dtype=self.dtype)
            P_ani = AtomArrays(layout, dims=dims, comm=band_comm)

            if domain_comm.rank == 0:
                try:
                    P_nI = reader.wave_functions.proxy('projections', *index)
                except KeyError:
                    data = None
                else:
                    b1, b2 = P_ani.my_slice()  # my bands
                    data = P_nI[b1:b2].astype(ibzwfs.dtype)  # read from file
            else:
                data = None

            have_projections = broadcast(
                data is not None if domain_comm.rank == 0 else None,
                comm=domain_comm)

            if have_projections:
                P_ani.scatter_from(data)  # distribute over atoms
                wfs._P_ani = P_ani
            else:
                wfs._P_ani = None

        try:
            ibzwfs.fermi_levels = reader.wave_functions.fermi_levels / ha
        except AttributeError:
            # old gpw-file
            ibzwfs.fermi_levels = np.array(
                [reader.occupations.fermilevel / ha])

    def create_environment(self, grid):
        return self.params.environment.build(
            setups=self.setups,
            grid=grid, relpos_ac=self.relpos_ac, log=self.log,
            comm=self.communicators['w'])


def create_communicators(comm: MPIComm = None,
                         nibzkpts: int = 1,
                         domain: int | tuple[int, int, int] | None = None,
                         kpt: int = None,
                         band: int = None,
                         xp: ModuleType = np) -> dict[str, MPIComm]:
    parallelization = Parallelization(comm or world, nibzkpts)
    if domain is not None and not isinstance(domain, int):
        domain = prod(domain)
    parallelization.set(kpt=kpt,
                        domain=domain,
                        band=band)
    comms = parallelization.build_communicators()
    comms['w'] = comm

    # We replace size=1 MPI communications with serial_comm so that
    # serial_comm.sum(<cupy-array>) works: XXX
    comms = {key: comm if comm.size > 1 else serial_comm
             for key, comm in comms.items()}

    if xp is not np and not GPU_AWARE_MPI:
        comms = {key: CuPyMPI(comm) for key, comm in comms.items()}

    return comms


def create_fourier_filter(grid):
    gamma = 1.6

    h = ((grid.icell**2).sum(1)**-0.5 / grid.size).max()

    def filter(rgd, rcut, f_r, l=0):
        gcut = np.pi / h - 2 / rcut / gamma
        ftmp = rgd.filter(f_r, rcut * gamma, gcut, l)
        f_r[:] = ftmp[:len(f_r)]

    return filter


def normalize_initial_magmoms(
        atoms: Atoms,
        magmoms: ArrayLike2D | ArrayLike1D | float | None = None,
        force_spinpol_calculation: bool = False) -> tuple[Array2D, int]:
    """Convert magnetic moments to (natoms, 3)-shaped array.

    Also return number of wave function components (1, 2 or 4).

    >>> h = Atoms('H', magmoms=[1])
    >>> normalize_initial_magmoms(h)
    (array([[0., 0., 1.]]), 2)
    >>> normalize_initial_magmoms(h, [[1, 0, 0]])
    (array([[1., 0., 0.]]), 4)
    """
    magmom_av = np.zeros((len(atoms), 3))
    ncomponents = 2

    if magmoms is None:
        magmom_av[:, 2] = atoms.get_initial_magnetic_moments()
    elif isinstance(magmoms, float):
        magmom_av[:, 2] = magmoms
    else:
        magmoms = np.asarray(magmoms)
        if magmoms.ndim == 1:
            magmom_av[:, 2] = magmoms
        else:
            magmom_av[:] = magmoms
            ncomponents = 4

    if (ncomponents == 2 and
        not force_spinpol_calculation and
        not magmom_av[:, 2].any()):
        ncomponents = 1

    return magmom_av, ncomponents


def ____create_kpts(kpts: dict[str, Any], atoms: Atoms) -> BZPoints:
    if 'kpts' in kpts:
        bz = BZPoints(kpts['kpts'])
    elif 'path' in kpts:
        path = atoms.cell.bandpath(pbc=atoms.pbc, **kpts)
        bz = BZPoints(path.kpts)
    else:
        size, offset = kpts2sizeandoffsets(**kpts, atoms=atoms)
        bz = MonkhorstPackKPoints(size, offset)
    for c, periodic in enumerate(atoms.pbc):
        if not periodic and not np.allclose(bz.kpt_Kc[:, c], 0.0):
            raise ValueError('K-points can only be used with PBCs!')
    return bz


def calculate_number_of_bands(nbands: int | str | None,
                              setups: Setups,
                              charge: float,
                              initial_magmom_av: Array2D,
                              is_lcao: bool) -> int:
    nao = setups.nao
    nvalence = setups.nvalence - charge
    M = np.linalg.norm(initial_magmom_av.sum(0))

    orbital_free = any(setup.orbital_free for setup in setups)
    if orbital_free:
        return 1

    if nbands is None:
        # Number of bound partial waves:
        nbandsmax = sum(setup.get_default_nbands()
                        for setup in setups)
        N = int(np.ceil(1.2 * (nvalence + M) / 2)) + 4
        N = min(N, nbandsmax)
        if is_lcao and N > nao:
            N = nao
    elif isinstance(nbands, str):
        if nbands == 'nao':
            N = nao
        elif nbands[-1] == '%':
            cfgbands = (nvalence + M) / 2
            N = int(np.ceil(float(nbands[:-1]) / 100 * cfgbands))
        else:
            url = 'https://gpaw.readthedocs.io/documentation/basic.html'
            raise ValueError(
                f'Bad value for nbands: {nbands!r}.  '
                f'See {url}#manual-nbands for help')
    elif nbands <= 0:
        N = max(1, int(nvalence + M + 0.5) // 2 + (-nbands))
    else:
        N = nbands

    if N > nao and is_lcao:
        raise ValueError('Too many bands for LCAO calculation: '
                         f'{nbands}%d bands and only {nao} atomic orbitals!')

    if nvalence < 0:
        raise ValueError(
            f'Charge {charge} is not possible - not enough valence electrons')

    if nvalence > 2 * N:
        raise ValueError(
            f'Too few bands!  Electrons: {nvalence}, bands: {nbands}')

    return N


def create_uniform_grid(mode: str,
                        gpts,
                        cell,
                        pbc,
                        symmetries,
                        h: float | None = None,
                        interpolation: int | str | None = None,
                        ecut: float = None,
                        comm: MPIComm = serial_comm) -> UGDesc:
    """Create grid in a backwards compatible way."""
    cell = cell / Bohr
    if h is not None:
        h /= Bohr

    realspace = (mode != 'pw' and interpolation != 'fft')
    if realspace:
        zerobc = [not periodic for periodic in pbc]
    else:
        zerobc = [False] * 3

    if gpts is not None:
        size = gpts
    else:
        modeobj = SimpleNamespace(name=mode, ecut=ecut)
        size = get_number_of_grid_points(cell, h, modeobj, realspace,
                                         symmetries)
    return UGDesc(cell=cell, pbc=pbc, zerobc=zerobc, size=size, comm=comm)
