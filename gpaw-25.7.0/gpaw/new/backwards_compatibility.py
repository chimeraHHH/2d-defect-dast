from functools import cached_property
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.units import Bohr

from gpaw.band_descriptor import BandDescriptor
from gpaw.fftw import MEASURE
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.new import prod, zips
from gpaw.new.calculation import DFTCalculation
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.projections import Projections
from gpaw.pw.descriptor import PWDescriptor
from gpaw.utilities import pack_density
from gpaw.utilities.timing import nulltimer
from gpaw.wavefunctions.arrays import (PlaneWaveExpansionWaveFunctions,
                                       UniformGridWaveFunctions)


class PT:
    def __init__(self, ibzwfs):
        self.ibzwfs = ibzwfs

    def integrate(self, psit_nG, P_ani, q):
        pt_aiX = self.ibzwfs.wfs_qs[q][0].pt_aiX
        pt_aiX._lazy_init()
        pt_aiX._lfc.integrate(psit_nG, P_ani, q=0)

    def add(self, psit_nG, c_axi, q):
        self.ibzwfs.wfs_qs[q][0].pt_aiX._lfc.add(psit_nG, c_axi, q=0)

    def dict(self, shape):
        return self.ibzwfs.wfs_qs[0][0].pt_aiX.empty(shape,
                                                     self.ibzwfs.band_comm)


class FakeWFS:
    def __init__(self,
                 ibzwfs,
                 density,
                 potential,
                 setups,
                 comm,
                 occ_calc,
                 hamiltonian,
                 atoms: Atoms,
                 scale_pw_coefs=False):
        from gpaw.utilities.partition import AtomPartition
        self.timer = nulltimer
        self.setups = setups
        self.ibzwfs = ibzwfs
        self.density = density
        self.potential = potential
        self.hamiltonian = hamiltonian
        ibz = ibzwfs.ibz
        self.kd = kd = KPointDescriptor(ibz.bz.kpt_Kc, ibzwfs.nspins)
        kd.ibzk_kc = ibz.kpt_kc
        kd.weight_k = ibz.weight_k
        kd.sym_k = ibz.s_K
        kd.time_reversal_k = ibz.time_reversal_K
        kd.bz2ibz_k = ibz.bz2ibz_K
        kd.ibz2bz_k = ibz.ibz2bz_k
        kd.bz2bz_ks = ibz.bz2bz_Ks
        kd.nibzkpts = len(ibz)
        kd.symmetry = ibz.symmetries._old_symmetry
        kd.set_communicator(ibzwfs.kpt_comm)
        self.bd = BandDescriptor(ibzwfs.nbands, ibzwfs.band_comm)
        self.grid = density.nt_sR.desc
        self.gd = self.grid._gd
        atomdist = density.D_asii.layout.atomdist
        self.atom_partition = AtomPartition(atomdist.comm, atomdist.rank_a)
        # self.setups.set_symmetry(ibzwfs.ibz.symmetries.symmetry)
        self.occ_calc = occ_calc
        self.occupations = occ_calc.occ
        self.nvalence = int(round(density.nvalence))
        self.nvalence = density.nvalence
        # assert self.nvalence == density.nvalence
        self.world = comm
        if ibzwfs.fermi_levels is not None:
            self.fermi_levels = ibzwfs.fermi_levels
            if len(self.fermi_levels) == 1:
                self.fermi_level = self.fermi_levels[0]
        self.nspins = ibzwfs.nspins
        self.dtype = ibzwfs.dtype
        wfs = ibzwfs.wfs_qs[0][0]
        self.pd = None
        self.basis_functions = getattr(wfs,  # dft.scf_loop.hamiltonian,
                                       'basis', None)
        if isinstance(wfs, PWFDWaveFunctions):
            if hasattr(wfs.psit_nX.desc, 'ecut'):
                self.mode = 'pw'
                self.ecut = wfs.psit_nX.desc.ecut
                self.pd = PWDescriptor(self.ecut,
                                       self.gd, self.dtype, self.kd, _new=True)
                self.pwgrid = self.grid.new(dtype=self.dtype)
            else:
                self.mode = 'fd'
        else:
            self.mode = 'lcao'
            self.manytci = wfs.tci_derivatives.manytci
            if self.basis_functions is not None:
                self.ksl = SimpleNamespace(Mstart=self.basis_functions.Mstart,
                                           Mstop=self.basis_functions.Mstop)
        self.collinear = wfs.ncomponents < 4
        self.positions_set = True
        self.read_from_file_init_wfs_dm = ibzwfs.read_from_file_init_wfs_dm

        self.pt = PT(ibzwfs)
        self.scalapack_parameters = (None, 1, 1, 128)
        self.ngpts = prod(self.gd.N_c)
        if self.mode == 'pw' and scale_pw_coefs:
            self.scale = self.ngpts
        else:
            self.scale = 1
        self.fftwflags = MEASURE

    def apply_pseudo_hamiltonian(self, kpt, ham, a1, a2):
        desc = self.ibzwfs.wfs_qs[kpt.q][0].psit_nX.desc
        self.hamiltonian.apply(
            self.potential.vt_sR,
            None,
            self.ibzwfs,  # needed for hybrids
            getattr(ham, 'D_asii', None),  # needed for hybrids
            desc.from_data(data=a1),
            desc.from_data(data=a2),
            kpt.s)

    def calculate_occupation_numbers(self, fixed):
        self.ibzwfs.calculate_occs(
            self.occ_calc,
            fix_fermi_level=fixed)

    def empty(self, n, q):
        return np.empty((n,) +
                        self.ibzwfs.wfs_qs[q][0].psit_nX.data.shape[1:],
                        complex if self.mode == 'pw' else self.dtype)

    @cached_property
    def work_array(self):
        return np.empty(
            (self.bd.mynbands,) + self.ibzwfs.get_max_shape(),
            complex if self.mode == 'pw' else self.dtype)

    @cached_property
    def work_matrix_nn(self):
        from gpaw.matrix import Matrix
        return Matrix(
            self.bd.nbands, self.bd.nbands,
            dtype=self.dtype,
            dist=(self.bd.comm, self.bd.comm.size))

    @property
    def orthonormalized(self):
        return self.ibzwfs.wfs_qs[0][0].orthonormalized

    def orthonormalize(self, kpt=None):
        if kpt is None:
            kpts = list(self.ibzwfs)
        else:
            kpts = [self.ibzwfs.wfs_qs[kpt.q][kpt.s]]
        for wfs in kpts:
            wfs._P_ani = None
            wfs.orthonormalized = False
            wfs.orthonormalize()

    def make_preconditioner(self, blocksize):
        if self.mode == 'pw':
            from gpaw.wavefunctions.pw import Preconditioner
            return Preconditioner(self.pd.G2_qG, self.pd,
                                  _scale=self.ngpts**2)
        from gpaw.preconditioner import Preconditioner
        return Preconditioner(self.gd, self.hamiltonian.kin, self.dtype,
                              blocksize)

    def _get_wave_function_array(self, u, n, realspace=True, periodic=False):
        assert realspace and not periodic
        psit_X = self.kpt_u[u].wfs.psit_nX[n]
        if hasattr(psit_X, 'ifft'):
            psit_R = psit_X.ifft(grid=self.pwgrid, periodic=True)
            psit_R.multiply_by_eikr(psit_X.desc.kpt_c)
            return psit_R.data
        return psit_X.data

    def get_wave_function_array(self, n, k, s,
                                realspace=True,
                                periodic=False,
                                cut=False):
        assert not cut
        assert self.ibzwfs.band_comm.size == 1
        assert self.ibzwfs.kpt_comm.size == 1
        if self.mode == 'lcao':
            assert not realspace
            return self.kpt_qs[k][s].C_nM[n]
        psit_X = self.kpt_qs[k][s].wfs.psit_nX[n]
        if not realspace:
            return psit_X.data
        if self.mode == 'pw':
            psit_R = psit_X.ifft(grid=self.pwgrid, periodic=True)
            if not periodic:
                psit_R.multiply_by_eikr(psit_X.desc.kpt_c)
        else:
            psit_R = psit_X
            if periodic:
                psit_R.multiply_by_eikr(-psit_R.desc.kpt_c)
        return psit_R.data

    def collect_projections(self, k, s):
        return self.kpt_qs[k][s].projections.collect()

    def collect_eigenvalues(self, k, s):
        return self.ibzwfs.wfs_qs[k][s].eig_n.copy()

    @cached_property
    def kpt_u(self):
        return [kpt
                for kpt_s in self.kpt_qs
                for kpt in kpt_s]

    @cached_property
    def kpt_qs(self):
        return [[KPT(self.mode, wfs, self.atom_partition, self.scale,
                     self.pd, self.gd)
                 for wfs in wfs_s]
                for wfs_s in self.ibzwfs.wfs_qs]

    def integrate(self, a_nX, b_nX, global_integral):
        if self.mode == 'fd':
            return self.gd.integrate(a_nX, b_nX, global_integral)
        x = self.pd.integrate(a_nX, b_nX, global_integral)
        return self.ngpts**2 * x


class KPT:
    def __init__(self, mode, wfs, atom_partition, scale, pd, gd):
        self.mode = mode
        self.scale = scale
        self.wfs = wfs
        self.pd = pd
        self.gd = gd

        try:
            I1 = 0
            nproj_a = []
            for a, shape in enumerate(wfs.P_ani.layout.shape_a):
                I2 = I1 + prod(shape)
                nproj_a.append(I2 - I1)
                I1 = I2
        except RuntimeError:
            pass
        else:
            self.projections = Projections(
                wfs.nbands,
                nproj_a,
                atom_partition,
                wfs.P_ani.comm,
                wfs.ncomponents < 4,
                wfs.spin,
                data=wfs.P_ani.data)

        self.s = wfs.spin if wfs.ncomponents < 4 else None
        self.k = wfs.k
        self.q = wfs.q
        self.weight = wfs.spin_degeneracy * wfs.weight
        self.weightk = wfs.weight
        if isinstance(wfs, PWFDWaveFunctions):
            self.psit_nX = wfs.psit_nX
        else:
            self.C_nM = wfs.C_nM.data
            self.S_MM = wfs.S_MM.data
            self.P_aMi = wfs.P_aMi
        if mode == 'fd':
            self.phase_cd = wfs.psit_nX.desc.phase_factor_cd

    @property
    def P_ani(self):
        return self.wfs.P_ani

    @property
    def eps_n(self):
        return self.wfs.myeig_n

    @property
    def f_n(self):
        f_n = self.wfs.myocc_n * self.weight
        f_n.flags.writeable = False
        return f_n

    @f_n.setter
    def f_n(self, val):
        self.wfs.myocc_n[:] = val / self.weight

    @property
    def psit_nG(self):
        if not hasattr(self, 'psit_nX'):
            return None
        data = self.psit_nX.data
        if self.scale == 1:
            return data
        if 1:  # isinstance(data, np.ndarray):
            return data * self.scale
        data.scale *= self.scale
        return data

    @cached_property
    def psit(self):
        band_comm = self.psit_nX.comm
        if self.mode == 'pw':
            return PlaneWaveExpansionWaveFunctions(
                self.wfs.nbands, self.pd, self.wfs.dtype,
                self.psit_nG,
                kpt=self.q,
                dist=(band_comm, band_comm.size),
                spin=self.s,
                collinear=self.wfs.ncomponents != 4)
        return UniformGridWaveFunctions(
            self.wfs.nbands, self.gd, self.wfs.dtype,
            self.psit_nX.data,
            kpt=self.q,
            dist=(band_comm, band_comm.size),
            spin=self.s,
            collinear=self.wfs.ncomponents != 4)


class FakeDensity:
    def __init__(self, dft: DFTCalculation):
        self.setups = dft.setups
        self.D_asii = dft.density.D_asii
        self.atom_partition = dft._atom_partition
        try:
            self.interpolate = dft.pot_calc._interpolate_density
            self.finegd = dft.pot_calc.fine_grid._gd
        except AttributeError:
            pass
        self.nt_sR = dft.density.nt_sR
        self.nt_sG = self.nt_sR.data
        self.gd = self.nt_sR.desc._gd
        self._densities = dft.densities()
        self.ncomponents = len(self.nt_sG)
        self.nspins = self.ncomponents % 3
        self.collinear = self.ncomponents < 4

    @cached_property
    def D_asp(self):
        D_asp = self.setups.empty_atomic_matrix(self.ncomponents,
                                                self.atom_partition)
        D_asp.update({a: np.array([pack_density(D_ii) for D_ii in D_sii.real])
                      for a, D_sii in self.D_asii.items()})
        return D_asp

    @cached_property
    def nt_sg(self):
        return self.interpolate(self.nt_sR)[0].data

    def interpolate_pseudo_density(self):
        pass

    def get_all_electron_density(self, *, atoms, gridrefinement):
        n_sr = self._densities.all_electron_densities(
            grid_refinement=gridrefinement).scaled(1 / Bohr, Bohr**3)
        return n_sr.data, n_sr.desc._gd


class FakeHamiltonian:
    def __init__(self, ibzwfs, density, potential, pot_calc,
                 e_total_free=np.nan,
                 e_xc=np.nan):
        self.pot_calc = pot_calc
        self.ibzwfs = ibzwfs
        self.density = density
        self.potential = potential
        try:
            self.finegd = self.pot_calc.fine_grid._gd
        except AttributeError:
            pass
        self.grid = potential.vt_sR.desc
        self.e_total_free = e_total_free
        self.e_xc = e_xc

    def update(self, dens, wfs, kin_en_using_band=True):
        self.potential, _ = self.pot_calc.calculate(
            self.density, self.ibzwfs, self.potential.vHt_x)

        energies = self.potential.energies
        self.e_xc = energies['xc']
        self.e_coulomb = energies['coulomb']
        self.e_zero = energies['zero']
        self.e_external = energies['external']

        if kin_en_using_band:
            self.e_kinetic0 = energies['kinetic']
        else:
            self.e_kinetic0 = self.ibzwfs.calculate_kinetic_energy(
                wfs.hamiltonian, self.density)
            self.ibzwfs.energies['exx_kinetic'] = 0.0
            energies['kinetic'] = self.e_kinetic0

    def get_energy(self, e_entropy, wfs, kin_en_using_band=True, e_sic=None):
        self.e_band = self.ibzwfs.energies['band']
        if kin_en_using_band:
            self.e_kinetic = self.e_kinetic0 + self.e_band
        else:
            self.e_kinetic = self.e_kinetic0
        self.e_entropy = e_entropy
        if 0:
            print(self.e_kinetic0,
                  self.e_band,
                  self.e_coulomb,
                  self.e_external,
                  self.e_zero,
                  self.e_xc,
                  self.e_entropy)
        self.e_total_free = (self.e_kinetic + self.e_coulomb +
                             self.e_external + self.e_zero + self.e_xc +
                             self.e_entropy)

        if e_sic is not None:
            self.e_sic = e_sic
            self.e_total_free += e_sic

        self.e_total_extrapolated = (
            self.e_total_free +
            self.ibzwfs.energies['extrapolation'])

        return self.e_total_free

    def restrict_and_collect(self, vxct_sg):
        fine_grid = self.pot_calc.fine_grid
        vxct_sr = fine_grid.empty(len(vxct_sg))
        vxct_sr.data[:] = vxct_sg
        vxct_sR = self.grid.empty(len(vxct_sg))
        for vxct_r, vxct_R in zips(vxct_sr, vxct_sR):
            self.pot_calc.restrict(vxct_r, vxct_R)
        return vxct_sR.data

    @property
    def xc(self):
        return self.pot_calc.xc.xc

    def dH(self, P, out):
        for a, I1, I2 in P.indices:
            dH_ii = self.potential.dH_asii[a][P.spin]
            out.array[:, I1:I2] = np.dot(P.array[:, I1:I2], dH_ii)
