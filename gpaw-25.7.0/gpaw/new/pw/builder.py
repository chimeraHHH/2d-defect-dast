from __future__ import annotations

import os
import warnings
from functools import cached_property

import numpy as np
from ase.units import Ha

from gpaw.core import PWDesc, UGDesc
from gpaw.core.domain import Domain
from gpaw.core.matrix import Matrix
from gpaw.core.plane_waves import PWArray
from gpaw.new import zips
from gpaw.new.builder import create_uniform_grid
from gpaw.new.gpw import as_double_precision
from gpaw.new.pw.bloechl_poisson import BloechlPAWPoissonSolver
from gpaw.new.pw.hamiltonian import PWHamiltonian, SpinorPWHamiltonian
from gpaw.new.pw.hybrids import PWHybridHamiltonian
from gpaw.new.pw.hybridsk import PWHybridHamiltonianK
from gpaw.new.pw.paw_poisson import SlowPAWPoissonSolver
from gpaw.new.pw.poisson import make_poisson_solver
from gpaw.new.pw.pot_calc import PlaneWavePotentialCalculator
from gpaw.new.pwfd.builder import PWFDDFTComponentsBuilder
from gpaw.new.xc import create_functional
from gpaw.typing import Array1D


class PWDFTComponentsBuilder(PWFDDFTComponentsBuilder):
    def __init__(self,
                 atoms,
                 params,
                 *,
                 comm=None,
                 log=None):
        mode = params.mode
        self.ecut = mode.ecut / Ha
        # mode.dedecut ???
        super().__init__(atoms, params, comm=comm, log=log)

        self._nct_ag = None
        self._tauct_ag = None

        nthreads = int(os.environ.get('OMP_NUM_THREADS', '') or '1')
        if nthreads > 1:
            warnings.warn(
                'Using OMP_NUM_THREADS>1 in PW-mode is not useful!')
        # We should just distribute the atom evenly, but that is not compatible
        # with LCAO initialization!
        # return AtomDistribution.from_number_of_atoms(len(self.relpos_ac),
        #                                              self.communicators['d'])

    def create_uniform_grids(self):
        grid = create_uniform_grid(
            'pw',
            self.params.gpts,
            self.atoms.cell,
            self.atoms.pbc,
            self.ibz.symmetries,
            h=self.params.h,
            interpolation=self.params.interpolation or 'fft',
            ecut=self.ecut,
            comm=self.communicators['d'])
        fine_grid = grid.new(size=grid.size_c * 2)
        # decomposition=[2 * d for d in grid.decomposition]
        return grid, fine_grid

    def create_wf_description(self) -> Domain:
        return PWDesc(ecut=self.ecut,
                      cell=self.grid.cell,
                      comm=self.grid.comm,
                      dtype=self.dtype)

    def create_xc_functional(self):
        return create_functional(self._xc,
                                 self.fine_grid, self.xp)

    @cached_property
    def interpolation_desc(self):
        """Plane-wave set used for interpolating from corse to fine grid."""
        # By default, the size of the grid used for the FFT's (self.grid)
        # will acommodate G-vectors up to 2 * self.ecut, but the grid-size
        # could have been set using h=... or gpts=...
        ecut = min(2 * self.ecut, self.grid.ekin_max())
        return PWDesc(ecut=ecut,
                      cell=self.grid.cell,
                      comm=self.grid.comm)

    @cached_property
    def electrostatic_potential_desc(self):
        if self.fast_poisson_solver:
            return self.interpolation_desc
        return self.interpolation_desc.new(ecut=8 * self.ecut)

    @cached_property
    def fast_poisson_solver(self) -> bool:
        fast = self.params.poissonsolver.params.get('fast', False)
        if fast:
            # Only works for gaussian compensation charges at the moment:
            fast = False
            for s in self.setups:
                if not hasattr(s, 'data'):
                    break
                if s.data.shape_function['type'] != 'gauss':
                    break
            else:  # no break
                fast = True
        return fast

    def get_pseudo_core_densities(self):
        if self._nct_ag is None:
            self._nct_ag = self.setups.create_pseudo_core_densities(
                self.interpolation_desc, self.relpos_ac, self.atomdist,
                xp=self.xp)
        return self._nct_ag

    def get_pseudo_core_ked(self):
        if self._tauct_ag is None:
            self._tauct_ag = self.setups.create_pseudo_core_ked(
                self.interpolation_desc, self.relpos_ac, self.atomdist)
        return self._tauct_ag

    def create_poisson_solver(self, env):
        psparams = self.params.poissonsolver.params.copy() or {'strength': 1.0}
        psparams.pop('fast', False)

        if self.fast_poisson_solver:
            grid = self.grid
        else:
            grid = self.fine_grid

        pw = self.electrostatic_potential_desc
        ps = make_poisson_solver(pw,
                                 grid,
                                 self.charge,
                                 env,
                                 **psparams)

        if self.fast_poisson_solver:
            cutoff_a = [s.data.shape_function['rc'] for s in self.setups]
            return BloechlPAWPoissonSolver(
                pw, cutoff_a, ps, self.relpos_ac, self.atomdist, self.xp)

        return SlowPAWPoissonSolver(
            self.interpolation_desc,
            self.setups,
            ps, self.relpos_ac, self.atomdist, self.xp)

    def create_potential_calculator(self):
        env = self.create_environment(self.fine_grid)
        return PlaneWavePotentialCalculator(
            self.grid, self.fine_grid,
            self.interpolation_desc,
            self.setups,
            self.xc,
            self.create_poisson_solver(env),
            relpos_ac=self.relpos_ac,
            atomdist=self.atomdist,
            soc=self.soc,
            xp=self.xp,
            environment=env,
            extensions=self.get_extensions())

    def create_hamiltonian_operator(self, blocksize=10):
        if self.ncomponents < 4:
            if self.xc.exx_fraction == 0.0:
                return PWHamiltonian(self.grid, self.wf_desc, self.xp)
            assert self.communicators['d'].size == 1
            assert self.communicators['k'].size == 1
            assert self.nbands % self.communicators['b'].size == 0
            if self.dtype is float:
                return PWHybridHamiltonian(
                    self.grid, self.wf_desc, self.xc, self.setups,
                    self.relpos_ac, self.atomdist,
                    comp_charge_in_real_space=self.params.experimental.get(
                        'ccirs'))
            else:
                return PWHybridHamiltonianK(
                    self.grid, self.wf_desc, self.xc, self.setups,
                    self.relpos_ac, self.atomdist, self.log,
                    self.communicators['k'],
                    self.communicators['w'])
        return SpinorPWHamiltonian(self.qspiral_v)

    def convert_wave_functions_from_uniform_grid(self,
                                                 C_nM: Matrix,
                                                 basis_set,
                                                 kpt_c,
                                                 q):
        if self.params.experimental.get('fast_pw_init', True):
            if self.ncomponents < 4:
                from gpaw.core.pwacf import PWAtomCenteredFunctions
                pw = self.wf_desc.new(kpt=kpt_c)
                phit_aJG = PWAtomCenteredFunctions(
                    [setup.basis_functions_J for setup in self.setups],
                    self.relpos_ac,
                    pw,
                    atomdist=self.atomdist,
                    xp=self.xp)
                psit_nG = pw.empty(self.nbands,
                                   comm=self.communicators['b'],
                                   xp=self.xp)
                mynbands, M = C_nM.dist.shape
                phit_aJG.multiply(C_nM.to_xp(self.xp).to_dtype(pw.dtype),
                                  out_nG=psit_nG[:mynbands])
                return psit_nG

        lcao_dtype = complex if \
            np.issubdtype(self.dtype, np.complexfloating) else float

        grid = self.grid.new(kpt=kpt_c, dtype=lcao_dtype)
        pw = self.wf_desc.new(kpt=kpt_c, dtype=lcao_dtype)
        if self.dtype != lcao_dtype:
            pw_correct = self.wf_desc.new(kpt=kpt_c, dtype=self.dtype)

        if np.issubdtype(self.dtype, np.complexfloating):
            emikr_R = grid.eikr(-kpt_c)

        mynbands, M = C_nM.dist.shape
        if self.ncomponents < 4:
            psit_nG = pw.empty(self.nbands, self.communicators['b'])
            psit_nR = grid.zeros(mynbands)
            basis_set.lcao_to_grid(C_nM.data, psit_nR.data, q)

            for psit_R, psit_G in zips(psit_nR, psit_nG, strict=False):
                if np.issubdtype(self.dtype, np.complexfloating):
                    psit_R.data *= emikr_R
                psit_R.fft(out=psit_G)

            if self.dtype != lcao_dtype:
                psit2_nG = pw_correct.empty(self.nbands,
                                            self.communicators['b'])
                psit2_nG.data[:] = psit_nG.data
                return psit2_nG.to_xp(self.xp)
            return psit_nG.to_xp(self.xp)
        else:
            psit_nsG = pw.empty((self.nbands, 2), self.communicators['b'])
            psit_sR = grid.empty(2)
            C_nsM = C_nM.data.reshape((mynbands, 2, M // 2))
            for psit_sG, C_sM in zips(psit_nsG, C_nsM, strict=False):
                psit_sR.data[:] = 0.0
                basis_set.lcao_to_grid(C_sM, psit_sR.data, q)
                psit_sR.data *= emikr_R
                for psit_G, psit_R in zips(psit_sG, psit_sR):
                    psit_R.fft(out=psit_G)
            return psit_nsG

    def read_ibz_wave_functions(self, reader):
        ibzwfs = super().read_ibz_wave_functions(reader)

        if 'coefficients' not in reader.wave_functions:
            return ibzwfs

        singlep = reader.get('precision', 'double') == 'single'
        c = reader.bohr**1.5
        if reader.version < 0:
            c = 1  # very old gpw file
        elif reader.version < 4:
            c /= self.grid.size_c.prod()

        index_kG = reader.wave_functions.indices

        if self.ncomponents == 4:
            shape = (self.nbands, 2)
        else:
            shape = (self.nbands,)

        for wfs in ibzwfs:
            pw = self.wf_desc.new(kpt=wfs.kpt_c)
            if wfs.spin == 0:
                check_g_vector_ordering(self.grid, pw, index_kG[wfs.k])

            index = (wfs.spin, wfs.k) if self.ncomponents != 4 else (wfs.k,)
            data = reader.wave_functions.proxy('coefficients', *index)
            data.scale = c
            data.length_of_last_dimension = pw.shape[-1]

            if self.communicators['w'].size == 1 and not singlep:
                orig_shape = data.shape
                data.shape = shape + pw.shape
                wfs.psit_nX = pw.from_data(data)
                data.shape = orig_shape
            else:
                band_comm = self.communicators['b']
                wfs.psit_nX = PWArray(pw, shape, comm=band_comm)
                mynbands = (self.nbands +
                            band_comm.size - 1) // band_comm.size
                n1 = min(band_comm.rank * mynbands, self.nbands)
                n2 = min((band_comm.rank + 1) * mynbands, self.nbands)
                if pw.comm.rank == 0:
                    assert wfs.psit_nX.mydims[0] == n2 - n1
                    data = data[n1:n2]  # read from file
                else:
                    data = [None] * (n2 - n1)
                for psit_G, array in zips(wfs.psit_nX, data):
                    if singlep:
                        psit_G.scatter_from(as_double_precision(array))
                    else:
                        psit_G.scatter_from(array)

        return ibzwfs


def check_g_vector_ordering(grid: UGDesc,
                            pw: PWDesc,
                            index_G: Array1D) -> None:
    size = tuple(grid.size)
    if np.issubdtype(pw.dtype, np.floating):
        size = (size[0], size[1], size[2] // 2 + 1)
    index0_G = pw.indices(size)
    nG = len(index0_G)
    assert (index0_G == index_G[:nG]).all()
    assert (index_G[nG:] == -1).all()
