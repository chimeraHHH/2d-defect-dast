from math import pi

import numpy as np

from gpaw.new.builder import DFTComponentsBuilder
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunctions
from gpaw.new.lcao.eigensolver import LCAOEigensolver
from gpaw.new.lcao.hamiltonian import LCAOHamiltonian
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.core.arrays import XArrayWithNoData


class PWFDDFTComponentsBuilder(DFTComponentsBuilder):
    def __init__(self,
                 atoms,
                 params,
                 *,
                 comm=None,
                 log=None):
        super().__init__(atoms, params, comm=comm, log=log)
        qspiral = params.mode.qspiral
        self.qspiral_v = (None if qspiral is None else
                          qspiral @ self.grid.icell * (2 * pi))

    def create_eigensolver(self, hamiltonian):
        from gpaw.dft import DefaultEigensolver
        es = self.params.eigensolver
        if isinstance(es, DefaultEigensolver):
            es = es.from_param({'name': 'davidson', **es.params})
        return es.build(
            self.nbands,
            self.wf_desc,
            self.communicators['b'],
            hamiltonian,
            self.params.convergence.get('bands', 'occupied'),
            self.setups,
            self.atoms)

    def read_ibz_wave_functions(self, reader):
        kpt_comm, band_comm, domain_comm = (self.communicators[x]
                                            for x in 'kbd')

        def create_wfs(spin: int, q: int, k: int, kpt_c, weight: float):
            psit_nG = XArrayWithNoData(
                comm=band_comm,
                dims=(self.nbands,),
                desc=self.wf_desc.new(kpt=kpt_c),
                xp=self.xp)
            wfs = PWFDWaveFunctions(
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                psit_nX=psit_nG,  # type: ignore
                setups=self.setups,
                relpos_ac=self.relpos_ac,
                atomdist=self.atomdist,
                ncomponents=self.ncomponents,
                qspiral_v=self.qspiral_v)

            return wfs

        ibzwfs = PWFDIBZWaveFunctions.create(
            ibz=self.ibz,
            ncomponents=self.ncomponents,
            create_wfs_func=create_wfs,
            kpt_comm=self.communicators['k'],
            kpt_band_comm=self.communicators['D'],
            comm=self.communicators['w'])

        # Set eigenvalues, occupations, etc..
        self.read_wavefunction_values(reader, ibzwfs)

        return ibzwfs

    def create_ibz_wave_functions(self, basis, potential):
        from gpaw.new.lcao.builder import create_lcao_ibzwfs

        if self.params.random:
            return self.create_random_ibz_wave_functions()

        # sl_default = self.params.parallel['sl_default']
        # sl_lcao = self.params.parallel['sl_lcao'] or sl_default

        lcao_dtype = complex if \
            np.issubdtype(self.dtype, np.complexfloating) else float

        lcaonbands = min(self.nbands,
                         basis.Mmax * (2 if self.ncomponents == 4 else 1))
        lcao_ibzwfs, _ = create_lcao_ibzwfs(
            basis,
            self.ibz, self.communicators, self.setups,
            self.relpos_ac, self.grid, lcao_dtype,
            lcaonbands, self.ncomponents, self.atomdist, self.nelectrons)

        self.log('\nDiagonalizing LCAO Hamiltonian', flush=True)

        hamiltonian = LCAOHamiltonian(basis)
        LCAOEigensolver(basis).iterate(
            lcao_ibzwfs, None, potential, hamiltonian)

        self.log('Converting LCAO to grid', flush=True)

        def create_wfs(spin, q, k, kpt_c, weight):
            lcaowfs = lcao_ibzwfs.wfs_qs[q][spin]
            assert lcaowfs.spin == spin

            # Convert to PW-coefs in PW-mode:
            psit_nX = self.convert_wave_functions_from_uniform_grid(
                lcaowfs.C_nM, basis, kpt_c, q)

            mylcaonbands, nao = lcaowfs.C_nM.dist.shape
            mynbands = len(psit_nX.data)
            eig_n = np.empty(self.nbands)
            eig_n[:lcaonbands] = lcaowfs._eig_n
            eig_n[lcaonbands:] = 100.0  # set high value for random wfs.
            if mylcaonbands < mynbands:
                psit_nX[mylcaonbands:].randomize(
                    seed=self.communicators['w'].rank)
            wfs = PWFDWaveFunctions(
                psit_nX=psit_nX,
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                setups=self.setups,
                relpos_ac=self.relpos_ac,
                atomdist=self.atomdist,
                ncomponents=self.ncomponents,
                qspiral_v=self.qspiral_v)
            wfs._eig_n = eig_n
            return wfs

        return PWFDIBZWaveFunctions.create(
            ibz=self.ibz,
            ncomponents=self.ncomponents,
            create_wfs_func=create_wfs,
            kpt_comm=self.communicators['k'],
            kpt_band_comm=self.communicators['D'],
            comm=self.communicators['w'])

    def create_random_ibz_wave_functions(self):
        self.log('Initializing wave functions with random numbers')

        def create_wfs(spin, q, k, kpt_c, weight):
            desc = self.wf_desc.new(kpt=kpt_c)
            psit_nX = desc.empty(
                dims=(self.nbands,),
                comm=self.communicators['b'],
                xp=self.xp)
            psit_nX.randomize()

            wfs = PWFDWaveFunctions(
                psit_nX=psit_nX,
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                setups=self.setups,
                relpos_ac=self.relpos_ac,
                atomdist=self.atomdist,
                ncomponents=self.ncomponents,
                qspiral_v=self.qspiral_v)

            return wfs

        return PWFDIBZWaveFunctions.create(
            ibz=self.ibz,
            ncomponents=self.ncomponents,
            create_wfs_func=create_wfs,
            kpt_comm=self.communicators['k'],
            kpt_band_comm=self.communicators['D'],
            comm=self.communicators['w'])
