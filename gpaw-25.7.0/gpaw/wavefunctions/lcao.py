import numpy as np
from ase.units import Bohr
from ase.utils.timing import timer

# from gpaw import debug
from gpaw.directmin.etdm_lcao import LCAOETDM
from gpaw.directmin.tools import loewdin_lcao, gramschmidt_lcao
from gpaw.lcao.atomic_correction import (DenseAtomicCorrection,
                                         SparseAtomicCorrection)
# from gpaw.lcao.overlap import NewTwoCenterIntegrals as NewTCI
from gpaw.lcao.tci import TCIExpansions
from gpaw.lfc import BasisFunctions
from gpaw.utilities import unpack_hermitian
from gpaw.utilities.blas import mmm, gemmdot
from gpaw.utilities.tools import tri2full
from gpaw.wavefunctions.base import WaveFunctions
from gpaw.wavefunctions.mode import Mode


class LCAO(Mode):
    name = 'lcao'

    def __init__(self, atomic_correction=None, interpolation=3,
                 force_complex_dtype=False):
        self.atomic_correction = atomic_correction
        self.interpolation = interpolation
        Mode.__init__(self, force_complex_dtype)

    def __call__(self, *args, **kwargs):
        return LCAOWaveFunctions(*args,
                                 atomic_correction=self.atomic_correction,
                                 **kwargs)

    def __repr__(self):
        return f'LCAO({self.todict()})'

    def todict(self):
        dct = Mode.todict(self)
        dct['interpolation'] = self.interpolation
        return dct


def update_phases(C_unM, q_u, ibzk_qc, spos_ac, oldspos_ac, setups, Mstart):
    """Complex-rotate coefficients compensating discontinuous phase shift.

    This changes the coefficients to counteract the phase discontinuity
    of overlaps when atoms move across a cell boundary."""

    # We don't want to apply any phase shift unless we crossed a cell
    # boundary.  So we round the shift to either 0 or 1.
    #
    # Example: spos_ac goes from 0.01 to 0.99 -- this rounds to 1 and
    # we apply the phase.  If someone moves an atom by half a cell
    # without crossing a boundary, then we are out of luck.  But they
    # should have reinitialized from LCAO anyway.
    phase_qa = np.exp(2j * np.pi *
                      np.dot(ibzk_qc, (spos_ac - oldspos_ac).T.round()))

    for q, C_nM in zip(q_u, C_unM):
        if C_nM is None:
            continue
        for a in range(len(spos_ac)):
            M1 = setups.M_a[a] - Mstart
            M2 = M1 + setups[a].nao
            M1 = max(0, M1)
            C_nM[:, M1:M2] *= phase_qa[q, a]  # (may truncate M2)


# replace by class to make data structure perhaps a bit less confusing
def get_r_and_offsets(nl, spos_ac, cell_cv):
    r_and_offset_aao = {}

    def add(a1, a2, R_c, offset):
        if (a1, a2) not in r_and_offset_aao:
            r_and_offset_aao[(a1, a2)] = []
        r_and_offset_aao[(a1, a2)].append((R_c, offset))

    for a1, spos1_c in enumerate(spos_ac):
        a2_a, offsets = nl.get_neighbors(a1)
        for a2, offset in zip(a2_a, offsets):
            spos2_c = spos_ac[a2] + offset

            R_c = np.dot(spos2_c - spos1_c, cell_cv)
            add(a1, a2, R_c, offset)
            if a1 != a2 or offset.any():
                add(a2, a1, -R_c, -offset)

    return r_and_offset_aao


class LCAOWaveFunctions(WaveFunctions):
    mode = 'lcao'

    def __init__(self, ksl, gd, nvalence, setups, bd,
                 dtype, world, kd, kptband_comm, timer,
                 atomic_correction=None, collinear=True):
        WaveFunctions.__init__(self, gd, nvalence, setups, bd,
                               dtype, collinear, world, kd,
                               kptband_comm, timer)
        self.ksl = ksl
        self.S_qMM = None
        self.T_qMM = None
        self.P_aqMi = None
        self.debug_tci = False

        if atomic_correction is None:
            atomic_correction = 'sparse' if ksl.using_blacs else 'dense'

        if atomic_correction == 'sparse':
            self.atomic_correction_cls = SparseAtomicCorrection
        else:
            assert atomic_correction == 'dense'
            self.atomic_correction_cls = DenseAtomicCorrection

        # self.tci = NewTCI(gd.cell_cv, gd.pbc_c, setups, kd.ibzk_qc, kd.gamma)
        with self.timer('TCI: Evaluate splines'):
            self.tciexpansions = TCIExpansions.new_from_setups(setups)

        self.basis_functions = BasisFunctions(gd,
                                              [setup.basis_functions_J
                                               for setup in setups],
                                              kd,
                                              dtype=dtype,
                                              cut=True)

        self.coefficients_read_from_file = False
        self.set_orthonormalized(False)

    def set_orthonormalized(self, flag):
        self.orthonormalized = flag

    @timer('Orthonormalize')
    def orthonormalize(self, kpt=None, type='gramschmidt'):
        assert type == 'gramschmidt' or type == 'loewdin'
        if kpt is None:
            for kpt in self.kpt_u:
                self.orthonormalize(kpt)
            self.orthonormalized = True
            return
        if type == 'loewdin':
            kpt.C_nM[:] = loewdin_lcao(kpt.C_nM, kpt.S_MM.conj())
        elif type == 'gramschmidt':
            kpt.C_nM[:] = gramschmidt_lcao(kpt.C_nM, kpt.S_MM.conj())

    def empty(self, n=(), global_array=False, realspace=False):
        if realspace:
            return self.gd.empty(n, self.dtype, global_array)
        else:
            if isinstance(n, int):
                n = (n,)
            nao = self.setups.nao
            return np.empty(n + (nao,), self.dtype)

    def __str__(self):
        s = 'Wave functions: LCAO\n'
        s += '  Diagonalizer: %s\n' % self.ksl.get_description()
        s += ('  Atomic Correction: %s\n'
              % self.atomic_correction_cls.description)
        s += '  Data-type: %s\n' % self.dtype.__name__
        return s

    def set_eigensolver(self, eigensolver):
        WaveFunctions.set_eigensolver(self, eigensolver)
        if eigensolver:
            if isinstance(eigensolver, LCAOETDM):
                eigensolver.initialize(self.gd, self.dtype, self.bd.nbands,
                                       self.kd.nibzkpts, self.setups.nao,
                                       self.ksl.using_blacs,
                                       self.bd.comm.size, self.kpt_u)
            else:
                eigensolver.initialize(self.gd, self.dtype, self.setups.nao,
                                       self.ksl)

    def set_positions(self, spos_ac, atom_partition=None, move_wfs=False):
        oldspos_ac = self.spos_ac
        with self.timer('Basic WFS set positions'):
            WaveFunctions.set_positions(self, spos_ac, atom_partition)

        with self.timer('Basis functions set positions'):
            self.basis_functions.set_positions(spos_ac)

        if self.ksl is not None:
            self.basis_functions.set_matrix_distribution(self.ksl.Mstart,
                                                         self.ksl.Mstop)

        Mstop = self.ksl.Mstop
        Mstart = self.ksl.Mstart

        # if self.ksl.using_blacs:  # XXX
        #     S and T have been distributed to a layout with blacs, so
        #     discard them to force reallocation from scratch.
        #
        #     TODO: evaluate S and T when they *are* distributed, thus saving
        #     memory and avoiding this problem
        for kpt in self.kpt_u:
            kpt.S_MM = None
            kpt.T_MM = None

        # Free memory in case of old matrices:
        self.S_qMM = self.T_qMM = self.P_aqMi = None

        if self.dtype == complex and oldspos_ac is not None:
            update_phases([kpt.C_nM for kpt in self.kpt_u],
                          [kpt.q for kpt in self.kpt_u],
                          self.kd.ibzk_qc, spos_ac, oldspos_ac,
                          self.setups, Mstart)

        self.timer.start('mktci')
        manytci = self.tciexpansions.get_manytci_calculator(
            self.setups, self.gd, spos_ac, self.kd.ibzk_qc, self.dtype,
            self.timer)
        self.timer.stop('mktci')
        self.manytci = manytci
        self.newtci = manytci.tci

        my_atom_indices = self.basis_functions.my_atom_indices
        self.timer.start('ST tci')
        newS_qMM, newT_qMM = manytci.O_qMM_T_qMM(self.gd.comm,
                                                 Mstart, Mstop,
                                                 self.ksl.using_blacs)
        self.timer.stop('ST tci')
        self.timer.start('P tci')
        P_qIM = manytci.P_qIM(my_atom_indices)
        self.timer.stop('P tci')
        self.P_aqMi = manytci.P_aqMi(my_atom_indices)
        self.P_qIM = P_qIM  # XXX atomic correction

        self.atomic_correction = self.atomic_correction_cls.new_from_wfs(self)

        # TODO
        #   OK complex/conj, periodic images
        #   OK scalapack
        #   derivatives/forces
        #   sparse
        #   use symmetry/conj tricks to reduce calculations
        #   enable caching of spherical harmonics

        self.atomic_correction.add_overlap_correction(newS_qMM)
        self.allocate_arrays_for_projections(my_atom_indices)

        newS_qMM = self.ksl.distribute_overlap_matrix(newS_qMM, root=-1)
        newT_qMM = self.ksl.distribute_overlap_matrix(newT_qMM, root=-1)

        self.positions_set = True

        for kpt in self.kpt_u:
            q = kpt.q
            kpt.S_MM = newS_qMM[q]
            kpt.T_MM = newT_qMM[q]
        self.S_qMM = newS_qMM
        self.T_qMM = newT_qMM

        # Elpa wants to reuse the decomposed form of S_qMM.
        # We need to keep track of the existence of that object here,
        # since this is where we change S_qMM.  Hence, expect this to
        # become arrays after the first diagonalization:
        self.decomposed_S_qMM = [None] * len(self.S_qMM)
        self.set_orthonormalized(False)

    def initialize(self, density, hamiltonian, spos_ac):
        # Note: The above line exists also in set_positions.
        # This is guaranteed to be correct, but we can probably remove one.
        # Of course no human can understand the initialization process,
        # so this will be some other day.
        self.timer.start('LCAO WFS Initialize')
        if density.nt_sG is None:
            if self.kpt_u[0].f_n is None or self.kpt_u[0].C_nM is None:
                density.initialize_from_atomic_densities(self.basis_functions)
            else:
                # We have the info we need for a density matrix, so initialize
                # from that instead of from scratch.  This will be the case
                # after set_positions() during a relaxation
                density.initialize_from_wavefunctions(self)
            # Initialize GLLB-potential from basis function orbitals
            if hamiltonian.xc.type == 'GLLB':
                hamiltonian.xc.initialize_from_atomic_orbitals(
                    self.basis_functions)

        else:
            # After a restart, nt_sg doesn't exist yet, so we'll have to
            # make sure it does.  Of course, this should have been taken care
            # of already by this time, so we should improve the code elsewhere
            density.calculate_normalized_charges_and_mix()

        hamiltonian.update(density)
        self.timer.stop('LCAO WFS Initialize')

        return 0, 0

    def initialize_wave_functions_from_lcao(self):
        """Fill the calc.wfs.kpt_[u].psit_nG arrays with useful data.

        Normally psit_nG is NOT used in lcao mode, but some extensions
        (like ase.dft.wannier) want to have it.
        This code is adapted from fd.py / initialize_from_lcao_coefficients()
        and fills psit_nG with data constructed from the current lcao
        coefficients (kpt.C_nM).

        (This may or may not work in band-parallel case!)
        """
        from gpaw.wavefunctions.arrays import UniformGridWaveFunctions
        bfs = self.basis_functions
        for kpt in self.kpt_u:
            kpt.psit = UniformGridWaveFunctions(
                self.bd.nbands, self.gd, self.dtype, kpt=kpt.q, dist=None,
                spin=kpt.s, collinear=True)
            kpt.psit_nG[:] = 0.0
            bfs.lcao_to_grid(kpt.C_nM, kpt.psit_nG[:self.bd.mynbands], kpt.q)

    def initialize_wave_functions_from_restart_file(self):
        """Dummy function to ensure compatibility to fd mode"""
        self.initialize_wave_functions_from_lcao()

    def add_orbital_density(self, nt_G, kpt, n):
        rank, q = self.kd.get_rank_and_index(kpt.k)
        u = q * self.nspins + kpt.s
        assert rank == self.kd.comm.rank
        assert self.kpt_u[u] is kpt
        psit_G = self._get_wave_function_array(u, n, realspace=True)
        self.add_realspace_orbital_to_density(nt_G, psit_G)

    def calculate_density_matrix(self, f_n, C_nM, rho_MM=None):
        self.timer.start('Calculate density matrix')
        rho_MM = self.ksl.calculate_density_matrix(f_n, C_nM, rho_MM)
        self.timer.stop('Calculate density matrix')
        return rho_MM

        if 1:
            # XXX Should not conjugate, but call gemm(..., 'c')
            # Although that requires knowing C_Mn and not C_nM.
            # that also conforms better to the usual conventions in literature
            Cf_Mn = C_nM.T.conj() * f_n
            self.timer.start('gemm')
            mmm(1.0, Cf_Mn, 'N', C_nM, 'N', 0.0, rho_MM)
            self.timer.stop('gemm')
            self.timer.start('band comm sum')
            self.bd.comm.sum(rho_MM)
            self.timer.stop('band comm sum')
        else:
            # Alternative suggestion. Might be faster. Someone should test this
            from gpaw.utilities.blas import r2k
            C_Mn = C_nM.T.copy()
            r2k(0.5, C_Mn, f_n * C_Mn, 0.0, rho_MM)
            tri2full(rho_MM)

    def calculate_atomic_density_matrices_with_occupation(self, D_asp, f_un):
        # ac = self.atomic_correction
        # if ac.implements_distributed_projections():
        #     D2_asp = ac.redistribute(self, D_asp, type='asp', op='forth')
        #     WaveFunctions.calculate_atomic_density_matrices_with_occupation(
        #         self, D2_asp, f_un)
        #     D3_asp = ac.redistribute(self, D2_asp, type='asp', op='back')
        #     for a in D_asp:
        #         D_asp[a][:] = D3_asp[a]
        # else:
        WaveFunctions.calculate_atomic_density_matrices_with_occupation(
            self, D_asp, f_un)

    def calculate_density_matrix_delta(self, d_nn, C_nM, rho_MM=None):
        self.timer.start('Calculate density matrix')
        rho_MM = self.ksl.calculate_density_matrix_delta(d_nn, C_nM, rho_MM)
        self.timer.stop('Calculate density matrix')
        return rho_MM

    def add_to_density_from_k_point_with_occupation(self, nt_sG, kpt, f_n):
        """Add contribution to pseudo electron-density. Do not use the standard
        occupation numbers, but ones given with argument f_n."""
        # Custom occupations are used in calculation of response potential
        # with GLLB-potential
        if kpt.rho_MM is None:
            rho_MM = self.calculate_density_matrix(f_n, kpt.C_nM)
            if hasattr(kpt, 'c_on'):
                assert self.bd.comm.size == 1
                d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands),
                                dtype=kpt.C_nM.dtype)
                for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                    assert abs(c_n.imag).max() < 1e-14
                    d_nn += ne * np.outer(c_n.conj(), c_n).real
                rho_MM += self.calculate_density_matrix_delta(d_nn, kpt.C_nM)
        else:
            rho_MM = kpt.rho_MM
        self.timer.start('Construct density')
        self.basis_functions.construct_density(rho_MM, nt_sG[kpt.s], kpt.q)
        self.timer.stop('Construct density')

    def add_to_kinetic_density_from_k_point(self, taut_G, kpt):
        raise NotImplementedError('Kinetic density calculation for LCAO '
                                  'wavefunctions is not implemented.')

    def calculate_forces(self, hamiltonian, F_av):
        self.timer.start('LCAO forces')

        Fref_av = np.zeros_like(F_av)
        self.forcecalc = LCAOforces(self.ksl, self.dtype, self.gd,
                                    self.bd, self.kd, self.kpt_u, self.nspins,
                                    self.basis_functions, self.newtci,
                                    self.P_aqMi, self.setups,
                                    self.manytci, hamiltonian,
                                    self, self.spos_ac,
                                    self.timer, Fref_av)

        F_av[:, :] = self.forcecalc.get_forces_sum_GS()
        self.timer.stop('LCAO forces')

    def _get_wave_function_array(self, u, n, realspace=True, periodic=False):
        # XXX Taking kpt is better than taking u
        kpt = self.kpt_u[u]
        C_M = kpt.C_nM[n]

        if realspace:
            psit_G = self.gd.zeros(dtype=self.dtype)
            self.basis_functions.lcao_to_grid(C_M, psit_G, kpt.q)
            if periodic and self.dtype == complex:
                k_c = self.kd.ibzk_kc[kpt.k]
                return self.gd.plane_wave(-k_c) * psit_G
            return psit_G
        else:
            return C_M

    def write(self, writer, write_wave_functions=False):
        WaveFunctions.write(self, writer)
        if write_wave_functions:
            self.write_wave_functions(writer)

    def write_wave_functions(self, writer):
        writer.add_array(
            'coefficients',
            (self.nspins, self.kd.nibzkpts, self.bd.nbands, self.setups.nao),
            dtype=self.dtype)
        for s in range(self.nspins):
            for k in range(self.kd.nibzkpts):
                C_nM = self.collect_array('C_nM', k, s)
                writer.fill(C_nM * Bohr**-1.5)

    def read(self, reader):
        WaveFunctions.read(self, reader)
        r = reader.wave_functions
        if 'coefficients' in r:
            self.read_wave_functions(r)

    def read_wave_functions(self, reader):
        c = 1.0 if getattr(reader, 'version', 3) >= 4 else Bohr**1.5
        for kpt in self.kpt_u:
            C_nM = reader.proxy('coefficients', kpt.s, kpt.k)
            kpt.C_nM = self.bd.empty(self.setups.nao, dtype=self.dtype)
            for myn, C_M in enumerate(kpt.C_nM):
                n = self.bd.global_index(myn)
                # XXX number of bands could have been rounded up!
                if n >= len(C_nM):
                    break
                C_M[:] = C_nM[n] * c

        self.coefficients_read_from_file = True

    def estimate_memory(self, mem):
        nq = len(self.kd.ibzk_qc)
        nao = self.setups.nao
        ni_total = sum([setup.ni for setup in self.setups])
        itemsize = mem.itemsize[self.dtype]
        mem.subnode('C [qnM]', nq * self.bd.mynbands * nao * itemsize)
        nM1, nM2 = self.ksl.get_overlap_matrix_shape()
        mem.subnode('S, T [2 x qmm]', 2 * nq * nM1 * nM2 * itemsize)
        mem.subnode('P [aqMi]', nq * nao * ni_total // self.gd.comm.size)
        # self.tci.estimate_memory(mem.subnode('TCI'))
        self.basis_functions.estimate_memory(mem.subnode('BasisFunctions'))
        self.eigensolver.estimate_memory(mem.subnode('Eigensolver'),
                                         self.dtype)


class LCAOforces:

    def __init__(self, ksl, dtype, gd, bd, kd, kpt_u, nspins, bfs, newtci,
                 P_aqMi, setups, manytci, hamiltonian, wfs, spos_ac,
                 timer, Fref_av):
        """ Object which calculates LCAO forces """

        self.ksl = ksl
        self.nao = ksl.nao
        self.mynao = ksl.mynao
        self.dtype = dtype
        self.newtci = newtci
        self.manytci = manytci
        self.P_aqMi = P_aqMi
        self.gd = gd
        self.bd = bd
        self.kd = kd
        self.kpt_u = kpt_u
        self.nspins = nspins
        self.bfs = bfs
        self.spos_ac = spos_ac
        self.Mstart = ksl.Mstart
        self.Mstop = ksl.Mstop
        self.setups = setups
        self.hamiltonian = hamiltonian
        self.wfs = wfs
        self.timer = timer
        self.Fref_av = Fref_av
        self.my_atom_indices = bfs.my_atom_indices
        self.atom_indices = bfs.atom_indices
        self.dH_asp = hamiltonian.dH_asp

        from gpaw.kohnsham_layouts import BlacsOrbitalLayouts
        self.isblacs = isinstance(self.ksl, BlacsOrbitalLayouts)

        if not self.isblacs:
            self.timer.start('TCI derivative')
            self.dThetadR_qvMM, self.dTdR_qvMM = self.manytci.O_qMM_T_qMM(
                self.gd.comm, self.Mstart, self.Mstop, False, derivative=True)
            self.dPdR_aqvMi = self.manytci.P_aqMi(self.bfs.my_atom_indices,
                                                  derivative=True)

            self.gd.comm.sum(self.dThetadR_qvMM)
            self.gd.comm.sum(self.dTdR_qvMM)
            self.timer.stop('TCI derivative')
            self.rhoT_uMM, self.ET_uMM = self.get_den_mat_and_E()

    def get_forces_sum_GS(self):
        """ This function calculates ground state forces in LCAO mode """
        if not self.isblacs:
            F_av = np.zeros_like(self.Fref_av)
            Fkin_av = self.get_kinetic_term()
            Fpot_av = self.get_pot_term()
            Ftheta_av = self.get_den_mat_term()
            Frho_av = self.get_den_mat_paw_term()
            Fatom_av = self.get_atomic_density_term()
            F_av += Fkin_av + Fpot_av + Ftheta_av + Frho_av + Fatom_av
        else:
            F_av = np.zeros_like(self.Fref_av)
            Fpot_av = self.get_pot_term_blacs()
            Fkin_av, Ftheta_av = self.get_kin_and_den_term_blacs()
            Fatom_av, Frho_av = self.get_at_den_and_den_paw_blacs()

            F_av += Fkin_av + Fpot_av + Ftheta_av + Frho_av + Fatom_av

        self.timer.start('Wait for sum')
        self.ksl.orbital_comm.sum(F_av)
        if self.bd.comm.rank == 0:
            self.kd.comm.sum(F_av, 0)
        self.timer.stop('Wait for sum')

        return F_av

    def _slices(self, indices):
        for a in indices:
            M1 = self.bfs.M_a[a] - self.Mstart
            M2 = M1 + self.setups[a].nao
            if M2 > 0:
                yield a, max(0, M1), M2

    def slices(self):
        return self._slices(self.atom_indices)

    def my_slices(self):
        return self._slices(self.my_atom_indices)

    def get_den_mat_and_E(self):
        #
        #         -----                    -----
        #          \    -1                  \    *
        # E      =  )  S     H    rho     =  )  c     eps  f  c
        #  mu nu   /    mu x  x z    z nu   /    n mu    n  n  n nu
        #         -----                    -----
        #          x z                       n
        #
        # We use the transpose of that matrix.  The first form is used
        # if rho is given, otherwise the coefficients are used.
        self.timer.start('Initial')
        if self.kpt_u[0].rho_MM is None:
            rhoT_uMM = []
            ET_uMM = []
            self.timer.start('Get density matrix')
            for kpt in self.kpt_u:
                rhoT_MM = self.ksl.get_transposed_density_matrix(kpt.f_n,
                                                                 kpt.C_nM)
                rhoT_uMM.append(rhoT_MM)
                ET_MM = self.ksl.get_transposed_density_matrix(kpt.f_n *
                                                               kpt.eps_n,
                                                               kpt.C_nM)
                ET_uMM.append(ET_MM)
                if hasattr(kpt, 'c_on'):
                    # XXX does this work with BLACS/non-BLACS/etc.?
                    assert self.bd.comm.size == 1
                    d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands),
                                    dtype=kpt.C_nM.dtype)
                    for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                        d_nn += ne * np.outer(c_n.conj(), c_n)
                    rhoT_MM += self.ksl.get_transposed_density_matrix_delta(
                        d_nn, kpt.C_nM)
                    ET_MM += self.ksl.get_transposed_density_matrix_delta(
                        d_nn * kpt.eps_n, kpt.C_nM)
            self.timer.stop('Get density matrix')
        else:
            rhoT_uMM = []
            ET_uMM = []
            for kpt in self.kpt_u:
                H_MM = self.wfs.eigensolver.calculate_hamiltonian_matrix(
                    self.hamiltonian, self.wfs, kpt)
                tri2full(H_MM)
                S_MM = kpt.S_MM.copy()
                tri2full(S_MM)
                ET_MM = np.linalg.solve(S_MM, gemmdot(H_MM,
                                                      kpt.rho_MM)).T.copy()
                del S_MM, H_MM
                rhoT_MM = kpt.rho_MM.T.copy()
                rhoT_uMM.append(rhoT_MM)
                ET_uMM.append(ET_MM)
        self.timer.stop('Initial')
        return rhoT_uMM, ET_uMM

    def get_kinetic_term(self):
        """Calculate Kinetic energy term in LCAO"""
        Fkin_av = np.zeros_like(self.Fref_av)
        self.timer.start('TCI derivative')
        # Kinetic energy contribution
        #
        #           ----- d T
        #  a         \       mu nu
        # F += 2 Re   )   -------- rho
        #            /    d R         nu mu
        #           -----    mu nu
        #        mu in a; nu
        #
        Fkin_av = np.zeros_like(Fkin_av)
        for u, kpt in enumerate(self.kpt_u):
            dEdTrhoT_vMM = (self.dTdR_qvMM[kpt.q] *
                            self.rhoT_uMM[u][np.newaxis]).real
            # XXX load distribution!
            for a, M1, M2 in self.my_slices():
                Fkin_av[a, :] += \
                    2.0 * dEdTrhoT_vMM[:, M1:M2].sum(-1).sum(-1)
        self.timer.stop('TCI derivative')

        return Fkin_av

    def get_den_mat_term(self):
        """Calculate density matrix term in LCAO"""
        Ftheta_av = np.zeros_like(self.Fref_av)
        # Density matrix contribution due to basis overlap
        #
        #            ----- d Theta
        #  a          \           mu nu
        # F  += -2 Re  )   ------------  E
        #             /        d R        nu mu
        #            -----        mu nu
        #         mu in a; nu
        #
        Ftheta_av = np.zeros_like(Ftheta_av)
        for u, kpt in enumerate(self.kpt_u):
            dThetadRE_vMM = (self.dThetadR_qvMM[kpt.q] *
                             self.ET_uMM[u][np.newaxis]).real
            for a, M1, M2 in self.my_slices():
                Ftheta_av[a, :] += \
                    -2.0 * dThetadRE_vMM[:, M1:M2].sum(-1).sum(-1)

        return Ftheta_av

    def get_pot_term(self):
        """Calculate potential term"""
        Fpot_av = np.zeros_like(self.Fref_av)
        # Potential contribution
        #
        #           -----      /  d Phi  (r)
        #  a         \        |        mu    ~
        # F += -2 Re  )       |   ---------- v (r)  Phi  (r) dr rho
        #            /        |     d R                nu          nu mu
        #           -----    /         a
        #        mu in a; nu
        #
        self.timer.start('Potential')
        vt_sG = self.hamiltonian.vt_sG
        Fpot_av = np.zeros_like(Fpot_av)
        for u, kpt in enumerate(self.kpt_u):
            vt_G = vt_sG[kpt.s]
            Fpot_av += self.bfs.calculate_force_contribution(vt_G,
                                                             self.rhoT_uMM[u],
                                                             kpt.q)
        self.timer.stop('Potential')

        return Fpot_av

    def get_den_mat_paw_term(self):
        """Calcualte PAW correction"""
        # TO DO: split this function into
        # _get_den_mat_paw_term (which calculate Frho_av) and
        # get_paw_correction (which calculate ZE_MM)
        # Density matrix contribution from PAW correction
        #
        #           -----                        -----
        #  a         \      a                     \     b
        # F +=  2 Re  )    Z      E        - 2 Re  )   Z      E
        #            /      mu nu  nu mu          /     mu nu  nu mu
        #           -----                        -----
        #           mu nu                    b; mu in a; nu
        #
        # with
        #                  b*
        #         -----  dP
        #   b      \       i mu    b   b
        #  Z     =  )   -------- dS   P
        #   mu nu  /     dR        ij  j nu
        #         -----    b mu
        #           ij
        #
        self.timer.start('Paw correction')
        Frho_av = np.zeros_like(self.Fref_av)
        for u, kpt in enumerate(self.kpt_u):
            work_MM = np.zeros((self.mynao, self.nao), self.dtype)
            ZE_MM = None
            for b in self.my_atom_indices:
                setup = self.setups[b]
                dO_ii = np.asarray(setup.dO_ii, self.dtype)
                dOP_iM = np.zeros((setup.ni, self.nao), self.dtype)
                mmm(1.0, dO_ii, 'N', self.P_aqMi[b][kpt.q], 'C', 0.0, dOP_iM)
                for v in range(3):
                    mmm(1.0,
                        self.dPdR_aqvMi[b][kpt.q][v][self.Mstart:self.Mstop],
                        'N',
                        dOP_iM, 'N',
                        0.0, work_MM)
                    ZE_MM = (work_MM * self.ET_uMM[u]).real
                    for a, M1, M2 in self.slices():
                        dE = 2 * ZE_MM[M1:M2].sum()
                        Frho_av[a, v] -= dE  # the "b; mu in a; nu" term
                        Frho_av[b, v] += dE  # the "mu nu" term
        self.timer.stop('Paw correction')
        return Frho_av

    def _get_den_mat_paw_term(self):
        # THIS doesn't work in parallel
        # Density matrix contribution from PAW correction
        #
        #           -----                        -----
        #  a         \      a                     \     b
        # F +=  2 Re  )    Z      E        - 2 Re  )   Z      E
        #            /      mu nu  nu mu          /     mu nu  nu mu
        #           -----                        -----
        #           mu nu                    b; mu in a; nu
        #
        # with
        #                  b*
        #         -----  dP
        #   b      \       i mu    b   b
        #  Z     =  )   -------- dS   P
        #   mu nu  /     dR        ij  j nu
        #         -----    b mu
        #           ij
        #
        Frho_av = np.zeros_like(self.Fref_av)
        self.timer.start('add paw correction')
        ZE_MM = self.get_paw_correction()
        for u, kpt in enumerate(self.kpt_u):
            for b in self.my_atom_indices:
                for v in range(3):
                    for a, M1, M2 in self.slices():
                        dE = 2 * ZE_MM[u, b, v, M1:M2].sum()
                        Frho_av[a, v] -= dE.real  # the "b; mu in a; nu" term
                        Frho_av[b, v] += dE.real  # the "mu nu" term
        self.timer.stop('add paw correction')
        return Frho_av

    def get_paw_correction(self):
        # THIS doesn't work in parallel
        #  <Phi_nu|pt_i>O_ii<dPt_i/dR|Phi_mu>
        self.timer.start('get paw correction')
        ZE_MM = np.zeros((len(self.kpt_u), len(self.my_atom_indices), 3,
                          self.mynao, self.nao), self.dtype)
        for u, kpt in enumerate(self.kpt_u):
            work_MM = np.zeros((self.mynao, self.nao), self.dtype)
            for b in self.my_atom_indices:
                setup = self.setups[b]
                dO_ii = np.asarray(setup.dO_ii, self.dtype)
                dOP_iM = np.zeros((setup.ni, self.nao), self.dtype)
                mmm(1.0, dO_ii, 'N', self.P_aqMi[b][kpt.q], 'C', 0.0, dOP_iM)
                for v in range(3):
                    mmm(1.0,
                        self.dPdR_aqvMi[b][kpt.q][v][self.Mstart:self.Mstop],
                        'N',
                        dOP_iM, 'N',
                        0.0, work_MM)
                    ZE_MM[u, b, v, :, :] = (work_MM * self.ET_uMM[u]).real
        self.timer.stop('get paw correction')
        return ZE_MM

    def get_atomic_density_term(self):
        Fatom_av = np.zeros_like(self.Fref_av)
        # Atomic density contribution
        #            -----                         -----
        #  a          \     a                       \     b
        # F  += -2 Re  )   A      rho       + 2 Re   )   A      rho
        #             /     mu nu    nu mu          /     mu nu    nu mu
        #            -----                         -----
        #            mu nu                     b; mu in a; nu
        #
        #                  b*
        #         ----- d P
        #  b       \       i mu   b   b
        # A     =   )   ------- dH   P
        #  mu nu   /    d R       ij  j nu
        #         -----    b mu
        #           ij
        #
        self.timer.start('Atomic Hamiltonian force')
        Fatom_av = np.zeros_like(Fatom_av)
        for u, kpt in enumerate(self.kpt_u):
            for b in self.my_atom_indices:
                H_ii = np.asarray(unpack_hermitian(self.dH_asp[b][kpt.s]),
                                  self.dtype)
                if len(H_ii) == 0:
                    # gemmdot does not like empty matrices!
                    # (has been fixed in the new code)
                    continue
                HP_iM = gemmdot(H_ii, np.ascontiguousarray(
                                self.P_aqMi[b][kpt.q].T.conj()))
                for v in range(3):
                    dPdR_Mi = \
                        self.dPdR_aqvMi[b][kpt.q][v][self.Mstart:self.Mstop]
                    ArhoT_MM = \
                        (gemmdot(dPdR_Mi, HP_iM) * self.rhoT_uMM[u]).real
                    for a, M1, M2 in self.slices():
                        dE = 2 * ArhoT_MM[M1:M2].sum()
                        Fatom_av[a, v] += dE  # the "b; mu in a; nu" term
                        Fatom_av[b, v] -= dE  # the "mu nu" term
        self.timer.stop('Atomic Hamiltonian force')

        return Fatom_av

    def get_den_mat_block_blacs(self, f_n, C_nM, redistributor):
        rho1_mm = self.ksl.calculate_blocked_density_matrix(f_n,
                                                            C_nM).conj()
        rho_mm = redistributor.redistribute(rho1_mm)
        return rho_mm

    def get_pot_term_blacs(self):
        Fpot_av = np.zeros_like(self.Fref_av)
        from gpaw.blacs import BlacsGrid, Redistributor
        self.grid = BlacsGrid(self.ksl.block_comm, self.gd.comm.size,
                              self.bd.comm.size)
        self.blocksize1 = -(-self.nao // self.grid.nprow)
        self.blocksize2 = -(-self.nao // self.grid.npcol)
        desc = self.grid.new_descriptor(self.nao, self.nao,
                                        self.blocksize1, self.blocksize2)
        vt_sG = self.hamiltonian.vt_sG
        self.rhoT_umm = []
        self.ET_umm = []
        self.redistributor = Redistributor(self.grid.comm,
                                           self.ksl.mmdescriptor, desc)
        Fpot_av = np.zeros_like(self.Fref_av)
        for u, kpt in enumerate(self.kpt_u):
            self.timer.start('Get density matrix')
            rhoT_mm = self.get_den_mat_block_blacs(kpt.f_n, kpt.C_nM,
                                                   self.redistributor)
            self.rhoT_umm.append(rhoT_mm)
            self.timer.stop('Get density matrix')
            self.timer.start('Potential')
            rhoT_mM = self.ksl.distribute_to_columns(rhoT_mm, desc)
            vt_G = vt_sG[kpt.s]
            Fpot_av += self.bfs.calculate_force_contribution(vt_G, rhoT_mM,
                                                             kpt.q)
            del rhoT_mM
            self.timer.stop('Potential')

        return Fpot_av

    def get_kin_and_den_term_blacs(self):
        Fkin_av_sum = np.zeros_like(self.Fref_av)
        Ftheta_av_sum = np.zeros_like(self.Fref_av)
        # pcutoff_a = [max([pt.get_cutoff() for pt in setup.pt_j])
        #              for setup in self.setups]
        # phicutoff_a = [max([phit.get_cutoff() for phit in setup.phit_j])
        #                for setup in self.setups]
        # XXX should probably use bdsize x gdsize instead
        # That would be consistent with some existing grids
        # I'm not sure if this is correct
        # XXX what are rows and columns actually?
        dH_asp = self.hamiltonian.dH_asp
        self.timer.start('Get density matrix')
        for kpt in self.kpt_u:
            ET_mm = self.get_den_mat_block_blacs(kpt.f_n * kpt.eps_n, kpt.C_nM,
                                                 self.redistributor)
            self.ET_umm.append(ET_mm)
        self.timer.stop('Get density matrix')
        self.M1start = self.blocksize1 * self.grid.myrow
        self.M2start = self.blocksize2 * self.grid.mycol
        self.M1stop = min(self.M1start + self.blocksize1, self.nao)
        self.M2stop = min(self.M2start + self.blocksize2, self.nao)
        self.m1max = self.M1stop - self.M1start
        self.m2max = self.M2stop - self.M2start
        # from gpaw.lcao.overlap import TwoCenterIntegralCalculator
        self.timer.start('Prepare TCI loop')
        self.M_a = self.bfs.M_a
        Fkin2_av = np.zeros_like(self.Fref_av)
        Ftheta2_av = np.zeros_like(self.Fref_av)
        self.atompairs = self.newtci.a1a2.get_atompairs()
        self.timer.start('broadcast dH')
        self.alldH_asp = {}
        for a in range(len(self.setups)):
            gdrank = self.bfs.sphere_a[a].rank
            if gdrank == self.gd.rank:
                dH_sp = dH_asp[a]
            else:
                ni = self.setups[a].ni
                dH_sp = np.empty((self.nspins, ni * (ni + 1) // 2))
            self.gd.comm.broadcast(dH_sp, gdrank)
            # okay, now everyone gets copies of dH_sp
            self.alldH_asp[a] = dH_sp
        self.timer.stop('broadcast dH')
        # This will get sort of hairy.  We need to account for some
        # three-center overlaps, such as:
        #
        #         a1
        #      Phi   ~a3    a3  ~a3     a2     a2,a1
        #   < ----  |p  > dH   <p   |Phi  > rho
        #      dR
        #
        # To this end we will loop over all pairs of atoms (a1, a3),
        # and then a sub-loop over (a3, a2).
        self.timer.stop('Prepare TCI loop')
        self.timer.start('Not so complicated loop')
        for (a1, a2) in self.atompairs:
            if a1 >= a2:
                # Actually this leads to bad load balance.
                # We should take a1 > a2 or a1 < a2 equally many times.
                # Maybe decide which of these choices
                # depending on whether a2 % 1 == 0
                continue
            m1start = self.M_a[a1] - self.M1start
            m2start = self.M_a[a2] - self.M2start
            if m1start >= self.blocksize1 or m2start >= self.blocksize2:
                continue  # (we have only one block per CPU)
            nm1 = self.setups[a1].nao
            nm2 = self.setups[a2].nao
            m1stop = min(m1start + nm1, self.m1max)
            m2stop = min(m2start + nm2, self.m2max)
            if m1stop <= 0 or m2stop <= 0:
                continue
            m1start = max(m1start, 0)
            m2start = max(m2start, 0)
            J1start = max(0, self.M1start - self.M_a[a1])
            J2start = max(0, self.M2start - self.M_a[a2])
            M1stop = J1start + m1stop - m1start
            J2stop = J2start + m2stop - m2start
            dThetadR_qvmm, dTdR_qvmm = self.newtci.dOdR_dTdR(a1, a2)
            for u, kpt in enumerate(self.kpt_u):
                rhoT_mm = self.rhoT_umm[u][m1start:m1stop, m2start:m2stop]
                ET_mm = self.ET_umm[u][m1start:m1stop, m2start:m2stop]
                Fkin_v = 2.0 * (dTdR_qvmm[kpt.q][:, J1start:M1stop,
                                                 J2start:J2stop] *
                                rhoT_mm[np.newaxis]).real.sum(-1).sum(-1)
                Ftheta_v = 2.0 * (dThetadR_qvmm[kpt.q][:, J1start:M1stop,
                                                       J2start:J2stop] *
                                  ET_mm[np.newaxis]).real.sum(-1).sum(-1)
                Fkin2_av[a1] += Fkin_v
                Fkin2_av[a2] -= Fkin_v
                Ftheta2_av[a1] -= Ftheta_v
                Ftheta2_av[a2] += Ftheta_v
        Fkin_av = Fkin2_av
        Ftheta_av = Ftheta2_av
        self.timer.stop('Not so complicated loop')

        Fkin_av_sum += Fkin_av
        Ftheta_av_sum += Ftheta_av

        return Fkin_av_sum, Ftheta_av_sum

    def get_at_den_and_den_paw_blacs(self):
        Fatom_av = np.zeros_like(self.Fref_av)
        Frho_av = np.zeros_like(self.Fref_av)
        Fatom_av_sum = np.zeros_like(self.Fref_av)
        Frho_av_sum = np.zeros_like(self.Fref_av)
        self.dHP_and_dSP_aauim = {}
        self.a2values = {}
        for (a2, a3) in self.atompairs:
            if a3 not in self.a2values:
                self.a2values[a3] = []
            self.a2values[a3].append(a2)

        self.timer.start('Complicated loop')
        for a1, a3 in self.atompairs:
            if a1 == a3:
                # Functions reside on same atom, so their overlap
                # does not change when atom is displaced
                continue
            m1start = self.M_a[a1] - self.M1start
            if m1start >= self.blocksize1:
                continue
            nm1 = self.setups[a1].nao
            m1stop = min(m1start + nm1, self.m1max)
            if m1stop <= 0:
                continue
            dPdR_qvim = self.newtci.dPdR(a3, a1)
            if dPdR_qvim is None:
                continue
            dPdR_qvmi = -dPdR_qvim.transpose(0, 1, 3, 2).conj()
            m1start = max(m1start, 0)
            J1start = max(0, self.M1start - self.M_a[a1])
            J1stop = J1start + m1stop - m1start
            dPdR_qvmi = dPdR_qvmi[:, :, J1start:J1stop, :].copy()
            for a2 in self.a2values[a3]:
                m2start = self.M_a[a2] - self.M2start
                if m2start >= self.blocksize2:
                    continue
                nm2 = self.setups[a2].nao
                m2stop = min(m2start + nm2, self.m2max)
                if m2stop <= 0:
                    continue
                m2start = max(m2start, 0)
                J2start = max(0, self.M2start - self.M_a[a2])
                J2stop = J2start + m2stop - m2start
                if (a2, a3) in self.dHP_and_dSP_aauim:
                    dHP_uim, dSP_uim = self.dHP_and_dSP_aauim[(a2, a3)]
                else:
                    P_qim = self.newtci.P(a3, a2)
                    if P_qim is None:
                        continue
                    P_qmi = P_qim.transpose(0, 2, 1).conj()
                    P_qmi = P_qmi[:, J2start:J2stop].copy()
                    dH_sp = self.alldH_asp[a3]
                    dS_ii = self.setups[a3].dO_ii
                    dHP_uim = []
                    dSP_uim = []
                    for u, kpt in enumerate(self.kpt_u):
                        dH_ii = unpack_hermitian(dH_sp[kpt.s])
                        dHP_im = np.dot(P_qmi[kpt.q], dH_ii).T.conj()
                        # XXX only need nq of these,
                        # but the looping is over all u
                        dSP_im = np.dot(P_qmi[kpt.q], dS_ii).T.conj()
                        dHP_uim.append(dHP_im)
                        dSP_uim.append(dSP_im)
                        self.dHP_and_dSP_aauim[(a2, a3)] = dHP_uim, dSP_uim
                for u, kpt in enumerate(self.kpt_u):
                    rhoT_mm = self.rhoT_umm[u][m1start:m1stop, m2start:m2stop]
                    ET_mm = self.ET_umm[u][m1start:m1stop, m2start:m2stop]
                    dPdRdHP_vmm = np.dot(dPdR_qvmi[kpt.q], dHP_uim[u])
                    dPdRdSP_vmm = np.dot(dPdR_qvmi[kpt.q], dSP_uim[u])
                    Fatom_c = 2.0 * (dPdRdHP_vmm *
                                     rhoT_mm).real.sum(-1).sum(-1)
                    Frho_c = 2.0 * (dPdRdSP_vmm *
                                    ET_mm).real.sum(-1).sum(-1)
                    Fatom_av[a1] += Fatom_c
                    Fatom_av[a3] -= Fatom_c
                    Frho_av[a1] -= Frho_c
                    Frho_av[a3] += Frho_c
        self.timer.stop('Complicated loop')

        Fatom_av_sum += Fatom_av
        Frho_av_sum += Frho_av

        return Fatom_av_sum, Frho_av_sum
