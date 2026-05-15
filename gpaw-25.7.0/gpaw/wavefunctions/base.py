import numpy as np
from ase.units import Ha

from gpaw.projections import Projections
from gpaw.utilities import pack_density
from gpaw.utilities.blas import axpy, mmm
from gpaw.utilities.partition import AtomPartition


class WaveFunctions:
    """...

    setups:
        List of setup objects.
    symmetry:
        Symmetry object.
    kpt_u:
        List of **k**-point objects.
    nbands: int
        Number of bands.
    nspins: int
        Number of spins.
    dtype: dtype
        Data type of wave functions (float or complex).
    bzk_kc: ndarray
        Scaled **k**-points used for sampling the whole
        Brillouin zone - values scaled to [-0.5, 0.5).
    ibzk_kc: ndarray
        Scaled **k**-points in the irreducible part of the
        Brillouin zone.
    weight_k: ndarray
        Weights of the **k**-points in the irreducible part
        of the Brillouin zone (summing up to 1).
    kpt_comm:
        MPI-communicator for parallelization over **k**-points.
    """

    def __init__(self, gd, nvalence, setups, bd, dtype, collinear,
                 world, kd, kptband_comm, timer):
        self.gd = gd
        self.nspins = kd.nspins
        self.collinear = collinear
        self.nvalence = nvalence
        self.bd = bd
        self.dtype = dtype
        assert dtype == float or dtype == complex
        self.world = world
        self.kd = kd
        self.kptband_comm = kptband_comm
        self.timer = timer
        self.atom_partition = None

        self.kpt_qs = kd.create_k_points(self.gd.sdisp_cd, collinear)
        self.kpt_u = [kpt for kpt_s in self.kpt_qs for kpt in kpt_s]

        self.occupations = None
        self.fermi_levels = None

        self.eigensolver = None
        self.positions_set = False
        self.spos_ac = None

        self.set_setups(setups)

    @property
    def fermi_level(self):
        assert len(self.fermi_levels) == 1
        return self.fermi_levels[0]

    def summary(self, log):
        log(eigenvalue_string(self))

        func = None
        if hasattr(self.eigensolver, 'dm_helper'):
            func = getattr(self.eigensolver.dm_helper.func, 'name', None)
        elif hasattr(self.eigensolver, 'odd'):
            func = getattr(self.eigensolver.odd, 'name', None)
        if func is None:
            pass
        elif 'SIC' in func:
            self.summary_func(log)

        if self.fermi_levels is None:
            return

        if len(self.fermi_levels) == 1:
            log(f'Fermi level: {self.fermi_levels[0] * Ha:.5f}\n')
        else:
            f1, f2 = (f * Ha for f in self.fermi_levels)
            log(f'Fermi levels: {f1:.5f}, {f2:.5f}\n')

    def set_setups(self, setups):
        self.setups = setups

    def set_eigensolver(self, eigensolver):
        self.eigensolver = eigensolver

    def add_realspace_orbital_to_density(self, nt_G, psit_G):
        if psit_G.dtype == float:
            axpy(1.0, psit_G**2, nt_G)
        else:
            assert psit_G.dtype == complex
            axpy(1.0, psit_G.real**2, nt_G)
            axpy(1.0, psit_G.imag**2, nt_G)

    def add_orbital_density(self, nt_G, kpt, n):
        self.add_realspace_orbital_to_density(nt_G, kpt.psit_nG[n])

    def calculate_band_energy(self):
        e_band = 0.0
        for kpt in self.kpt_u:
            e_band += np.dot(kpt.f_n, kpt.eps_n)

        try:  # DCSF needs this ...
            e_band += self.occupations.calculate_band_energy(self)
        except AttributeError:
            pass

        return self.kptband_comm.sum_scalar(e_band)

    def calculate_density_contribution(self, nt_sG):
        """Calculate contribution to pseudo density from wave functions.

        Array entries are written to (not added to)."""
        nt_sG.fill(0.0)
        for kpt in self.kpt_u:
            self.add_to_density_from_k_point(nt_sG, kpt)
        self.kptband_comm.sum(nt_sG)

        self.timer.start('Symmetrize density')
        for nt_G in nt_sG:
            self.kd.symmetry.symmetrize(nt_G, self.gd)
        self.timer.stop('Symmetrize density')

    def add_to_density_from_k_point(self, nt_sG, kpt):
        self.add_to_density_from_k_point_with_occupation(nt_sG, kpt, kpt.f_n)

    def get_orbital_density_matrix(self, a, kpt, n):
        """Add the nth band density from kpt to density matrix D_sp"""
        ni = self.setups[a].ni
        D_sii = np.zeros((self.nspins, ni, ni))
        P_i = kpt.P_ani[a][n]
        D_sii[kpt.s] += np.outer(P_i.conj(), P_i).real
        D_sp = [pack_density(D_ii) for D_ii in D_sii]
        return D_sp

    def calculate_atomic_density_matrices_k_point(self, D_sii, kpt, a, f_n):
        if kpt.rho_MM is not None:
            P_Mi = self.P_aqMi[a][kpt.q]
            rhoP_Mi = np.zeros_like(P_Mi)
            D_ii = np.zeros(D_sii[kpt.s].shape, kpt.rho_MM.dtype)
            mmm(1.0, kpt.rho_MM, 'N', P_Mi, 'N', 0.0, rhoP_Mi)
            mmm(1.0, P_Mi, 'C', rhoP_Mi, 'N', 0.0, D_ii)
            D_sii[kpt.s] += D_ii.real
        else:
            if self.collinear:
                P_ni = kpt.projections[a]
                D_sii[kpt.s] += np.dot(P_ni.T.conj() * f_n, P_ni).real
            else:
                P_nsi = kpt.projections[a]
                D_ssii = np.einsum('nsi,n,nzj->szij',
                                   P_nsi.conj(), f_n, P_nsi)
                D_sii[0] += (D_ssii[0, 0] + D_ssii[1, 1]).real
                D_sii[1] += 2 * D_ssii[0, 1].real
                D_sii[2] += 2 * D_ssii[0, 1].imag
                D_sii[3] += (D_ssii[0, 0] - D_ssii[1, 1]).real

        if hasattr(kpt, 'c_on'):
            for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                d_nn = ne * np.outer(c_n.conj(), c_n)
                D_sii[kpt.s] += np.dot(P_ni.T.conj(), np.dot(d_nn, P_ni)).real

    def calculate_atomic_density_matrices(self, D_asp):
        """Calculate atomic density matrices from projections."""
        f_un = [kpt.f_n for kpt in self.kpt_u]
        self.calculate_atomic_density_matrices_with_occupation(D_asp, f_un)

    def calculate_atomic_density_matrices_with_occupation(self, D_asp, f_un):
        """Calculate atomic density matrices from projections with
        custom occupation f_un."""

        # Parameter check (if user accidentally passes f_n instead of f_un)
        if f_un[0] is not None:  # special case for transport calculations...
            assert isinstance(f_un[0], np.ndarray)
        # Varying f_n used in calculation of response part of GLLB-potential
        for a, D_sp in D_asp.items():
            ni = self.setups[a].ni
            D_sii = np.zeros((len(D_sp), ni, ni))
            for f_n, kpt in zip(f_un, self.kpt_u):
                self.calculate_atomic_density_matrices_k_point(D_sii, kpt, a,
                                                               f_n)
            D_sp[:] = [pack_density(D_ii) for D_ii in D_sii]
            self.kptband_comm.sum(D_sp)

        self.symmetrize_atomic_density_matrices(D_asp)

    def symmetrize_atomic_density_matrices(self, D_asp):
        if len(self.kd.symmetry.op_scc) == 0:
            return

        D_asp.redistribute(self.atom_partition.as_serial())
        self.setups.atomrotations.symmetrize_atomic_density_matrices(
            D_asp, a_sa=self.kd.symmetry.a_sa)
        D_asp.redistribute(self.atom_partition)

    def calculate_occupation_numbers(self, fix_fermi_level=False):
        if self.collinear and self.nspins == 1:
            degeneracy = 2
        else:
            degeneracy = 1

        f_qn, fermi_levels, e_entropy = self.occupations.calculate(
            nelectrons=self.nvalence / degeneracy,
            eigenvalues=[kpt.eps_n * Ha for kpt in self.kpt_u],
            weights=[kpt.weightk for kpt in self.kpt_u],
            fermi_levels_guess=(self.fermi_levels * Ha
                                if self.fermi_levels is not None else None),
            fix_fermi_level=fix_fermi_level)

        if not fix_fermi_level or self.fermi_levels is None:
            self.fermi_levels = np.array(fermi_levels) / Ha

        for f_n, kpt in zip(f_qn, self.kpt_u):
            kpt.f_n = f_n * (kpt.weightk * degeneracy)

        return e_entropy * degeneracy / Ha

    def set_positions(self, spos_ac, atom_partition=None):
        self.positions_set = False
        # rank_a = self.gd.get_ranks_from_positions(spos_ac)
        # atom_partition = AtomPartition(self.gd.comm, rank_a)
        # XXX pass AtomPartition around instead of spos_ac?
        # All the classes passing around spos_ac end up needing the ranks
        # anyway.

        if atom_partition is None:
            rank_a = self.gd.get_ranks_from_positions(spos_ac)
            atom_partition = AtomPartition(self.gd.comm, rank_a)

        if self.atom_partition is not None and self.kpt_u[0].P_ani is not None:
            with self.timer('Redistribute'):
                for kpt in self.kpt_u:
                    P = kpt.projections
                    assert self.atom_partition == P.atom_partition
                    kpt.projections = P.redist(atom_partition)
                    assert atom_partition == kpt.projections.atom_partition

        self.atom_partition = atom_partition
        self.kd.symmetry.check(spos_ac)
        self.spos_ac = spos_ac

    def allocate_arrays_for_projections(self, my_atom_indices):  # XXX unused
        if not self.positions_set and self.kpt_u[0]._projections is not None:
            # Projections have been read from file - don't delete them!
            pass
        else:
            nproj_a = [setup.ni for setup in self.setups]
            for kpt in self.kpt_u:
                kpt.projections = Projections(
                    self.bd.nbands, nproj_a,
                    self.atom_partition,
                    self.bd.comm,
                    collinear=self.collinear, spin=kpt.s, dtype=self.dtype)

    def collect_eigenvalues(self, k, s):
        return self.collect_array('eps_n', k, s)

    def collect_occupations(self, k, s):
        return self.collect_array('f_n', k, s)

    def collect_array(self, name, k, s, subset=None):
        """Helper method for collect_eigenvalues and collect_occupations.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_qs = self.kpt_qs
        kpt_rank, q = self.kd.get_rank_and_index(k)
        if self.kd.comm.rank == kpt_rank:
            a_nx = getattr(kpt_qs[q][s], name)

            if subset is not None:
                a_nx = a_nx[subset]

            # Domain master send this to the global master
            if self.gd.comm.rank == 0:
                if self.bd.comm.size == 1:
                    if kpt_rank == 0:
                        return a_nx
                    else:
                        self.kd.comm.ssend(a_nx, 0, 1301)
                else:
                    b_nx = self.bd.collect(a_nx)
                    if self.bd.comm.rank == 0:
                        if kpt_rank == 0:
                            return b_nx
                        else:
                            self.kd.comm.ssend(b_nx, 0, 1301)

        elif self.world.rank == 0 and kpt_rank != 0:
            # Only used to determine shape and dtype of receiving buffer:
            a_nx = getattr(kpt_qs[0][0], name)

            if subset is not None:
                a_nx = a_nx[subset]

            b_nx = np.zeros((self.bd.nbands,) + a_nx.shape[1:],
                            dtype=a_nx.dtype)
            self.kd.comm.receive(b_nx, kpt_rank, 1301)
            return b_nx

        return np.zeros(0)  # see comment in get_wave_function_array() method

    def collect_auxiliary(self, value, k, s, shape=1, dtype=float):
        """Helper method for collecting band-independent scalars/arrays.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_rank, q = self.kd.get_rank_and_index(k)

        if self.kd.comm.rank == kpt_rank:
            if isinstance(value, str):
                a_o = getattr(self.kpt_qs[q][s], value)
            else:
                u = q * self.nspins + s
                a_o = value[u]  # assumed list

            # Make sure data is a mutable object
            a_o = np.asarray(a_o)

            if a_o.dtype is not dtype:
                a_o = a_o.astype(dtype)

            # Domain master send this to the global master
            if self.gd.comm.rank == 0:
                if kpt_rank == 0:
                    return a_o
                else:
                    self.kd.comm.send(a_o, 0, 1302)

        elif self.world.rank == 0 and kpt_rank != 0:
            b_o = np.zeros(shape, dtype=dtype)
            self.kd.comm.receive(b_o, kpt_rank, 1302)
            return b_o

    def collect_projections(self, k, s):
        """Helper method for collecting projector overlaps across domains.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, send to the global master."""

        kpt_rank, q = self.kd.get_rank_and_index(k)

        if self.kd.comm.rank == kpt_rank:
            kpt = self.kpt_qs[q][s]
            P_nI = kpt.projections.collect()
            if self.world.rank == 0:
                return P_nI
            if P_nI is not None:
                self.kd.comm.send(np.ascontiguousarray(P_nI), 0, tag=117)
        if self.world.rank == 0:
            nproj = sum(setup.ni for setup in self.setups)
            if not self.collinear:
                nproj *= 2
            P_nI = np.empty((self.bd.nbands, nproj), self.dtype)
            self.kd.comm.receive(P_nI, kpt_rank, tag=117)
            return P_nI

    def get_wave_function_array(self, n, k, s, realspace=True, periodic=False):
        """Return pseudo-wave-function array on master.

        n: int
            Global band index.
        k: int
            Global IBZ k-point index.
        s: int
            Spin index (0 or 1).
        realspace: bool
            Transform plane wave or LCAO expansion coefficients to real-space.

        For the parallel case find the ranks in kd.comm and bd.comm
        that contains to (n, k, s), and collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_rank, q = self.kd.get_rank_and_index(k)
        band_rank, myn = self.bd.who_has(n)

        rank = self.world.rank

        if (self.kd.comm.rank == kpt_rank and
            self.bd.comm.rank == band_rank):
            u = q * self.nspins + s
            psit_G = self._get_wave_function_array(u, myn,
                                                   realspace, periodic)

            if realspace:
                psit_G = self.gd.collect(psit_G)

            if rank == 0:
                return psit_G

            # Domain master send this to the global master
            if self.gd.comm.rank == 0:
                psit_G = np.ascontiguousarray(psit_G)
                self.world.ssend(psit_G, 0, 1398)

        if rank == 0:
            # allocate full wave function and receive
            shape = () if self.collinear else (2,)
            psit_G = self.empty(shape, global_array=True,
                                realspace=realspace)
            # XXX this will fail when using non-standard nesting
            # of communicators.
            world_rank = (kpt_rank * self.gd.comm.size *
                          self.bd.comm.size +
                          band_rank * self.gd.comm.size)
            self.world.receive(psit_G, world_rank, 1398)
            return psit_G

        # We return a number instead of None on all the slaves.  Most of
        # the time the return value will be ignored on the slaves, but
        # in some cases it will be multiplied by some other number and
        # then ignored.  Allowing for this will simplify some code here
        # and there.
        return np.nan

    def get_homo_lumo(self, spin=None, _gllb=False):
        """Return HOMO and LUMO eigenvalues."""
        if spin is None:
            if self.nspins == 1:
                return self.get_homo_lumo(0)
            h0, l0 = self.get_homo_lumo(0)
            h1, l1 = self.get_homo_lumo(1)
            return np.array([max(h0, h1), min(l0, l1)])

        if _gllb:
            # Backwards compatibility (see test/gllb/test_metallic.py test)
            n = self.nvalence // 2
        else:
            nocc = 0.0
            for kpt in self.kpt_u:
                if kpt.s == spin:
                    nocc += kpt.f_n.sum()
            nocc = self.kptband_comm.sum_scalar(nocc) * self.nspins / 2
            n = int(round(nocc))

        band_rank, myn = self.bd.who_has(n - 1)
        homo = -np.inf
        if self.bd.comm.rank == band_rank:
            for kpt in self.kpt_u:
                if kpt.s == spin:
                    homo = max(kpt.eps_n[myn], homo)
        homo = self.world.max_scalar(homo)

        lumo = np.inf
        if n < self.bd.nbands:  # there are not enough bands for LUMO
            band_rank, myn = self.bd.who_has(n)
            if self.bd.comm.rank == band_rank:
                for kpt in self.kpt_u:
                    if kpt.s == spin:
                        lumo = min(kpt.eps_n[myn], lumo)
            lumo = self.world.min_scalar(lumo)

        return np.array([homo, lumo])

    def write(self, writer):
        writer.write(version=2, ha=Ha)
        writer.write(fermi_levels=self.fermi_levels * Ha)
        writer.write(kpts=self.kd)
        self.write_projections(writer)
        self.write_eigenvalues(writer)
        self.write_occupations(writer)

    def write_projections(self, writer):
        nproj = sum(setup.ni for setup in self.setups)

        if self.collinear:
            shape = (self.nspins, self.kd.nibzkpts, self.bd.nbands, nproj)
        else:
            shape = (self.kd.nibzkpts, self.bd.nbands, 2, nproj)

        writer.add_array('projections', shape, self.dtype)

        for s in range(self.nspins):
            for k in range(self.kd.nibzkpts):
                P_nI = self.collect_projections(k, s)
                if not self.collinear and P_nI is not None:
                    P_nI.shape = (self.bd.nbands, 2, nproj)
                writer.fill(P_nI)

    def write_eigenvalues(self, writer):
        if self.collinear:
            shape = (self.nspins, self.kd.nibzkpts, self.bd.nbands)
        else:
            shape = (self.kd.nibzkpts, self.bd.nbands)

        writer.add_array('eigenvalues', shape)
        for s in range(self.nspins):
            for k in range(self.kd.nibzkpts):
                writer.fill(self.collect_eigenvalues(k, s) * Ha)

    def write_occupations(self, writer):

        if self.collinear:
            shape = (self.nspins, self.kd.nibzkpts, self.bd.nbands)
        else:
            shape = (self.kd.nibzkpts, self.bd.nbands)

        writer.add_array('occupations', shape)
        for s in range(self.nspins):
            for k in range(self.kd.nibzkpts):
                # Scale occupation numbers when writing:
                # XXX fix this in the code also ...
                weight = self.kd.weight_k[k] * 2 / self.nspins
                writer.fill(self.collect_occupations(k, s) / weight)

    def read(self, reader):
        wfs_reader = reader.wave_functions
        # Backward compatibility:
        # Take parameters from main reader
        if 'ha' not in wfs_reader:
            wfs_reader.ha = reader.ha
        if 'version' not in wfs_reader:
            wfs_reader.version = reader.version

        if reader.version >= 3:
            self.fermi_levels = wfs_reader.fermi_levels / wfs_reader.ha
        else:
            o = reader.occupations
            self.fermi_levels = np.array(
                [o.fermilevel + o.split / 2,
                 o.fermilevel - o.split / 2]) / wfs_reader.ha
            if self.occupations.name != 'fixmagmom':
                assert o.split == 0.0
                self.fermi_levels = self.fermi_levels[:1]

        if reader.version >= 2:
            kpts = wfs_reader.kpts
            assert np.allclose(kpts.ibzkpts, self.kd.ibzk_kc)
            assert np.allclose(kpts.bzkpts, self.kd.bzk_kc)
            assert (kpts.bz2ibz == self.kd.bz2ibz_k).all()
            assert np.allclose(kpts.weights, self.kd.weight_k)

        if 'projections' in wfs_reader:
            self.read_projections(wfs_reader)
        self.read_eigenvalues(wfs_reader, wfs_reader.version <= 0)
        self.read_occupations(wfs_reader, wfs_reader.version <= 0)

    def read_projections(self, reader):
        nslice = self.bd.get_slice()
        nproj_a = [setup.ni for setup in self.setups]
        atom_partition = AtomPartition(self.gd.comm,
                                       np.zeros(len(nproj_a), int))
        for u, kpt in enumerate(self.kpt_u):
            if self.collinear:
                index = (kpt.s, kpt.k)
            else:
                index = (kpt.k,)
            kpt.projections = Projections(
                self.bd.nbands, nproj_a,
                atom_partition, self.bd.comm,
                collinear=self.collinear, spin=kpt.s, dtype=self.dtype)
            if self.gd.comm.rank == 0:
                P_nI = reader.proxy('projections', *index)[nslice]
                if not self.collinear:
                    P_nI.shape = (self.bd.mynbands, -1)
                kpt.projections.matrix.array[:] = P_nI

    def read_eigenvalues(self, reader, old=False):
        nslice = self.bd.get_slice()
        for u, kpt in enumerate(self.kpt_u):
            if self.collinear:
                index = (kpt.s, kpt.k)
            else:
                index = (kpt.k,)
            eps_n = reader.proxy('eigenvalues', *index)[nslice]
            x = self.bd.mynbands - len(eps_n)  # missing bands?
            if x > 0:
                # Working on a real fix to this parallelization problem ...
                eps_n = np.pad(eps_n, (0, x), 'constant')
            if not old:  # skip for old tar-files gpw's
                eps_n /= reader.ha
            kpt.eps_n = eps_n

    def read_occupations(self, reader, old=False):
        nslice = self.bd.get_slice()
        for u, kpt in enumerate(self.kpt_u):
            if self.collinear:
                index = (kpt.s, kpt.k)
            else:
                index = (kpt.k,)
            f_n = reader.proxy('occupations', *index)[nslice]
            x = self.bd.mynbands - len(f_n)  # missing bands?
            if x > 0:
                # Working on a real fix to this parallelization problem ...
                f_n = np.pad(f_n, (0, x), 'constant')
            if not old:  # skip for old tar-files gpw's
                f_n *= kpt.weight
            kpt.f_n = f_n

    def summary_func(self, log):

        pot = None
        if hasattr(self.eigensolver, 'dm_helper'):
            pot = self.eigensolver.dm_helper.func
        elif hasattr(self.eigensolver, 'odd'):
            pot = self.eigensolver.odd

        f_sn = {}
        for kpt in self.kpt_u:
            u = kpt.s * self.kd.nibzkpts + kpt.q
            f_sn[u] = kpt.f_n

        log("Diagonal elements of Lagrange matrix:")
        if self.nspins == 1:
            header = " Band         L_ii  " \
                     "Occupancy"
            log(header)

            lagr = pot.lagr_diag_s[0]
            for i, x in enumerate(lagr):
                log('%5d  %11.5f  %9.5f' % (
                    i, Ha * x, f_sn[0][i]))

        if self.nspins == 2:
            if self.kd.comm.size > 1:
                if self.kd.comm.rank == 0:
                    # occupation numbers
                    size = np.array([0])
                    self.kd.comm.receive(size, 1)
                    f_2n = np.zeros(shape=(int(size[0])))
                    self.kd.comm.receive(f_2n, 1)
                    f_sn[1] = f_2n

                    # orbital energies
                    size = np.array([0])
                    self.kd.comm.receive(size, 1)
                    lagr_1 = np.zeros(shape=(int(size[0])))
                    self.kd.comm.receive(lagr_1, 1)

                else:
                    # occupations
                    size = np.array([f_sn[1].shape[0]])
                    self.kd.comm.send(size, 0)
                    self.kd.comm.send(f_sn[1], 0)

                    # orbital energies
                    a = pot.lagr_diag_s[1].copy()
                    size = np.array([a.shape[0]])
                    self.kd.comm.send(size, 0)
                    self.kd.comm.send(a, 0)

                    del a

            if self.kd.comm.rank == 0:
                log('                  Up                 '
                    '  Down')
                log(' Band         L_ii   Occupancy  '
                    ' Band      L_ii   Occupancy')

                lagr_0 = pot.lagr_diag_s[0]
                lagr_labeled_0 = {}
                for c, x in enumerate(pot.lagr_diag_s[0]):
                    lagr_labeled_0[str(round(x, 12))] = c

                if self.kd.comm.size == 1:
                    lagr_1 = pot.lagr_diag_s[1]
                    lagr_labeled_1 = {}
                    for c, x in enumerate(
                            pot.lagr_diag_s[1]):
                        lagr_labeled_1[str(round(x, 12))] = c
                else:
                    lagr_labeled_1 = {}
                    for c, x in enumerate(lagr_1):
                        lagr_labeled_1[str(round(x, 12))] = c

                for x, y in zip(lagr_0, lagr_1):
                    i0 = lagr_labeled_0[str(round(x, 12))]
                    i1 = lagr_labeled_1[str(round(y, 12))]

                    log('%5d  %11.5f  %9.5f'
                        '%5d  %11.5f  %9.5f' %
                        (i0, Ha * x,
                         f_sn[0][i0],
                         i1,
                         Ha * y,
                         f_sn[1][i1]))

        log(flush=True)

        sic_n = pot.e_sic_by_orbitals
        if pot.name == 'PZ-SIC':
            log('Perdew-Zunger SIC')
        elif 'SPZ' in pot.name:
            log('Self-Interaction Corrections:\n')
            sf = pot.scalingf
        else:
            raise NotImplementedError

        if self.nspins == 2 and self.kd.comm.size > 1:
            if self.kd.comm.rank == 0:
                size = np.array([0])
                self.kd.comm.receive(size, 1)
                sic_n2 = np.zeros(shape=(int(size[0]), 2),
                                  dtype=float)
                self.kd.comm.receive(sic_n2, 1)
                sic_n[1] = sic_n2

                if 'SPZ' in pot.name:
                    sf_2 = np.zeros(shape=(int(size[0]), 1),
                                    dtype=float)
                    self.kd.comm.receive(sf_2, 1)
                    sf[1] = sf_2
            else:
                size = np.array([sic_n[1].shape[0]])
                self.kd.comm.send(size, 0)
                self.kd.comm.send(sic_n[1], 0)

                if 'SPZ' in pot.name:
                    self.kd.comm.send(sf[1], 0)

        if self.kd.comm.rank == 0:
            for s in range(self.nspins):
                if self.nspins == 2:
                    log('Spin: %3d ' % (s))
                header = """\
            Self-Har.  Self-XC   Hartree + XC  Scaling
            energy:    energy:   energy:       Factors:"""
                log(header)

                occupied = f_sn[s] > 1.0e-10
                f_sn_occ = f_sn[s][occupied]
                occupied_indices = np.where(occupied)[0]
                u_s = 0.0
                xc_s = 0.0
                for i in range(len(sic_n[s])):

                    u = sic_n[s][i][0]
                    xc = sic_n[s][i][1]
                    if 'SPZ' in pot.name:
                        f = (sf[s][i], sf[s][i])
                    else:
                        f = (pot.beta_c, pot.beta_x)

                    log('band: %3d ' %
                        (occupied_indices[i]), end='')
                    log('%11.6f%11.6f%11.6f %8.3f%7.3f' %
                        (-Ha * u / (f[0] * f_sn_occ[i]),
                         -Ha * xc / (f[1] * f_sn_occ[i]),
                         -Ha * (u / (f[0] * f_sn_occ[i]) +
                                xc / (f[1] * f_sn_occ[i])),
                         f[0], f[1]), end='')
                    log(flush=True)
                    u_s += u / (f[0] * f_sn_occ[i])
                    xc_s += xc / (f[1] * f_sn_occ[i])
                log('--------------------------------'
                    '-------------------------')
                log('Total     ', end='')
                log('%11.6f%11.6f%11.6f' %
                    (-Ha * u_s,
                     -Ha * xc_s,
                     -Ha * (u_s + xc_s)
                     ), end='')
                log("\n")
                log(flush=True)


def eigenvalue_string(wfs, comment=' '):
    """Write eigenvalues and occupation numbers into a string.

    The parameter comment can be used to comment out non-numers,
    for example to escape it for gnuplot.
    """
    tokens = []

    def add(*line):
        for token in line:
            tokens.append(token)
        tokens.append('\n')

    def eigs(k, s):
        eps_n = wfs.collect_eigenvalues(k, s)
        return eps_n * Ha

    def occs(k, s):
        occ_n = wfs.collect_occupations(k, s)
        return occ_n / wfs.kd.weight_k[k]

    if len(wfs.kd.ibzk_kc) == 1:
        if wfs.nspins == 1:
            add(comment, 'Band  Eigenvalues  Occupancy')
            eps_n = eigs(0, 0)
            f_n = occs(0, 0)
            if wfs.world.rank == 0:
                for n in range(wfs.bd.nbands):
                    add('%5d  %11.5f  %9.5f' % (n, eps_n[n], f_n[n]))
        else:
            add(comment, '                  Up                     Down')
            add(comment, 'Band  Eigenvalues  Occupancy  Eigenvalues  '
                'Occupancy')
            epsa_n = eigs(0, 0)
            epsb_n = eigs(0, 1)
            fa_n = occs(0, 0)
            fb_n = occs(0, 1)
            if wfs.world.rank == 0:
                for n in range(wfs.bd.nbands):
                    add('%5d  %11.5f  %9.5f  %11.5f  %9.5f' %
                        (n, epsa_n[n], fa_n[n], epsb_n[n], fb_n[n]))
        return ''.join(tokens)

    if len(wfs.kd.ibzk_kc) > 2:
        add('Showing only first 2 kpts')
        print_range = 2
    else:
        add('Showing all kpts')
        print_range = len(wfs.kd.ibzk_kc)

    if wfs.nvalence / 2. > 2:
        m = int(wfs.nvalence / 2. - 2)
    else:
        m = 0
    if wfs.bd.nbands - wfs.nvalence / 2. > 2:
        j = int(wfs.nvalence / 2. + 2)
    else:
        j = int(wfs.bd.nbands)

    if wfs.nspins == 1:
        add(comment, 'Kpt  Band  Eigenvalues  Occupancy')
        for i in range(print_range):
            eps_n = eigs(i, 0)
            f_n = occs(i, 0)
            if wfs.world.rank == 0:
                for n in range(m, j):
                    add('%3i %5d  %11.5f  %9.5f' % (i, n, eps_n[n], f_n[n]))
                add()
    else:
        add(comment, '                     Up                     Down')
        add(comment, 'Kpt  Band  Eigenvalues  Occupancy  Eigenvalues  '
            'Occupancy')

        for i in range(print_range):
            epsa_n = eigs(i, 0)
            epsb_n = eigs(i, 1)
            fa_n = occs(i, 0)
            fb_n = occs(i, 1)
            if wfs.world.rank == 0:
                for n in range(m, j):
                    add('%3i %5d  %11.5f  %9.5f  %11.5f  %9.5f' %
                        (i, n, epsa_n[n], fa_n[n], epsb_n[n], fb_n[n]))
                add()
    return ''.join(tokens)
