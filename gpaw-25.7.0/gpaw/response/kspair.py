from __future__ import annotations

import numpy as np
from functools import cached_property

from gpaw.projections import Projections, serial_comm
from gpaw.response import ResponseGroundStateAdapter, ResponseContext, timer
from gpaw.response.pw_parallelization import Blocks1D


class IrreducibleKPoint:
    """Irreducible k-point data pertaining to a certain set of transitions."""

    def __init__(self, ik, eps_h, f_h, Ph, psit_hG, h_myt):
        """Construct the IrreducibleKPoint data object.

        The data is indexed by the composite band and spin index h = (n, s),
        which can be unfolded to the local transition index myt.
        """
        self.ik = ik             # Irreducible k-point index
        self.eps_h = eps_h       # Eigenvalues
        self.f_h = f_h           # Occupation numbers
        self.Ph = Ph             # PAW projections
        self.psit_hG = psit_hG   # Pseudo wave function plane-wave components
        self.h_myt = h_myt       # myt -> h index mapping

    @cached_property
    def nh(self):
        nh = len(self.eps_h)
        assert len(self.f_h) == nh
        assert self.Ph.nbands == nh
        assert len(self.psit_hG) == nh

        return nh

    @property
    def eps_myt(self):
        return self.eps_h[self.h_myt]

    @property
    def f_myt(self):
        return self.f_h[self.h_myt]

    def projectors_in_transition_index(self, Ph):
        Pmyt = Ph.new(nbands=len(self.h_myt), bcomm=None)
        Pmyt.array[:] = Ph.array[self.h_myt]
        return Pmyt


class KohnShamKPointPair:
    """Data of pairs of Kohn-Sham orbital pertaining to transitions k -> k'."""

    def __init__(self, K1, K2, ikpt1, ikpt2, transitions, tblocks):
        """Construct the KohnShamKPointPair from the k-point data of k and k'.

        K1, K2 : int, int
            k-point indices of k and k'
        ikpt1, ikpt2 : IrreducibleKPoint, IrreducibleKPoint
            k-point data of the two specific k-points in the irreducible part
            of the BZ which are related to K1 and K2 by symmetry respectively.
        """

        self.K1 = K1
        self.K2 = K2
        self.ikpt1 = ikpt1
        self.ikpt2 = ikpt2
        self.transitions = transitions
        self.tblocks = tblocks

    def get_all(self, in_mytx):
        """Get a certain data array with all transitions"""
        return self.tblocks.all_gather(in_mytx)

    @property
    def deps_myt(self):
        return self.ikpt2.eps_myt - self.ikpt1.eps_myt

    @property
    def df_myt(self):
        return self.ikpt2.f_myt - self.ikpt1.f_myt

    def get_local_band_indices(self):
        n1_t, n2_t = self.transitions.get_band_indices()
        n1_myt = n1_t[self.tblocks.myslice]
        n2_myt = n2_t[self.tblocks.myslice]
        return n1_myt, n2_myt

    def get_local_spin_indices(self):
        s1_t, s2_t = self.transitions.get_spin_indices()
        s1_myt = s1_t[self.tblocks.myslice]
        s2_myt = s2_t[self.tblocks.myslice]
        return s1_myt, s2_myt

    def get_local_intraband_mask(self):
        intraband_t = self.transitions.get_intraband_mask()
        return intraband_t[self.tblocks.myslice]


class KohnShamKPointPairExtractor:
    """Functionality to extract KohnShamKPointPairs from a
    ResponseGroundStateAdapter."""

    def __init__(self, gs, context, *,
                 transitions_blockcomm, kpts_blockcomm):
        """
        Parameters
        ----------
        gs : ResponseGroundStateAdapter
        context : ResponseContext
        transitions_blockcomm : gpaw.mpi.Communicator
            Communicator to distribute band and spin transitions
        kpts_blockcomm : gpaw.mpi.Communicator
            Communicator over which the k-point are distributed
        """
        assert isinstance(gs, ResponseGroundStateAdapter)
        self.gs = gs
        assert isinstance(context, ResponseContext)
        self.context = context

        if self.gs.is_parallelized():
            assert self.context.comm is self.gs.world
            # We assume no grid-parallelization in `map_who_has()`
            assert self.gs.gd.comm.size == 1

        self.transitions_blockcomm = transitions_blockcomm
        self.kpts_blockcomm = kpts_blockcomm

        # Prepare to distribute transitions
        self.tblocks = None

        # Prepare to redistribute kptdata
        self.rrequests = []
        self.srequests = []

    @timer('Get Kohn-Sham pairs')
    def get_kpoint_pairs(self, k1_pc, k2_pc,
                         transitions) -> KohnShamKPointPair | None:
        """Get all pairs of Kohn-Sham orbitals for transitions k -> k'

        (n1_t, k1_p, s1_t) -> (n2_t, k2_p, s2_t)

        Here, t is a composite band and spin transition index accounted for by
        the input PairTransitions object, whereas p indexes the k-point that
        each rank of the k-point block communicator needs to extract."""
        assert k1_pc.shape == k2_pc.shape

        # Distribute transitions and extract data for transitions in
        # this process' block
        self.tblocks = Blocks1D(self.transitions_blockcomm, len(transitions))

        K1, ikpt1 = self.get_kpoints(k1_pc, transitions.n1_t, transitions.s1_t)
        K2, ikpt2 = self.get_kpoints(k2_pc, transitions.n2_t, transitions.s2_t)

        # The process might not have a Kohn-Sham k-point pair to return, due to
        # the distribution over kpts_blockcomm
        if self.kpts_blockcomm.rank not in range(len(k1_pc)):
            return None

        assert K1 is not None and ikpt1 is not None
        assert K2 is not None and ikpt2 is not None

        return KohnShamKPointPair(K1, K2, ikpt1, ikpt2,
                                  transitions, self.tblocks)

    def get_kpoints(self, k_pc, n_t, s_t):
        """Get the process' own k-point data and help other processes
        extracting theirs."""
        assert len(n_t) == len(s_t)
        assert len(k_pc) <= self.kpts_blockcomm.size

        # Use the data extraction factory to extract the kptdata
        kptdata = self.extract_kptdata(k_pc, n_t, s_t)

        if self.kpts_blockcomm.rank not in range(len(k_pc)):
            return None, None  # The process has no data of its own

        assert kptdata is not None
        K = kptdata[0]
        ikpt = IrreducibleKPoint(*kptdata[1:])

        return K, ikpt

    @timer('Extracting data from the ground state calculator object')
    def extract_kptdata(self, k_pc, n_t, s_t):
        """Extract the input data needed to construct the IrreducibleKPoints.
        """
        if self.gs.is_parallelized():
            return self.parallel_extract_kptdata(k_pc, n_t, s_t)
        else:
            return self.serial_extract_kptdata(k_pc, n_t, s_t)
            # Useful for debugging:
            # return self.parallel_extract_kptdata(k_pc, n_t, s_t)

    def parallel_extract_kptdata(self, k_pc, n_t, s_t):
        """Extract the k-point data from a parallelized calculator."""
        (myK, myik, myu_eu,
         myn_eueh, ik_r2,
         nrh_r2, eh_eur2reh,
         rh_eur2reh, h_r1rh,
         h_myt) = self.get_parallel_extraction_protocol(k_pc, n_t, s_t)

        (eps_r1rh, f_r1rh,
         P_r1rhI, psit_r1rhG,
         eps_r2rh, f_r2rh,
         P_r2rhI, psit_r2rhG) = self.allocate_transfer_arrays(myik, nrh_r2,
                                                              ik_r2, h_r1rh)

        # Do actual extraction
        for myu, myn_eh, eh_r2reh, rh_r2reh in zip(myu_eu, myn_eueh,
                                                   eh_eur2reh, rh_eur2reh):

            eps_eh, f_eh, P_ehI = self.extract_wfs_data(myu, myn_eh)

            for r2, (eh_reh, rh_reh) in enumerate(zip(eh_r2reh, rh_r2reh)):
                if eh_reh:
                    eps_r2rh[r2][rh_reh] = eps_eh[eh_reh]
                    f_r2rh[r2][rh_reh] = f_eh[eh_reh]
                    P_r2rhI[r2][rh_reh] = P_ehI[eh_reh]

            # Wavefunctions are heavy objects which can only be extracted
            # for one band index at a time, handle them seperately
            self.add_wave_function(myu, myn_eh, eh_r2reh,
                                   rh_r2reh, psit_r2rhG)

        self.distribute_extracted_data(eps_r1rh, f_r1rh, P_r1rhI, psit_r1rhG,
                                       eps_r2rh, f_r2rh, P_r2rhI, psit_r2rhG)

        # Some processes may not have to return a k-point
        if myik is None:
            data = None
        else:
            eps_h, f_h, Ph, psit_hG = self.collect_kptdata(
                myik, h_r1rh, eps_r1rh, f_r1rh, P_r1rhI, psit_r1rhG)
            data = myK, myik, eps_h, f_h, Ph, psit_hG, h_myt

        # Wait for communication to finish
        with self.context.timer('Waiting to complete mpi.send'):
            while self.srequests:
                self.context.comm.wait(self.srequests.pop(0))

        return data

    @timer('Create data extraction protocol')
    def get_parallel_extraction_protocol(self, k_pc, n_t, s_t):
        """Figure out how to extract data efficiently in parallel."""
        comm = self.context.comm
        get_extraction_info = self.create_get_extraction_info()

        # (K, ik) for each process
        mykpt = (None, None)

        # Extraction protocol
        myu_eu = []
        myn_eueh = []

        # Data distribution protocol
        nrh_r2 = np.zeros(comm.size, dtype=int)
        ik_r2 = [None for _ in range(comm.size)]
        eh_eur2reh = []
        rh_eur2reh = []
        h_r1rh = [list([]) for _ in range(comm.size)]

        # h to t index mapping
        t_myt = self.tblocks.myslice
        n_myt, s_myt = n_t[t_myt], s_t[t_myt]
        h_myt = np.empty(self.tblocks.nlocal, dtype=int)

        nt = len(n_t)
        assert nt == len(s_t)
        t_t = np.arange(nt)
        nh = 0
        for p, k_c in enumerate(k_pc):  # p indicates the receiving process
            K = self.gs.kpoints.kptfinder.find(k_c)
            ik = self.gs.kd.bz2ibz_k[K]
            for r2 in range(p * self.tblocks.blockcomm.size,
                            min((p + 1) * self.tblocks.blockcomm.size,
                                comm.size)):
                ik_r2[r2] = ik

            if p == self.kpts_blockcomm.rank:
                mykpt = (K, ik)

            # Find out who should store the data in KSKPpoint
            r2_t, myt_t = self.map_who_has(p, t_t)

            # Find out how to extract data
            # In the ground state, kpts are indexed by u=(s, k)
            for s in set(s_t):
                thiss_myt = s_myt == s
                thiss_t = s_t == s
                t_ct = t_t[thiss_t]
                n_ct = n_t[thiss_t]
                r2_ct = r2_t[t_ct]

                # Find out where data is in GS
                u = ik * self.gs.nspins + s
                myu, r1_ct, myn_ct = get_extraction_info(u, n_ct, r2_ct)

                # If the process is extracting or receiving data,
                # figure out how to do so
                if comm.rank in np.append(r1_ct, r2_ct):
                    # Does this process have anything to send?
                    thisr1_ct = r1_ct == comm.rank
                    if np.any(thisr1_ct):
                        eh_r2reh = [list([]) for _ in range(comm.size)]
                        rh_r2reh = [list([]) for _ in range(comm.size)]
                        # Find composite indeces h = (n, s)
                        n_et = n_ct[thisr1_ct]
                        n_eh = np.unique(n_et)
                        # Find composite local band indeces
                        myn_eh = np.unique(myn_ct[thisr1_ct])

                        # Where to send the data
                        r2_et = r2_ct[thisr1_ct]
                        for r2 in np.unique(r2_et):
                            thisr2_et = r2_et == r2
                            # What ns are the process sending?
                            n_reh = np.unique(n_et[thisr2_et])
                            eh_reh = []
                            for n in n_reh:
                                eh_reh.append(np.where(n_eh == n)[0][0])
                            # How to send it
                            eh_r2reh[r2] = eh_reh
                            nreh = len(eh_reh)
                            rh_r2reh[r2] = np.arange(nreh) + nrh_r2[r2]
                            nrh_r2[r2] += nreh

                        myu_eu.append(myu)
                        myn_eueh.append(myn_eh)
                        eh_eur2reh.append(eh_r2reh)
                        rh_eur2reh.append(rh_r2reh)

                    # Does this process have anything to receive?
                    thisr2_ct = r2_ct == comm.rank
                    if np.any(thisr2_ct):
                        # Find unique composite indeces h = (n, s)
                        n_rt = n_ct[thisr2_ct]
                        n_rn = np.unique(n_rt)
                        nrn = len(n_rn)
                        h_rn = np.arange(nrn) + nh
                        nh += nrn

                        # Where to get the data from
                        r1_rt = r1_ct[thisr2_ct]
                        for r1 in np.unique(r1_rt):
                            thisr1_rt = r1_rt == r1
                            # What ns are the process getting?
                            n_reh = np.unique(n_rt[thisr1_rt])
                            # Where to put them
                            for n in n_reh:
                                h = h_rn[np.where(n_rn == n)[0][0]]
                                h_r1rh[r1].append(h)

                                # h to t mapping
                                thisn_myt = n_myt == n
                                thish_myt = np.logical_and(thisn_myt,
                                                           thiss_myt)
                                h_myt[thish_myt] = h

        return (*mykpt, myu_eu, myn_eueh, ik_r2, nrh_r2,
                eh_eur2reh, rh_eur2reh, h_r1rh, h_myt)

    def create_get_extraction_info(self):
        """Creator component of the extraction information factory."""
        if self.gs.is_parallelized():
            return self.get_parallel_extraction_info
        else:
            return self.get_serial_extraction_info

    @staticmethod
    def get_serial_extraction_info(u, n_ct, r2_ct):
        """Figure out where to extract the data from in the gs calc"""
        # Let the process extract its own data
        myu = u  # The process has access to all data
        r1_ct = r2_ct
        myn_ct = n_ct

        return myu, r1_ct, myn_ct

    def get_parallel_extraction_info(self, u, n_ct, *unused):
        """Figure out where to extract the data from in the gs calc"""
        gs = self.gs
        # Find out where data is in GS
        k, s = divmod(u, gs.nspins)
        kptrank, q = gs.kd.who_has(k)
        myu = q * gs.nspins + s
        r1_ct, myn_ct = [], []
        for n in n_ct:
            bandrank, myn = gs.bd.who_has(n)
            # XXX this will fail when using non-standard nesting
            # of communicators.
            r1 = (kptrank * gs.gd.comm.size * gs.bd.comm.size
                  + bandrank * gs.gd.comm.size)
            r1_ct.append(r1)
            myn_ct.append(myn)

        return myu, np.array(r1_ct), np.array(myn_ct)

    @timer('Allocate transfer arrays')
    def allocate_transfer_arrays(self, myik, nrh_r2, ik_r2, h_r1rh):
        """Allocate arrays for intermediate storage of data."""
        kptex = self.gs.kpt_u[0]
        Pshape = kptex.projections.array.shape
        Pdtype = kptex.projections.matrix.dtype
        psitdtype = kptex.psit.array.dtype

        # Number of h-indeces to receive
        nrh_r1 = [len(h_rh) for h_rh in h_r1rh]

        # if self.kpts_blockcomm.rank in range(len(ik_p)):
        if myik is not None:
            ng = self.gs.global_pd.ng_q[myik]
            eps_r1rh, f_r1rh, P_r1rhI, psit_r1rhG = [], [], [], []
            for nrh in nrh_r1:
                if nrh >= 1:
                    eps_r1rh.append(np.empty(nrh))
                    f_r1rh.append(np.empty(nrh))
                    P_r1rhI.append(np.empty((nrh,) + Pshape[1:], dtype=Pdtype))
                    psit_r1rhG.append(np.empty((nrh, ng), dtype=psitdtype))
                else:
                    eps_r1rh.append(None)
                    f_r1rh.append(None)
                    P_r1rhI.append(None)
                    psit_r1rhG.append(None)
        else:
            eps_r1rh, f_r1rh, P_r1rhI, psit_r1rhG = None, None, None, None

        eps_r2rh, f_r2rh, P_r2rhI, psit_r2rhG = [], [], [], []
        for nrh, ik in zip(nrh_r2, ik_r2):
            if nrh:
                eps_r2rh.append(np.empty(nrh))
                f_r2rh.append(np.empty(nrh))
                P_r2rhI.append(np.empty((nrh,) + Pshape[1:], dtype=Pdtype))
                ng = self.gs.global_pd.ng_q[ik]
                psit_r2rhG.append(np.empty((nrh, ng), dtype=psitdtype))
            else:
                eps_r2rh.append(None)
                f_r2rh.append(None)
                P_r2rhI.append(None)
                psit_r2rhG.append(None)

        return (eps_r1rh, f_r1rh, P_r1rhI, psit_r1rhG,
                eps_r2rh, f_r2rh, P_r2rhI, psit_r2rhG)

    def map_who_has(self, p, t_t):
        """Convert k-point and transition index to global world rank
        and local transition index"""
        trank_t, myt_t = np.divmod(t_t, self.tblocks.blocksize)
        return p * self.tblocks.blockcomm.size + trank_t, myt_t

    @timer('Extracting eps, f and P_I from wfs')
    def extract_wfs_data(self, myu, myn_eh):
        kpt = self.gs.kpt_u[myu]
        # Get eig and occ
        eps_eh, f_eh = kpt.eps_n[myn_eh], kpt.f_n[myn_eh] / kpt.weight

        # Get projections
        assert kpt.projections.atom_partition.comm.size == 1
        P_ehI = kpt.projections.array[myn_eh]

        return eps_eh, f_eh, P_ehI

    @timer('Extracting wave function from wfs')
    def add_wave_function(self, myu, myn_eh,
                          eh_r2reh, rh_r2reh, psit_r2rhG):
        """Add the plane wave coefficients of the smooth part of
        the wave function to the psit_r2rtG arrays."""
        kpt = self.gs.kpt_u[myu]

        for eh_reh, rh_reh, psit_rhG in zip(eh_r2reh, rh_r2reh, psit_r2rhG):
            if eh_reh:
                for eh, rh in zip(eh_reh, rh_reh):
                    psit_rhG[rh] = kpt.psit_nG[myn_eh[eh]]

    @timer('Distributing kptdata')
    def distribute_extracted_data(self, eps_r1rh, f_r1rh, P_r1rhI, psit_r1rhG,
                                  eps_r2rh, f_r2rh, P_r2rhI, psit_r2rhG):
        """Send the extracted data to appropriate destinations"""
        comm = self.context.comm
        # Store the data extracted by the process itself
        rank = comm.rank
        # Check if there is actually some data to store
        if eps_r2rh[rank] is not None:
            eps_r1rh[rank] = eps_r2rh[rank]
            f_r1rh[rank] = f_r2rh[rank]
            P_r1rhI[rank] = P_r2rhI[rank]
            psit_r1rhG[rank] = psit_r2rhG[rank]

        # Receive data
        if eps_r1rh is not None:  # The process may not be receiving anything
            for r1, (eps_rh, f_rh,
                     P_rhI, psit_rhG) in enumerate(zip(eps_r1rh, f_r1rh,
                                                       P_r1rhI, psit_r1rhG)):
                # Check if there is any data to receive
                if r1 != rank and eps_rh is not None:
                    rreq1 = comm.receive(eps_rh, r1, tag=201, block=False)
                    rreq2 = comm.receive(f_rh, r1, tag=202, block=False)
                    rreq3 = comm.receive(P_rhI, r1, tag=203, block=False)
                    rreq4 = comm.receive(psit_rhG, r1, tag=204, block=False)
                    self.rrequests += [rreq1, rreq2, rreq3, rreq4]

        # Send data
        for r2, (eps_rh, f_rh,
                 P_rhI, psit_rhG) in enumerate(zip(eps_r2rh, f_r2rh,
                                                   P_r2rhI, psit_r2rhG)):
            # Check if there is any data to send
            if r2 != rank and eps_rh is not None:
                sreq1 = comm.send(eps_rh, r2, tag=201, block=False)
                sreq2 = comm.send(f_rh, r2, tag=202, block=False)
                sreq3 = comm.send(P_rhI, r2, tag=203, block=False)
                sreq4 = comm.send(psit_rhG, r2, tag=204, block=False)
                self.srequests += [sreq1, sreq2, sreq3, sreq4]

        with self.context.timer('Waiting to complete mpi.receive'):
            while self.rrequests:
                comm.wait(self.rrequests.pop(0))

    @timer('Collecting kptdata')
    def collect_kptdata(self, myik, h_r1rh,
                        eps_r1rh, f_r1rh, P_r1rhI, psit_r1rhG):
        """From the extracted data, collect the IrreducibleKPoint data arrays
        """
        # Allocate data arrays
        maxh_r1 = [max(h_rh) for h_rh in h_r1rh if h_rh]
        if maxh_r1:
            nh = max(maxh_r1) + 1
        else:  # Carry around empty arrays
            assert self.tblocks.a == self.tblocks.b
            nh = 0
        eps_h = np.empty(nh)
        f_h = np.empty(nh)
        Ph = self.new_projections(nh)
        psit_hG = self.new_wfs(nh, self.gs.global_pd.ng_q[myik])

        # Store extracted data in the arrays
        for (h_rh, eps_rh,
             f_rh, P_rhI, psit_rhG) in zip(h_r1rh, eps_r1rh,
                                           f_r1rh, P_r1rhI, psit_r1rhG):
            if h_rh:
                eps_h[h_rh] = eps_rh
                f_h[h_rh] = f_rh
                Ph.array[h_rh] = P_rhI
                psit_hG[h_rh] = psit_rhG

        return eps_h, f_h, Ph, psit_hG

    def new_projections(self, nh):
        proj = self.gs.kpt_u[0].projections
        # We have to initialize the projections by hand, because
        # Projections.new() interprets nbands == 0 to imply that it should
        # inherit the preexisting number of bands...
        return Projections(nh, proj.nproj_a, proj.atom_partition, serial_comm,
                           proj.collinear, proj.spin, proj.matrix.dtype)

    def new_wfs(self, nh, nG):
        assert self.gs.dtype == self.gs.kpt_u[0].psit.array.dtype
        return np.empty((nh, nG), self.gs.dtype)

    def serial_extract_kptdata(self, k_pc, n_t, s_t):
        """Extract the k-point data from a serial calculator.

        Since all the processes can access all of the data, each process
        extracts the data of its own k-point without any need for
        communication."""
        if self.kpts_blockcomm.rank not in range(len(k_pc)):
            # No data to extract
            return None

        # Find k-point indeces
        k_c = k_pc[self.kpts_blockcomm.rank]
        K = self.gs.kpoints.kptfinder.find(k_c)
        ik = self.gs.kd.bz2ibz_k[K]

        (myu_eu, myn_eurn, nh,
         h_eurn, h_myt) = self.get_serial_extraction_protocol(ik, n_t, s_t)

        # Allocate transfer arrays
        eps_h = np.empty(nh)
        f_h = np.empty(nh)
        Ph = self.new_projections(nh)
        psit_hG = self.new_wfs(nh, self.gs.pd.ng_q[ik])

        # Extract data from the ground state
        for myu, myn_rn, h_rn in zip(myu_eu, myn_eurn, h_eurn):
            kpt = self.gs.kpt_u[myu]
            with self.context.timer('Extracting eps, f and P_I from wfs'):
                eps_h[h_rn] = kpt.eps_n[myn_rn]
                f_h[h_rn] = kpt.f_n[myn_rn] / kpt.weight
                Ph.array[h_rn] = kpt.projections.array[myn_rn]

            with self.context.timer('Extracting wave function from wfs'):
                for myn, h in zip(myn_rn, h_rn):
                    psit_hG[h] = kpt.psit_nG[myn]

        return K, ik, eps_h, f_h, Ph, psit_hG, h_myt

    @timer('Create data extraction protocol')
    def get_serial_extraction_protocol(self, ik, n_t, s_t):
        """Figure out how to extract data efficiently in serial."""

        # Only extract the transitions handled by the process itself
        t_myt = self.tblocks.myslice
        n_myt = n_t[t_myt]
        s_myt = s_t[t_myt]

        # In the ground state, kpts are indexed by u=(s, k)
        myu_eu = []
        myn_eurn = []
        nh = 0
        h_eurn = []
        h_myt = np.empty(self.tblocks.nlocal, dtype=int)
        for s in set(s_myt):
            thiss_myt = s_myt == s
            n_ct = n_myt[thiss_myt]

            # Find unique composite h = (n, u) indeces
            n_rn = np.unique(n_ct)
            nrn = len(n_rn)
            h_eurn.append(np.arange(nrn) + nh)
            nh += nrn

            # Find mapping between h and the transition index
            for n, h in zip(n_rn, h_eurn[-1]):
                thisn_myt = n_myt == n
                thish_myt = np.logical_and(thisn_myt, thiss_myt)
                h_myt[thish_myt] = h

            # Find out where data is
            u = ik * self.gs.nspins + s
            # The process has access to all data
            myu = u
            myn_rn = n_rn

            myu_eu.append(myu)
            myn_eurn.append(myn_rn)

        return myu_eu, myn_eurn, nh, h_eurn, h_myt
