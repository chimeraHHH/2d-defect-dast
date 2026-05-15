import numpy as np
from abc import ABC, abstractmethod

from gpaw.utilities.progressbar import ProgressBar
from gpaw.typing import Vector

from gpaw.response import ResponseGroundStateAdapter, ResponseContext, timer
from gpaw.response.frequencies import ComplexFrequencyDescriptor
from gpaw.response.symmetry import QSymmetryAnalyzer
from gpaw.response.kspair import (KohnShamKPointPair,
                                  KohnShamKPointPairExtractor)
from gpaw.response.pw_parallelization import block_partition
from gpaw.response.pair_transitions import PairTransitions


class PairFunction(ABC):
    r"""Pair function data object.

    In the GPAW response module, a pair function is understood as any function
    which can be written as a sum over the eigenstate transitions with a given
    crystal momentum difference q,
             __
             \
    pf(q) =  /  pf_αα' δ_{q,q_{α',α}}
             ‾‾
             α,α'

    where pf_αα' is an arbitrary matrix element.
    """

    def __init__(self, q_c: Vector):
        self.q_c = np.asarray(q_c)
        self.array = self.zeros()

    @abstractmethod
    def zeros(self):
        """Generate an array of zeros, representing the pair function."""


class DynamicPairFunction(PairFunction):
    r"""Dynamic pair function data object.

    A dynamic pair function is not only a function of the crystal momentum q,
    but also of the complex frequency z = ω + iη:
               __
               \
    pf(q,z) =  /  pf_αα'(z) δ_{q,q_{α',α}}
               ‾‾
               α,α'

    Typically, this will be some generalized (linear) susceptibility, which is
    defined by the Kubo formula,

                   i           ˰          ˰
    χ_BA(t-t') = - ‾ θ(t-t') <[B_0(t-t'), A]>_0
                   ħ

    and can be written in its Lehmann representation as a function of frequency
    in the upper half complex frequency plane,

               __      ˰        ˰
               \    <α|B|α'><α'|A|α>
    χ_BA(z) =  /   ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ (n_α - n_α')
               ‾‾   ħz - (E_α' - E_α)
               α,α'

    where E_α and n_α are the eigenstate energies and occupations respectively.

    For more information, please refer to [Skovhus T., PhD. Thesis, 2021]."""

    def __init__(self, q_c: Vector, zd: ComplexFrequencyDescriptor):
        self.zd = zd
        super().__init__(q_c)


class PairFunctionIntegrator(ABC):
    r"""Baseclass for computing pair functions in the Kohn-Sham system.

    The implementation is currently restricted collinear periodic crystals in
    absence of spin-orbit coupling.

    In the Kohn-Sham system, pair functions can be constructed
    straight-forwardly as a sum over transitions between Kohn-Sham eigenstates
    at k and k + q,
                __  __  __                          __
             1  \   \   \                        1  \
    pf(q) =  ‾  /   /   /   pf_nks,n'k+qs'(q) =  ‾  /  pf_T(q)
             V  ‾‾  ‾‾  ‾‾                       V  ‾‾
                k  n,n' s,s'                        T

    where V is the crystal volume and T is a composite index encoding all
    relevant transitions:

    T: (n, k, s) -> (n', k + q, s')

    The sum over transitions can be split into two steps: (1) an integral over
    k-points k inside the 1st Brillouin Zone and (2) a sum over band and spin
    transitions t:

    t (composite transition index): (n, s) -> (n', s')
                __               __  __                __
             1  \             1  \   \              1  \
    pf(q) =  ‾  /  pf_T(q) =  ‾  /   /  pf_kt(q) =  ‾  /  (...)_k
             V  ‾‾            V  ‾‾  ‾‾             V  ‾‾
                T                k   t                 k

    In the code, the k-point integral is handled by a KPointPairIntegral
    object, while the sum over band and spin transitions t is carried out in
    the self.add_integrand() method, which also defines the specific pair
    function in question.

    KPointPairIntegral:
       __
    1  \
    ‾  /  (...)_k
    V  ‾‾
       k

    self.add_integrand():
                __              __   __
                \               \    \
    (...)_k  =  /  pf_kt(q)  =  /    /   pf_nks,n'k+qs'(q)
                ‾‾              ‾‾   ‾‾
                t               n,n' s,s'

    In practise, the integration is carried out by letting the
    KPointPairIntegral extract individual KohnShamKPointPair objects, which
    contain all relevant information about the Kohn-Sham eigenstates at k and
    k + q for a number of specified spin and band transitions t.
    KPointPairIntegral.weighted_kpoint_pairs() generates these kptpairs along
    with their integral weights such that self._integrate() can construct the
    pair functions in a flexible, yet general manner.
    """

    def __init__(self, gs, context, qsymmetry=True, nblocks=1):
        """Construct the PairFunctionIntegrator

        Parameters
        ----------
        gs : ResponseGroundStateAdaptable
        context : ResponseContextInput
        qsymmetry : QSymmetryInput
        nblocks : int
            Distribute the pair function into nblocks. Useful when the pair
            function itself becomes a large array (read: memory limiting).
        """
        self.gs = ResponseGroundStateAdapter.from_input(gs)
        self.context = ResponseContext.from_input(context)
        self.qsymmetry = QSymmetryAnalyzer.from_input(qsymmetry)

        # Communicators for distribution of memory and work
        (self.blockcomm,
         self.intrablockcomm) = self.create_communicators(nblocks)
        self.nblocks = self.blockcomm.size

        # The KohnShamKPointPairExtractor class handles extraction of k-point
        # pairs from the ground state
        self.kptpair_extractor = KohnShamKPointPairExtractor(
            self.gs, self.context,
            # Distribution of work:
            # t-transitions are distributed through blockcomm,
            # k-points through intrablockcomm.
            transitions_blockcomm=self.blockcomm,
            kpts_blockcomm=self.intrablockcomm)

    @timer('Integrate pair function')
    def _integrate(self, out: PairFunction, transitions: PairTransitions):
        """In-place pair function integration

        Parameters
        ----------
        out : PairFunction
            Output data structure
        transitions : PairTransitions
            Band and spin transitions to integrate.

        Returns
        -------
        symmetries : QSymmetries
        """
        symmetries, generator = self.qsymmetry.analyze(
            out.q_c, self.gs.kpoints, self.context)

        # Perform the actual integral as a point integral over k-point pairs
        integral = KPointPairPointIntegral(self.kptpair_extractor, generator)
        weighted_kptpairs = integral.weighted_kpoint_pairs(transitions)
        pb = ProgressBar(self.context.fd)  # pb with a generator is awkward
        for _, _ in pb.enumerate([None] * integral.ni):
            kptpair, weight = next(weighted_kptpairs)
            if weight is not None:
                assert kptpair is not None
                self.add_integrand(kptpair, weight, out)

        # Sum over the k-points, which have been distributed between processes
        with self.context.timer('Sum over distributed k-points'):
            self.intrablockcomm.sum(out.array)

        return symmetries

    @abstractmethod
    def add_integrand(self, kptpair: KohnShamKPointPair, weight,
                      out: PairFunction):
        """Add the relevant integrand of the outer k-point integral to the
        output data structure 'out', weighted by 'weight' and constructed
        from the provided KohnShamKPointPair 'kptpair'.

        This method effectively defines the pair function in question.
        """

    def create_communicators(self, nblocks):
        """Create MPI communicators to distribute the memory needed to store
        large arrays and parallelize calculations when possible.

        Parameters
        ----------
        nblocks : int
            Separate large arrays into n different blocks. Each process
            allocates memory for the large arrays. By allocating only a
            fraction/block of the total arrays, the memory requirements are
            eased.

        Returns
        -------
        blockcomm : gpaw.mpi.Communicator
            Communicate between processes belonging to different memory blocks.
            In every communicator, there is one process for each block of
            memory, so that all blocks are represented.
            If nblocks < comm.size, there will be size // nblocks different
            processes that allocate memory for the same block of the large
            arrays. Thus, there will be also size // nblocks different block
            communicators, grouping the processes into sets that allocate the
            entire arrays between them.
        intrablockcomm : gpaw.mpi.Communicator
            Communicate between processes belonging to the same memory block.
            There will be size // nblocks processes per memory block.
        """
        comm = self.context.comm
        blockcomm, intrablockcomm = block_partition(comm, nblocks)

        return blockcomm, intrablockcomm

    def get_band_and_spin_transitions(self, spincomponent, nbands=None,
                                      bandsummation='pairwise'):
        """Get band and spin transitions (n, s) -> (n', s') to integrate."""
        # Defaults to inclusion of all bands in the ground state calculator
        if nbands is None:
            nbands = self.gs.nbands
        assert nbands <= self.gs.nbands
        return PairTransitions.from_transitions_domain_arguments(
            spincomponent, nbands, self.gs.nocc1, self.gs.nocc2,
            self.gs.nspins, bandsummation)

    def get_basic_info_string(self):
        """Get basic information about the ground state and parallelization."""
        nspins = self.gs.nspins
        nbands, nocc1, nocc2 = self.gs.nbands, self.gs.nocc1, self.gs.nocc2
        nk = self.gs.kd.nbzkpts
        nik = self.gs.kd.nibzkpts

        csize = self.context.comm.size
        knsize = self.intrablockcomm.size
        bsize = self.blockcomm.size

        isl = ['',
               'The pair function integration is based on a ground state '
               'with:',
               f'    Number of spins: {nspins}',
               f'    Number of bands: {nbands}',
               f'    Number of completely occupied bands: {nocc1}',
               f'    Number of partially occupied bands: {nocc2}',
               f'    Number of kpoints: {nk}',
               f'    Number of irreducible kpoints: {nik}',
               '',
               'The pair function integration is performed in parallel with:',
               f'    comm.size: {csize}',
               f'    intrablockcomm.size: {knsize}',
               f'    blockcomm.size: {bsize}']

        return '\n'.join(isl)

    @staticmethod
    def get_band_and_transitions_info_string(nbands, nt):
        isl = []  # info string list
        if nbands is None:
            isl.append('    Bands included: All')
        else:
            isl.append(f'    Number of bands included: {nbands}')
        isl.append('Resulting in:')
        isl.append(f'    A total number of band and spin transitions of: {nt}')
        return '\n'.join(isl)


class KPointPairIntegral(ABC):
    r"""Baseclass for reciprocal space integrals of the first Brillouin Zone,
    where the integrand is a sum over transitions between any number of states
    at the wave vectors k and k + q (referred to as a k-point pair).

    Definition (V is the total crystal hypervolume and D is the dimension of
    the crystal):
       __
    1  \                 1    /
    ‾  /  (...)_k  =  ‾‾‾‾‾‾  |dk (...)_k
    V  ‾‾             (2π)^D  /
       k

    NB: In the current implementation, the dimension is fixed to 3. This is
    sensible for pair functions which are functions of position (such as the
    four-component Kohn-Sham susceptibility tensor), and in most circumstances
    a change in dimensionality can be accomplished simply by adding an extra
    prefactor to the integral elsewhere.
    """

    def __init__(self, kptpair_extractor, generator):
        """Construct a KPointPairIntegral corresponding to a given q-point.

        Parameters
        ----------
        kptpair_extractor : KohnShamKPointPairExtractor
            Object responsible for extracting all relevant information about
            the k-point pairs from the underlying ground state calculation.
        generator : KPointDomainGenerator
            Object responsible for generating the k-point integration domain
            based on the symmetries of the q-point in question.
        """
        self.gs = kptpair_extractor.gs
        self.kptpair_extractor = kptpair_extractor
        self.q_c = generator.symmetries.q_c

        # Prepare the k-point pair integral
        bzk_kc, weight_k = self.get_kpoint_domain(generator)
        bzk_ipc, weight_i = self.slice_kpoint_domain(bzk_kc, weight_k)
        self._domain = (bzk_ipc, weight_i)
        self.ni = len(weight_i)

    def weighted_kpoint_pairs(self, transitions):
        r"""Generate all k-point pairs in the integral along with their
        integral weights.

        The reciprocal space integral is estimated as the sum over a discrete
        k-point domain. The domain will genererally depend on the integration
        method as well as the symmetry of the crystal.

        Definition:
                                        __
           1    /            ~     1    \   (2π)^D
        ‾‾‾‾‾‾  |dk (...)_k  =  ‾‾‾‾‾‾  /   ‾‾‾‾‾‾ w_kr (...)_kr
        (2π)^D  /               (2π)^D  ‾‾  Nk V0
                                        kr
                                __
                             ~  \
                             =  /  iw_kr (...)_kr
                                ‾‾
                                kr

        The sum over kr denotes the reduced k-point domain specified by the
        integration method (a reduced selection of Nkr points from the ground
        state k-point grid of Nk total points in the entire 1BZ). Each point
        is weighted by its k-point volume in the ground state k-point grid

                      (2π)^D
        kpointvol  =  ‾‾‾‾‾‾,
                      Nk V0

        and an additional individual k-point weight w_kr specific to the
        integration method (V0 denotes the cell volume). Together with the
        integral prefactor, these make up the integral weight

                   1
        iw_kr = ‾‾‾‾‾‾ kpointvol w_kr
                (2π)^D

        Parameters
        ----------
        transitions : PairTransitions
            Band and spin transitions to integrate.
        """
        # Calculate prefactors
        outer_prefactor = 1 / (2 * np.pi)**3
        V = self.crystal_volume()  # V = Nk * V0
        kpointvol = (2 * np.pi)**3 / V
        prefactor = outer_prefactor * kpointvol

        # Generate k-point pairs
        for k_pc, weight in zip(*self._domain):
            if weight is None:
                integral_weight = None
            else:
                integral_weight = prefactor * weight
            kptpair = self.kptpair_extractor.get_kpoint_pairs(
                k_pc, k_pc + self.q_c, transitions)
            yield kptpair, integral_weight

    @abstractmethod
    def get_kpoint_domain(self, generator):
        """Use the KPointDomainGenerator to define and weight the k-integral.

        Returns
        -------
        bzk_kc : np.array
            k-points to integrate in relative coordinates.
        weight_k : np.array
            Integral weight of each k-point in the integral.
        """

    def slice_kpoint_domain(self, bzk_kc, weight_k):
        """When integrating over k-points, slice the domain in pieces with one
        k-point per process each.

        Returns
        -------
        bzk_ipc : nd.array
            k-points (relative) coordinates for each process for each iteration
        """
        comm = self.kptpair_extractor.kpts_blockcomm
        rank, size = comm.rank, comm.size

        nk = bzk_kc.shape[0]
        ni = (nk + size - 1) // size
        bzk_ipc = [bzk_kc[i * size:(i + 1) * size] for i in range(ni)]

        # Extract the weight corresponding to the process' own k-point pair
        weight_ip = [weight_k[i * size:(i + 1) * size] for i in range(ni)]
        weight_i = [None] * len(weight_ip)
        for i, w_p in enumerate(weight_ip):
            if rank in range(len(w_p)):
                weight_i[i] = w_p[rank]

        return bzk_ipc, weight_i

    def crystal_volume(self):
        """Calculate the total crystal volume, V = Nk * V0, corresponding to
        the ground state k-point grid."""
        return self.gs.kd.nbzkpts * self.gs.volume


class KPointPairPointIntegral(KPointPairIntegral):
    r"""Reciprocal space integral of k-point pairs in the first Brillouin Zone,
    estimated as a point integral over all k-points of the ground state k-point
    grid.

    Definition:
                                   __
       1    /           ~     1    \   (2π)^D
    ‾‾‾‾‾‾  |dk (...)_k =  ‾‾‾‾‾‾  /   ‾‾‾‾‾‾ (...)_k
    (2π)^D  /              (2π)^D  ‾‾  Nk V0
                                   k

    """

    def get_kpoint_domain(self, generator):
        # Generate k-point domain in relative coordinates
        K_gK = generator.group_kpoints()
        bzk_kc = np.array([self.gs.kd.bzk_kc[K_K[0]] for
                           K_K in K_gK])
        # Generate k-point weights
        weight_k = np.array([generator.get_kpoint_weight(k_c)
                             for k_c in bzk_kc])
        return bzk_kc, weight_k
