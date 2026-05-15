import numpy as np

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.projections import Projections
from gpaw.utilities.partition import AtomPartition
from gpaw.wavefunctions.arrays import PlaneWaveExpansionWaveFunctions
from gpaw.pw.descriptor import PWDescriptor


class KPoint:
    def __init__(self,
                 proj,  # projections
                 f_n,  # occupations numbers between 0 and 1
                 k_c,  # k-vector in units of reciprocal cell
                 weight,  # weight of k-point
                 dPdR_aniv=[]):
        self.proj = proj
        self.f_n = f_n
        self.k_c = k_c
        self.weight = weight
        self.dPdR_aniv = dPdR_aniv


class PWKPoint(KPoint):
    def __init__(self, psit, *args):  # plane-wave expansion of wfs
        self.psit = psit
        KPoint.__init__(self, *args)


class RSKPoint(KPoint):
    def __init__(self, u_nR, *args):
        self.u_nR = u_nR
        KPoint.__init__(self, *args)


def to_real_space(psit, na=0, nb=None):
    pd = psit.pd
    comm = pd.comm
    S = comm.size
    q = psit.kpt
    nbands = len(psit.array)
    nb = nb or nbands
    u_nR = pd.gd.empty(nbands, pd.dtype, global_array=True)
    for n1 in range(0, nbands, S):
        n2 = min(n1 + S, nbands)
        u_G = pd.alltoall1(psit.array[n1:n2], q)
        if u_G is not None:
            n = n1 + comm.rank
            u_nR[n] = pd.ifft(u_G, local=True, safe=False, q=q)
        for n in range(n1, n2):
            comm.broadcast(u_nR[n], n - n1)

    return u_nR[na:nb]


def get_kpt(wfs, k, spin, n1, n2):
    k_c = wfs.kd.ibzk_kc[k]
    weight = wfs.kd.weight_k[k]

    if wfs.world.size == wfs.gd.comm.size:
        # Easy:
        kpt = wfs.kpt_qs[k][spin]
        psit = kpt.psit.view(n1, n2)
        proj = kpt.projections.view(n1, n2)
        f_n = kpt.f_n[n1:n2]
    else:
        # Need to redistribute things:
        gd = wfs.gd.new_descriptor(comm=wfs.world)
        kd = KPointDescriptor([k_c])
        pd = PWDescriptor(wfs.ecut, gd, wfs.dtype, kd, wfs.fftwflags)
        psit = PlaneWaveExpansionWaveFunctions(n2 - n1,
                                               pd,
                                               dtype=wfs.dtype,
                                               spin=spin)
        for n in range(n1, n2):
            psit_G = wfs.get_wave_function_array(n, k, spin,
                                                 realspace=False, cut=False)
            if isinstance(psit_G, float):
                psit_G = None
            else:
                psit_G = psit_G[:pd.ngmax]
            psit._distribute(psit_G, psit.array[n - n1])

        P_nI = wfs.collect_projections(k, spin)
        if wfs.world.rank == 0:
            P_nI = P_nI[n1:n2]
        natoms = len(wfs.setups)
        rank_a = np.zeros(natoms, int)
        atom_partition = AtomPartition(wfs.world, rank_a)
        nproj_a = [setup.ni for setup in wfs.setups]
        proj = Projections(n2 - n1,
                           nproj_a,
                           atom_partition,
                           spin=spin,
                           dtype=wfs.dtype,
                           data=P_nI)

        rank_a = np.linspace(0, wfs.world.size, len(wfs.setups),
                             endpoint=False).astype(int)
        atom_partition = AtomPartition(wfs.world, rank_a)
        proj = proj.redist(atom_partition)

        f_n = wfs.collect_occupations(k, spin)
        if wfs.world.rank != 0:
            f_n = np.empty(n2 - n1)
        else:
            f_n = f_n[n1:n2]
        wfs.world.broadcast(f_n, 0)

    f_n = f_n / (weight * (2 // wfs.nspins))  # scale to [0, 1]

    return PWKPoint(psit, proj, f_n, k_c, weight)
