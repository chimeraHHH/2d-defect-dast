import numpy as np

from gpaw.blacs import BlacsGrid
from gpaw.blacs import Redistributor


def collect_uX(kd, comm, a_uX, s, k):
    # Comm is a communicator orthogonal to kd.comm (ie, domainband_comm)
    Xshape = a_uX[0].shape
    dtype = a_uX[0].dtype
    kpt_rank, q = kd.get_rank_and_index(k)
    u = q * kd.nspins + s
    if kd.comm.rank == kpt_rank:
        a_X = a_uX[u]
        # Comm master send to the global master
        if comm.rank == 0:
            if kpt_rank == 0:
                # assert world.rank == 0
                return a_X
            else:
                kd.comm.ssend(a_X, 0, 2018)
    elif comm.rank == 0 and kpt_rank != 0:
        # assert world.rank == 0
        a_X = np.empty(Xshape, dtype=dtype)
        kd.comm.receive(a_X, kpt_rank, 2018)
        return a_X


def write_uX(kd, comm, writer, name, a_uX):
    ushape = (kd.nspins, kd.nibzkpts)
    Xshape = a_uX[0].shape
    dtype = a_uX[0].dtype
    writer.add_array(name, ushape + Xshape, dtype=dtype)
    for s in range(kd.nspins):
        for k in range(kd.nibzkpts):
            a_X = collect_uX(kd, comm, a_uX, s, k)
            writer.fill(a_X)


def read_uX(kpt_u, reader, name):
    a_uX = []
    for kpt in kpt_u:
        indices = (kpt.s, kpt.k)
        # TODO: does this read on all the comm ranks in vain?
        a_X = reader.proxy(name, *indices)[:]
        a_uX.append(a_X)
    return a_uX


def distribute_nM(ksl, a_nM):
    if not ksl.using_blacs:
        return a_nM

    dtype = a_nM.dtype
    ksl.nMdescriptor.checkassert(a_nM)
    if ksl.gd.rank != 0:
        a_nM = ksl.nM_unique_descriptor.zeros(dtype=dtype)

    nM2mm = Redistributor(ksl.block_comm, ksl.nM_unique_descriptor,
                          ksl.mmdescriptor)

    a_mm = ksl.mmdescriptor.empty(dtype=dtype)
    nM2mm.redistribute(a_nM, a_mm, ksl.bd.nbands, ksl.nao)
    return a_mm


def collect_MM(ksl, a_mm):
    if not ksl.using_blacs:
        return a_mm

    dtype = a_mm.dtype
    NM = ksl.nao
    grid = BlacsGrid(ksl.block_comm, 1, 1)
    MM_descriptor = grid.new_descriptor(NM, NM, NM, NM)
    mm2MM = Redistributor(ksl.block_comm,
                          ksl.mmdescriptor,
                          MM_descriptor)

    a_MM = MM_descriptor.empty(dtype=dtype)
    mm2MM.redistribute(a_mm, a_MM)
    return a_MM


def collect_uMM(kd, ksl, a_uMM, s, k):
    return collect_wuMM(kd, ksl, [a_uMM], 0, s, k)


def collect_wuMM(kd, ksl, a_wuMM, w, s, k):
    # This function is based on
    # gpaw/wavefunctions/base.py: WaveFunctions.collect_auxiliary()

    dtype = a_wuMM[0][0].dtype
    NM = ksl.nao
    kpt_rank, q = kd.get_rank_and_index(k)
    u = q * kd.nspins + s
    if kd.comm.rank == kpt_rank:
        a_MM = a_wuMM[w][u]

        # Collect within blacs grid
        a_MM = collect_MM(ksl, a_MM)

        # KSL master send a_MM to the global master
        if ksl.block_comm.rank == 0:
            if kpt_rank == 0:
                assert ksl.world.rank == 0
                # I have it already
                return a_MM
            else:
                kd.comm.send(a_MM, 0, 2017)
                return None
    elif ksl.block_comm.rank == 0 and kpt_rank != 0:
        assert ksl.world.rank == 0
        a_MM = np.empty((NM, NM), dtype=dtype)
        kd.comm.receive(a_MM, kpt_rank, 2017)
        return a_MM


def distribute_MM(ksl, a_MM):
    if not ksl.using_blacs:
        return a_MM

    dtype = a_MM.dtype
    NM = ksl.nao
    grid = BlacsGrid(ksl.block_comm, 1, 1)
    MM_descriptor = grid.new_descriptor(NM, NM, NM, NM)
    MM2mm = Redistributor(ksl.block_comm,
                          MM_descriptor,
                          ksl.mmdescriptor)
    if ksl.block_comm.rank != 0:
        a_MM = MM_descriptor.empty(dtype=dtype)

    a_mm = ksl.mmdescriptor.empty(dtype=dtype)
    MM2mm.redistribute(a_MM, a_mm)
    return a_mm


def write_uMM(kd, ksl, writer, name, a_uMM):
    return write_wuMM(kd, ksl, writer, name, [a_uMM], wlist=[0])


def write_wuMM(kd, ksl, writer, name, a_wuMM, wlist):
    NM = ksl.nao
    dtype = a_wuMM[0][0].dtype
    writer.add_array(name,
                     (len(wlist), kd.nspins, kd.nibzkpts, NM, NM),
                     dtype=dtype)
    for w in wlist:
        for s in range(kd.nspins):
            for k in range(kd.nibzkpts):
                a_MM = collect_wuMM(kd, ksl, a_wuMM, w, s, k)
                writer.fill(a_MM)


def read_uMM(kpt_u, ksl, reader, name):
    return read_wuMM(kpt_u, ksl, reader, name, wlist=[0])[0]


def read_wuMM(kpt_u, ksl, reader, name, wlist):
    a_wuMM = []
    for w in wlist:
        a_uMM = []
        for kpt in kpt_u:
            indices = (w, kpt.s, kpt.k)
            # TODO: does this read on all the ksl ranks in vain?
            a_MM = reader.proxy(name, *indices)[:]
            a_MM = distribute_MM(ksl, a_MM)
            a_uMM.append(a_MM)
        a_wuMM.append(a_uMM)
    return a_wuMM
