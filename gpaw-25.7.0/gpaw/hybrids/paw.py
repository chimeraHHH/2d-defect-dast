from __future__ import annotations
from typing import TYPE_CHECKING
from typing import NamedTuple, Dict, List

import numpy as np

from gpaw.mpi import broadcast
from gpaw.utilities import (pack_atomic_matrices, unpack_atomic_matrices,
                            unpack_density, unpack_hermitian, packed_index)


class PAWThings(NamedTuple):
    VC_aii: Dict[int, np.ndarray | None]
    VV_aii: Dict[int, np.ndarray]  # distributed
    Delta_aiiL: List[np.ndarray]


def calculate_paw_stuff(wfs, dens) -> List[PAWThings]:
    D_asp = dens.D_asp
    comm = D_asp.partition.comm
    if comm.size != wfs.world.size:
        D_sP = pack_atomic_matrices(D_asp)
        D_sP = broadcast(D_sP if comm.rank == 0 else None, comm=comm)
        D_asp = unpack_atomic_matrices(D_sP, wfs.setups)
        rank_a = np.linspace(0, wfs.world.size, len(wfs.setups),
                             endpoint=False).astype(int)
        D_asp = {a: D_sp for a, D_sp in D_asp.items()
                 if rank_a[a] == wfs.world.rank}

    VV_saii: List[Dict[int, np.ndarray]] = [{} for s in range(dens.nspins)]
    for a, D_sp in D_asp.items():
        data = wfs.setups[a]
        for VV_aii, D_p in zip(VV_saii, D_sp):
            D_ii = unpack_density(D_p) * (dens.nspins / 2)
            VV_ii = pawexxvv(data.M_pp, D_ii)
            VV_aii[a] = VV_ii

    Delta_aiiL = []
    VC_aii: Dict[int, np.ndarray | None] = {}
    for a, data in enumerate(wfs.setups):
        Delta_aiiL.append(data.Delta_iiL)
        if data.X_p is None:
            VC_aii[a] = None
        else:
            VC_aii[a] = unpack_hermitian(data.X_p)

    return [PAWThings(VC_aii, VV_aii, Delta_aiiL)
            for VV_aii in VV_saii]


def python_pawexxvv(M_pp, D_ii):
    """PAW correction for valence-valence EXX energy."""
    ni = len(D_ii)
    V_ii = np.empty((ni, ni))
    for i1 in range(ni):
        for i2 in range(ni):
            V = 0.0
            for i3 in range(ni):
                p13 = packed_index(i1, i3, ni)
                for i4 in range(ni):
                    p24 = packed_index(i2, i4, ni)
                    V += M_pp[p13, p24] * D_ii[i3, i4]
            V_ii[i1, i2] = V
    return V_ii


pawexxvv = python_pawexxvv


if not TYPE_CHECKING:
    try:
        from _gpaw import pawexxvv  # noqa: F811
    except ImportError:
        import warnings
        warnings.warn('Please recompile GPAW binary. Using python '
                      'version of pawexxvv instead of faster c version.')
