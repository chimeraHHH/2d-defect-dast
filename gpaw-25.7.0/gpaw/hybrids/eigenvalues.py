"""Calculate non self-consistent eigenvalues for hybrid functionals."""
from __future__ import annotations
import functools
import json
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
from ase.units import Ha
from gpaw.calculator import GPAW as GPAWOld
from gpaw import GPAW
from gpaw.new.ase_interface import ASECalculator
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.mpi import serial_comm
from gpaw.pw.descriptor import PWDescriptor
from gpaw.pw.lfc import PWLFC
from gpaw.typing import Array3D
from gpaw.xc import XC
from gpaw.xc.kernel import XCNull
from gpaw.xc.tools import vxc

from gpaw.hybrids import parse_name
from gpaw.hybrids.coulomb import coulomb_interaction
from gpaw.hybrids.kpts import RSKPoint, get_kpt, to_real_space
from gpaw.hybrids.paw import calculate_paw_stuff
from gpaw.hybrids.symmetry import Symmetry


def non_self_consistent_eigenvalues(
        calc: Union[GPAWOld, ASECalculator, str, Path],
        xcname: str,
        n1: int = 0,
        n2: int = 0,
        kpt_indices: List[int] = None,
        snapshot: Union[str, Path] = None,
        ftol: float = 1e-9) -> tuple[Array3D,
                                     Array3D,
                                     Array3D]:
    """Calculate non self-consistent eigenvalues for a hybrid functional.

    Based on a self-consistent DFT calculation (calc).  Only eigenvalues n1 to
    n2 - 1 for the IBZ indices in kpt_indices are calculated
    (default is all bands and all k-points). EXX integrals involving
    states with occupation numbers less than ftol are skipped.  Use
    snapshot='name.json' to get snapshots for each k-point finished.

    Returns three (nspins, nkpts, n2 - n1)-shaped ndarrays
    with contributions to the eigenvalues in eV:

    >>> nsceigs = non_self_consistent_eigenvalues
    >>> eig_dft, vxc_dft, vxc_hyb = nsceigs('<gpw-file>', xcname='PBE0')
    >>> eig_hyb = eig_dft - vxc_dft + vxc_hyb
    """

    if not isinstance(calc, (GPAWOld, ASECalculator)):
        if calc == '<gpw-file>':  # for doctest
            return (np.zeros((1, 1, 1)),
                    np.zeros((1, 1, 1)),
                    np.zeros((1, 1, 1)))
        calc = GPAW(Path(calc), txt=None, parallel={'band': 1, 'kpt': 1})

    assert isinstance(calc, (GPAWOld, ASECalculator))
    wfs = calc.wfs

    if n2 <= 0:
        n2 += wfs.bd.nbands

    if kpt_indices is None:
        kpt_indices = np.arange(wfs.kd.nibzkpts).tolist()

    path = Path(snapshot) if snapshot is not None else None

    e_dft_sin = np.zeros(0)
    v_dft_sin = np.zeros(0)

    # sl=semilocal, nl=nonlocal
    v_hyb_sl_sin = np.zeros(0)
    v_hyb_nl_sin: Optional[List[List[np.ndarray]]] = None

    if path:
        e_dft_sin, v_dft_sin, v_hyb_sl_sin, v_hyb_nl_sin = read_snapshot(path)

    xcname, exx_fraction, omega, yukawa = parse_name(xcname)

    if v_dft_sin.size == 0:
        xc = XC(xcname)
        e_dft_sin, v_dft_sin, v_hyb_sl_sin = _semi_local(
            calc, xc, n1, n2, kpt_indices)
        write_snapshot(e_dft_sin, v_dft_sin, v_hyb_sl_sin, v_hyb_nl_sin,
                       path, wfs.world)
    # Non-local hybrid contribution
    if v_hyb_nl_sin is None:
        v_hyb_nl_sin = [[] for s in range(wfs.nspins)]

    # Find missing indices:
    kpt_indices_s = [kpt_indices[len(v_hyb_nl_in):]
                     for v_hyb_nl_in in v_hyb_nl_sin]

    if any(len(kpt_indices) > 0 for kpt_indices in kpt_indices_s):
        for s, v_hyb_nl_n in _non_local(calc, n1, n2, kpt_indices_s,
                                        ftol, omega, yukawa):
            v_hyb_nl_sin[s].append(v_hyb_nl_n * exx_fraction)
            write_snapshot(e_dft_sin, v_dft_sin, v_hyb_sl_sin, v_hyb_nl_sin,
                           path, wfs.world)

    return (e_dft_sin * Ha,
            v_dft_sin * Ha,
            (v_hyb_sl_sin + v_hyb_nl_sin) * Ha)


def _semi_local(calc: GPAWOld | ASECalculator,
                xc,
                n1: int,
                n2: int,
                kpt_indices: List[int]
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wfs = calc.wfs
    nspins = wfs.nspins
    e_dft_sin = np.array([[calc.get_eigenvalues(k, spin)[n1:n2]
                           for k in kpt_indices]
                          for spin in range(nspins)])
    v_dft_sin = vxc(calc.gs_adapter(), n1=n1, n2=n2)[:, kpt_indices]
    if isinstance(xc.kernel, XCNull):
        v_hyb_sl_sin = np.zeros_like(v_dft_sin)
    else:
        v_hyb_sl_sin = vxc(calc.gs_adapter(), xc, n1=n1, n2=n2)[:, kpt_indices]
    return e_dft_sin / Ha, v_dft_sin / Ha, v_hyb_sl_sin / Ha


def _non_local(calc: GPAWOld | ASECalculator,
               n1: int,
               n2: int,
               kpt_indices_s: List[List[int]],
               ftol: float,
               omega: float,
               yukawa: bool) -> Generator[Tuple[int, np.ndarray], None, None]:
    wfs = calc.wfs
    kd = wfs.kd
    dens = calc.density

    nocc = max(((kpt.f_n / kpt.weight) > ftol).sum()
               for kpt in wfs.kpt_u)
    nocc = kd.comm.max_scalar(wfs.bd.comm.sum_scalar(int(nocc)))

    coulomb = coulomb_interaction(omega, wfs.gd, kd, yukawa=yukawa)
    sym = Symmetry(kd)

    paw_s = calculate_paw_stuff(wfs, dens)

    for spin, kpt_indices in enumerate(kpt_indices_s):
        if len(kpt_indices) == 0:
            continue
        kpts2 = [get_kpt(wfs, k, spin, 0, nocc) for k in range(kd.nibzkpts)]
        for i in kpt_indices:
            kpt1 = get_kpt(wfs, i, spin, n1, n2)
            v_n = _calculate_eigenvalues(
                kpt1, kpts2, paw_s[spin], kd, coulomb, sym, wfs, calc.spos_ac)
            wfs.world.sum(v_n)
            yield spin, v_n


def _calculate_eigenvalues(kpt1, kpts2, paw, kd, coulomb, sym, wfs, spos_ac):
    pd = kpt1.psit.pd
    gd = pd.gd.new_descriptor(comm=serial_comm)
    comm = wfs.world
    size = comm.size
    rank = comm.rank

    nsym = len(kd.symmetry.op_scc)
    assert len(kpts2) == kd.nibzkpts

    N1 = len(kpt1.psit.array)
    N2 = len(kpts2[0].psit.array)

    size1, size2 = layout(N1, N2, size)
    assert size1 * size2 == size
    B1 = (N1 + size1 - 1) // size1
    B2 = (N2 + size2 - 1) // size2
    rank1, rank2 = divmod(rank, size2)
    n1a = min(B1 * rank1, N1)
    n1b = min(n1a + B1, N1)
    n2a = min(B2 * rank2, N2)
    n2b = min(n2a + B2, N2)

    u1_nR = to_real_space(kpt1.psit, n1a, n1b)
    proj1all = kpt1.proj.broadcast()
    proj1 = proj1all.view(n1a, n1b)

    E_n = np.zeros(N1)
    e_n = E_n[n1a:n1b]
    e_nn = np.empty((n1b - n1a, n2b - n2a))

    for i2, kpt2 in enumerate(kpts2):
        u2_nR = to_real_space(kpt2.psit, n2a, n2b)
        rskpt0 = RSKPoint(u2_nR,
                          kpt2.proj.broadcast().view(n2a, n2b),
                          kpt2.f_n[n2a:n2b],
                          kpt2.k_c,
                          kpt2.weight)
        for k, i in enumerate(kd.bz2ibz_k):
            if i != i2:
                continue
            s = kd.sym_k[k] + kd.time_reversal_k[k] * nsym
            rskpt2 = sym.apply_symmetry(s, rskpt0, wfs, spos_ac)
            q_c = rskpt2.k_c - kpt1.k_c
            qd = KPointDescriptor([-q_c])
            pd12 = PWDescriptor(pd.ecut, gd, pd.dtype, kd=qd)
            ghat = PWLFC([data.ghat_l for data in wfs.setups], pd12)
            ghat.set_positions(spos_ac)
            v_G = coulomb.get_potential(pd12)
            Q_annL = [np.einsum('mi, ijL, nj -> mnL',
                                proj1[a],
                                Delta_iiL,
                                rskpt2.proj[a].conj())
                      for a, Delta_iiL in enumerate(paw.Delta_aiiL)]
            rho_nG = ghat.pd.empty(n2b - n2a, u1_nR.dtype)

            for n1, u1_R in enumerate(u1_nR):
                for u2_R, rho_G in zip(rskpt2.u_nR, rho_nG):
                    rho_G[:] = ghat.pd.fft(u1_R * u2_R.conj())

                ghat.add(rho_nG,
                         {a: Q_nnL[n1] for a, Q_nnL in enumerate(Q_annL)})

                for n2, rho_G in enumerate(rho_nG):
                    vrho_G = v_G * rho_G
                    e = ghat.pd.integrate(rho_G, vrho_G).real
                    e_nn[n1, n2] = e / kd.nbzkpts
            e_n -= e_nn.dot(rskpt2.f_n)

    for a, VV_ii in paw.VV_aii.items():
        P_ni = proj1all[a]
        vv_n = np.einsum('ni, ij, nj -> n',
                         P_ni.conj(), VV_ii, P_ni).real
        if paw.VC_aii[a] is not None:
            vc_n = np.einsum('ni, ij, nj -> n',
                             P_ni.conj(), paw.VC_aii[a], P_ni).real
        else:
            vc_n = 0.0
        E_n -= (2 * vv_n + vc_n)

    return E_n


def write_snapshot(e_dft_sin: np.ndarray,
                   v_dft_sin: np.ndarray,
                   v_hyb_sl_sin: np.ndarray,
                   v_hyb_nl_sin: Optional[List[List[np.ndarray]]],
                   path: Optional[Path],
                   comm) -> None:
    """Write to json-file what has been calculated so far."""
    if comm.rank == 0 and path:
        dct = {'e_dft_sin': e_dft_sin.tolist(),
               'v_dft_sin': v_dft_sin.tolist(),
               'v_hyb_sl_sin': v_hyb_sl_sin.tolist()}
        if v_hyb_nl_sin is not None:
            dct['v_hyb_nl_sin'] = [[v_n.tolist()
                                    for v_n in v_in]
                                   for v_in in v_hyb_nl_sin]
        path.write_text(json.dumps(dct, indent=0))


def read_snapshot(snapshot: Path
                  ) -> Tuple[np.ndarray,
                             np.ndarray,
                             np.ndarray,
                             Optional[List[List[np.ndarray]]]]:
    """Read from json-file what has already been calculated."""
    if snapshot.is_file():
        dct = json.loads(snapshot.read_text())
        v_hyb_nl_sin = dct.get('v_hyb_nl_sin')
        if v_hyb_nl_sin is not None:
            v_hyb_nl_sin = [[np.array(v_n)
                             for v_n in v_in]
                            for v_in in v_hyb_nl_sin]
        return (np.array(dct['e_dft_sin']),
                np.array(dct['v_dft_sin']),
                np.array(dct['v_hyb_sl_sin']),
                v_hyb_nl_sin)
    return np.array([[[]]]), np.array([[[]]]), np.array([[[]]]), None


@functools.lru_cache()
def layout(n1: int, n2: int, size: int) -> Tuple[int, int]:
    """Distribute n1*n2 matrix over s1*s2=size blocks.

    Returns s1, s2.

    >>> layout(10, 10, 8)
    (4, 2)
    """
    candidates: List[Tuple[float, int, int]] = []
    for s1 in range(1, size + 1):
        s2, r = divmod(size, s1)
        if r > 0:
            continue
        fitness = (1 - idle(n1, s1)) * (1 - idle(n2, s2))
        candidates.append((fitness, s1, s2))
    return max(candidates)[1:]


def idle(n: int, s: int) -> float:
    """Idle fraction (helper function for layout() function)."""
    b = (n + s - 1) // s
    return 1 - n / (b * s)
