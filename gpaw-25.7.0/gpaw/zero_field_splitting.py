"""Zero-field splitting.

See::

    Spin decontamination for magnetic dipolar coupling calculations:
    Application to high-spin molecules and solid-state spin qubits

    Timur Biktagirov, Wolf Gero Schmidt, and Uwe Gerstmann

    Phys. Rev. Research 2, 022024(R) – Published 30 April 2020

"""
from math import pi
from typing import List, Tuple, Dict

import numpy as np
from ase.units import Bohr, Ha, _c, _e, _hplanck

from gpaw.calculator import GPAW
from gpaw.grid_descriptor import GridDescriptor
from gpaw.typing import Array1D, Array2D, Array4D
from gpaw.hyperfine import alpha  # fine-structure constant: ~ 1 / 137
from gpaw.setup import Setup
from gpaw.pw.lfc import PWLFC
from gpaw.pw.descriptor import PWDescriptor
from gpaw.mpi import serial_comm


def zfs(calc: GPAW,
        method: int = 1) -> Array2D:
    """Zero-field splitting.

    Calculate magnetic dipole coupling tensor in eV.
    """
    (kpt1, kpt2), = calc.wfs.kpt_qs  # spin-polarized and gamma only

    nocc1 = (kpt1.f_n > 0.5).sum()
    nocc2 = (kpt2.f_n > 0.5).sum()

    assert nocc1 == nocc2 + 2, (nocc1, nocc2)

    if method == 1:
        wf1 = WaveFunctions.from_calc(calc, 0, nocc1 - 2, nocc1)
        wf12 = [wf1]
    else:
        wf1 = WaveFunctions.from_calc(calc, 0, 0, nocc1)
        wf2 = WaveFunctions.from_calc(calc, 1, 0, nocc2)
        wf12 = [wf1, wf2]

    D_vv = np.zeros((3, 3))

    if calc.world.rank == 0:
        compensation_charge = create_compensation_charge(wf1.setups,
                                                         wf1.pd,
                                                         calc.spos_ac)
        for wfa in wf12:
            for wfb in wf12:
                d_vv = zfs1(wfa, wfb, compensation_charge)
                D_vv += d_vv

    calc.world.broadcast(D_vv, 0)

    return D_vv


class WaveFunctions:
    def __init__(self,
                 psit_nR: Array4D,
                 P_ani: Dict[int, Array2D],
                 spin: int,
                 setups: List[Setup],
                 gd: GridDescriptor = None,
                 pd: PWDescriptor = None):
        """Container for wave function in real-space and projections."""

        self.pd = pd or PWDescriptor(ecut=None, gd=gd)
        self.psit_nR = psit_nR
        self.P_ani = P_ani
        self.spin = spin
        self.setups = setups

    @staticmethod
    def from_calc(calc: GPAW, spin: int, n1: int, n2: int) -> 'WaveFunctions':
        """Create WaveFunctions object GPAW calculation."""
        kpt = calc.wfs.kpt_qs[0][spin]
        gd = calc.wfs.gd.new_descriptor(pbc_c=np.ones(3, bool),
                                        comm=serial_comm)
        psit_nR = gd.empty(n2 - n1)
        for band, psit_R in enumerate(psit_nR):
            psit_R[:] = calc.get_pseudo_wave_function(
                band + n1,
                spin=spin) * Bohr**1.5

        return WaveFunctions(psit_nR,
                             kpt.projections.as_dict_on_master(n1, n2),
                             spin,
                             calc.setups,
                             gd=gd)

    def __len__(self) -> int:
        return len(self.psit_nR)


def create_compensation_charge(setups: List[Setup],
                               pd: PWDescriptor,
                               spos_ac: Array2D) -> PWLFC:
    compensation_charge = PWLFC([data.ghat_l for data in setups], pd)
    compensation_charge.set_positions(spos_ac)
    return compensation_charge


def zfs1(wf1: WaveFunctions,
         wf2: WaveFunctions,
         compensation_charge: PWLFC) -> Array2D:
    """Compute dipole coupling."""
    pd = wf1.pd
    setups = wf1.setups
    N2 = len(wf2)

    G_G = pd.G2_qG[0]**0.5
    G_G[0] = 1.0
    G_Gv = pd.get_reciprocal_vectors(add_q=False) / G_G[:, np.newaxis]

    n_sG = pd.zeros(2)
    for n_G, wf in zip(n_sG, [wf1, wf2]):
        D_aii = {}
        Q_aL = {}
        for a, P_ni in wf.P_ani.items():
            D_ii = np.einsum('ni, nj -> ij', P_ni, P_ni)
            D_aii[a] = D_ii
            Q_aL[a] = np.einsum('ij, ijL -> L', D_ii, setups[a].Delta_iiL)

        for psit_R in wf.psit_nR:
            n_G += pd.fft(psit_R**2)

        compensation_charge.add(n_G, Q_aL)

    nn_G = (n_sG[0] * n_sG[1].conj()).real
    D_vv = zfs2(pd, G_Gv, nn_G)

    n_nG = pd.empty(N2)
    for n1, psit1_R in enumerate(wf1.psit_nR):
        D_anii = {}
        Q_anL = {}
        for a, P1_ni in wf1.P_ani.items():
            D_nii = np.einsum('i, nj -> nij', P1_ni[n1], wf2.P_ani[a])
            D_anii[a] = D_nii
            Q_anL[a] = np.einsum('nij, ijL -> nL',
                                 D_nii, setups[a].Delta_iiL)

        for n_G, psit2_R in zip(n_nG, wf2.psit_nR):
            n_G[:] = pd.fft(psit1_R * psit2_R)

        compensation_charge.add(n_nG, Q_anL)

        nn_G = (n_nG * n_nG.conj()).sum(axis=0).real
        D_vv -= zfs2(pd, G_Gv, nn_G)

    D_vv -= np.trace(D_vv) / 3 * np.eye(3)  # should be traceless

    sign = 1.0 if wf1.spin == wf2.spin else -1.0

    return sign * alpha**2 * pi * D_vv * Ha


def zfs2(pd: PWDescriptor,
         G_Gv: Array2D,
         nn_G: Array1D) -> Array2D:
    """Integral."""
    D_vv = np.einsum('gv, gw, g -> vw', G_Gv, G_Gv, nn_G)
    D_vv *= 2 * pd.gd.dv / pd.gd.N_c.prod()
    return D_vv


def convert_tensor(D_vv: Array2D,
                   unit: str = 'eV') -> Tuple[float, float, Array1D, Array2D]:
    """Convert 3x3 tensor to D, E and easy axis.

    Input tensor must be in eV and the result can be returned in
    eV, μeV, MHz or 1/cm according to the value of *unit*
    (must be one of "eV", "ueV", "MHz", "1/cm").

    >>> D_vv = np.diag([1, 2, 3])
    >>> D, E, axis, _ = convert_tensor(D_vv)
    >>> D
    4.5
    >>> E
    0.5
    >>> axis
    array([0., 0., 1.])
    """
    if unit == 'ueV':
        scale = 1e6
    elif unit == 'MHz':
        scale = _e / _hplanck * 1e-6
    elif unit == '1/cm':
        scale = _e / _hplanck / _c / 100
    elif unit == 'eV':
        scale = 1.0
    else:
        raise ValueError(f'Unknown unit: {unit}')

    (e1, e2, e3), U = np.linalg.eigh(D_vv * scale)

    if abs(e1) > abs(e3):
        D = 1.5 * e1
        E = 0.5 * (e2 - e3)
        axis = U[:, 0]
    else:
        D = 1.5 * e3
        E = 0.5 * (e2 - e1)
        axis = U[:, 2]

    return float(D), float(E), axis, D_vv * scale


def main(argv: List[str] = None) -> Array2D:
    """CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        prog='python3 -m gpaw.zero_field_splitting',
        description='...')
    add = parser.add_argument
    add('file', metavar='input-file',
        help='GPW-file with wave functions.')
    add('-u', '--unit', default='ueV', choices=['ueV', 'MHz', '1/cm'],
        help='Unit.  Must be "ueV" (micro-eV, default), "MHz" or "1/cm".')
    add('-m', '--method', type=int, default=1)

    args = parser.parse_intermixed_args(argv)

    calc = GPAW(args.file)

    D_vv = zfs(calc, args.method)
    D, E, axis, D_vv = convert_tensor(D_vv, args.unit)

    unit = args.unit
    if unit == 'ueV':
        unit = 'μeV'

    print('D_ij = (' +
          ',\n        '.join('(' + ', '.join(f'{d:10.3f}' for d in D_v) + ')'
                             for D_v in D_vv) +
          ') ', unit)
    print('i, j = x, y, z')
    print()
    print(f'D = {D:.3f} {unit}')
    print(f'E = {E:.3f} {unit}')
    x, y, z = axis
    print(f'axis = ({x:.3f}, {y:.3f}, {z:.3f})')

    return D_vv


if __name__ == '__main__':
    main()
