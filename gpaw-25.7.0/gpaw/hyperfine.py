"""Hyperfine parameters.

See:

    First-principles calculations of defects in oxygen-deficient
    silica exposed to hydrogen

    Peter E. Blöchl

    Phys. Rev. B 62, 6158 – Published 1 September 2000

    https://doi.org/10.1103/PhysRevB.62.6158

"""
import argparse
from math import pi
from typing import List

import ase.units as units
import numpy as np
from scipy.integrate import simpson

from gpaw import get_scipy_version
from gpaw.atom.aeatom import Channel
from gpaw.atom.configurations import configurations
from gpaw.atom.radialgd import RadialGridDescriptor
from gpaw.calculator import GPAW
from gpaw.gaunt import gaunt
from gpaw.grid_descriptor import GridDescriptor
from gpaw.pw.descriptor import PWDescriptor
from gpaw.setup import Setup
from gpaw.typing import Array1D, Array2D, Array3D
from gpaw.utilities import unpack_density
from gpaw.xc.functional import XCFunctional

# Fine-structure constant: (~1/137)
alpha = 0.5 * units._mu0 * units._c * units._e**2 / units._hplanck

G_FACTOR_E = 2.00231930436256


def hyperfine_parameters(calc: GPAW,
                         exclude_core=False) -> Array3D:
    r"""Calculate isotropic and anisotropic hyperfine coupling parameters.

    One tensor (:math:`A_{ij}`) per atom is returned in eV units.
    In Hartree atomic units, we have the isotropic part
    :math:`a = \text{Tr}(\mathbf{A}) / 3`:

    .. math::

        a = \frac{2 \alpha^2 g_e m_e}{3 m_p}
            \int \delta_T(\mathbf{r}) \rho_s(\mathbf{r}) d\mathbf{r},

    and the anisotropic part :math:`\mathbf{A} - a`:

    .. math::

        \frac{\alpha^2 g_e m_e}{4 \pi m_p}
        \int \frac{3 r_i r_j - \delta_{ij} r^2}{r^5}
        \rho_s(\mathbf{r}) d\mathbf{r}.

    Remember to multiply each tensor by the g-factors of the nuclei.

    Use ``exclude_core=True`` to exclude contribution from "frozen" core.
    """
    dens = calc.density
    nt_sR = dens.nt_sG
    A_avv = smooth_part(
        nt_sR[0] - nt_sR[1],
        dens.gd,
        calc.atoms.get_scaled_positions())

    D_asp = calc.density.D_asp
    for a, D_sp in D_asp.items():
        density_sii = unpack_density(D_sp)
        setup = calc.wfs.setups[a]

        A_vv = paw_correction(density_sii,
                              setup,
                              calc.hamiltonian.xc,
                              exclude_core)

        A_avv[a] += A_vv

    A_avv *= pi * alpha**2 * G_FACTOR_E * units._me / units._mp * units.Ha

    return A_avv


def smooth_part(spin_density_R: Array3D,
                gd: GridDescriptor,
                spos_ac: Array2D,
                ecut: float = None) -> Array3D:
    """Contribution from pseudo spin-density."""
    pd = PWDescriptor(ecut, gd)
    spin_density_G = pd.fft(spin_density_R)
    G_Gv = pd.get_reciprocal_vectors(add_q=False)
    # eiGR_aG = np.exp(-1j * spos_ac.dot(gd.cell_cv).dot(G_Gv.T))
    eiGR_aG = np.exp(-1j * spos_ac @ gd.cell_cv @ G_Gv.T)

    # Isotropic term:
    W1_a = pd.integrate(spin_density_G, eiGR_aG) / gd.dv * (2 / 3)

    spin_density_G[0] = 0.0
    G2_G = pd.G2_qG[0].copy()
    G2_G[0] = 1.0
    spin_density_G /= G2_G

    # Anisotropic term:
    W_vva = np.empty((3, 3, len(spos_ac)))
    for v1 in range(3):
        for v2 in range(3):
            W_a = pd.integrate(G_Gv[:, v1] * G_Gv[:, v2] * spin_density_G,
                               eiGR_aG)
            W_vva[v1, v2] = -W_a / gd.dv

    W_a = np.trace(W_vva) / 3
    for v in range(3):
        W_vva[v, v] -= W_a
        W_vva[v, v] += W1_a

    return W_vva.transpose((2, 0, 1))


# Normalization constants for xy, yz, 3z^2-r^2, xz, x^2-y^2:
Y2_m = (np.array([15 / 4,
                  15 / 4,
                  5 / 16,
                  15 / 4,
                  15 / 16]) / pi)**0.5
# Second derivatives:
Y2_mvv = np.array([[[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0]],
                   [[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0]],
                   [[-2, 0, 0],
                    [0, -2, 0],
                    [0, 0, 4]],
                   [[0, 0, 1],
                    [0, 0, 0],
                    [1, 0, 0]],
                   [[2, 0, 0],
                    [0, -2, 0],
                    [0, 0, 0]]])


def paw_correction(density_sii: Array3D,
                   setup: Setup,
                   xc: XCFunctional = None,
                   exclude_core: bool = False) -> Array2D:
    """Corrections from 1-center expansions of spin-density."""
    # Spherical part:
    spin_density_ii = density_sii[0] - density_sii[1]
    D0_jj = expand(spin_density_ii, setup.l_j, l=0)[0]

    phit_jg = np.array(setup.data.phit_jg)
    phi_jg = np.array(setup.data.phi_jg)

    rgd = setup.rgd

    # Spin-density from pseudo density:
    nt0 = phit_jg[:, 0].dot(D0_jj).dot(phit_jg[:, 0]) / (4 * pi)**0.5

    # All-electron contribution diverges as r^-beta and must be integrated
    # over a small region of size rT:
    n0_g = np.einsum('ab, ag, bg -> g', D0_jj, phi_jg, phi_jg) / (4 * pi)**0.5
    if not exclude_core and setup.Nc > 0 and xc is not None:
        n0_g += core_contribution(density_sii, setup, xc)

    beta = 2 * (1 - (1 - (setup.Z * alpha)**2)**0.5)
    rT = setup.Z * alpha**2
    n0 = integrate(n0_g, rgd, rT, beta)

    W1 = (n0 - nt0) * 2 / 3  # isotropic result

    # Now the anisotropic part from the l=2 part of the density:
    D2_mjj = expand(spin_density_ii, setup.l_j, 2)
    dn2_mg = np.einsum('mab, ag, bg -> mg', D2_mjj, phi_jg, phi_jg)
    dn2_mg -= np.einsum('mab, ag, bg -> mg', D2_mjj, phit_jg, phit_jg)
    A_m = dn2_mg[:, 1:].dot(rgd.dr_g[1:] / rgd.r_g[1:]) / 5
    A_m *= Y2_m
    # W_vv: Array2D = Y2_mvv.T.dot(A_m)  # type: ignore
    W_vv = Y2_mvv.T @ A_m
    W = np.trace(W_vv) / 3
    for v in range(3):
        W_vv[v, v] -= W
        W_vv[v, v] += W1

    return W_vv


def expand(D_ii: Array2D,
           l_j: List[int],
           l: int) -> Array3D:
    """Get expansion coefficients."""
    G_LLm = gaunt(lmax=2)[:, :, l**2:(l + 1)**2]
    D_mjj = np.empty((2 * l + 1, len(l_j), len(l_j)))
    i1a = 0
    for j1, l1 in enumerate(l_j):
        i1b = i1a + 2 * l1 + 1
        i2a = 0
        for j2, l2 in enumerate(l_j):
            i2b = i2a + 2 * l2 + 1
            D_mjj[:, j1, j2] = np.einsum('ab, abm -> m',
                                         D_ii[i1a:i1b, i2a:i2b],
                                         G_LLm[l1**2:(l1 + 1)**2,
                                               l2**2:(l2 + 1)**2])
            i2a = i2b
        i1a = i1b
    return D_mjj


def delta(r: Array1D, rT: float) -> Array1D:
    """Extended delta function."""
    return 2 / rT / (1 + 2 * r / rT)**2


def integrate(n0_g: Array1D,
              rgd: RadialGridDescriptor,
              rT: float,
              beta: float) -> float:
    """Take care of r^-beta divergence."""
    r_g = rgd.r_g
    a_i = np.polyfit(r_g[1:5], n0_g[1:5] * r_g[1:5]**beta, 3)
    r4 = r_g[4]
    dr = rT / 50
    n = max(int(r4 / dr / 2) * 2 + 1, 3)
    r_j = np.linspace(0, r4, n)

    b_i = np.arange(3, -1, -1) + 1 - beta
    d0 = 2 / rT  # delta(0, rT)
    d1 = -8 / rT**2
    n0 = a_i.dot(d0 * r4**b_i / b_i + d1 * r4**(b_i + 1) / (b_i + 1))

    d_j = delta(r_j, rT) - (d0 + d1 * r_j)

    head_j = d_j * np.polyval(a_i, r_j)
    head_j[1:] *= r_j[1:]**-beta
    n0 += simpson(head_j, x=r_j)

    tail_g = n0_g[4:] * delta(r_g[4:], rT)
    if get_scipy_version() >= [1, 11]:
        n0 += simpson(tail_g, x=r_g[4:])
    else:
        n0 += simpson(tail_g, x=r_g[4:], even='first')

    return n0


def core_contribution(density_sii: Array3D,
                      setup: Setup,
                      xc: XCFunctional) -> Array1D:
    """Calculate spin-density from "frozen" core."""
    # Spherical part:
    D_sjj = [expand(density_ii, setup.l_j, 0)[0]
             for density_ii in density_sii]
    phi_jg = np.array(setup.data.phi_jg)
    rgd = setup.rgd

    # Densities with frozen core:
    n_sg = np.einsum('ag, sab, bg -> sg',
                     phi_jg, D_sjj, phi_jg) / (4 * pi)**0.5
    n_sg += setup.data.nc_g * (0.5 / (4 * pi)**0.5)

    # Potential:
    v_sg = np.zeros_like(n_sg)
    xc.calculate_spherical(rgd, n_sg, v_sg)
    vr_sg = v_sg * rgd.r_g
    vr_sg -= setup.Z
    vr_sg += rgd.poisson(n_sg.sum(axis=0))

    # Find first bound s-state included in PAW potential:
    for n, l in zip(setup.n_j, setup.l_j):
        if l == 0:
            assert n > 0
            n0 = n
            break
    else:
        assert False, (setup.n_j, setup.l_j)

    # Initial guesses for core s-states:
    eigs = [e for n, l, f, e in configurations[setup.symbol][1]
            if n < n0 and l == 0]

    # Solve spherical scalar-relativistic Schrödinger equation:
    core_spin_density_g = rgd.zeros()
    sign = 1.0
    for vr_g in vr_sg:
        channel = Channel(l=0, f_n=[1] * len(eigs))
        channel.e_n = eigs
        channel.phi_ng = rgd.empty(len(eigs))
        channel.solve2(vr_g, rgd=rgd, scalar_relativistic=True, Z=setup.Z)
        assert channel.solve2ok
        core_spin_density_g += sign * channel.calculate_density()
        sign = -1.0

    return core_spin_density_g


# From https://en.wikipedia.org/wiki/Gyromagnetic_ratio
# Units: MHz/T
gyromagnetic_ratios = {'H': (1, 42.577478518),
                       'He': (3, -32.434),
                       'Li': (7, 16.546),
                       'C': (13, 10.7084),
                       'N': (14, 3.077),
                       'O': (17, -5.772),
                       'F': (19, 40.052),
                       'Na': (23, 11.262),
                       'Al': (27, 11.103),
                       'Si': (29, -8.465),
                       'P': (31, 17.235),
                       'Fe': (57, 1.382),
                       'Cu': (63, 11.319),
                       'Zn': (67, 2.669),
                       'Xe': (129, -11.777)}


def main(argv: List[str] = None) -> None:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        prog='python3 -m gpaw.hyperfine',
        description='Calculate hyperfine parameters.')
    add = parser.add_argument
    add('file', metavar='input-file',
        help='GPW-file (with or without wave functions).')
    add('-g', '--g-factors',
        help='G-factors.  Example: "-g H:5.6,O:-0.76".')
    add('-u', '--units', default='ueV', choices=['ueV', 'MHz'],
        help='Units.  Must be "ueV" (micro-eV, default) or "MHz".')
    add('-x', '--exclude-core', action='store_true')
    add('-d', '--diagonalize', action='store_true',
        help='Show eigenvalues of tensor.')

    args = parser.parse_intermixed_args(argv)

    calc = GPAW(args.file)
    atoms = calc.get_atoms()

    symbols = atoms.symbols
    magmoms = atoms.get_magnetic_moments()
    total_magmom = atoms.get_magnetic_moment()

    g_factors = {symbol: ratio * 1e6 * 4 * pi * units._mp / units._e
                 for symbol, (n, ratio) in gyromagnetic_ratios.items()}

    if args.g_factors:
        for symbol, value in (part.split(':')
                              for part in args.g_factors.split(',')):
            g_factors[symbol] = float(value)

    if args.units == 'ueV':
        scale = 1e6
        unit = 'μeV'
    else:
        scale = units._e / units._hplanck * 1e-6
        unit = 'MHz'

    A_avv = hyperfine_parameters(calc, args.exclude_core)

    print('Hyperfine coupling paramters '
          f'in {unit}:\n')

    if args.diagonalize:
        columns = ['1.', '2.', '3.']
    else:
        columns = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']
    print('  atom  magmom      ', '       '.join(columns))

    used = {}
    for a, A_vv in enumerate(A_avv):
        symbol = symbols[a]
        magmom = magmoms[a]
        g_factor = g_factors.get(symbol, 1.0)
        used[symbol] = g_factor
        A_vv *= g_factor * scale
        if args.diagonalize:
            numbers = np.linalg.eigvalsh(A_vv).tolist()
        else:
            numbers = [A_vv[0, 0], A_vv[1, 1], A_vv[2, 2],
                       A_vv[1, 2], A_vv[0, 2], A_vv[0, 1]]
        print(f'{a:3} {symbol:>2}  {magmom:6.3f}',
              ''.join(f'{x:9.2f}' for x in numbers))

    print('\nCore correction',
          'NOT included!' if args.exclude_core else 'included')
    print(f'Total magnetic moment: {total_magmom:.3f}')

    print('\nG-factors used:')
    for symbol, g in used.items():
        print(f'{symbol:2} {g:10.3f}')


if __name__ == '__main__':
    main()
