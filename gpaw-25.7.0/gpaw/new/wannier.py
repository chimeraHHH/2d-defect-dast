from __future__ import annotations
from math import factorial as fac

import numpy as np
from ase.units import Bohr

from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.spline import Spline
from gpaw.typing import Array2D


def get_wannier_integrals(ibzwfs: IBZWaveFunctions,
                          grid,
                          s: int,
                          k: int,
                          k1: int,
                          G_c,
                          nbands=None) -> Array2D:
    """Calculate integrals for maximally localized Wannier functions."""
    ibzwfs.make_sure_wfs_are_read_from_gpw_file()
    assert s <= ibzwfs.nspins
    # XXX not for the kpoint/spin parallel case
    assert ibzwfs.comm.size == 1
    wfs = ibzwfs.wfs_qs[k][s].to_uniform_grid_wave_functions(grid, None)
    wfs1 = ibzwfs.wfs_qs[k1][s].to_uniform_grid_wave_functions(grid, None)
    # Get pseudo part
    psit_nR = wfs.psit_nX.data
    psit1_nR = wfs1.psit_nX.data
    Z_nn = grid._gd.wannier_matrix(psit_nR, psit1_nR, G_c, nbands)
    # Add corrections
    add_wannier_correction(Z_nn, G_c, wfs, wfs1, nbands)
    grid.comm.sum(Z_nn)
    return Z_nn


def add_wannier_correction(Z_nn, G_c, wfs, wfs1, nbands):
    r"""Calculate the correction to the wannier integrals.

    See: (Eq. 27 ref1)::

                      -i G.r
        Z   = <psi | e      |psi >
         nm       n             m

                       __                __
               ~      \              a  \     a*   a    a
        Z    = Z    +  ) exp[-i G . R ]  )   P   dO    P
         nmx    nmx   /__            x  /__   ni   ii'  mi'

                       a                 ii'

    Note that this correction is an approximation that assumes the
    exponential varies slowly over the extent of the augmentation sphere.

    ref1: Thygesen et al, Phys. Rev. B 72, 125119 (2005)
    """
    P_ani = wfs.P_ani
    P1_ani = wfs1.P_ani
    for a, P_ni in P_ani.items():
        P_ni = P_ani[a][:nbands]
        P1_ni = P1_ani[a][:nbands]
        dO_ii = wfs.setups[a].dO_ii
        e = np.exp(-2.j * np.pi * np.dot(G_c, wfs.relpos_ac[a]))
        Z_nn += e * P_ni.conj() @ dO_ii @ P1_ni.T


def initial_wannier(ibzwfs: IBZWaveFunctions,
                    initialwannier, kpointgrid, fixedstates,
                    edf, spin, nbands):
    """Initial guess for the shape of wannier functions.

    Use initial guess for wannier orbitals to determine rotation
    matrices U and C.
    """

    from ase.dft.wannier import rotation_from_projection
    proj_knw = get_projections(ibzwfs, initialwannier, spin)
    U_kww = []
    C_kul = []
    for fixed, proj_nw in zip(fixedstates, proj_knw):
        U_ww, C_ul = rotation_from_projection(proj_nw[:nbands],
                                              fixed,
                                              ortho=True)
        U_kww.append(U_ww)
        C_kul.append(C_ul)

    return C_kul, np.asarray(U_kww)


def get_projections(ibzwfs: IBZWaveFunctions,
                    locfun: str | list[tuple],
                    spin=0):
    """Project wave functions onto localized functions

    Determine the projections of the Kohn-Sham eigenstates
    onto specified localized functions of the format::

      locfun = [[spos_c, l, sigma], [...]]

    spos_c can be an atom index, or a scaled position vector. l is
    the angular momentum, and sigma is the (half-) width of the
    radial gaussian.

    Return format is::

      f_kni = <psi_kn | f_i>

    where psi_kn are the wave functions, and f_i are the specified
    localized functions.

    As a special case, locfun can be the string 'projectors', in which
    case the bound state projectors are used as localized functions.
    """
    if isinstance(locfun, str):
        assert locfun == 'projectors'
        f_kin = []
        for wfs in ibzwfs:
            if wfs.spin == spin:
                f_in = []
                for a, P_ni in wfs.P_ani.items():
                    i = 0
                    setup = wfs.setups[a]
                    for l, n in zip(setup.l_j, setup.n_j):
                        if n >= 0:
                            for j in range(i, i + 2 * l + 1):
                                f_in.append(P_ni[:, j])
                        i += 2 * l + 1
                f_kin.append(f_in)
        f_kni = np.array(f_kin).transpose(0, 2, 1)
        return f_kni.conj()

    nkpts = len(ibzwfs.ibz)
    nbf = np.sum([2 * l + 1 for pos, l, a in locfun])
    f_knB = np.zeros((nkpts, ibzwfs.nbands, nbf), ibzwfs.dtype)
    relpos_ac = ibzwfs.wfs_qs[0][0].relpos_ac

    spos_bc = []
    splines_b = []
    for spos_c, l, sigma in locfun:
        if isinstance(spos_c, int):
            spos_c = relpos_ac[spos_c]
        spos_bc.append(spos_c)
        alpha = .5 * Bohr**2 / sigma**2
        r = np.linspace(0, 10. * sigma, 500)
        f_g = (fac(l) * (4 * alpha)**(l + 3 / 2.) *
               np.exp(-alpha * r**2) /
               (np.sqrt(4 * np.pi) * fac(2 * l + 1)))
        splines_b.append([Spline.from_data(l, rmax=r[-1], f_g=f_g)])

    assert ibzwfs.domain_comm.size == 1

    for wfs in ibzwfs:
        if wfs.spin != spin:
            continue
        psit_nX = wfs.psit_nX
        lf_blX = psit_nX.desc.atom_centered_functions(
            splines_b, spos_bc, cut=True)
        f_bnl = lf_blX.integrate(psit_nX)
        f_knB[wfs.q] = f_bnl.data
    return f_knB.conj()
