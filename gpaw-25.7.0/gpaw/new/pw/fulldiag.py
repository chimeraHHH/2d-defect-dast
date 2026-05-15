from __future__ import annotations

import numpy as np
from gpaw.core.atom_arrays import AtomArrays
from gpaw.core.matrix import Matrix, create_distribution
from gpaw.core.plane_waves import (PWAtomCenteredFunctions,
                                   PWArray, PWDesc)
from gpaw.core.uniform_grid import UGArray
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.typing import Array2D
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.wave_functions import WaveFunctions
from gpaw.new.potential import Potential
from gpaw.new.smearing import OccupationNumberCalculator


def pw_matrix(pw: PWDesc,
              pt_aiG: PWAtomCenteredFunctions,
              dH_aii: AtomArrays,
              dS_aii: list[Array2D],
              vt_R: UGArray,
              dedtaut_R: UGArray | None,
              comm) -> tuple[Matrix, Matrix]:
    """Calculate H and S matrices in plane-wave basis.

    :::

                 _ _     _ _
            /  -iG.r ~  iG.r _
      O   = | e      O e    dr
       GG'  /

    :::

      ~   ^   ~ _    _ _     --- ~a _ _a    a  ~  _  _a
      H = T + v(r) δ(r-r') + <   p (r-R ) ΔH   p (r'-R )
                             ---  i         ij  j
                             aij

    :::

      ~     _ _     --- ~a _ _a    a  ~  _  _a
      S = δ(r-r') + <   p (r-R ) ΔS   p (r'-R )
                    ---  i         ij  j
                    aij
    """
    assert pw.dtype == complex
    npw = pw.shape[0]
    dist = create_distribution(npw, npw, comm, -1, 1)
    H_GG = dist.matrix(complex)
    S_GG = dist.matrix(complex)
    G1, G2 = dist.my_row_range()

    x_G = pw.empty()
    assert isinstance(x_G, PWArray)  # Fix this!
    x_R = vt_R.desc.new(dtype=complex).zeros()
    assert isinstance(x_R, UGArray)  # Fix this!
    dv = pw.dv

    for G in range(G1, G2):
        x_G.data[:] = 0.0
        x_G.data[G] = 1.0
        x_G.ifft(out=x_R)
        x_R.data *= vt_R.data
        x_R.fft(out=x_G)
        H_GG.data[G - G1] = dv * x_G.data

    if dedtaut_R is not None:
        G_Gv = pw.reciprocal_vectors()
        for G in range(G1, G2):
            for v in range(3):
                x_G.data[:] = 0.0
                x_G.data[G] = 1j * G_Gv[G, v]
                x_G.ifft(out=x_R)
                x_R.data *= dedtaut_R.data
                x_R.fft(out=x_G)
                H_GG.data[G - G1] += -0.5j * dv * G_Gv[:, v] * x_G.data

    H_GG.add_to_diagonal(dv * pw.ekin_G[G1:G2])
    S_GG.data[:] = 0.0
    S_GG.add_to_diagonal(dv)

    pt_aiG._lazy_init()
    assert pt_aiG._lfc is not None
    f_GI = pt_aiG._lfc.expand()
    nI = f_GI.shape[1]
    dH_II = np.zeros((nI, nI))
    dS_II = np.zeros((nI, nI))
    I1 = 0
    for a, dH_ii in dH_aii.items():
        dS_ii = dS_aii[a]
        I2 = I1 + len(dS_ii)
        dH_II[I1:I2, I1:I2] = dH_ii
        dS_II[I1:I2, I1:I2] = dS_ii
        I1 = I2

    H_GG.data += (f_GI[G1:G2].conj() @ dH_II) @ f_GI.T
    S_GG.data += (f_GI[G1:G2].conj() @ dS_II) @ f_GI.T

    return H_GG, S_GG


def diagonalize(potential: Potential,
                ibzwfs: IBZWaveFunctions,
                occ_calc: OccupationNumberCalculator,
                nbands: int,
                nelectrons: float) -> IBZWaveFunctions:
    """Diagonalize hamiltonian in plane-wave basis."""
    vt_sR = potential.vt_sR
    dH_asii = potential.dH_asii
    dedtaut_sR: UGArray | list[None] = [None] * len(vt_sR)
    if potential.dedtaut_sR is not None:
        dedtaut_sR = potential.dedtaut_sR

    band_comm = ibzwfs.band_comm

    wfs_qs: list[list[WaveFunctions]] = []
    for wfs_s in ibzwfs.wfs_qs:
        wfs_qs.append([])
        for wfs in wfs_s:
            dS_aii = [setup.dO_ii for setup in wfs.setups]
            assert isinstance(wfs, PWFDWaveFunctions)
            assert isinstance(wfs.pt_aiX, PWAtomCenteredFunctions)
            pw = wfs.psit_nX.desc
            H_GG, S_GG = pw_matrix(pw,
                                   wfs.pt_aiX,
                                   dH_asii[:, wfs.spin],
                                   dS_aii,
                                   vt_sR[wfs.spin],
                                   dedtaut_sR[wfs.spin],
                                   band_comm)

            eig_n = H_GG.eigh(S_GG, limit=nbands)
            H_GG.complex_conjugate()
            assert eig_n[0] > -1000, 'See issue #241'
            psit_nG = pw.empty(nbands, comm=band_comm)
            mynbands, nG = psit_nG.data.shape
            maxmynbands = (nbands + band_comm.size - 1) // band_comm.size
            C_nG = H_GG.new(
                dist=(band_comm, band_comm.size, 1, maxmynbands, 1))
            H_GG.redist(C_nG)
            psit_nG.data[:] = C_nG.data[:mynbands]
            new_wfs = PWFDWaveFunctions.from_wfs(wfs, psit_nX=psit_nG)
            new_wfs._eig_n = eig_n
            wfs_qs[-1].append(new_wfs)

    new_ibzwfs = IBZWaveFunctions(
        ibzwfs.ibz,
        ncomponents=ibzwfs.ncomponents,
        wfs_qs=wfs_qs,
        kpt_comm=ibzwfs.kpt_comm,
        kpt_band_comm=ibzwfs.kpt_band_comm,
        comm=ibzwfs.comm)

    new_ibzwfs.calculate_occs(occ_calc, nelectrons)

    return new_ibzwfs
