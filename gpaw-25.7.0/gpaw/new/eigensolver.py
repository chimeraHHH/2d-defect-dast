from __future__ import annotations

import numpy as np

from ase.units import Ha

from gpaw.new.density import Density
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.potential import Potential
from gpaw.new.energies import DFTEnergies
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pot_calc import PotentialCalculator
from gpaw.mpi import broadcast_float
from gpaw.typing import Array1D


class Eigensolver:
    direct = False

    def iterate(self,
                ibzwfs: IBZWaveFunctions,
                density: Density,
                potential: Potential,
                hamiltonian: Hamiltonian,
                pot_calc: PotentialCalculator,
                energies: DFTEnergies) -> tuple[float, float, DFTEnergies]:
        raise NotImplementedError

    def postprocess(self, ibzwfs, density, potential, hamiltonian):
        pass

    def iterate_kpt(self, wfs, weight_n, iter_func, **fkwargs):
        had_eigs_and_occs = wfs.has_eigs and wfs.has_occs
        if had_eigs_and_occs:
            eig_old = wfs.myeig_n
        eigs_error = iter_func(wfs=wfs, weight_n=weight_n, **fkwargs)
        if had_eigs_and_occs:
            eig_error = np.max(weight_n * np.abs(eig_old - wfs.myeig_n),
                               initial=0)
        else:  # no eigenvalues to compare with
            eig_error = np.inf
        return eigs_error, eig_error


def calculate_weights(converge_bands: int | str,
                      ibzwfs: IBZWaveFunctions) -> list[Array1D | None]:
    """Calculate convergence weights for all eigenstates."""
    weight_un = []
    nu = len(ibzwfs.wfs_qs) * ibzwfs.nspins
    nbands = ibzwfs.nbands

    if converge_bands == 'occupied':
        # Converge occupied bands:
        for wfs in ibzwfs:
            if wfs.has_occs:
                # Methfessel-Paxton or cold-smearing distributions can give
                # negative occupation numbers - so we take the absolute value:
                weight_n = np.abs(wfs.myocc_n)
            else:
                # No eigenvalues yet:
                return [None] * nu
            weight_un.append(weight_n)
        return weight_un

    if converge_bands == 'all':
        converge_bands = nbands

    if not isinstance(converge_bands, str):
        # Converge fixed number of bands:
        n = converge_bands
        if n < 0:
            n += nbands
            assert n >= 0
        for wfs in ibzwfs:
            weight_n = np.zeros(wfs.n2 - wfs.n1)
            m = max(wfs.n1, min(n, wfs.n2)) - wfs.n1
            weight_n[:m] = 1.0
            weight_un.append(weight_n)
        return weight_un

    # Converge states with energy up to CBM + delta:
    assert converge_bands.startswith('CBM+')
    delta = float(converge_bands[4:]) / Ha

    if ibzwfs.fermi_levels is None:
        return [None] * nu

    efermi = np.mean(ibzwfs.fermi_levels)

    # Find CBM:
    cbm = np.inf
    nocc_u = np.empty(nu, int)
    for u, wfs in enumerate(ibzwfs):
        n = (wfs.eig_n < efermi).sum()  # number of occupied bands
        nocc_u[u] = n
        if n < nbands:
            cbm = min(cbm, wfs.eig_n[n])

    # If all k-points don't have the same number of occupied bands,
    # then it's a metal:
    n0 = int(broadcast_float(float(nocc_u[0]), ibzwfs.kpt_comm))
    metal = bool(ibzwfs.kpt_comm.sum_scalar(float((nocc_u != n0).any())))
    if metal:
        cbm = efermi
    else:
        cbm = ibzwfs.kpt_comm.min_scalar(cbm)

    ecut = cbm + delta

    for wfs in ibzwfs:
        weight_n = (wfs.myeig_n < ecut).astype(float)
        if wfs.eig_n[-1] < ecut:
            # We don't have enough bands!
            weight_n[:] = np.inf
        weight_un.append(weight_n)

    return weight_un
