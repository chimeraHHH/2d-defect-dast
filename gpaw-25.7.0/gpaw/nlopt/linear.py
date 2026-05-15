import numpy as np
from ase.units import Ha

from gpaw.mpi import world


def get_chi_tensor(
        nlodata,
        freqs=[1.0], eta=0.05,
        ftol=1e-4, Etol=1e-6, eshift=0.0,
        band_n=None, out_name=None):
    """
    Calculate full linear susceptibility tensor for nonmagnetic semiconductors;
    array will be saved to disk if out_name is given.

    Parameters
    ----------
    nlodata
        Data object of type NLOData.
    freqs
        Excitation frequency array (a numpy array or list).
    eta
        Broadening, a number or an array (default 0.05 eV).
    Etol, ftol
        Tolerance in energy and occupancy to consider degeneracy.
    eshift
        Bandgap correction.
    band_n
        List of bands in the sum (default 0 to nb).
    out_name
        If it is given: output filename.

    Returns
    -------
    np.ndarray:
        Full linear susceptibility tensor (3, 3, nw).

    """

    # Covert inputs in eV to Ha
    freqs = np.array(freqs)
    nw = len(freqs)
    w_lc = (freqs + 1j * eta) / Ha
    Etol /= Ha
    eshift /= Ha

    # Load the required data
    k_info = nlodata.distribute()
    if k_info:
        tmp = list(k_info.values())[0]
        nb = len(tmp[1])
        if band_n is None:
            band_n = list(range(nb))

    # Initialize the outputs
    sum_vvl = np.zeros((3, 3, nw), complex)

    # Do the calculations
    for _, (we, f_n, E_n, p_vnn) in k_info.items():
        tmp = np.zeros((3, 3, nw), complex)
        for v1 in range(3):
            for v2 in range(v1, 3):
                sum_l = calc_chi(
                    w_lc, f_n, E_n, p_vnn, [v1, v2],
                    band_n, ftol, Etol, eshift=eshift)
                tmp[v1, v2] = sum_l
                tmp[v2, v1] = sum_l
        # Add it to previous with a weight
        sum_vvl += tmp * we

    world.sum(sum_vvl)

    # Multiply prefactors (4pi from eps0, 8pi^3 from BZ)
    prefactor = 4 * np.pi / (2 * np.pi)**3
    chi_vvl = prefactor * sum_vvl

    # Save it to the file
    if world.rank == 0 and out_name is not None:
        tmp = chi_vvl.reshape(9, nw)
        lin = np.vstack((freqs, tmp))
        np.save(out_name, lin)

    return chi_vvl


def calc_chi(
        w_l, f_n, E_n, p_vnn, pol_v,
        band_n=None, ftol=1e-4, Etol=1e-6, eshift=0):
    """
    Loop over bands for computing the response

    Input:
        w_l             Complex frequency array
        f_n             Fermi levels
        E_n             Energies
        p_vnn           Momentum matrix elements
        pol_v           Tensor element
        band_n          Band list
        Etol, ftol      Tol. in energy and fermi to consider degeneracy
        eshift          Bandgap correction
    Output:
        sum_l           Output sum value (array with length of w_l)
    """

    # Initialize variables
    nb = len(f_n)
    if band_n is None:
        band_n = list(range(nb))
    sum_l = np.zeros(w_l.size, complex)

    # Loop over bands
    for nni in band_n:
        for mmi in band_n:
            # Use TRS to reduce
            if mmi <= nni:
                continue
            fnm = f_n[nni] - f_n[mmi]
            Emn = E_n[mmi] - E_n[nni] + fnm * eshift
            if np.abs(fnm) < ftol or np.abs(Emn) < Etol:
                continue
            # *2 for real, /2 for TRS, *2 for m<n
            sum_l += 2 * fnm * np.real(
                p_vnn[pol_v[0], nni, mmi] * p_vnn[pol_v[1], mmi, nni]) \
                / (Emn * (Emn**2 - w_l**2))

    return sum_l
