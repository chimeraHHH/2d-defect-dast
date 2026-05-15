import numpy as np
from ase.parallel import parprint
from ase.units import _e, Bohr, Ha, J
from ase.utils.timing import Timer

from gpaw.mpi import world
from gpaw.nlopt.matrixel import get_derivative, get_rml
from gpaw.utilities.progressbar import ProgressBar


def get_shg(
        nlodata,
        freqs=[1.0],
        eta=0.05,
        pol='yyy',
        eshift=0.0,
        gauge='lg',
        ftol=1e-4, Etol=1e-6,
        band_n=None,
        out_name='shg.npy'):
    """
    Calculate SHG spectrum within the independent particle approximation (IPA)
    for nonmagnetic semiconductors.

    Output: shg.npy file with numpy array containing the spectrum and
    frequencies.

    Parameters
    ----------
    nlodata
        Data object of class NLOData. Contains energies, occupancies and
        momentum matrix elements.
    freqs
        Excitation frequency array (a numpy array or list).
    eta
        Broadening, a number or an array (default 0.05 eV).
    pol
        Tensor element (default 'yyy').
    gauge
        Choose the gauge ('lg' or 'vg').
    Etol, ftol
        Tol. in energy and fermi to consider degeneracy.
    band_n
        List of bands in the sum (default 0 to nb).
    out_name
        Output filename (default 'shg.npy').

    """

    # Start a timer
    timer = Timer()
    parprint(f'Calculating SHG spectrum (in {world.size:d} cores).')

    # Covert inputs in eV to Ha
    freqs = np.array(freqs)
    nw = len(freqs)
    w_lc = (freqs + 1e-12 + 1j * eta) / Ha  # Add small value to avoid 0
    Etol /= Ha
    eshift /= Ha

    # Useful variables
    pol_v = ['xyz'.index(ii) for ii in pol]

    # Use the TRS to reduce calculation time
    w_l = np.hstack((-w_lc[-1::-1], w_lc))
    nw = 2 * nw
    parprint(f'Calculation in the {gauge} gauge for element {pol}.')

    # Load the required data
    with timer('Load and distribute the data'):
        k_info = nlodata.distribute()
        if k_info:
            tmp = list(k_info.values())[0]
            nb = len(tmp[1])
            nk = len(k_info) * world.size  # Approximately
            if band_n is None:
                band_n = list(range(nb))
            mem = 6 * 3 * nk * nb**2 * 16 / 2**20
            parprint(f'At least {mem:.2f} MB of memory is required.')

    # Initial call to print 0% progress
    count = 0
    ncount = len(k_info)
    if world.rank == 0:
        pb = ProgressBar()

    # Initialize the outputs
    sum2_l = np.zeros((nw), complex)
    sum3_l = np.zeros((nw), complex)

    # Do the calculations
    for _, (we, f_n, E_n, p_vnn) in k_info.items():
        # Which gauge
        if gauge == 'vg':
            with timer('Sum over bands'):
                tmp = shg_velocity_gauge(
                    w_l, f_n, E_n, p_vnn, pol_v,
                    band_n, ftol, Etol, eshift)
        elif gauge == 'lg':
            with timer('Position matrix elements calculation'):
                r_vnn, D_vnn = get_rml(E_n, p_vnn, pol_v, Etol=Etol)

            with timer('Compute generalized derivative'):
                rd_vvnn = get_derivative(E_n, r_vnn, D_vnn, pol_v, Etol=Etol)

            with timer('Sum over bands'):
                tmp = shg_length_gauge(
                    w_l, f_n, E_n, r_vnn, rd_vvnn, D_vnn, pol_v,
                    band_n, ftol, Etol, eshift)
        else:
            parprint('Gauge ' + gauge + ' not implemented.')
            raise NotImplementedError

        # Add it to previous with a weight
        sum2_l += tmp[0] * we
        sum3_l += tmp[1] * we

        # Print the progress
        if world.rank == 0:
            pb.update(count / ncount)
        count += 1

    if world.rank == 0:
        pb.finish()
    with timer('Gather data from cores'):
        world.sum(sum2_l)
        world.sum(sum3_l)

    # Make the output in SI unit
    chi_l = make_output(gauge, sum2_l, sum3_l)
    # A multi-col output
    nw = len(freqs)
    chi_l = chi_l[nw:] + chi_l[nw - 1::-1]
    shg = np.vstack((freqs, chi_l))

    # Save it to the file
    if world.rank == 0:
        np.save(out_name, shg)

        # Print the timing
        timer.write()

    return shg


def shg_velocity_gauge(
        w_l, f_n, E_n, p_vnn, pol_v,
        band_n=None, ftol=1e-4, Etol=1e-6, eshift=0):
    """
    Loop over bands for computing in velocity gauge

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
        sum2_l, sum3_l  Output 2 and 3 bands terms
    """

    # Initialize variables
    nb = len(f_n)
    if band_n is None:
        band_n = list(range(nb))
    sum2_l = np.zeros(w_l.size, complex)
    sum3_l = np.zeros(w_l.size, complex)

    # Loop over bands
    for nni in band_n:
        for mmi in band_n:
            # Remove non important term using TRS
            if mmi <= nni:
                continue

            # Useful variables
            fnm = f_n[nni] - f_n[mmi]
            Emn = E_n[mmi] - E_n[nni] + fnm * eshift

            # Comute the 2-band term
            if np.abs(Emn) > Etol and np.abs(fnm) > ftol:
                pnml = (p_vnn[pol_v[0], nni, mmi]
                        * (p_vnn[pol_v[1], mmi, nni]
                            * (p_vnn[pol_v[2], mmi, mmi]
                                - p_vnn[pol_v[2], nni, nni])
                            + p_vnn[pol_v[2], mmi, nni]
                            * (p_vnn[pol_v[1], mmi, mmi]
                                - p_vnn[pol_v[1], nni, nni])) / 2)
                sum2_l += 1j * fnm * np.imag(pnml) * \
                    (1 / (Emn**4 * (w_l - Emn)) -
                        16 / (Emn**4 * (2 * w_l - Emn)))

            # Loop over the last band index
            for lli in band_n:
                fnl = f_n[nni] - f_n[lli]
                fml = f_n[mmi] - f_n[lli]

                # Do not do zero calculations
                if np.abs(fnl) < ftol and np.abs(fml) < ftol:
                    continue

                # Compute the susceptibility with 1/w form
                Eln = E_n[lli] - E_n[nni] + fnl * eshift
                Eml = E_n[mmi] - E_n[lli] - fml * eshift
                pnml = (p_vnn[pol_v[0], nni, mmi]
                        * (p_vnn[pol_v[1], mmi, lli]
                            * p_vnn[pol_v[2], lli, nni]
                            + p_vnn[pol_v[2], mmi, lli]
                            * p_vnn[pol_v[1], lli, nni]))
                pnml = 1j * np.imag(pnml) / 2

                # Compute the divergence-free terms
                if np.abs(Emn) > Etol and np.abs(
                        Eml) > Etol and np.abs(Eln) > Etol:
                    ftermD = (16 / (Emn**3 * (2 * w_l - Emn))
                              * (fnl / (Emn - 2 * Eln)
                                 + fml / (Emn - 2 * Eml))) \
                        + fnl / (Eln**3 * (2 * Eln - Emn)
                                 * (w_l - Eln)) \
                        + fml / (Eml**3 * (2 * Eml - Emn)
                                 * (w_l - Eml))
                    sum3_l += pnml * ftermD

    return sum2_l, sum3_l


def shg_length_gauge(
        w_l, f_n, E_n, r_vnn, rd_vvnn, D_vnn, pol_v,
        band_n=None, ftol=1e-4, Etol=1e-6, eshift=0):
    """
    Loop over bands for computing in length gauge

    Input:
        w_l             Complex frequency array
        f_n             Fermi levels
        E_n             Energies
        r_vnn           Momentum matrix elements
        rd_vvnn         Generalized derivative of position
        D_vnn           Velocity difference
        pol_v           Tensor element
        band_n          Band list
        Etol, ftol      Tol. in energy and fermi to consider degeneracy
        eshift          Band gap correction in eV
    Output:
        sum2_l, sum3_l  Output 2 and 3 bands terms
    """

    # Initialize variables
    nb = len(f_n)
    if band_n is None:
        band_n = list(range(nb))
    sum2_l = np.zeros(w_l.size, complex)
    sum3_l = np.zeros(w_l.size, complex)

    # Loop over bands
    for nni in band_n:
        for mmi in band_n:
            # Remove the non important term using TRS
            if mmi <= nni:
                continue
            fnm = f_n[nni] - f_n[mmi]
            Emn = E_n[mmi] - E_n[nni] + fnm * eshift

            # Two band part
            if np.abs(fnm) > ftol:
                tmp = 2 * np.imag(
                    r_vnn[pol_v[0], nni, mmi]
                    * (rd_vvnn[pol_v[1], pol_v[2], mmi, nni]
                        + rd_vvnn[pol_v[2], pol_v[1], mmi, nni])) \
                    / (Emn * (2 * w_l - Emn))
                tmp += np.imag(
                    r_vnn[pol_v[1], mmi, nni]
                    * rd_vvnn[pol_v[2], pol_v[0], nni, mmi]
                    + r_vnn[pol_v[2], mmi, nni]
                    * rd_vvnn[pol_v[1], pol_v[0], nni, mmi]) \
                    / (Emn * (w_l - Emn))
                tmp += np.imag(
                    r_vnn[pol_v[0], nni, mmi]
                    * (r_vnn[pol_v[1], mmi, nni]
                        * D_vnn[pol_v[2], mmi, nni]
                        + r_vnn[pol_v[2], mmi, nni]
                        * D_vnn[pol_v[1], mmi, nni])) \
                    * (1 / (w_l - Emn)
                        - 4 / (2 * w_l - Emn)) / Emn**2
                tmp -= np.imag(
                    r_vnn[pol_v[1], mmi, nni]
                    * rd_vvnn[pol_v[0], pol_v[2], nni, mmi]
                    + r_vnn[pol_v[2], mmi, nni]
                    * rd_vvnn[pol_v[0], pol_v[1], nni, mmi]) \
                    / (2 * Emn * (w_l - Emn))
                sum2_l += 1j * fnm * tmp / 2  # 1j imag

            # Three band term
            for lli in band_n:
                fnl = f_n[nni] - f_n[lli]
                fml = f_n[mmi] - f_n[lli]
                Eml = E_n[mmi] - E_n[lli] - fml * eshift
                Eln = E_n[lli] - E_n[nni] + fnl * eshift
                # Do not do zero calculations
                if (np.abs(fnm) < ftol and np.abs(fnl) < ftol
                        and np.abs(fml) < ftol):
                    continue
                if np.abs(Eln - Eml) < Etol:
                    continue

                rnml = np.real(
                    r_vnn[pol_v[0], nni, mmi]
                    * (r_vnn[pol_v[1], mmi, lli]
                        * r_vnn[pol_v[2], lli, nni]
                        + r_vnn[pol_v[2], mmi, lli]
                        * r_vnn[pol_v[1], lli, nni])) / (2 * (Eln - Eml))
                if np.abs(fnm) > ftol:
                    sum3_l += 2 * fnm / (2 * w_l - Emn) * rnml
                if np.abs(fnl) > ftol:
                    sum3_l += -fnl / (w_l - Eln) * rnml
                if np.abs(fml) > ftol:
                    sum3_l += fml / (w_l - Eml) * rnml

    return sum2_l, sum3_l


def make_output(gauge, sum2_l, sum3_l):
    """
    Multiply prefactors and return second-order chi in SI units [m / V]

    Input:
        gauge       Chosen gauge
        sum2_l      2-bands term
        sum3_l      3-bands term
    Output:
        chi_l       Output chi as an array
    """

    # 4 * pi from vacuum permittivty eps_0 in atomic units
    prefactor = 4.0 * np.pi
    # Pi factors from BZ volume
    prefactor /= (2.0 * np.pi)**3
    # atomic units [Bohr * elementary charge / Hartee] to SI units [m / V]
    prefactor *= (Bohr * 1e-10 * _e) / (Ha / J)

    if gauge == 'lg':
        chi_l = prefactor * (1j * sum2_l + sum3_l)
    elif gauge == 'vg':
        chi_l = prefactor * 1j / 2 * (sum2_l + sum3_l)
    else:
        parprint('Gauge ' + gauge + ' not implemented.')
        raise NotImplementedError

    return chi_l
