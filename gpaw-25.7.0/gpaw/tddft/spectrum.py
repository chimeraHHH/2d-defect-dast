import re
import numpy as np

from gpaw import __version__ as version
from gpaw.mpi import world
from gpaw.tddft.units import au_to_as, au_to_fs, au_to_eV, rot_au_to_cgs
from gpaw.tddft.folding import FoldedFrequencies
from gpaw.tddft.folding import Folding


def calculate_fourier_transform(x_t, y_ti, foldedfrequencies, velocity=False):
    ff = foldedfrequencies
    X_w = ff.frequencies
    envelope = ff.folding.envelope

    # Construct integration weights:
    # We use trapezoidal rule except the end point is accounted with
    # full weight. This ensures better numerical compatibility
    # with the on-the-fly Fourier integrators.
    # This shouldn't affect in usual scenarios as the envelope
    # should damp the data to zero at the end point in any case.
    dx_t1 = x_t[1:] - x_t[:-1]
    dx_t = 0.5 * (np.insert(dx_t1, 0, 0.0) + np.append(dx_t1, dx_t1[-1]))

    env_t = envelope(x_t)
    Ienv = np.sum(dx_t * env_t)

    if velocity:
        y_ti -= np.sum((dx_t * env_t)[:, None] * y_ti, axis=0) / Ienv

    # Integrate
    f_wt = np.exp(1.0j * np.outer(X_w, x_t))
    y_it = np.swapaxes(y_ti, 0, 1)

    Y_wi = np.tensordot(f_wt, dx_t * env_t * y_it, axes=(1, 1))
    print('Sinc contamination', env_t[-1])
    return Y_wi


def read_td_file_data(fname, remove_duplicates=True):
    """Read data from time-dependent data file.

    Parameters
    ----------
    fname
        File path
    remove_duplicates
        If true, remove data from overlapping time values.
        The first encountered values are kept.

    Returns
    -------
    time_t
        Array of time values
    data_ti
        Array of data values
    """
    # Read data
    data_tj = np.loadtxt(fname)
    time_t = data_tj[:, 0]
    data_ti = data_tj[:, 1:]

    # Remove duplicates due to abruptly stopped and restarted calculation
    if remove_duplicates:
        flt_t = np.ones_like(time_t, dtype=bool)
        maxtime = time_t[0]
        for t in range(1, len(time_t)):
            # Note about ">=" here:
            # The equality is included here in order to
            # retain step-like data (for example, the data both just before
            # and just after the kick is kept).
            if time_t[t] >= maxtime:
                maxtime = time_t[t]
            else:
                flt_t[t] = False
        time_t = time_t[flt_t]
        data_ti = data_ti[flt_t]
        ndup = len(flt_t) - flt_t.sum()
        if ndup > 0:
            print('Removed %d duplicates' % ndup)
    return time_t, data_ti


def read_td_file_kicks(fname):
    """Read kicks from time-dependent data file.

    Parameters
    ----------
    fname
        File path

    Returns
    -------
    kick_i
        List of kicks.
        Each kick is a dictionary with keys
        ``strength_v`` and ``time``.
    """
    def parse_kick_line(line):
        # Kick
        regexp = (r"Kick = \["
                  r"(?P<k0>[-+0-9\.e\ ]+), "
                  r"(?P<k1>[-+0-9\.e\ ]+), "
                  r"(?P<k2>[-+0-9\.e\ ]+)\]")
        m = re.search(regexp, line)
        assert m is not None, 'Kick not found'
        kick_v = np.array([float(m.group('k%d' % v)) for v in range(3)])
        # Time
        regexp = r"Time = (?P<time>[-+0-9\.e\ ]+)"
        m = re.search(regexp, line)
        if m is None:
            print('time not found')
            time = 0.0
        else:
            time = float(m.group('time'))
        velocity = 'velocity' in line
        return kick_v, time, velocity

    # Search kicks
    kick_i = []
    with open(fname) as f:
        for line in f:
            if line.startswith('# Kick'):
                kick_v, time, velocity = parse_kick_line(line)
                kick_i.append({'strength_v': kick_v, 'time': time,
                              'velocity': velocity})
    return kick_i


def clean_td_data(kick_i, time_t, data_ti):
    """Prune time-dependent data.

    This function checks that there is only one kick
    in the kick list and moves time zero to the kick
    time (discarding all preceding data).

    Parameters
    ----------
    kick_i
        List of kicks.
    time_t
        Array of time values
    data_ti
        Array of data values

    Returns
    -------
    kick_i
        List of kicks.
        Each kick is a dictionary with keys
        ``strength_v`` and ``time``.

    Raises
    ------
    RuntimeError
        If kick list contains multiple kicks.
    """
    # Check kicks
    if len(kick_i) > 1:
        raise RuntimeError('Multiple kicks')
    kick = kick_i[0]
    kick_v = kick['strength_v']
    velocity = kick['velocity']
    kick_time = kick['time']

    # Discard times before kick
    flt_t = time_t >= kick_time
    time_t = time_t[flt_t]
    data_ti = data_ti[flt_t]

    # Move time zero to kick time
    time_t -= kick_time
    assert time_t[0] == 0.0

    return kick_v, velocity, time_t, data_ti


def read_dipole_moment_file(fname, remove_duplicates=True):
    """Read time-dependent dipole moment data file.

    Parameters
    ----------
    fname
        File path
    remove_duplicates
        If true, remove data from overlapping time values.
        The first encountered values are kept.

    Returns
    -------
    kick_i
        List of kicks.
        Each kick is a dictionary with keys
        ``strength_v`` and ``time``.
    time_t
        Array of time values
    norm_t
        Array of norm values
    dm_tv
        Array of dipole moment values
    """
    time_t, data_ti = read_td_file_data(fname, remove_duplicates)
    kick_i = read_td_file_kicks(fname)
    norm_t = data_ti[:, 0]
    dm_tv = data_ti[:, 1:]
    return kick_i, time_t, norm_t, dm_tv


def calculate_polarizability(kick_v, time_t, dm_tv,
                             foldedfrequencies, velocity=False):
    if not velocity:
        dm_tv = dm_tv - dm_tv[0]

    alpha_wv = calculate_fourier_transform(time_t, dm_tv, foldedfrequencies,
                                           velocity=velocity)

    kick_magnitude = np.sqrt(np.sum(kick_v**2))
    alpha_wv /= kick_magnitude
    return alpha_wv


def calculate_photoabsorption(kick_v, time_t, dm_tv,
                              foldedfrequencies, velocity=False):
    omega_w = foldedfrequencies.frequencies
    alpha_wv = calculate_polarizability(kick_v, time_t, dm_tv,
                                        foldedfrequencies,
                                        velocity=velocity)
    if velocity:
        abs_wv = 2 / np.pi * alpha_wv.real
    else:
        abs_wv = 2 / np.pi * omega_w[:, np.newaxis] * alpha_wv.imag

    kick_magnitude = np.sqrt(np.sum(kick_v**2))
    abs_wv *= kick_v / kick_magnitude
    return abs_wv


def read_magnetic_moment_file(fname, remove_duplicates=True):
    """Read time-dependent magnetic moment data file.

    Parameters
    ----------
    fname
        File path
    remove_duplicates
        If true, remove data from overlapping time values.
        The first encountered values are kept.

    Returns
    -------
    kick_i
        List of kicks.
        Each kick is a dictionary with keys
        ``strength_v`` and ``time``.
    time_t
        Array of time values
    mm_tv
        Array of magnetic moment values
    """
    time_t, mm_tv = read_td_file_data(fname, remove_duplicates)
    kick_i = read_td_file_kicks(fname)
    return kick_i, time_t, mm_tv


def calculate_rotatory_strength_components(kick_v, time_t, mm_tv,
                                           foldedfrequencies):
    assert np.all(mm_tv[0] == 0.0)
    mm_wv = calculate_fourier_transform(time_t, mm_tv, foldedfrequencies)
    kick_magnitude = np.sqrt(np.sum(kick_v**2))
    rot_wv = mm_wv.real / (np.pi * kick_magnitude)
    return rot_wv


def write_spectrum(dipole_moment_file, spectrum_file,
                   folding, width, e_min, e_max, delta_e,
                   title, symbol, calculate):
    def str_list(v_i, fmt='%g'):
        return '[%s]' % ', '.join(map(lambda v: fmt % v, v_i))

    kick_i, time_t, _, dm_tv = read_dipole_moment_file(dipole_moment_file)
    kick_v, velocity, time_t, dm_tv = clean_td_data(kick_i, time_t, dm_tv)
    dt_t = time_t[1:] - time_t[:-1]

    freqs = np.arange(e_min, e_max + 0.5 * delta_e, delta_e)
    folding = Folding(folding, width)
    ff = FoldedFrequencies(freqs, folding)
    omega_w = ff.frequencies
    spec_wv = calculate(kick_v, time_t, dm_tv, ff, velocity=velocity)

    # Write spectrum file header
    with open(spectrum_file, 'w') as f:
        def w(s):
            f.write('%s\n' % s)

        w('# %s spectrum from real-time propagation' % title)
        w('# GPAW version: %s' % version)
        w('# Total time = %.4f fs, Time steps = %s as' %
          (dt_t.sum() * au_to_fs,
           str_list(np.unique(np.around(dt_t, 6)) * au_to_as, '%.4f')))
        w('# Kick = %s' % str_list(kick_v))
        w('# %sian folding, Width = %.4f eV = %lf Hartree'
          ' <=> FWHM = %lf eV' %
          (folding.folding, folding.width * au_to_eV, folding.width,
           folding.fwhm * au_to_eV))

        col_i = []
        data_iw = [omega_w * au_to_eV]
        for v in range(len(kick_v)):
            h = '{}_{}'.format(symbol, 'xyz'[v])
            if spec_wv.dtype == complex:
                col_i.append('Re[%s]' % h)
                data_iw.append(spec_wv[:, v].real)
                col_i.append('Im[%s]' % h)
                data_iw.append(spec_wv[:, v].imag)
            else:
                col_i.append(h)
                data_iw.append(spec_wv[:, v])

        w('# %10s %s' % ('om (eV)', ' '.join(['%20s' % s for s in col_i])))

    # Write spectrum file data
    with open(spectrum_file, 'ab') as f:
        np.savetxt(f, np.array(data_iw).T,
                   fmt='%12.6lf' + (' %20.10le' * len(col_i)))

    return folding.envelope(time_t[-1])


def photoabsorption_spectrum(dipole_moment_file: str,
                             spectrum_file: str,
                             folding: str = 'Gauss',
                             width: float = 0.2123,
                             e_min: float = 0.0,
                             e_max: float = 30.0,
                             delta_e: float = 0.05):
    """Calculates photoabsorption spectrum from the time-dependent
    dipole moment.

    The spectrum is represented as a dipole strength function
    in units of 1/eV. Thus, the resulting spectrum should integrate
    to the number of valence electrons in the system.

    Parameters
    ----------
    dipole_moment_file
        Name of the time-dependent dipole moment file from which
        the spectrum is calculated
    spectrum_file
        Name of the spectrum file
    folding
        Gaussian (``'Gauss'``) or Lorentzian (``'Lorentz'``) folding
    width
        Width of the Gaussian (sigma) or Lorentzian (Gamma)
        Gaussian =     1/(sigma sqrt(2pi)) exp(-(1/2)(omega/sigma)^2)
        Lorentzian =  (1/pi) (1/2) Gamma / [omega^2 + ((1/2) Gamma)^2]
    e_min
        Minimum energy shown in the spectrum (eV)
    e_max
        Maximum energy shown in the spectrum (eV)
    delta_e
        Energy resolution (eV)
    """
    if world.rank == 0:
        print('Calculating photoabsorption spectrum from file "%s"'
              % dipole_moment_file)

        def calculate(*args, **kwargs):
            return (calculate_photoabsorption(*args, **kwargs)
                    / au_to_eV)
        sinc = write_spectrum(dipole_moment_file, spectrum_file,
                              folding, width, e_min, e_max, delta_e,
                              'Photoabsorption', 'S', calculate)
        print('Sinc contamination %.8f' % sinc)
        print('Calculated photoabsorption spectrum saved to file "%s"'
              % spectrum_file)


def polarizability_spectrum(dipole_moment_file, spectrum_file,
                            folding='Gauss', width=0.2123,
                            e_min=0.0, e_max=30.0, delta_e=0.05):
    """Calculates polarizability spectrum from the time-dependent
    dipole moment.

    Parameters:

    dipole_moment_file: string
        Name of the time-dependent dipole moment file from which
        the spectrum is calculated
    spectrum_file: string
        Name of the spectrum file
    folding: 'Gauss' or 'Lorentz'
        Whether to use Gaussian or Lorentzian folding
    width: float
        Width of the Gaussian (sigma) or Lorentzian (Gamma)
        Gaussian =     1/(sigma sqrt(2pi)) exp(-(1/2)(omega/sigma)^2)
        Lorentzian =  (1/pi) (1/2) Gamma / [omega^2 + ((1/2) Gamma)^2]
    e_min: float
        Minimum energy shown in the spectrum (eV)
    e_max: float
        Maximum energy shown in the spectrum (eV)
    delta_e: float
        Energy resolution (eV)
    """
    if world.rank == 0:
        print('Calculating polarizability spectrum from file "%s"'
              % dipole_moment_file)

        def calculate(*args, **kwargs):
            return calculate_polarizability(*args, **kwargs) / au_to_eV**2
        sinc = write_spectrum(dipole_moment_file, spectrum_file,
                              folding, width, e_min, e_max, delta_e,
                              'Polarizability', 'alpha', calculate)
        print('Sinc contamination %.8f' % sinc)
        print('Calculated polarizability spectrum saved to file "%s"'
              % spectrum_file)


def rotatory_strength_spectrum(magnetic_moment_files, spectrum_file,
                               folding='Gauss', width=0.2123,
                               e_min=0.0, e_max=30.0, delta_e=0.05):
    """Calculates rotatory strength spectrum from the time-dependent
    magnetic moment.

    Parameters
    ----------
    magnetic_moment_files: list of string
        Time-dependent magnetic moment files for x, y, and z kicks
    spectrum_file: string
        Name of the spectrum file
    folding: 'Gauss' or 'Lorentz'
        Whether to use Gaussian or Lorentzian folding
    width: float
        Width of the Gaussian (sigma) or Lorentzian (Gamma)
        Gaussian =     1/(sigma sqrt(2pi)) exp(-(1/2)(omega/sigma)^2)
        Lorentzian =  (1/pi) (1/2) Gamma / [omega^2 + ((1/2) Gamma)^2]
    e_min: float
        Minimum energy shown in the spectrum (eV)
    e_max: float
        Maximum energy shown in the spectrum (eV)
    delta_e: float
        Energy resolution (eV)
    """
    if world.rank != 0:
        return

    freqs = np.arange(e_min, e_max + 0.5 * delta_e, delta_e)
    folding = Folding(folding, width)
    ff = FoldedFrequencies(freqs, folding)
    omega_w = ff.frequencies * au_to_eV
    rot_w = np.zeros_like(omega_w)

    tot_time = np.inf
    time_steps = []
    kick_strength = None
    for v, fpath in enumerate(magnetic_moment_files):
        kick_i, time_t, mm_tv = read_magnetic_moment_file(fpath)
        kick_v, velocity, time_t, mm_tv = clean_td_data(kick_i, time_t, mm_tv)

        tot_time = min(tot_time, time_t[-1])
        time_steps.append(np.around(time_t[1:] - time_t[:-1], 6))

        # Check kicks
        for v0 in range(3):
            if v0 == v:
                continue
            if kick_v[v0] != 0.0:
                raise RuntimeError('The magnetic moment files must be '
                                   'for kicks in x, y, and z directions.')
        if kick_strength is None:
            kick_strength = np.sqrt(np.sum(kick_v**2))
        if np.sqrt(np.sum(kick_v**2)) != kick_strength:
            raise RuntimeError('The magnetic moment files must have '
                               'been calculated with the same kick strength.')

        rot_wv = calculate_rotatory_strength_components(kick_v, time_t,
                                                        mm_tv, ff)
        rot_w += rot_wv[:, v]

    rot_w *= rot_au_to_cgs * 1e40 / au_to_eV

    # Unique non-zero time steps
    time_steps = np.unique(time_steps)
    time_steps = time_steps[time_steps != 0]

    with open(spectrum_file, 'w') as fd:
        steps_str = ', '.join(f'{val:.4f}' for val in time_steps * au_to_as)
        lines = ['Rotatory strength spectrum from real-time propagations',
                 f'Total time = {tot_time * au_to_fs:.4f} fs, '
                 f'Time steps = [{steps_str}] as',
                 f'Kick strength = {kick_strength}',
                 f'{folding.folding}ian folding, '
                 f'width = {folding.width * au_to_eV:.4f} eV '
                 f'<=> FWHM = {folding.fwhm * au_to_eV:.4f} eV']
        fd.write('# ' + '\n# '.join(lines) + '\n')
        fd.write(f'# {"Energy (eV)":>12} {"R (1e-40 cgs / eV)":>20}\n')

        data_wi = np.vstack((omega_w, rot_w)).T
        np.savetxt(fd, data_wi,
                   fmt='%14.6lf' + (' %20.10le' * (data_wi.shape[1] - 1)))
