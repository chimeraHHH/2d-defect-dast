import numpy as np

import gpaw
from ase.units import Hartree, Bohr
from gpaw.utilities.folder import Folder


def get_dielectric(exlist, volume,
                   emin=0,
                   emax=None,
                   de=None,
                   energyunit='eV',
                   width=0.08,  # width in energyunit
                   comment=None,
                   form='v'
                   ):
    """Evaluate the dielectric function

    Parameters:
    =============== ===================================================
    ``exlist``      ExcitationList
    ``volume``      Unit cell volume in Angstrom**3
    ``emin``        min. energy, set to zero if not given
    ``emax``        max. energy, set to cover all energies if not given
    ``de``          energy spacing
    ``energyunit``  Energy unit 'eV' or 'nm', default 'eV'
    ``width``       folding width in terms of the chosen energyunit
    =============== ===================================================
    all energies in [eV]

    Returns
    =======
    energy: energy values
    eps1: real part of the dielectric function
    eps2: imaginary part of the dielectric function
    N: real part of the refractive index
    K: imaginary part of the refractive index
    R: reflection coefficient
    """

    x = []
    y = []
    for ex in exlist:
        x.append(ex.get_energy() * Hartree)
        y.append(ex.get_oscillator_strength(form))

    if energyunit != 'eV':
        raise RuntimeError('currently only eV is supported')

    pre = 4. * np.pi / volume * Bohr**3 * Hartree**2
    energies, values = Folder(width, 'RealLorentzPole').fold(
        x, y, de, emin, emax)
    eps1 = 1 + pre * values.T[0]
    energies, values = Folder(width, 'ImaginaryLorentzPole').fold(
        x, y, de, emin, emax)
    eps2 = pre * values.T[0]

    N = np.sqrt(0.5 * (np.sqrt(eps1**2 + eps2**2) + eps1))
    K = np.sqrt(0.5 * (np.sqrt(eps1**2 + eps2**2) - eps1))
    R = ((N - 1)**2 + K**2) / ((N + 1)**2 + K**2)

    return energies, eps1, eps2, N, K, R


def dielectric(exlist,
               volume,
               filename=None,
               emin=0,
               emax=None,
               de=None,
               energyunit='eV',
               width=0.08,  # width in energyunit
               comment=None,
               form='v'
               ):
    """Write out the dielectric function

    Parameters:
    =============== ===================================================
    ``exlist``      ExcitationList
    ``volume``      Unit cell volume in Angstrom**3
    ``filename``    File name for the output file, STDOUT if not given
    ``emin``        min. energy, set to zero if not given
    ``emax``        max. energy, set to cover all energies if not given
    ``de``          energy spacing
    ``energyunit``  Energy unit 'eV' or 'nm', default 'eV'
    ``width``       folding width in terms of the chosen energyunit
    =============== ===================================================
    all energies in [eV]
    """

    with open(filename, 'w') as out:
        if comment:
            print('#', comment, file=out)

        print('# Dielec function', file=out)
        print('# GPAW version:', gpaw.__version__, file=out)
        print(f'# width={width} [{energyunit}]', file=out)
        if form == 'r':
            print('# length form', file=out)
        else:
            assert form == 'v'
            print('# velocity form', file=out)
        print(
            '# om [{}]  eps1/eps0  eps2/eps0   n     k     R'
            .format(energyunit), file=out)

        energies, eps1, eps2, N, K, R = get_dielectric(
            exlist=exlist, volume=volume, emin=emin, emax=emax, de=de,
            energyunit=energyunit, width=width, form=form)

        for e, e1, e2, n, k, r in zip(energies, eps1, eps2, N, K, R):
            print('%10.5f %12.7e %12.7e %12.7e %12.7e %12.7e' %
                  (e, e1, e2, n, k, r), file=out)

        if filename is not None:
            out.close()
