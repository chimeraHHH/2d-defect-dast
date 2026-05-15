import numpy as np
from gpaw.utilities import unpack_hermitian


def kinetic_energies(psit, projections, setups):
    """Calculate kinetic energies of states."""
    T_G = 0.5 * psit.pd.G2_qG[psit.kpt]
    ekin_n = []
    for psit_G in psit.array:
        ekin_n.append(psit.pd.integrate(psit_G, T_G * psit_G).real)

    ekincore = 0.0
    ekin_n = np.array(ekin_n)
    for a, setup in enumerate(setups):
        K_ii = unpack_hermitian(setup.K_p)
        P_ni = projections[a]
        ekin_n += np.einsum('ni, ij, nj -> n', P_ni.conj(), K_ii, P_ni).real
        ekincore += setup.Kc

    return ekincore, ekin_n
