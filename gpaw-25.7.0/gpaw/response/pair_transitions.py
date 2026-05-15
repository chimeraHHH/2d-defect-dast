from __future__ import annotations

import numpy as np


class PairTransitions:
    """Bookkeeping object for transitions in band and spin indices.

    All transitions between different band and spin indices (for a given pair
    of k-points k and k + q) are accounted for via single transition index t,

    t (composite transition index): (n, s) -> (n', s')
    """

    def __init__(self, n1_t, n2_t, s1_t, s2_t):
        """Construct the PairTransitions object.

        Parameters
        ----------
        n1_t : np.array
            Band index of k-point k for each transition t.
        n2_t : np.array
            Band index of k-point k + q for each transition t.
        s1_t : np.array
            Spin index of k-point k for each transition t.
        s2_t : np.array
            Spin index of k-point k + q for each transition t.
        """
        self.n1_t = n1_t
        self.n2_t = n2_t
        self.s1_t = s1_t
        self.s2_t = s2_t

        assert len(n2_t) == len(self)
        assert len(s1_t) == len(self)
        assert len(s2_t) == len(self)

    def __len__(self):
        return len(self.n1_t)

    def get_band_indices(self):
        return self.n1_t, self.n2_t

    def get_spin_indices(self):
        return self.s1_t, self.s2_t

    def get_intraband_mask(self):
        """Get mask for selecting intraband transitions."""
        intraband_t = (self.n1_t == self.n2_t) & (self.s1_t == self.s2_t)
        return intraband_t

    @classmethod
    def from_transitions_domain_arguments(cls, spincomponent,
                                          nbands, nocc1, nocc2, nspins,
                                          bandsummation) -> PairTransitions:
        """Generate the band and spin transitions integration domain.

        The integration domain is determined by the spin rotation (from spin
        index s to spin index s'), the number of bands and spins in the
        underlying ground state calculation as well as the band summation
        scheme.

        The integration domain automatically excludes transitions between two
        occupied bands and two unoccupied bands respectively.

        Parameters
        ----------
        spincomponent : str
            Spin component (μν) of the pair function.
            Currently, '00', 'uu', 'dd', '+-' and '-+' are implemented.
        nbands : int
            Maximum band index to include.
        nocc1 : int
            Number of completely filled bands in the ground state calculation
        nocc2 : int
            Number of non-empty bands in the ground state calculation
        nspins : int
            Number of spin channels in the ground state calculation (1 or 2)
        bandsummation : str
            Band (and spin) summation scheme for pairs of Kohn-Sham orbitals
            'pairwise': sum over pairs of bands (and spins)
            'double': double sum over band (and spin) indices.
        """
        n1_M, n2_M = get_band_transitions_domain(bandsummation, nbands,
                                                 nocc1=nocc1,
                                                 nocc2=nocc2)
        s1_S, s2_S = get_spin_transitions_domain(bandsummation,
                                                 spincomponent, nspins)

        n1_t, n2_t, s1_t, s2_t = transitions_in_composite_index(n1_M, n2_M,
                                                                s1_S, s2_S)

        return cls(n1_t, n2_t, s1_t, s2_t)


def get_band_transitions_domain(bandsummation, nbands, nocc1=None, nocc2=None):
    """Get all pairs of bands to sum over

    Parameters
    ----------
    bandsummation : str
        Band summation method
    nbands : int
        number of bands
    nocc1 : int
        number of completely filled bands
    nocc2 : int
        number of non-empty bands

    Returns
    -------
    n1_M : ndarray
        band index 1, M = (n1, n2) composite index
    n2_M : ndarray
        band index 2, M = (n1, n2) composite index
    """
    _get_band_transitions_domain =\
        create_get_band_transitions_domain(bandsummation)
    n1_M, n2_M = _get_band_transitions_domain(nbands)

    return remove_null_transitions(n1_M, n2_M, nocc1=nocc1, nocc2=nocc2)


def create_get_band_transitions_domain(bandsummation):
    """Creator component deciding how to carry out band summation."""
    if bandsummation == 'pairwise':
        return get_pairwise_band_transitions_domain
    elif bandsummation == 'double':
        return get_double_band_transitions_domain
    raise ValueError(bandsummation)


def get_double_band_transitions_domain(nbands):
    """Make a simple double sum"""
    n_n = np.arange(0, nbands)
    m_m = np.arange(0, nbands)
    n_nm, m_nm = np.meshgrid(n_n, m_m)
    n_M, m_M = n_nm.flatten(), m_nm.flatten()

    return n_M, m_M


def get_pairwise_band_transitions_domain(nbands):
    """Make a sum over all pairs"""
    n_n = range(0, nbands)
    n_M = []
    m_M = []
    for n in n_n:
        m_m = range(n, nbands)
        n_M += [n] * len(m_m)
        m_M += m_m

    return np.array(n_M), np.array(m_M)


def remove_null_transitions(n1_M, n2_M, nocc1=None, nocc2=None):
    """Remove pairs of bands, between which transitions are impossible"""
    n1_newM = []
    n2_newM = []
    for n1, n2 in zip(n1_M, n2_M):
        if nocc1 is not None and (n1 < nocc1 and n2 < nocc1):
            continue  # both bands are fully occupied
        elif nocc2 is not None and (n1 >= nocc2 and n2 >= nocc2):
            continue  # both bands are completely unoccupied
        n1_newM.append(n1)
        n2_newM.append(n2)

    return np.array(n1_newM), np.array(n2_newM)


def get_spin_transitions_domain(bandsummation, spincomponent, nspins):
    """Get structure of the sum over spins

    Parameters
    ----------
    bandsummation : str
        Band summation method
    spincomponent : str
        Spin component (μν) of the pair function.
        Currently, '00', 'uu', 'dd', '+-' and '-+' are implemented.
    nspins : int
        number of spin channels in ground state calculation

    Returns
    -------
    s1_s : ndarray
        spin index 1, S = (s1, s2) composite index
    s2_S : ndarray
        spin index 2, S = (s1, s2) composite index
    """
    _get_spin_transitions_domain =\
        create_get_spin_transitions_domain(bandsummation)
    return _get_spin_transitions_domain(spincomponent, nspins)


def create_get_spin_transitions_domain(bandsummation):
    """Creator component deciding how to carry out spin summation."""
    if bandsummation == 'pairwise':
        return get_pairwise_spin_transitions_domain
    elif bandsummation == 'double':
        return get_double_spin_transitions_domain
    raise ValueError(bandsummation)


def get_double_spin_transitions_domain(spincomponent, nspins):
    """Usual spin rotations forward in time"""
    if nspins == 1:
        if spincomponent == '00':
            s1_S = [0]
            s2_S = [0]
        else:
            raise ValueError(spincomponent, nspins)
    else:
        if spincomponent == '00':
            s1_S = [0, 1]
            s2_S = [0, 1]
        elif spincomponent == 'uu':
            s1_S = [0]
            s2_S = [0]
        elif spincomponent == 'dd':
            s1_S = [1]
            s2_S = [1]
        elif spincomponent == '+-':
            s1_S = [0]  # spin up
            s2_S = [1]  # spin down
        elif spincomponent == '-+':
            s1_S = [1]  # spin down
            s2_S = [0]  # spin up
        else:
            raise ValueError(spincomponent)

    return np.array(s1_S), np.array(s2_S)


def get_pairwise_spin_transitions_domain(spincomponent, nspins):
    """In a sum over pairs, transitions including a spin rotation may have to
    include terms, propagating backwards in time."""
    if spincomponent in ['+-', '-+']:
        assert nspins == 2
        return np.array([0, 1]), np.array([1, 0])
    else:
        return get_double_spin_transitions_domain(spincomponent, nspins)


def transitions_in_composite_index(n1_M, n2_M, s1_S, s2_S):
    """Use a composite index t for transitions (n, s) -> (n', s')."""
    n1_MS, s1_MS = np.meshgrid(n1_M, s1_S)
    n2_MS, s2_MS = np.meshgrid(n2_M, s2_S)
    return n1_MS.flatten(), n2_MS.flatten(), s1_MS.flatten(), s2_MS.flatten()
