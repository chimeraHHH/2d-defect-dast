import pytest
import numpy as np

from ase.io.ulm import ulmopen
from gpaw.test.gpwfile import response_band_cutoff


@pytest.mark.response
@pytest.mark.parametrize('gs', list(response_band_cutoff))
def test_response_band_cutoff(in_tmp_dir, gpw_files, gs, gpaw_new):
    if gpaw_new and gs == 'v2br4_pw':
        pytest.skip('interpolation=3 not implemented')
    nbands = response_band_cutoff[gs]
    with ulmopen(gpw_files[gs]) as reader:
        eps_skn = reader.wave_functions.eigenvalues
        nconv = reader.parameters.convergence['bands']
    print(gs)
    assert nbands < nconv
    possible_cutoffs = get_nbands_cutoff_list(eps_skn, nconv)
    print(possible_cutoffs)
    assert nbands in possible_cutoffs


def get_nbands_cutoff_list(eps_skn, nconv, atol=1e-3):
    """ Possible cutoffs for response calc
    Returns the set all allowed band cutoffs in a response calculation.
    Assures that there are no  degeneracies at the edge of the cutoff

    >>> eps = np.array([[[0, 1, 2, 2, 3, 4, 5, 5]]])
    >>> get_nbands_cutoff_list(eps, 7)
    {1, 2, 4, 5, 6}
    """
    allset = set(range(nconv + 1))
    # Loop over spins and k-points
    for eps_kn in eps_skn:
        for eps_n in eps_kn:
            # check degenerate eigenvalues
            cutlist = np.isclose(eps_n[:nconv - 1], eps_n[1:nconv],
                                 atol=atol)
            cutlist = np.argwhere(~cutlist)
            # cutoff is allowed index + 1
            cutlist += 1
            thisset = set(cutlist.flatten())
            # find minimum cutoff that works for all k
            allset = thisset & allset

    return {int(n) for n in allset}
