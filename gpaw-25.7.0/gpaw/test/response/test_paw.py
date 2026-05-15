import numpy as np
import pytest
from functools import partial

from ase.units import Ha
from ase.data import chemical_symbols

from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.qpd import SingleQPWDescriptor
from gpaw.response.groundstate import ResponsePAWDataset
from gpaw.response.paw import (calculate_pair_density_correction,
                               calculate_matrix_element_correction)
from gpaw.response.site_paw import calculate_site_matrix_element_correction
from gpaw.response.localft import add_LSDA_trans_fxc

from gpaw.setup import create_setup
from gpaw.sphere.rshe import calculate_reduced_rshe


def setups():
    for symbol in chemical_symbols:
        try:
            setup = create_setup(symbol)
        except FileNotFoundError:
            pass
        else:
            yield setup


@pytest.mark.response
@pytest.mark.serial
@pytest.mark.parametrize('setup', setups())
def test_paw_corrections(setup):
    radial_points = 2**10
    if setup.symbol in {'I', 'Hg', 'Pb'}:
        # More points where needed, for performance.
        # https://gitlab.com/gpaw/gpaw/-/issues/984
        radial_points *= 4

    G_Gv = np.zeros((5, 3))
    G_Gv[:, 0] = np.linspace(0, 20, 5)
    pawdata = ResponsePAWDataset(setup, radial_points=radial_points)
    calculate_pair_density_correction(G_Gv, pawdata=pawdata)


@pytest.mark.response
def test_paw_correction_consistency(gpw_files):
    """Test consistency of the pair density PAW corrections."""
    gs = ResponseGroundStateAdapter.from_gpw_file(gpw_files['fe_pw'])

    # Set up plane-wave basis
    ecut = 50  # eV
    qpd = SingleQPWDescriptor.from_q([0., 0., 0.5],  # N-point
                                     ecut / Ha, gs.gd)
    qG_Gv = qpd.get_reciprocal_vectors(add_q=True)

    # Calculate ordinary pair density corrections
    pawdata = gs.pawdatasets.by_atom[0]
    Q1_Gii = calculate_pair_density_correction(qG_Gv, pawdata=pawdata)

    # Calculate pair density as a generalized matrix element
    # Expand unity in real spherical harmonics
    rgd = pawdata.xc_correction.rgd
    Y_nL = pawdata.xc_correction.Y_nL
    f_ng = np.ones((Y_nL.shape[0], rgd.N))
    rshe, _ = calculate_reduced_rshe(rgd, f_ng, Y_nL, lmax=0)
    # Calculate correction
    Q2_Gii = calculate_matrix_element_correction(qG_Gv, pawdata, rshe)

    assert Q2_Gii == pytest.approx(Q1_Gii, rel=1e-3, abs=1e-5)


@pytest.mark.response
@pytest.mark.serial
def test_site_paw_correction_consistency(gpw_files):
    """Test consistency of generalized matrix elements."""
    gs = ResponseGroundStateAdapter.from_gpw_file(gpw_files['fe_pw'])

    # Expand the LDA fxc kernel in real spherical harmonics
    pawdata = gs.pawdatasets.by_atom[0]
    micro_setup = gs.micro_setups[0]
    add_fxc = partial(add_LSDA_trans_fxc, fxc='ALDA')
    rshe, _ = micro_setup.expand_function(add_fxc, wmin=1e-8)

    # Calculate PAW correction with G + q = 0
    qG_Gv = np.zeros((1, 3))
    nF_Gii = calculate_matrix_element_correction(qG_Gv, pawdata, rshe)

    # Calculate PAW correction with site cutoff exceeding the augmentation
    # sphere radius
    augr = gs.get_aug_radii()[0]
    rcut_p = [augr * 1.5]
    drcut = augr * 0.25
    lambd_p = [0.5]
    nF_pii = calculate_site_matrix_element_correction(pawdata, rshe,
                                                      rcut_p, drcut, lambd_p)

    assert nF_Gii[0] == pytest.approx(nF_pii[0])
