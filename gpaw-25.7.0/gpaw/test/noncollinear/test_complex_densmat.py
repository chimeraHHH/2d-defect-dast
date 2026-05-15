import numpy as np
import pytest

from gpaw.new.pot_calc import calculate_non_local_potential1
from gpaw.new.xc import create_functional
from gpaw.setup import create_setup
from gpaw.core import UGDesc
from gpaw.xc import XC


@pytest.mark.soc
def test_energy_from_complex_densmat():

    # Set up objects and matrices

    setup = create_setup('Ga')
    grid = UGDesc(cell=[1, 1, 1], size=[9, 9, 9])
    xc = create_functional(XC('LDA', collinear=False), grid)
    soc = True
    err = 1.0e-6
    D_sii = np.zeros((4, setup.ni, setup.ni), complex)
    D_sii[0, 0, 0] = 2

    # Calculate non-local potential energy for some random state

    P_sm = np.array([[0.2 + 0.3j, 0.1 + 0.2j, 0.3 + 0.4j],
                     [0.4 - 0.5j, 0.2 + 0.3j, 0.6 - 0.7j]])

    D_ssmm = np.einsum('si, zj -> szij', P_sm.conj(), P_sm)
    D_sii[:, 1:4, 1:4] = [D_ssmm[0, 0] + D_ssmm[1, 1],
                          D_ssmm[0, 1] + D_ssmm[1, 0],
                          -1j * (D_ssmm[0, 1] - D_ssmm[1, 0]),
                          D_ssmm[0, 0] - D_ssmm[1, 1]]

    def calc_energies(D_sii):
        _, energies = calculate_non_local_potential1(
            setup, xc, D_sii, np.zeros(1), soc, [], 0)
        return energies

    energies1 = calc_energies(D_sii)

    assert energies1['kinetic_correction'] == pytest.approx(
        0.04340694003, abs=err)
    assert energies1['coulomb'] == pytest.approx(-5.5575386716, abs=err)
    assert energies1['zero'] == pytest.approx(-2.432694074696, abs=err)
    assert energies1['xc'] == pytest.approx(1.5938337327, abs=err)

    # Rotate the state 90 degrees around the z-axis (x -> y, y -> -x, z -> z).
    # Assert that this does not change the energies.

    # First rotate the spins
    P_sm = [P_sm[0, :] * (1 - 1j), P_sm[1, :] * (1 + 1j)] / np.sqrt(2)
    # Then rotate the density
    P_sm = np.matmul([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], P_sm.T).T

    D_ssmm = np.einsum('si, zj -> szij', P_sm.conj(), P_sm)
    D_sii[:, 1:4, 1:4] = [D_ssmm[0, 0] + D_ssmm[1, 1],
                          D_ssmm[0, 1] + D_ssmm[1, 0],
                          -1j * (D_ssmm[0, 1] - D_ssmm[1, 0]),
                          D_ssmm[0, 0] - D_ssmm[1, 1]]

    energies2 = calc_energies(D_sii)

    assert energies2['kinetic_correction'] == pytest.approx(
        energies1['kinetic_correction'], abs=err)
    assert energies2['coulomb'] == pytest.approx(energies1['coulomb'], abs=err)
    assert energies2['zero'] == pytest.approx(energies1['zero'], abs=err)
    assert energies2['xc'] == pytest.approx(energies1['xc'], abs=err)

    # Assert that only the kinetic energy changes when the density
    # matrix is forced to be real

    energies3 = calc_energies(D_sii.real)

    assert energies3['kinetic_correction'] == pytest.approx(
        0.0446930609623, abs=err)
    assert energies3['coulomb'] == pytest.approx(energies1['coulomb'], abs=err)
    assert energies3['zero'] == pytest.approx(energies1['zero'], abs=err)
    assert energies3['xc'] == pytest.approx(energies1['xc'], abs=err)
