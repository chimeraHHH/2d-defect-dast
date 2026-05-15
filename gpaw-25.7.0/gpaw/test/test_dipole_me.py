from ase.units import Bohr
from gpaw.new.ase_interface import GPAW
from gpaw.utilities.dipole import dipole_matrix_elements_from_calc


def test_dipole_me(gpw_files):
    """Check dipole matrix-elements for H2 molecule."""
    calc = GPAW(gpw_files['h2_pw'])

    # Method 1: evaluate all-electron wave functions on fine grid:
    psi0, psi1 = (
        calc.dft.ibzwfs.get_all_electron_wave_function(n)
        for n in [0, 1])
    if psi0 is not None:
        d1_v = (psi0 * psi1).moment() * Bohr

    # Method 2: use pseudo wave function + PAW corrections:
    d2_nnv = dipole_matrix_elements_from_calc(calc, n1=0, n2=2)[0]

    assert abs(d2_nnv[0, 0] - calc.atoms.cell.sum(0) / 2).max() < 0.04
    assert abs(d2_nnv[1, 1] - calc.atoms.cell.sum(0) / 2).max() < 0.04
    if psi0 is not None:
        assert abs(d2_nnv[0, 1] - d1_v).max() < 1e-3
