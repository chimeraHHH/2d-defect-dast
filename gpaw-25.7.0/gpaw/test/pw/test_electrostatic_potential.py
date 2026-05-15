from gpaw import GPAW


def test_electrostatic_potential(gpw_files):
    """Make sure the whole array is returned also when parallelizing
    over plane-wave coefficients."""
    calc = GPAW(gpw_files['h2_pw'])
    v = calc.get_electrostatic_potential()
    assert v.shape == tuple(calc.hamiltonian.finegd.N_c)
