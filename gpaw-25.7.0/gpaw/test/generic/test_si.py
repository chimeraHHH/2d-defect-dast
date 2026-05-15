from gpaw import GPAW
import pytest
import numpy as np


def test_generic_si(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['si8_fd'])
    eigs = calc.get_eigenvalues(kpt=0)
    e1 = calc.get_potential_energy()

    calc = GPAW(gpw_files['si8_fd']).fixed_density()
    eigs2 = calc.get_eigenvalues(kpt=0)
    print('Orginal', eigs)
    print('Fixdensity', eigs2)
    print('Difference', eigs2 - eigs)

    assert np.fabs(eigs2 - eigs)[:-1].max() < 3e-5
    assert e1 == pytest.approx(-36.767, abs=0.003)
