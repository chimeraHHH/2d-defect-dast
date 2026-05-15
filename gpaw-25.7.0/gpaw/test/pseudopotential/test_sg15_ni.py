import pytest
from gpaw.setup import create_setup


def test_sg15_ni():
    try:
        s = create_setup('Ni', 'PBE', type='sg15')
    except FileNotFoundError:
        pytest.skip('No SG15 PP for Ni found')
    f_sJ = s.calculate_initial_occupation_numbers(2.0, False, 0.0, 2)
    print(f_sJ)
    up, dn = f_sJ.sum(axis=1)
    assert up - dn == pytest.approx(2.0)
