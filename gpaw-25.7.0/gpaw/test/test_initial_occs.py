import pytest
from gpaw.setup import create_setup


@pytest.mark.parametrize('symbol, M, n',
                         [('Pd', 1, 6),  # 4d -> 5s
                          ('Fe', 3, 5),
                          ('V', 3, 5),
                          ('Ti', 2, 5)])
def test_initial_occs(symbol, M, n):
    s = create_setup(symbol)
    f_si = s.calculate_initial_occupation_numbers(magmom=M,
                                                  hund=False,
                                                  charge=0,
                                                  nspins=2)
    print(f_si)
    magmom = (f_si[0] - f_si[1]).sum()
    assert abs(magmom - M) < 1e-10
    N = ((f_si[0] - f_si[1]) != 0).sum()
    assert n == N, 'Wrong # of values have changed'


def test_ca():
    """Our Ca PAW-potential only has completely filled partial waves
    (3s, 3p and 4s).  So, a magmom of 1 can't be accomodated.  GPAW
    should still work though!  Same problem for Ne.
    """
    from ase.data import chemical_symbols
    for sy in chemical_symbols:
        try:
            s = create_setup(sy)
        except FileNotFoundError:
            continue
        f_si = s.calculate_initial_occupation_numbers(
            magmom=1, hund=False, charge=0, nspins=2)
        print(sy, f_si)
