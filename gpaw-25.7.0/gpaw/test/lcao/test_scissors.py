import pytest
from ase import Atoms
from gpaw.new.ase_interface import GPAW
from gpaw.lcao.scissors import non_self_consistent_scissors_shift as nsc_shift
from gpaw.spinorbit import soc_eigenstates
from gpaw.mpi import world


def test_scissors():
    """Opens gap in one of two isolated H2 moleculs."""
    h2 = Atoms('2H2', [[0, 0, 0], [0, 0, 0.74],
                       [4, 0, 0], [4, 0, 0.74]])
    h2.center(vacuum=3.0)
    d = 1.0
    h2.calc = GPAW(mode='lcao',
                   basis='sz(dzp)',
                   eigensolver={'name': 'scissors',
                                'shifts': [(-d, d, 2)]},
                   symmetry='off',
                   parallel={'domain': 1,
                             'band': world.size,
                             'sl_auto': True},
                   txt=None)
    h2.get_potential_energy()
    eigs1 = h2.calc.get_eigenvalues()

    i, ii, iii, iv = eigs1
    assert ii - i == pytest.approx(d, abs=0.01)
    assert iv - iii == pytest.approx(d, abs=0.01)

    # Non self-consistent:
    eigs2 = nsc_shift([(-d, d, 2)], h2.calc.dft)[0, 0]
    assert eigs2 == pytest.approx(eigs1)

    # Check also fixed-density calculations:
    calc = h2.calc.fixed_density(kpts=[[0, 0, 0]])
    eigs3 = calc.get_eigenvalues()
    assert eigs3 == pytest.approx(eigs1)

    # SOC corrections:
    eigs4 = soc_eigenstates(calc).eigenvalues()[0]
    assert eigs4[::2] == pytest.approx(eigs1)
    assert eigs4[1::2] == pytest.approx(eigs1)


if __name__ == '__main__':
    test_scissors()
