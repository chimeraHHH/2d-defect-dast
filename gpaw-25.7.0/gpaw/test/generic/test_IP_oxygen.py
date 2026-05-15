import numpy as np
import pytest
from ase import Atoms
from gpaw import GPAW


def test_generic_IP_oxygen():
    a = 6.0
    O = Atoms('O',
              [(a / 2, a / 2 + 0.5, a / 2)],
              magmoms=[2],
              pbc=False,
              cell=(a, a + 1, a))
    O.calc = GPAW(mode='fd', gpts=(32, 36, 32), nbands=4)
    e0 = O.get_potential_energy()

    O.calc = GPAW(mode='fd', gpts=(32, 36, 32), nbands=4, charge=1)

    e1 = O.get_potential_energy()

    h0 = -26.43361787
    l0 = np.inf
    h1 = -35.55077324
    l1 = -20.81227577
    assert O.calc.get_homo_lumo(0) == pytest.approx([h0, l0], rel=0.001)
    assert O.calc.get_homo_lumo(1) == pytest.approx([h1, l1], rel=0.001)
    assert O.calc.get_homo_lumo() == pytest.approx([h0, l1], rel=0.001)

    print(e1 - e0)
    assert abs(e1 - e0 - 13.989) < 0.04

    energy_tolerance = 0.004
    assert e0 == pytest.approx(-1.88477, abs=energy_tolerance)
    assert e1 == pytest.approx(12.11080, abs=energy_tolerance)

    # The first ionization energy for LDA oxygen is from this paper:
    # In-Ho Lee, Richard M. Martin, Phys. Rev. B 56 7197 (1997)
