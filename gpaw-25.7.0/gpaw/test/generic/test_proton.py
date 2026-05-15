"""Test calculation for bare proton.

Also, its interaction with an external potential in the form of a point charge
is tested.
"""
import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Hartree

from gpaw import GPAW
from gpaw.external import PointChargePotential


@pytest.mark.old_gpaw_only
def test_generic_proton(in_tmp_dir):
    a = 4.5
    H = Atoms('H', [(a / 2, a / 2, a / 2)],
              pbc=0,
              cell=(a, a, a))
    H.calc = GPAW(mode='fd', nbands=1, h=0.2, charge=1, txt='H.txt')
    e0 = H.get_potential_energy()
    assert abs(e0 + H.calc.get_reference_energy()) < 0.014

    # Test the point charge potential with a smooth cutoff:
    pcp = PointChargePotential([-1], rc2=5.5, width=1.5)
    H.calc = H.calc.new(external=pcp)
    E = []
    F = []
    D = np.linspace(2, 6, 30)
    for d in D:
        pcp.set_positions([[a / 2, a / 2, a / 2 + d]])
        e = H.get_potential_energy() - e0
        f1 = H.get_forces()
        f2 = pcp.get_forces(H.calc)
        eref = -1 / d * Bohr * Hartree
        print(d, e, eref, abs(f1 + f2).max())
        if d < 4.0:
            error = e + 1 / d * Bohr * Hartree
            assert abs(error) < 0.01, error
        assert abs(f1 + f2).max() < 0.01
        E.append(e)
        F.append(f1[0, 2])

    E = np.array(E)
    FF = (E[2:] - E[:-2]) / (D[2] - D[0])  # numerical force
    print(abs(F[1:-1] - FF).max())
    assert abs(F[1:-1] - FF).max() < 0.1

    if not True:
        import matplotlib.pyplot as plt
        plt.plot(D, E)
        plt.plot(D, -1 / D * Bohr * Hartree)
        plt.plot(D, F)
        plt.plot(D[1:-1], FF)
        plt.show()
