import numpy as np
import pytest
from ase import Atoms

from gpaw import GPAW
from gpaw.utilities.sic import NSCFSIC


@pytest.mark.old_gpaw_only
@pytest.mark.sic
@pytest.mark.serial
def test_sic_nscfsic(in_tmp_dir):
    atoms = ['He', 'Be']  # ,'Ne'] # Ne deviates already 2.5 eV
    EE = []
    EREF = [-79.4, -399.8, -3517.6]
    rng = np.random.default_rng(42)

    for a in atoms:
        s = Atoms(a)
        s.center(vacuum=4.0)
        calc = GPAW(mode='fd', h=0.15, txt=a + '.txt')
        s.calc = calc
        s.get_potential_energy()
        EE.append(NSCFSIC(calc, rng=rng).calculate())

    print("Difference to table VI of Phys. Rev. B 23, 5048 in eV")
    # https://journals.aps.org/prb/abstract/10.1103/PhysRevB.23.5048
    print("%10s%10s%10s%10s" % ("atom", "ref.", "gpaw", "diff"))
    for a, er, e in zip(atoms, EREF, EE):
        print("%10s%10.2f%10.2f%10.2f" % (a, er, e, er - e))
        assert er == pytest.approx(e, abs=0.1)
        # Arbitrary 0.1 eV tolerance for non-self consistent SIC
        # Note that Ne already deviates 2.5 eV
