# this test should coverage the save and restore of
# fermi-levels when using fixmagmom:
#
# yes, fermi-level-splitting sounds a little bit strange
import numpy as np
import pytest
from ase import Atoms

from gpaw import GPAW, FermiDirac, MixerSum


def test_fermisplit(in_tmp_dir):
    calc = GPAW(mode='fd',
                occupations=FermiDirac(width=0.1, fixmagmom=True),
                mixer=MixerSum(beta=0.05, nmaxold=3, weight=50.0),
                convergence={'energy': 0.1, 'eigenstates': 1.5e-1,
                             'density': 1.5e-1})
    atoms = Atoms('Cr', pbc=False)
    atoms.center(vacuum=4)
    mm = [1] * 1
    mm[0] = 6.
    atoms.set_initial_magnetic_moments(mm)
    atoms.calc = calc
    atoms.get_potential_energy()

    ef1 = calc.get_fermi_levels().mean()
    efsplit1 = np.ptp(calc.get_fermi_levels())

    ef3 = calc.get_fermi_levels()
    calc.write('test.gpw')

    # check number one: is the splitting value saved?
    readtest = GPAW('test.gpw')
    ef2 = readtest.get_fermi_levels().mean()
    efsplit2 = np.ptp(readtest.get_fermi_levels())

    # numpy arrays
    ef4 = readtest.get_fermi_levels()

    # These values should be identic
    assert ef1 == pytest.approx(ef2, abs=1e-9)
    assert efsplit1 == pytest.approx(efsplit2, abs=1e-9)
    assert ef3.mean() == pytest.approx(ef1, abs=1e-9)
    assert ef3.mean() == pytest.approx(ef2, abs=1e-9)
    assert ef3.mean() == pytest.approx(ef4.mean(), abs=1e-9)
    assert ef3[0] - ef3[1] == pytest.approx(ef4[0] - ef4[1], abs=1e-9)
    assert efsplit1 == pytest.approx(ef4[0] - ef4[1], abs=1e-9)
