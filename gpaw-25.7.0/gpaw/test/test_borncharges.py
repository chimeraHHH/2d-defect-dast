import pytest
import numpy as np
from glob import glob

from ase.build import mx2, molecule
from gpaw import GPAW
from gpaw.borncharges import born_charges_wf


def test_born_charges_wf(in_tmp_dir, gpw_files, cleanup=True):
    gpw_file = gpw_files["hbn_pw_nosym"]
    calc = GPAW(gpw_file, txt=None)
    atoms = calc.get_atoms()

    Z_t = np.array([np.diag([-2.83, -2.83, -0.35]),
                    np.diag([2.83, 2.83, 0.35])])

    Z_avv = born_charges_wf(atoms, calc, cleanup=cleanup)['Z_avv']

    assert Z_t == pytest.approx(Z_avv, abs=1e-2)

    if cleanup:
        flist = glob('disp*.gpw')
        assert len(flist) == 0


def test_born_charges_symmetry(gpw_files):
    gpw_file = gpw_files["hbn_pw"]
    calc = GPAW(gpw_file, txt=None)
    atoms = calc.get_atoms()

    with pytest.raises(AssertionError):
        born_charges_wf(atoms, calc)


@pytest.mark.parametrize('atoms', [mx2('MoS2', vacuum=7.5),
                                   molecule('H2', cell=[10, 10, 10])])
def test_born_charges_ionic_wf(in_tmp_dir, atoms):

    atoms.center()

    Z_a = atoms.numbers
    Z_t = np.array([za * np.eye(3) for za in Z_a])

    # we do not need a calculator
    Z_avv = born_charges_wf(atoms, None, ionic_only=True)['Z_avv']

    assert Z_t == pytest.approx(Z_avv)
