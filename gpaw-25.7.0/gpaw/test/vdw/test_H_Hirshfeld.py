"""Test Hirshfeld for spin/no spin consistency."""
import pytest
from ase import Atoms
from ase.parallel import parprint

from gpaw import GPAW, FermiDirac
from gpaw.analyse.hirshfeld import HirshfeldPartitioning


@pytest.mark.old_gpaw_only
def test_vdw_H_Hirshfeld():
    h = 0.25
    box = 3

    atoms = Atoms('H')
    atoms.center(vacuum=box)

    volumes = []
    for spinpol in [False, True]:
        calc = GPAW(mode='fd',
                    h=h,
                    occupations=FermiDirac(0.1, fixmagmom=spinpol),
                    spinpol=spinpol)
        calc.calculate(atoms)
        vol = HirshfeldPartitioning(calc).get_effective_volume_ratios()
        volumes.append(vol)
    parprint(volumes)
    assert volumes[0][0] == pytest.approx(volumes[1][0], abs=4e-9)
