from ase.spacegroup import crystal
from gpaw import GPAW
from gpaw import PW
import pytest


def test_symmetry_fractional_translations_big():
    'cristobalite'
    # no. 92 - tetragonal

    a = 5.0833674
    c = 7.0984738
    p0 = (0.2939118, 0.2939118, 0.0)
    p1 = (0.2412656, 0.0931314, 0.1739217)

    atoms = crystal(['Si', 'O'], basis=[p0, p1],
                    spacegroup=92, cellpar=[a, a, c, 90, 90, 90])

    # with fractional translations
    calc = GPAW(mode=PW(200),
                xc='LDA',
                kpts=(3, 3, 2),
                nbands=32,
                symmetry={'symmorphic': False},
                gpts=(16, 16, 20),
                eigensolver='rmm-diis')

    atoms.calc = calc
    energy_fractrans = atoms.get_potential_energy()

    assert len(calc.wfs.kd.ibzk_kc) == 3
    assert len(calc.wfs.kd.symmetry.op_scc) == 8

    # without fractional translations
    calc = GPAW(mode=PW(200),
                xc='LDA',
                kpts=(3, 3, 2),
                nbands=32,
                gpts=(16, 16, 20),
                eigensolver='rmm-diis')

    atoms.calc = calc
    energy_no_fractrans = atoms.get_potential_energy()

    assert len(calc.wfs.kd.ibzk_kc) == 6
    assert len(calc.wfs.kd.symmetry.op_scc) == 2

    assert energy_fractrans == pytest.approx(energy_no_fractrans, abs=1e-6)
