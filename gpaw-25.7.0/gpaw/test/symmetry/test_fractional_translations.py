from ase.spacegroup import crystal
from gpaw import GPAW
from gpaw import PW
import pytest


def test_symmetry_fractional_translations(in_tmp_dir):
    'sishovite'
    # no 136 - tetragonal

    a = 4.233944
    c = 2.693264
    p0 = (0, 0, 0)
    p1 = (0.306866, 0.306866, 0.0)

    atoms = crystal(['Si', 'O'], basis=[p0, p1],
                    spacegroup=136, cellpar=[a, a, c, 90, 90, 90])

    # with fractional translation
    calc = GPAW(mode=PW(),
                xc='LDA',
                kpts=(3, 3, 3),
                nbands=28,
                symmetry={'symmorphic': False},
                gpts=(18, 18, 12),
                eigensolver='rmm-diis',
                txt='with.txt')

    atoms.calc = calc
    energy_fractrans = atoms.get_potential_energy()

    assert len(calc.wfs.kd.ibzk_kc) == 6
    assert len(calc.wfs.kd.symmetry.op_scc) == 16

    # without fractional translations
    calc = GPAW(mode=PW(),
                xc='LDA',
                kpts=(3, 3, 3),
                nbands=28,
                gpts=(18, 18, 12),
                eigensolver='rmm-diis',
                txt='without.txt')

    atoms.calc = calc
    energy_no_fractrans = atoms.get_potential_energy()

    assert len(calc.wfs.kd.ibzk_kc) == 8
    assert len(calc.wfs.kd.symmetry.op_scc) == 8

    assert energy_fractrans == pytest.approx(energy_no_fractrans, abs=1e-5)
