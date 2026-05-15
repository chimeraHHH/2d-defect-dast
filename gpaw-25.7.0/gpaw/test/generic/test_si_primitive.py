import pytest
from ase.build import bulk
from gpaw import GPAW


def test_generic_si_primitive(in_tmp_dir):
    a = 5.475
    calc = GPAW(mode={'name': 'pw', 'ecut': 200},
                kpts=(4, 4, 4),
                occupations={'name': 'tetrahedron-method'},
                spinpol=True,
                nbands=5)
    atoms = bulk('Si', 'diamond', a=a)
    atoms.calc = calc
    E = atoms.get_potential_energy()
    assert E == pytest.approx(-11.8584, abs=0.001)

    homo, lumo = calc.get_homo_lumo()
    assert lumo - homo == pytest.approx(1.117, abs=0.002)

    calc.write('si_primitive.gpw', 'all')
    calc = GPAW('si_primitive.gpw',
                parallel={'domain': 1, 'band': 1},
                txt=None)
