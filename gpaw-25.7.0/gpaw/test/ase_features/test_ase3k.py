import pytest
from ase import Atoms
from ase.io import read
from ase.units import Ha

from gpaw import GPAW


@pytest.mark.ci
def test_no_cell():
    with pytest.raises(ValueError):
        H = Atoms('H', calculator=GPAW(mode='fd'))
        H.get_potential_energy()


@pytest.mark.parametrize('name', ['h2_pw', 'bcc_li_lcao'])
def test_read_txt(in_tmp_dir, gpw_files, name):
    gpw = gpw_files[name]
    calc = GPAW(gpw)
    e0 = calc.get_atoms().get_potential_energy()
    atoms = read(gpw.with_suffix('.txt'))
    e = atoms.get_potential_energy()
    assert e == pytest.approx(e0)
    if not calc.old:
        assert atoms.calc.energy_contributions['kinetic'] == pytest.approx(
            calc.dft.energies.kinetic * Ha)
