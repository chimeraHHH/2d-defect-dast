import pytest
from gpaw import restart
from gpaw.solvation.sjm import SJM


# Test wrting and reading of the SJM object into the gpw file
@pytest.mark.old_gpaw_only
# @pytest.mark.ci
def test_gpw(in_tmp_dir, atoms):
    atoms.calc.set(sj={'target_potential': None})

    E1 = atoms.get_potential_energy()
    atoms.calc.write('test.gpw', mode='all')

    atoms2, calc = restart('test.gpw', Class=SJM)
    E2 = atoms2.get_potential_energy()
    h = atoms2.calc.stuff_for_hamiltonian[0].effective_potential
    assert h.unsolv_backside
    assert E1 == E2
