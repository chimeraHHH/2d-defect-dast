import pytest


@pytest.mark.old_gpaw_only
@pytest.mark.ci
def test_scf(atoms):
    atoms.calc.set(sj={'target_potential': 3.64,
                       'excess_electrons': 0.02,
                       'tol': 0.5})
    atoms.get_potential_energy()
    # XXX Not assertive enough
