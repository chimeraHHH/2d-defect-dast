import pytest
from ase import Atoms

from gpaw import GPAW
from gpaw.bfield import BField


@pytest.mark.old_gpaw_only  # use extension=[BField(...)]
@pytest.mark.serial
def test_b_field(gpaw_new):
    """Hydrogen atom in a magnetic field."""
    L = 2.0
    atom = Atoms('H', magmoms=[1], cell=[L, L, L], pbc=True)
    atom.calc = GPAW(mode='pw')
    E1 = atom.get_potential_energy()
    a1, b1 = (atom.calc.get_eigenvalues(spin=s)[0] for s in [0, 1])

    B = 0.1

    # Collinear:
    atom.calc = GPAW(mode='pw',
                     external=BField((0, 0, B)))
    E2 = atom.get_potential_energy()
    a2, b2 = (atom.calc.get_eigenvalues(spin=s)[0] for s in [0, 1])

    assert E2 - E1 == pytest.approx(-B, abs=1e-6)
    assert a2 - a1 == pytest.approx(-B, abs=1e-6)
    assert b2 - b1 == pytest.approx(B, abs=1e-6)

    # Non-collinear:
    params = ({'magmoms': [(0.5, 0.5, 0)]}
              if gpaw_new else
              {'experimental': {'magmoms': [(0.5, 0.5, 0)]}})
    atom.calc = GPAW(mode='pw',
                     symmetry='off',
                     **params,
                     external=BField([B, 0, 0]))
    E3 = atom.get_potential_energy()
    a3, b3 = atom.calc.get_eigenvalues()

    assert E3 - E1 == pytest.approx(-B, abs=2e-5)
    assert a3 - a1 == pytest.approx(-B, abs=3e-5)

    if gpaw_new:
        totmom_v, magmom_av = (
            atom.calc.dft.density.calculate_magnetic_moments())
    else:
        totmom_v, magmom_av = atom.calc.density.estimate_magnetic_moments()
    assert totmom_v == pytest.approx([1, 0, 0], abs=1e-5)
    assert magmom_av[0] == pytest.approx([0.176, 0, 0], abs=1e-3)
