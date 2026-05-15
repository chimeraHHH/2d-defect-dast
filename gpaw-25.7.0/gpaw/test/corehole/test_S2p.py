import pytest
from ase.build import molecule
from ase.units import Bohr
from gpaw import GPAW
from gpaw.test import gen


@pytest.mark.old_gpaw_only
def test_S2p():
    """Compare number of electrons for different channels with 2p corehole"""
    atoms = molecule('SH2')
    atoms.center(vacuum=2)

    setup = gen('S', name='fch2p', corehole=(2, 1, 1))

    grf = 1
    calc = GPAW(mode='fd', h=0.3, spinpol=True,
                setups={'S': setup})
    atoms.calc = calc
    atoms.get_potential_energy()

    for spin, nelectrons in zip([0, 1], [8, 9]):
        n_g = calc.get_all_electron_density(spin=spin, gridrefinement=grf)
        ne = calc.density.gd.integrate(
            n_g, global_integral=False) * (Bohr / grf)**3
        assert ne == pytest.approx(nelectrons, abs=1e-5)
