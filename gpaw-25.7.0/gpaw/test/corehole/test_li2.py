"""Test all electron density for right interpretation of coreholes"""
import pytest
from ase.build import molecule
from ase.units import Bohr
from gpaw import GPAW, PoissonSolver
from gpaw.mixer import Mixer
from gpaw.test import gen


@pytest.mark.old_gpaw_only
def test_aed_with_corehole_li():
    """Compare number of electrons for different channels with corehole"""
    li_setup = gen('Li', name='fch1s', corehole=(1, 0, 1), xcname='PBE')
    grf = 1
    atoms = molecule('Li2')
    atoms.center(vacuum=2.5)

    calc = GPAW(mode='fd',
                xc='PBE',
                mixer=Mixer(),
                setups={0: li_setup},
                charge=-1,
                poissonsolver=PoissonSolver('fd'))
    atoms.calc = calc
    atoms.get_potential_energy()

    n_sg = calc.get_all_electron_density(gridrefinement=grf)

    ne_sz = calc.density.gd.integrate(
        n_sg, global_integral=False) * (Bohr / grf)**3
    assert ne_sz == pytest.approx(6.0, abs=1e-5)

    atoms.set_initial_magnetic_moments([0.66, .34])
    calc = calc.new(spinpol=True)
    atoms.calc = calc
    atoms.get_potential_energy()

    for sz in range(2):
        n_sg = calc.get_all_electron_density(spin=sz, gridrefinement=grf)
        ne_sz = calc.density.gd.integrate(
            n_sg, global_integral=False) * (Bohr / grf)**3
        assert ne_sz == pytest.approx(3.0, abs=1e-5)
