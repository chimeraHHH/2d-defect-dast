import gpaw.solvation as solv
from ase import Atoms
import numpy as np
import pytest


def test_solvation_nan_radius():
    atoms = Atoms('H')
    atoms.center(vacuum=3.0)
    kwargs = solv.get_HW14_water_kwargs()

    kwargs['cavity'].effective_potential.atomic_radii = {'H': np.nan}
    atoms.calc = solv.SolvationGPAW(mode='fd', xc='LDA', h=0.24, **kwargs)
    with pytest.raises(ValueError):
        atoms.get_potential_energy()
