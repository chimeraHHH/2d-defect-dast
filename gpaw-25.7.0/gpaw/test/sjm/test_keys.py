import pytest


@pytest.mark.old_gpaw_only
@pytest.mark.ci
@pytest.mark.serial
def test_keys(atoms):
    atoms.calc.set(
        sj={'excess_electrons': 1.,
            'jelliumregion': {'top': -2.,
                              'bottom': -4.,
                              'thickness': None,
                              'fix_bottom': False},
            'target_potential': 3,
            'pot_ref': 'wf',
            'tol': 0.01,
            'always_adjust': True,
            'grand_output': False,
            'max_iters': 100,
            'max_step': 3.,
            'slope': 5,
            'mixer': 1,
            'fdt': False,
            'slope_regression_depth': 4,
            'dirichlet': False,
            'cip': {'autoinner': {'nlayers': None,
                                  'threshold': 0.0001},
                    'inner_region': None,
                    'mu_pzc': None,
                    'phi_pzc': None,
                    'filter': 10}})

    atoms.calc.initialize(atoms)
    atoms.calc.initialize_positions(atoms)
