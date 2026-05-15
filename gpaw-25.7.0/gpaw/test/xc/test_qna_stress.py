import pytest
import numpy as np
from ase.parallel import parprint
from gpaw import GPAW
from gpaw.mpi import world


def numeric_stress(atoms, d=1e-6, component=None):
    cell = atoms.cell.copy()
    V = atoms.get_volume()
    for i in range(3):
        x = np.eye(3)
        if component == (i, i):
            x[i, i] += d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            eplus = atoms.get_potential_energy(force_consistent=True)

            x[i, i] -= 2 * d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            eminus = atoms.get_potential_energy(force_consistent=True)
            atoms.set_cell(cell, scale_atoms=True)
            return (eplus - eminus) / (2 * d * V)

        if (component == (i, (i - 2) % 3)) or (component == ((i - 2) % 3, i)):
            j = i - 2
            x[i, j] = d
            x[j, i] = d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            eplus = atoms.get_potential_energy(force_consistent=True)

            x[i, j] = -d
            x[j, i] = -d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            eminus = atoms.get_potential_energy(force_consistent=True)
            atoms.set_cell(cell, scale_atoms=True)
            return (eplus - eminus) / (4 * d * V)

    raise ValueError(f'Invalid component {component}')


@pytest.mark.old_gpaw_only
@pytest.mark.skipif(world.size > 1, reason='See #898')
def test_xc_qna_stress(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['Cu3Au_qna'])
    atoms = calc.get_atoms()
    atoms.set_cell(np.dot(atoms.cell,
                          [[1.02, 0, 0.03],
                           [0, 0.99, -0.02],
                           [0.2, -0.01, 1.03]]),
                   scale_atoms=True)

    s_analytical = atoms.get_stress(voigt=False)
    print(s_analytical)
    components = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    for componentid in [1]:
        component = components[componentid]
        s_numerical = numeric_stress(atoms, 1e-5, component)
        s_err = s_numerical - s_analytical.__getitem__(component)

        parprint('Analytical stress:', s_analytical.__getitem__(component))
        parprint('Numerical stress :', s_numerical)
        parprint('Error in stress  :', s_err)
        assert np.abs(s_err) < 0.002
