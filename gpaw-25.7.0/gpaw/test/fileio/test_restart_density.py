import numpy as np
from gpaw import GPAW, restart
import pytest
from ase.io.ulm import ulmopen


def get_restart_test_values(calc, skip_forces):
    # XXX the forces must be evaluated first because a force evaluation
    # somehow affects (changes) both the energy and the wf when restarting
    # from a gpw file. An issue was created (#1051) documenting this unexpected
    # and rather bizarre behavior.

    atoms = calc.get_atoms()
    if skip_forces:
        f = np.zeros((len(atoms), 3))
    else:
        f = atoms.get_forces()
    e = atoms.get_potential_energy()
    m = atoms.get_magnetic_moments()

    eig0 = calc.get_eigenvalues(spin=0)
    eig1 = calc.get_eigenvalues(spin=1)

    return e, f, m, eig0, eig1


def test_fileio_restart_density(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['na3_fd_density_restart'])

    # We don't care about forces working for new GPAW reading old gpw-file
    skip_forces = (not calc.old and
                   ulmopen(gpw_files['na3_fd_density_restart']).version < 4)
    e0, f0, m0, eig00, eig01 = get_restart_test_values(calc, skip_forces)

    # Write the restart file
    calc.write('tmp.gpw')

    # Try restarting from all the files
    atoms, calc = restart('tmp.gpw')
    e1, f1, m1, eig10, eig11 = get_restart_test_values(calc, skip_forces)

    print(e0, e1)
    assert e0 == pytest.approx(e1, abs=2e-3)
    print(f0, f1)
    for ff0, ff1 in zip(f0, f1):
        err = np.linalg.norm(ff0 - ff1)
        # for forces, we use larger tolerance
        assert err == pytest.approx(0.0, abs=4e-2)
    print(m0, m1)
    for mm0, mm1 in zip(m0, m1):
        assert mm0 == pytest.approx(mm1, abs=2e-3)
    print('A', eig00, eig10)
    for eig0, eig1 in zip(eig00, eig10):
        assert eig0 == pytest.approx(eig1, abs=5e-3)
    print('B', eig01, eig11)
    for eig0, eig1 in zip(eig01, eig11):
        assert eig0 == pytest.approx(eig1, abs=2e-2)

    # Check that after restart, everything is writable
    calc.write('tmp2.gpw')
