import pytest
import numpy as np

from gpaw import GPAW, restart


def get_restart_test_values(calc):
    atoms = calc.get_atoms()
    # XXX the forces must be evaluated first because a force evaluation
    # somehow affects (changes) both the energy and the wf when restarting
    # from a gpw file. An issue was created (#1051) documenting this unexpected
    # and rather bizarre behavior.
    f = atoms.get_forces()
    e = atoms.get_potential_energy()
    m = atoms.get_magnetic_moments()
    wf = calc.get_pseudo_wave_function(band=1)
    eig0 = calc.get_eigenvalues(spin=0)
    eig1 = calc.get_eigenvalues(spin=1)

    return wf, e, f, m, eig0, eig1


@pytest.fixture(params=['na3_pw_restart', 'na3_fd_restart'])
def gpwfile(request, gpw_files):
    return gpw_files[request.param]


@pytest.mark.old_gpaw_only
def test_fileio_restart(in_tmp_dir, gpwfile):
    # gpw restart file is written in fixture
    calc = GPAW(gpwfile)

    wf0, e0, f0, m0, eig00, eig01 = get_restart_test_values(calc=calc)

    # Write the restart file(s):
    calc.write('tmp1.gpw')
    calc.write('tmp.gpw', 'all')

    # Try restarting:
    _, calc = restart('tmp.gpw', txt=None)
    wf1, e1, f1, m1, eig10, eig11 = get_restart_test_values(calc=calc)

    # compare that the values are absolutely equal
    print(e0, e1)
    assert e0 == pytest.approx(e1, abs=1e-10)
    print(f0, f1)
    for ff0, ff1 in zip(f0, f1):
        err = np.linalg.norm(ff0 - ff1)
        assert err <= 1e-10
    print(m0, m1)
    for mm0, mm1 in zip(m0, m1):
        assert mm0 == pytest.approx(mm1, abs=1e-10)
    print("A", eig00, eig10)
    for eig0, eig1 in zip(eig00, eig10):
        assert eig0 == pytest.approx(eig1, abs=1e-10)
    print("B", eig01, eig11)
    for eig0, eig1 in zip(eig01, eig11):
        assert eig0 == pytest.approx(eig1, abs=1e-10)
    assert abs(wf1 - wf0).max() == pytest.approx(0, abs=1e-14)

    # Check that after restart, everything is writable
    calc.write("tmp3.gpw")
    calc.write("tmp4.gpw", "all")
