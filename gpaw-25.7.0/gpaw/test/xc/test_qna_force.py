import pytest
from ase.parallel import parprint
from gpaw import GPAW


@pytest.mark.old_gpaw_only
def test_xc_qna_force(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['Cu3Au_qna'], parallel=dict(domain=1))
    atoms = calc.get_atoms()
    # Displace atoms to have non-zero forces in the first place
    atoms[0].position[0] += 0.1

    dx_array = [-0.005, 0.000, 0.005]
    E = []

    for i, dx in enumerate(dx_array):

        atoms[0].position[0] += dx
        atoms.calc = calc
        E.append(atoms.get_potential_energy(force_consistent=True))
        if i == 1:
            F = atoms.get_forces()[0, 0]
        atoms[0].position[0] -= dx

    F_num = -(E[-1] - E[0]) / (dx_array[-1] - dx_array[0])
    F_err = F_num - F

    parprint('Analytical force = ', F)
    parprint('Numerical  force = ', F_num)
    parprint('Difference       = ', F_err)
    assert abs(F_err) < 0.01, F_err
    eerr = abs(E[-1] - 277.546724 + 0.12677918230332352)
    assert eerr < 1e-3, eerr
