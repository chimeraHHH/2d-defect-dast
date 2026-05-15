import pytest
from gpaw import GPAW, restart
from gpaw.mom import prepare_mom_calculation


@pytest.mark.mom
def test_mom_lcao_smearing(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['co_mom'])
    E_gs = calc.get_potential_energy()

    f_sn = []
    for spin in range(calc.get_number_of_spins()):
        f_n = calc.get_occupation_numbers(spin=spin)
        f_sn.append(f_n)

    ne0_gs = f_sn[0].sum()
    f_sn[0][3] -= 1.
    f_sn[0][5] += 1.

    # Test both MOM and fixed occupations with Gaussian smearing
    for i in [True, False]:
        atoms, calc = restart(gpw_files['co_mom'])

        # Excited-state calculation with Gaussian
        # smearing of the occupation numbers
        prepare_mom_calculation(calc, atoms,
                                numbers=f_sn,
                                width=0.01,
                                use_fixed_occupations=i)
        E_es = atoms.get_potential_energy()

        f_n0 = calc.get_occupation_numbers(spin=0)
        ne0_es = f_n0.sum()

        dE = E_es - E_gs
        dne0 = ne0_es - ne0_gs
        print(dE)
        assert dE == pytest.approx(9.8445603513, abs=0.01)
        assert dne0 == pytest.approx(0.0, abs=1e-16)
