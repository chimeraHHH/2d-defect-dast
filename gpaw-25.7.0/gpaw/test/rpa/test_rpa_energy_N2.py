import pytest

from ase.units import Hartree

from gpaw import GPAW
from gpaw.mpi import serial_comm
from gpaw.xc.rpa import RPACorrelation
from gpaw.hybrids.energy import non_self_consistent_energy as nsc_energy


@pytest.mark.rpa
@pytest.mark.response
def test_rpa_rpa_energy_N2(in_tmp_dir, gpw_files, scalapack):
    ecut = 25

    N2_calc = GPAW(gpw_files['n2_pw'], communicator=serial_comm)

    E_n2_pbe = N2_calc.get_potential_energy()
    E_n2_hf = nsc_energy(N2_calc, 'EXX')

    rpa = RPACorrelation(N2_calc, nfrequencies=8, ecut=[ecut])
    data = rpa.calculate_all_contributions()
    energy_w = data.energy_wi[:, 0]  # we only have 1 ecut
    assert energy_w == pytest.approx(
        [-1.7415, -1.7336, -1.4659, -0.4209, -0.0295, -0.0010]
        + [0.0] * 2, abs=0.001)
    E_n2_rpa = data.energy_i * Hartree

    N_calc = GPAW(gpw_files['n_pw'], communicator=serial_comm)
    E_n_pbe = N_calc.get_potential_energy()
    E_n_hf = nsc_energy(N_calc, 'EXX')

    rpa = RPACorrelation(N_calc, nfrequencies=8, ecut=[ecut])
    E_n_rpa = rpa.calculate()

    print('Atomization energies:')
    print('PBE: ', E_n2_pbe - 2 * E_n_pbe)
    print('HF: ', E_n2_hf - 2 * E_n_hf)
    print('HF+RPA: ', E_n2_hf - 2 * E_n_hf + E_n2_rpa[0] - 2 * E_n_rpa[0])

    assert E_n2_rpa - 2 * E_n_rpa == pytest.approx(-1.68, abs=0.02)
    assert (E_n2_hf - 2 * E_n_hf) == pytest.approx(
        [-10.47, 3.03, 0, 0, -0.06, 2.79], abs=0.01)
