import pytest
from gpaw.mpi import serial_comm
from gpaw import GPAW
from gpaw.xc.rpa import RPACorrelation


@pytest.mark.rpa
@pytest.mark.response
@pytest.mark.parametrize('gpwfilename', ['si_pw', 'si_noisy_kpoints'])
def test_rpa_energy_si(in_tmp_dir, gpw_files, gpwfilename):
    calc = GPAW(gpw_files[gpwfilename], communicator=serial_comm)
    calc.diagonalize_full_hamiltonian(nbands=50)

    ecut = 50
    rpa = RPACorrelation(calc, qsym=False, nfrequencies=8, ecut=[ecut])
    E_rpa_noqsym = rpa.calculate()

    rpa = RPACorrelation(calc, qsym=True, nfrequencies=8, ecut=[ecut])
    E_rpa_qsym = rpa.calculate()

    assert E_rpa_qsym == pytest.approx(-12.61, abs=0.01)
    assert E_rpa_qsym == pytest.approx(E_rpa_noqsym)
