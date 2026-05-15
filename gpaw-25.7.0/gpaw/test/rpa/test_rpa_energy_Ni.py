import pytest
from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.mpi import serial_comm
from gpaw.xc.rpa import RPACorrelation
from gpaw.xc.fxc import FXCCorrelation


@pytest.mark.rpa
@pytest.mark.response
def test_rpa_rpa_energy_Ni(in_tmp_dir):
    Ni = bulk('Ni', 'fcc')
    Ni.set_initial_magnetic_moments([0.7])

    calc = GPAW(mode='pw',
                kpts=(3, 3, 3),
                occupations=FermiDirac(0.001),
                setups={'Ni': '10'},
                communicator=serial_comm)
    Ni.calc = calc
    Ni.get_potential_energy()
    calc.diagonalize_full_hamiltonian(nbands=50)

    rpa = RPACorrelation(calc, nfrequencies=8, skip_gamma=True, ecut=[50])
    E_rpa = rpa.calculate()

    fxc = FXCCorrelation(calc, nlambda=16, nfrequencies=8, skip_gamma=True,
                         ecut=[50])
    E_fxc = fxc.calculate()

    assert E_rpa == pytest.approx(-7.826, abs=0.01)
    assert E_fxc == pytest.approx(-7.826, abs=0.01)
