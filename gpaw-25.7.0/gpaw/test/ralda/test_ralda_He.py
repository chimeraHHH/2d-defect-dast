import pytest
from ase import Atoms

from gpaw import GPAW, FermiDirac
from gpaw.mpi import serial_comm
from gpaw.xc.fxc import FXCCorrelation
from gpaw.xc.rpa import RPACorrelation

pytestmark = pytest.mark.usefixtures('module_tmp_path')


@pytest.fixture(scope='module')
def calc():
    a = 3.0
    atoms = Atoms('He', cell=[a, a, a], pbc=True)
    calc = GPAW(mode=dict(name='pw', ecut=200),
                kpts=dict(size=(2, 2, 2), gamma=True),
                nbands=2,
                txt='gpaw.txt',
                occupations=FermiDirac(0.001),
                # FXCCorrelation needs a serial-comm GPAW object:
                communicator=serial_comm)
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.diagonalize_full_hamiltonian(nbands=20)
    return calc


whyskip_rapbe = 'https://gitlab.com/gpaw/gpaw/-/issues/723'


@pytest.mark.rpa
@pytest.mark.response
@pytest.mark.parametrize('xc, kwargs, ref_energy', [
    ('RPA', dict(nlambda=16), -0.1054),
    ('rALDA', dict(unit_cells=[1, 1, 2]), -0.0560),
    pytest.param('rAPBE', dict(unit_cells=[1, 1, 2]), -0.0523,
                 marks=pytest.mark.skip(reason=whyskip_rapbe)),
    ('rALDA', dict(avg_scheme='wavevector'), -0.0241),
    ('rAPBE', dict(avg_scheme='wavevector'), -0.0288),
])
def test_ralda_ralda_energy_He(in_tmp_dir, scalapack, calc, xc, kwargs,
                               ref_energy):
    ecuts = [20, 30]
    fxc = FXCCorrelation(calc, xc=xc, ecut=ecuts, **kwargs)
    energy = fxc.calculate()[-1]

    assert energy == pytest.approx(ref_energy, abs=0.001)

    if xc == 'RPA':
        rpa = RPACorrelation(calc, nfrequencies=8, ecut=ecuts)
        E_rpa1 = rpa.calculate()[-1]
        assert E_rpa1 == pytest.approx(ref_energy, abs=0.001)
