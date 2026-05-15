from gpaw.tddft import TDDFT
from gpaw.tddft.ehrenfest import EhrenfestVelocityVerlet
import pytest


def test_tddft_ehrenfest_nacl(in_tmp_dir, gpw_files):

    td_calc = TDDFT(gpw_files['nacl_fd'], propagator='EFSICN')
    evv = EhrenfestVelocityVerlet(td_calc, 0.001)

    i = 0
    evv.get_energy()
    r = evv.x[1][2] - evv.x[0][2]
    # print 'E = ', [i, r, evv.Etot, evv.Ekin, evv.e_coulomb]

    for i in range(5):
        evv.propagate(1.0)
        evv.get_energy()
        r = evv.x[1][2] - evv.x[0][2]
        print('E = ', [i + 1, r, evv.Etot, evv.Ekin, evv.e_coulomb])

    assert r == pytest.approx(7.558883144, abs=1e-6)
    assert evv.Etot == pytest.approx(-0.10359175317017391, abs=1e-4)
