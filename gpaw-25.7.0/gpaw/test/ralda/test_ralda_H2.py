import pytest

from gpaw.mpi import world
from gpaw.xc.fxc import FXCCorrelation


@pytest.mark.rpa
@pytest.mark.response
@pytest.mark.parametrize('gpw, kwargs, ref_energy, abstol', [
    ('h2_pw280_fulldiag', dict(xc='rALDA', nblocks=min(4, world.size)),
     -0.8509, 1e-3),
    ('h2_pw280_fulldiag', dict(xc='rAPBE'), -0.74555, 1e-3),
    ('h_pw280_fulldiag', dict(xc='rALDA'), 0.002757, 1e-4),
    ('h_pw280_fulldiag', dict(xc='rAPBE', nblocks=min(4, world.size)),
     0.01365, 1e-4)])
def test_ralda_energy_H2(in_tmp_dir, gpw_files, scalapack, gpw,
                         kwargs,
                         ref_energy, abstol):
    gpw = gpw_files[gpw]
    fxc = FXCCorrelation(gpw, **kwargs, ecut=[200])

    energy = fxc.calculate()
    assert energy == pytest.approx(ref_energy, abs=abstol)
