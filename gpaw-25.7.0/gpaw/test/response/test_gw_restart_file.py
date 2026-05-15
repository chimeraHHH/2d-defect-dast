import pytest
import numpy as np
from gpaw.response.g0w0 import G0W0
from gpaw.mpi import world


class FragileG0W0(G0W0):
    def calculate_q(self, *args, **kwargs):
        if not hasattr(self, 'doom'):
            self.doom = 0
        self.doom += 1  # Advance doom
        if self.doom == 12:
            raise ValueError('Cthulhu awakens')
        G0W0.calculate_q(self, *args, **kwargs)


@pytest.mark.response
def test_restart_file(in_tmp_dir, gpw_files):
    kwargs = dict(bands=(3, 5),
                  nbands=9,
                  nblocks=world.size,
                  ecut=40,
                  kpts=[0, 1])
    gw = FragileG0W0(gpw_files['bn_pw'], **kwargs)
    with pytest.raises(ValueError, match='Cthulhu*'):
        gw.calculate()

    assert gw.doom == 12

    # Use FragileG0W0 also in the restart.
    # The FragileG0W0 cannot by itself calculate the full thing because
    # calculate_q is called 16 times in total. Thus, it must be that
    # it was helped by the previous calculation.
    gw = FragileG0W0(gpw_files['bn_pw'], **kwargs)
    results = gw.calculate()

    gw = G0W0(gpw_files['bn_pw'], filename='referencecalc', **kwargs)
    results2 = gw.calculate()

    assert np.allclose(results['qp'], results2['qp'])
