import numpy as np
import pytest
from ase.build import bulk

from gpaw import GPAW, PW, Mixer
from gpaw.mpi import world


@pytest.mark.stress
def test_pw_si_stress(in_tmp_dir, gpaw_new):
    xc = 'PBE'
    si = bulk('Si')
    si.calc = GPAW(mode=PW(200),
                   mixer=Mixer(0.7, 5, 50.0),
                   xc=xc,
                   kpts=(1, 1, 2),  # Run (1, 1, 2) to avoid gamma pt code
                   convergence={'energy': 1e-8},
                   parallel={'domain': min(2, world.size)},
                   txt='si_stress.txt')

    si.set_cell(np.dot(si.cell,
                       [[1.02, 0, 0.03],
                        [0, 0.99, -0.02],
                        [0.2, -0.01, 1.03]]),
                scale_atoms=True)

    si.get_potential_energy()

    if not gpaw_new:
        # Trigger nasty bug (fixed in !486):
        si.calc.wfs.pt.blocksize = si.calc.wfs.pd.maxmyng - 1

    s_analytical = si.get_stress()
    s_ref = [-0.16569446, -0.07630128, -0.1266625,
             -0.06144752, -0.02055657, 0.04574812]
    # si.calc.calculate_numerical_stress(si, 1e-5)
    print(s_analytical)
    s_err = s_analytical - s_ref
    assert np.all(abs(s_err) < 1e-4)
