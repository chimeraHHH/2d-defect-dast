# see changeset 4891
import pytest
import numpy as np
from ase import Atoms
from gpaw import GPAW
from gpaw.mpi import world


def test_parallel_kptpar(in_tmp_dir):
    a = 2.5
    H = Atoms('H', cell=[a, a, a], pbc=True)

    energy_tolerance = 0.0006

    base_params = dict(mode='fd', kpts=[6, 6, 1], spinpol=True)
    if world.size >= 3:
        calc = GPAW(**base_params,
                    parallel={'domain': world.size},
                    txt='H-a.txt')
        H.calc = calc
        e1 = H.get_potential_energy()
        assert H.calc.wfs.kd.comm.size == 1

        assert e1 == pytest.approx(-2.23708481, abs=energy_tolerance)

        if world.rank < 3:
            comm = world.new_communicator(np.array([0, 1, 2]))
            H.calc = GPAW(**base_params,
                          communicator=comm,
                          txt='H-b.txt')
            e2 = H.get_potential_energy()
            assert H.calc.wfs.kd.comm.size == 3
            assert e2 == pytest.approx(e1, abs=5e-9)
        else:
            comm = world.new_communicator(np.array(range(3, world.size)))
            H.calc = GPAW(**base_params,
                          communicator=comm,
                          parallel={'kpt': comm.size},
                          txt='H-b2.txt')
            e2 = H.get_potential_energy()
            assert e2 == pytest.approx(e1, abs=5e-9)
