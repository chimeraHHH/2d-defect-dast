from ase import Atoms
from gpaw import GPAW
from gpaw.mpi import rank, size


def test_parallel_compare():
    a = 3.0
    H = Atoms('H',
              cell=(a, a, a),
              pbc=True,
              calculator=GPAW(mode='fd'))
    if size > 1:
        H.positions[0, 0] += 0.01 * rank
        try:
            H.get_potential_energy()
        except ValueError as e:
            err_ranks = e.args[1]
            assert (err_ranks == range(1, size)).all()
        else:
            assert False
