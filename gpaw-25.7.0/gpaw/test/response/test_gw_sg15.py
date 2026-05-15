from ase.build import molecule
from gpaw import GPAW, PW
from gpaw.response.g0w0 import G0W0
from gpaw.mpi import world
import pytest


@pytest.mark.response
def test_gw_sg15(in_tmp_dir, add_cwd_to_setup_paths):
    from gpaw.test.pseudopotential.H_sg15 import pp_text
    if world.rank == 0:
        with open('H_ONCV_PBE-1.0.upf', 'w') as pp_file:
            print(pp_text, file=pp_file)
    world.barrier()
    sys = molecule('H2', pbc=True)
    sys.center(vacuum=2.5)

    calc = GPAW(setups='sg15',
                xc='PBE', mode=PW(ecut=300),
                convergence={'bands': 40,
                             'density': 1e-6},
                nbands=50, kpts=(2, 2, 2))
    sys.calc = calc
    sys.get_potential_energy()
    calc.write('gs.gpw', mode='all')

    gw = G0W0(calc='gs.gpw',
              bands=(1, 6),
              ecut=20,
              nblocksmax=True,
              filename='H2_g0w0_b11-15')
    results = gw.calculate()
    refs = pytest.approx([1.58177132, 3.33595453, 3.87312694, 4.95358262,
                         6.09555254], abs=2e-5)
    assert results['qp'][0][0] == refs
