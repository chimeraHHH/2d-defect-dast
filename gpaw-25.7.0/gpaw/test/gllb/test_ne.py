import pytest
from ase import Atoms, Atom
from gpaw import GPAW
from gpaw.atom.generator import Generator
from gpaw.atom.configurations import parameters
from gpaw.mpi import world


@pytest.mark.gllb
@pytest.mark.libxc
def test_gllb_ne(in_tmp_dir, add_cwd_to_setup_paths):
    atom = 'Ne'

    for xcname in ['GLLBSC', 'GLLB']:
        if world.rank == 0:
            g = Generator(atom, xcname=xcname, scalarrel=False, nofiles=True)
            g.run(**parameters[atom])
            eps = g.e_j[-1]
        else:
            eps = 0.0
        eps = world.sum_scalar(eps)
        world.barrier()

        a = 5
        Ne = Atoms([Atom(atom, (0, 0, 0))],
                   cell=(a, a, a), pbc=False)
        Ne.center()
        calc = GPAW(mode='fd', nbands=7, h=0.25, xc=xcname)
        Ne.calc = calc
        e = Ne.get_potential_energy()
        # Calculate the discontinuity
        homo, lumo = calc.get_homo_lumo()
        response = calc.hamiltonian.xc.response
        dxc_pot = response.calculate_discontinuity_potential(homo, lumo)
        response.calculate_discontinuity(dxc_pot)

        eps3d = calc.wfs.kpt_u[0].eps_n[3]
        # if world.rank == 0:
        assert eps == pytest.approx(eps3d, abs=1e-3)
        # Correct for small cell +0.14eV (since the test needs to be fast
        # in test suite)
        assert e + 0.147106041 == pytest.approx(0, abs=5e-2)
