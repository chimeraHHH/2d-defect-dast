import pytest
from gpaw.utilities.elpa import LibElpa
from ase.build import bulk
from gpaw import GPAW
from gpaw.mpi import world

# Run single SCF iteration and compare total energy with elpa vs. scalapack

pytestmark = pytest.mark.skipif(not LibElpa.have_elpa(),
                                reason='not LibElpa.have_elpa()')


def test_lcao_lcao_elpa_kpts(gpaw_new):
    if gpaw_new and world.size == 8:
        pytest.skip('Not implementted')

    energies = []

    for elpasolver in [None, '1stage', '2stage']:
        atoms = bulk('Al')
        calc = GPAW(mode='lcao', basis='sz(dzp)',
                    kpts=[2, 2, 2],
                    parallel=dict(sl_auto=True,
                                  use_elpa=elpasolver is not None,
                                  band=2 if world.size > 4 else 1,
                                  kpt=2 if world.size > 2 else 1,
                                  elpasolver=elpasolver),
                    convergence={'maximum iterations': 2},
                    txt='-')
        atoms.calc = calc
        E = atoms.get_potential_energy()
        energies.append(E)

        err = abs(E - energies[0])
        assert err < 1e-10, ' '.join(['err',
                                      str(err), 'energies:', str(energies)])
