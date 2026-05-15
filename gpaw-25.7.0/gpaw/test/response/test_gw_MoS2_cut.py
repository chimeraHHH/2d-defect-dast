import pytest
from ase import Atoms
from ase.lattice.hexagonal import Hexagonal
from gpaw import GPAW, FermiDirac
from gpaw.response.g0w0 import G0W0


@pytest.fixture
def gpwfile(in_tmp_dir):
    calc = GPAW(
        mode='pw',
        xc='PBE',
        nbands=16,
        convergence={'bands': 15},
        setups={'Mo': '6'},
        occupations=FermiDirac(0.001),
        kpts={'size': (3, 3, 1), 'gamma': True})

    a = 3.1604
    c = 10.0

    cell = Hexagonal(symbol='Mo',
                     latticeconstant={'a': a, 'c': c}).get_cell()
    layer = Atoms(symbols='MoS2', cell=cell, pbc=[True, True, False],
                  scaled_positions=[(0, 0, 0.5),
                                    (2 / 3, 1 / 3, 0.3 + 0.5),
                                    (2 / 3, 1 / 3, -0.3 + 0.5)])

    pos = layer.get_positions()
    pos[1][2] = pos[0][2] + 3.172 / 2
    pos[2][2] = pos[0][2] - 3.172 / 2
    layer.set_positions(pos)
    layer.calc = calc
    layer.get_potential_energy()
    fname = 'MoS2.gpw'
    calc.write(fname, mode='all')
    return fname


@pytest.mark.response
@pytest.mark.parametrize('integrate_gamma', ['sphere', 'reciprocal2D',
                                             '1BZ2D'])
def test_response_gw_MoS2_cut(scalapack, gpwfile, integrate_gamma):
    gw = G0W0(gpwfile,
              'gw-test',
              nbands=15,
              ecut=10,
              eta=0.2,
              frequencies={'type': 'nonlinear', 'domega0': 0.1},
              integrate_gamma=integrate_gamma,
              truncation='2D',
              kpts=[((1 / 3, 1 / 3, 0))],
              bands=(8, 10))

    e_qp = gw.calculate()['qp'][0, 0]

    paths = gw.savepckl()
    for path in paths.values():
        assert path.exists()

    results = {'sphere': (2.392, 7.337),
               '1BZ2D': (2.400, 7.311),
               'reciprocal2D': (2.406, 7.297)}
    ev, ec = results[integrate_gamma]
    assert e_qp[0] == pytest.approx(ev, abs=0.01)
    assert e_qp[1] == pytest.approx(ec, abs=0.01)
