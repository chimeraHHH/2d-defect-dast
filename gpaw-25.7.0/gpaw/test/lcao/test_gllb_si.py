import pytest
from ase.build import bulk
from gpaw import GPAW, LCAO

# This test calculates a GLLB quasiparticle gap with LCAO and verifies
# that it does not change from a reference value.  Note that the
# calculation, physically speaking, is garbage.


@pytest.fixture
def calc():
    return GPAW(mode=LCAO(interpolation=2),
                h=0.3,
                basis='sz(dzp)',
                xc='GLLBSC',
                kpts={'size': (2, 2, 2), 'gamma': True},
                convergence={'maximum iterations': 1},
                txt='si.txt')


@pytest.mark.gllb
@pytest.mark.libxc
def test_lcao_gllb_si(in_tmp_dir, calc):
    si = bulk('Si', 'diamond', a=5.421)
    si.calc = calc
    si.get_potential_energy()

    homo, lumo = calc.get_homo_lumo()
    response = calc.hamiltonian.xc.response
    dxc_pot = response.calculate_discontinuity_potential(homo, lumo)
    EKs, Dxc = response.calculate_discontinuity(dxc_pot)
    refgap = 3.02333
    gap = EKs + Dxc
    print('GAP', gap)
    assert gap == pytest.approx(refgap, abs=1e-4)
