"""Calculate diamond with various parallelizations with GLLBSC."""
import pytest
from gpaw.mpi import world
from ase.build import bulk
from gpaw import GPAW, Mixer


@pytest.mark.gllb
@pytest.mark.skipif(world.size < 4,
                    reason='world.size < 4')
def test_parallel_diamond_gllb(in_tmp_dir):
    xc = 'GLLBSC'
    KS_gap_ref = 4.180237125868162
    QP_gap_ref = 5.469387490357182
    # M. Kuisma et. al, https://doi.org/10.1103/PhysRevB.82.115106
    #     C: KS gap 4.14 eV, QP gap 5.41eV, expt. 5.48 eV
    KSb = []
    dxcb = []

    eigensolver = 'rmm-diis'

    for band in [1, 2, 4]:
        # Calculate ground state
        atoms = bulk('C', 'diamond', a=3.567)
        calc = GPAW(mode='fd',
                    h=0.15,
                    kpts=(4, 4, 4),
                    xc=xc,
                    nbands=8,
                    mixer=Mixer(0.5, 5, 50.0),
                    eigensolver=eigensolver,
                    parallel={'band': band})
        atoms.calc = calc
        atoms.get_potential_energy()

        # Calculate accurate KS-band gap from band structure
        bs_calc = calc.fixed_density(kpts={'path': 'GX', 'npoints': 12},
                                     symmetry='off',
                                     nbands=8,
                                     convergence={'bands': 8},
                                     eigensolver=eigensolver)
        # Get the accurate KS-band gap
        homo, lumo = bs_calc.get_homo_lumo()

        # Calculate the discontinuity potential with accurate band gap
        response = calc.hamiltonian.xc.response
        dxc_pot = response.calculate_discontinuity_potential(homo, lumo)

        # Calculate the discontinuity using the band structure calculator
        bs_response = bs_calc.hamiltonian.xc.response
        KS_gap, dxc = bs_response.calculate_discontinuity(dxc_pot)
        assert KS_gap == pytest.approx(lumo - homo, abs=1e-10)
        assert KS_gap == pytest.approx(KS_gap_ref, abs=1e-4)

        QP_gap = KS_gap + dxc
        assert QP_gap == pytest.approx(QP_gap_ref, abs=1e-4)
        KSb.append(KS_gap)
        dxcb.append(dxc)

    assert abs(KSb[0] - KSb[1]) < 1e-6
    assert abs(KSb[0] - KSb[2]) < 1e-6
    assert abs(dxcb[0] - dxcb[1]) < 1e-6
    assert abs(dxcb[0] - dxcb[2]) < 1e-6
