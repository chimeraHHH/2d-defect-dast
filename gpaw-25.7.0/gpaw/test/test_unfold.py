from ase import Atoms
import pytest
from gpaw.new.ase_interface import GPAW
from gpaw.unfold import Unfold, find_K_from_k


@pytest.mark.soc
def test_unfold_Ni(gpw_files, in_tmp_dir):
    # Collinear calculation
    gpw = 'fcc_Ni_col'
    calc_col = GPAW(gpw_files[gpw],
                    parallel={'domain': 1, 'band': 1})

    pc = calc_col.atoms.get_cell(complete=True)
    bp = pc.get_bravais_lattice().bandpath('GX', npoints=3)

    M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # Spin 0
    unfold = Unfold(name='Ni_defect_s0',
                    calc=gpw_files[gpw],
                    M=M,
                    spin=0,
                    spinorbit=False)
    e_mk, P_mk = unfold.get_spectral_weights(bp.kpts)
    N0 = len(e_mk)
    assert P_mk == pytest.approx(1, abs=1.0e-6)

    # Spin 1
    unfold = Unfold(name='Ni_defect_s1',
                    calc=gpw_files[gpw],
                    M=M,
                    spin=1,
                    spinorbit=False)
    e_mk, P_mk = unfold.get_spectral_weights(bp.kpts)
    N1 = len(e_mk)
    assert P_mk == pytest.approx(1, abs=1.0e-6)

    # Full bands including nscf spin-orbit
    unfold = Unfold(name='Ni_defect_soc',
                    calc=gpw_files[gpw],
                    M=M,
                    spinorbit=True)
    e_mk, P_mk = unfold.get_spectral_weights(bp.kpts)
    Nm = len(e_mk)
    assert P_mk == pytest.approx(1, abs=1.0e-6)
    assert Nm == N0 + N1

    # Non-collinear calculation with self-consistent spin–orbit
    gpw = 'fcc_Ni_ncolsoc'
    calc_ncol = GPAW(gpw_files[gpw],
                     parallel={'domain': 1, 'band': 1})
    pc = calc_ncol.atoms.get_cell(complete=True)

    bp = pc.get_bravais_lattice().bandpath('GX', npoints=3)

    M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    unfold = Unfold(name='Ni_defect_nc',
                    calc=gpw_files[gpw],
                    M=M)
    e_mk, P_mk = unfold.get_spectral_weights(bp.kpts)
    assert P_mk == pytest.approx(1, abs=1.0e-6)


def test_lcao(in_tmp_dir):
    atoms = Atoms('H', [[2.0, 2.0, 0.0]], cell=[4.0, 4.0, 0.9], pbc=True)
    atoms *= (1, 1, 2)
    atoms.calc = GPAW(mode='lcao',
                      basis='dzp',
                      kpts=(1, 1, 4),
                      txt='gs.txt')
    atoms.get_potential_energy()
    atoms.calc.write('gs.gpw', 'all')

    M = [[1, 0, 0], [0, 1, 0], [0, 0, 2]]

    kpts = [[0, 0, 0], [0, 0, 0.25]]
    Kpts = []
    for k in kpts:
        K = find_K_from_k(k, M)[0]
        Kpts.append(K)

    print(Kpts)

    calc_bands = GPAW('gs.gpw').fixed_density(
        kpts=Kpts,
        symmetry='off',
        nbands=4,
        convergence={'bands': 2})

    calc_bands.write('bands.gpw', 'all')

    unfold = Unfold(name='2',
                    calc='bands.gpw',
                    M=M,
                    spinorbit=False)
    unfold.get_spectral_weights(kpts)
