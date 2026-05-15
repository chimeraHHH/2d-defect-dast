import pytest
from io import StringIO

from ase.io import read
from ase.units import Ha

from gpaw import GPAW, FermiDirac
from gpaw.lrtddft import LrTDDFT
from gpaw.lrtddft2 import LrTDDFT2


jend = 12  # LUMO


@pytest.fixture
def C3H6O():
    atoms = read(StringIO("""10
https://cccbdb.nist.gov/ Geometry for C3H6O (Propylene oxide), CISD/6-31G*
O   0.8171  -0.7825  -0.2447
C  -1.5018   0.1019  -0.1473
H  -1.3989   0.3323  -1.2066
H  -2.0652  -0.8262  -0.0524
H  -2.0715   0.8983   0.3329
C  -0.1488  -0.0393   0.4879
H  -0.1505  -0.2633   1.5506
C   1.0387   0.6105  -0.0580
H   0.9518   1.2157  -0.9531
H   1.8684   0.8649   0.5908
"""), format='xyz')
    atoms.center(vacuum=3)

    atoms.calc = GPAW(mode='fd', h=0.3,
                      occupations=FermiDirac(width=0.1),
                      nbands=15, convergence={
                          'eigenstates': 1e-4,
                          'bands': jend},
                      txt=None)
    atoms.get_potential_energy()
    return atoms


@pytest.mark.old_gpaw_only
def test_lrtddft2(C3H6O, in_tmp_dir):
    """Test equivalence"""
    atoms = C3H6O

    istart = 10  # HOMO-1

    evs = atoms.calc.get_eigenvalues() / Ha
    energy_differences = evs[jend] - evs[istart:jend]

    lr = LrTDDFT(atoms.calc, restrict={'istart': istart, 'jend': jend})

    lr2 = LrTDDFT2('C3H6O_lr', atoms.calc, fxc='LDA',
                   min_occ=istart, max_unocc=jend)
    lr2.calculate()

    # check for Kohn-Sham properties

    for de, ks, ks2 in zip(energy_differences,
                           lr.kss, lr2.ks_singles.kss_list[::-1]):
        assert de == pytest.approx(ks.energy, 1e-10)
        assert de == pytest.approx(ks2.energy_diff, 1e-8)

        assert ks.mur == pytest.approx(ks2.dip_mom_r, 1e-3)
        assert ks.magn == pytest.approx(ks2.magn_mom, 1e-6)

    # check for TDDFT properties

    (w, S, R, Sx, Sy, Sz) = lr2.get_transitions()

    assert len(lr) == len(w)
    for i, ex in enumerate(lr):
        assert ex.energy * Ha == pytest.approx(w[i], 1e-6)
        f = ex.get_oscillator_strength()
        assert f[0] == pytest.approx(S[i], 1e-4)
        assert f[1] == pytest.approx(Sx[i], 1e-3)
        assert f[2] == pytest.approx(Sy[i], 1e-3)
        assert f[3] == pytest.approx(Sz[i], 1e-3)
        assert ex.get_rotatory_strength() == pytest.approx(R[i], 1e-4)
