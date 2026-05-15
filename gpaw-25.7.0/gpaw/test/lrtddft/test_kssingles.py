import pytest

from ase.build import molecule

from gpaw.mpi import world
from gpaw import GPAW
from gpaw.lrtddft.kssingle import KSSingles


@pytest.mark.lrtddft
def test_old_io(in_tmp_dir):
    """Test reading of old style output files"""
    fname = 'veryold.dat'
    if world.rank == 0:
        with open(fname, 'w') as f:
            f.write("""# KSSingles
2
0 1   0 0   0.129018 1   -4.7624e-02 -3.2340e-01 -4.6638e-01
0 1   1 0   0.129018 1   -4.7624e-02 -3.2340e-01 -4.6638e-01
""")
    world.barrier()

    kss = KSSingles.read(fname)
    assert len(kss) == 2

    fname = 'old.dat'
    if world.rank == 0:
        with open(fname, 'w') as f:
            f.write("""# KSSingles
2 float64
0.024
0 1  0 0  0.392407 2  8.82 2.91 7.98  1.0 2.7 7.8   1.4 1.2 1.08
0 2  0 0  0.402312 2  2.24 8.49 -5.24   2.3 8.26 -4.99038e-14
""")
    world.barrier()

    kss = KSSingles.read(fname)
    assert len(kss) == 2
    assert kss.restrict['eps'] == 0.024


@pytest.fixture
def ch4():
    ch4 = molecule('CH4')
    ch4.center(vacuum=2)
    ch4.calc = GPAW(mode='fd', h=0.25, nbands=8, txt=None)
    ch4.get_potential_energy()
    return ch4


@pytest.fixture
def ch4_kss(ch4):
    """Test multiplication with a number"""
    istart, jend = 1, 4
    kss = KSSingles(restrict={'eps': 0.9, 'istart': istart, 'jend': jend})
    kss.calculate(ch4)
    return kss


@pytest.mark.lrtddft
def test_io(in_tmp_dir, ch4):
    # full KSSingles
    kssfull = KSSingles(restrict={'eps': 0.9})
    kssfull.calculate(ch4)
    fullname = 'kssfull.dat'
    kssfull.write(fullname)
    world.barrier()

    # read full
    kss1 = KSSingles.read(fullname)
    assert len(kss1) == 16

    # restricted KSSingles
    istart, jend = 1, 4
    kss = KSSingles(restrict={'eps': 0.9, 'istart': istart, 'jend': jend})
    kss.calculate(ch4)
    f14name = 'kss_1_4.dat'
    kss.write(f14name)
    world.barrier()

    kss2 = KSSingles.read(f14name)
    assert len(kss2) == len(kss)
    assert kss2.restrict['istart'] == istart
    assert kss2.restrict['jend'] == jend

    # restrict when reading
    kss3 = KSSingles.read(fullname,
                          restrict={'istart': istart, 'jend': jend})
    assert len(kss3) == len(kss)
    assert kss3.restrict['istart'] == istart
    assert kss3.restrict['jend'] == jend


@pytest.mark.lrtddft
def test_mul(ch4_kss):
    """Test multiplication with a number"""
    ks0 = ch4_kss[0]
    ks1 = 1. * ks0
    assert (ks0.me == ks1.me).all()
    assert (ks0.magn == ks1.magn).all()


@pytest.mark.lrtddft
def test_add_sub(in_tmp_dir):
    """Test adding and subtracting"""
    fname = 'kss_with_magn.dat'
    if world.rank == 0:
        with open(fname, 'w') as f:
            # output for C3H6O, see gpaw/test/lrtddft/test_lrtddft2.py
            f.write("""# KSSingles
2 float64
{"istart": 10, "jend": 12}
10 12  0 0  0.2834260231 2.000000     -6.99002e-02 3.49407e-02 3.84815e-02   -6.12491e-02 2.91457e-02 3.60540e-02   5.66737e-04 -3.65251e-04 -1.34614e-03
11 12  0 0  0.2396446019 2.000000     8.73651e-02 -1.01160e-01 -2.82990e-01   7.60765e-02 -9.83790e-02 -2.78192e-01   6.38772e-04 -2.97370e-04 -8.74882e-04
""")  # noqa
    world.barrier()

    kss = KSSingles.read(fname)

    ks_add = kss[0] + (-1 * kss[1])
    ks_sub = kss[0] - kss[1]

    assert ks_add.me == pytest.approx(ks_sub.me)
    assert ks_add.mur == pytest.approx(ks_sub.mur)
    assert ks_add.muv == pytest.approx(ks_sub.muv)
    assert ks_add.magn == pytest.approx(ks_sub.magn)
