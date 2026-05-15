import pytest
from ase.build import molecule

import gpaw.mpi as mpi
from gpaw import GPAW
from gpaw.test import gen
from gpaw.xas import XAS


def folding_is_normalized(xas: XAS, dks, rel: float = 1e-5) -> bool:
    _, ys_cn = xas.get_oscillator_strength(dks=dks)

    ys_summed_c = ys_cn.sum(axis=1)
    xf, yf_cn = xas.get_spectra(fwhm=0.5, dks=dks)
    dxf = xf[1:] - xf[:-1]
    assert dxf == pytest.approx(dxf[0])
    yf_summed_c = yf_cn.sum(axis=1) * dxf[0]

    return yf_summed_c == pytest.approx(ys_summed_c, rel=rel)


@pytest.fixture
def s1s1ch_name():
    setupname = 'S1s1ch'
    gen('S', name=setupname, corehole=(1, 0, 1), gpernode=30, write_xml=True)
    return setupname


@pytest.fixture
def s2p1ch_name():
    setupname = 'S2p1ch'
    gen('S', name=setupname, corehole=(2, 1, 1), gpernode=30, write_xml=True)
    return setupname


def sh2(setupname):
    atoms = molecule('SH2')
    atoms.center(3)

    nbands = 6
    atoms.calc = GPAW(mode='fd', h=0.3, nbands=nbands,
                      setups={'S': setupname}, txt=None)
    atoms.get_potential_energy()

    return atoms


@pytest.fixture
def sh2_s1s1ch(s1s1ch_name):
    return sh2(s1s1ch_name)


@pytest.fixture
def sh2_s2p1ch(s2p1ch_name):
    return sh2(s2p1ch_name)


def test_sulphur_2p_spin_io(in_tmp_dir, add_cwd_to_setup_paths, s2p1ch_name):
    """Make sure this calculation does not fail
    because of get_spin_contamination"""
    atoms = molecule('SH2')
    atoms.center(3)

    atoms.set_initial_magnetic_moments([1, 0, 0])
    atoms.calc = GPAW(mode='fd', h=0.3, spinpol=True,
                      setups={'S': s2p1ch_name}, txt=None,
                      convergence={
                          'energy': 0.1, 'density': 0.1, 'eigenstates': 0.1})
    atoms.get_potential_energy()


def test_sulphur_1s_xas_tp(in_tmp_dir, add_cwd_to_setup_paths, sh2_s1s1ch):
    nbands = 6
    nocc = 4  # for SH2

    dks = 20
    xas = XAS(sh2_s1s1ch.calc)
    x, y_cn = xas.get_oscillator_strength(dks=dks)
    assert y_cn.shape == (3, nbands - nocc)
    assert x[0] == dks
    assert xas.nocc == nocc

    assert folding_is_normalized(xas, dks)


def test_sulphur_1s_xas_XCH(in_tmp_dir, add_cwd_to_setup_paths, sh2_s1s1ch):
    atoms = sh2_s1s1ch

    nbands = 6
    nocc = 4  # for SH2
    dks = 20

    calc = sh2_s1s1ch.calc.new(charge=-1)
    atoms.calc = calc

    atoms[0].magmom = 1
    atoms.get_potential_energy()

    xas = XAS(atoms.calc, relative_index_lumo=-1)
    x, y_cn = xas.get_oscillator_strength(dks=dks)
    assert xas.nocc == nocc
    assert y_cn.shape == (3, nbands - nocc)
    assert x[0] == dks
    assert folding_is_normalized(xas, dks)


def test_sulphur_2p_xas(in_tmp_dir, add_cwd_to_setup_paths, sh2_s2p1ch):
    dks = 20

    xas = XAS(sh2_s2p1ch.calc)
    assert folding_is_normalized(xas, dks)


def test_proj(in_tmp_dir, add_cwd_to_setup_paths, sh2_s1s1ch):
    atoms = sh2_s1s1ch

    dks = 20
    xas0 = XAS(atoms.calc)
    mefname = 'me.dat.npz'
    proj = [[1, 0, 0]]
    xas0.write(mefname)
    x0, y0_cn = xas0.get_oscillator_strength(
        dks=dks, proj=proj, proj_xyz=False)
    x1, y1_cn = xas0.get_oscillator_strength(
        dks=dks, proj_xyz=True)

    xas1 = XAS().restart(mefname)
    x0_1, y0_1_cn = xas1.get_oscillator_strength(
        dks=dks, proj=proj, proj_xyz=False)
    x1_1, y1_1_cn = xas1.get_oscillator_strength(
        dks=dks, proj_xyz=True)

    assert y1_cn.shape[0] == y0_cn.shape[0] + 2
    assert y1_cn.shape[1] == y0_cn.shape[1]
    assert x1 == pytest.approx(x0)
    assert y1_cn[0] == pytest.approx(y0_cn[0])

    assert x0 == pytest.approx(x0_1)
    assert x1 == pytest.approx(x1_1)
    assert y0_cn == pytest.approx(y0_1_cn)
    assert y1_cn == pytest.approx(y1_1_cn)


def test_parallel(in_tmp_dir, add_cwd_to_setup_paths, s2p1ch_name):
    atoms = molecule('SH2')
    atoms.center(3)

    # serial calculation
    fserial = f'serial_xas_rank{mpi.world.rank}.npz'
    comm = mpi.world.new_communicator([mpi.world.rank])
    print('serial, rank, size:', mpi.world.rank, comm.size)
    atoms.calc = GPAW(mode='fd', h=0.3, setups={'S': s2p1ch_name},
                      txt=None, communicator=comm)

    print('serial, atoms.calc.world.size:', atoms.calc.world.size)
    atoms.get_potential_energy()

    import time
    t0 = time.time()
    xas_s = XAS(atoms.calc)
    xas_s.write(fserial)
    t1 = time.time()
    print(t1 - t0)

    # parallel calculation
    fparallel = 'parallel_xas.npz'
    atoms.calc = GPAW(mode='fd', h=0.3, setups={'S': s2p1ch_name}, txt=None)
    print('parallel, atoms.calc.world.size:', atoms.calc.world.size)
    atoms.get_potential_energy()

    t0 = time.time()
    xas_p = XAS(atoms.calc)
    xas_p.write(fparallel)
    t1 = time.time()
    print(t1 - t0)

    dks = 20
    xs, ys = xas_s.get_oscillator_strength(dks=dks)
    xp, yp = xas_p.get_oscillator_strength(dks=dks)

    assert xs == pytest.approx(xp)
    assert ys == pytest.approx(yp)

    assert xs == pytest.approx(xp)
    assert ys == pytest.approx(yp)


def test_io(in_tmp_dir, add_cwd_to_setup_paths, sh2_s1s1ch):
    """Test that a direct calculation gives the same results as a calculation
    restarted from matrix element file."""
    dks = 20
    medata = 'xasme.dat'

    xas1 = XAS(sh2_s1s1ch.calc)
    xas1.write(medata)

    # define the XAS object by reading
    xas2 = XAS().restart(medata)

    x1, y1 = xas1.get_oscillator_strength(dks=dks)
    x2, y2 = xas2.get_oscillator_strength(dks=dks)
    assert x1 == pytest.approx(x2)
    assert y1 == pytest.approx(y2)
