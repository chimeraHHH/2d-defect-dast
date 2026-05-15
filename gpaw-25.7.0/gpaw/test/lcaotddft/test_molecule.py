import numpy as np
import pytest

from ase.utils import workdir

from gpaw import GPAW
from gpaw.mpi import world, serial_comm, broadcast
from gpaw.lcaotddft.wfwriter import WaveFunctionReader
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.frequencydensitymatrix import FrequencyDensityMatrix
from gpaw.lcaotddft.ksdecomposition import KohnShamDecomposition

from gpaw.test import only_on_master
from . import (parallel_options, calculate_time_propagation, calculate_error,
               check_txt_data, check_wfs)

pytestmark = pytest.mark.usefixtures('module_tmp_path')

parallel_i = parallel_options(fix_sl_auto=True)


@pytest.fixture(scope='module')
def nacl_spin(gpw_files):
    return gpw_files['nacl_spin']


@pytest.fixture(scope='module')
def nacl_nospin(gpw_files):
    return gpw_files['nacl_nospin']


@pytest.fixture(scope='module')
@only_on_master(world)
def initialize_system(nacl_nospin):
    comm = serial_comm
    calc = GPAW(nacl_nospin, communicator=comm)
    fdm = calculate_time_propagation(nacl_nospin,
                                     kick=np.ones(3) * 1e-5,
                                     communicator=comm,
                                     do_fdm=True)

    # Calculate ground state with full unoccupied space
    unocc_calc = calc.fixed_density(nbands='nao',
                                    communicator=comm,
                                    txt='unocc.out')
    unocc_calc.write('unocc.gpw', mode='all')
    return unocc_calc, fdm


@pytest.mark.rttddft
def test_propagated_wave_function(initialize_system, module_tmp_path):
    wfr = WaveFunctionReader(module_tmp_path / 'wf.ulm')
    coeff = wfr[-1].wave_functions.coefficients
    # Pick a few coefficients corresponding to non-degenerate states;
    # degenerate states should be normalized so that they can be compared
    coeff = coeff[np.ix_([0], [0], [0, 1, 4], [0, 1, 2])]
    # Normalize the wave function sign
    coeff = np.sign(coeff.real[..., 0, np.newaxis]) * coeff
    ref = [[[[1.6564776755628504e-02 + 1.2158943340143986e-01j,
              4.7464497657284752e-03 + 3.4917799444496286e-02j,
              8.2152048273399657e-07 - 1.6344333784831069e-06j],
             [1.5177089239371724e-01 + 7.6502712023931621e-02j,
              8.0497556154952932e-01 + 4.0573839188792121e-01j,
              -5.1505952970811632e-06 - 1.1507918955641119e-05j],
             [2.5116252101774323e+00 + 3.6776360873471503e-01j,
              1.9024613198566329e-01 + 2.7843314959952882e-02j,
              -1.3848736953929574e-05 - 2.6402210145403184e-05j]]]]
    err = calculate_error(coeff, ref)
    assert err < 1e-4


@pytest.mark.rttddft
@pytest.mark.parametrize('parallel', parallel_i)
def test_propagation(initialize_system, module_tmp_path, parallel,
                     gpw_files, in_tmp_dir):
    calculate_time_propagation(gpw_files['nacl_nospin'],
                               kick=np.ones(3) * 1e-5,
                               parallel=parallel)
    check_wfs(module_tmp_path / 'wf.ulm', 'wf.ulm', atol=1e-12)


@pytest.fixture(scope='module')
@only_on_master(world, broadcast=broadcast)
def dipole_moment_reference(initialize_system):
    from gpaw.tddft.spectrum import \
        read_dipole_moment_file, calculate_fourier_transform

    unocc_calc, fdm = initialize_system
    _, time_t, _, dm_tv = read_dipole_moment_file('dm.dat')
    dm_tv = dm_tv - dm_tv[0]
    dm_wv = calculate_fourier_transform(time_t, dm_tv,
                                        fdm.foldedfreqs_f[0])
    return dm_wv


@pytest.fixture(scope='module')
@only_on_master(world)
def ksd_reference(initialize_system):
    unocc_calc, fdm = initialize_system
    ksd = KohnShamDecomposition(unocc_calc)
    ksd.initialize(unocc_calc)
    return ksd, fdm


def ksd_transform_fdm(ksd, fdm):
    rho_iwp = np.empty((2, len(fdm.freq_w), len(ksd.w_p)), dtype=complex)
    rho_iwp[:] = np.nan + 1j * np.nan
    for i, rho_wuMM in enumerate([fdm.FReDrho_wuMM, fdm.FImDrho_wuMM]):
        for w in range(len(fdm.freq_w)):
            rho_uMM = rho_wuMM[w]
            rho_up = ksd.transform(rho_uMM)
            rho_iwp[i, w, :] = rho_up[0]
    return rho_iwp


@pytest.fixture(scope='module')
@only_on_master(world, broadcast=broadcast)
def ksd_transform_reference(ksd_reference):
    ksd, fdm = ksd_reference
    ref_rho_iwp = ksd_transform_fdm(ksd, fdm)
    return ref_rho_iwp


@pytest.fixture(scope='module', params=parallel_i)
def build_ksd(initialize_system, request):
    calc = GPAW('unocc.gpw', parallel=request.param, txt=None)
    if not calc.wfs.ksl.using_blacs and calc.wfs.bd.comm.size > 1:
        pytest.xfail('Band parallelization without scalapack '
                     'is not supported')
    ksd = KohnShamDecomposition(calc)
    ksd.initialize(calc)
    ksd.write('ksd.ulm')


@pytest.fixture(scope='module', params=parallel_i)
def load_ksd(build_ksd, request):
    calc = GPAW('unocc.gpw', parallel=request.param, txt=None)
    # Initialize positions in order to calculate density
    calc.initialize_positions()
    ksd = KohnShamDecomposition(calc, 'ksd.ulm')
    dmat = DensityMatrix(calc)
    fdm = FrequencyDensityMatrix(calc, dmat, 'fdm.ulm')
    return ksd, fdm


@pytest.fixture(scope='module')
def ksd_transform(load_ksd):
    ksd, fdm = load_ksd
    rho_iwp = ksd_transform_fdm(ksd, fdm)
    return rho_iwp


@pytest.mark.skip(reason='See #933')
@pytest.mark.rttddft
def test_ksd_transform(ksd_transform, ksd_transform_reference):
    ref_iwp = ksd_transform_reference
    rho_iwp = ksd_transform
    err = calculate_error(rho_iwp, ref_iwp)
    atol = 1e-18
    assert err < atol


@pytest.mark.skip(reason='See #933')
@pytest.mark.rttddft
def test_ksd_transform_real_only(load_ksd, ksd_transform_reference):
    ksd, fdm = load_ksd
    ref_iwp = ksd_transform_reference
    rho_iwp = np.empty((2, len(fdm.freq_w), len(ksd.w_p)), dtype=complex)
    rho_iwp[:] = np.nan + 1j * np.nan
    for i, rho_wuMM in enumerate([fdm.FReDrho_wuMM, fdm.FImDrho_wuMM]):
        for w in range(len(fdm.freq_w)):
            rho_uMM = rho_wuMM[w]
            rho_p = ksd.transform([rho_uMM[0].real], broadcast=True)[0] \
                + 1j * ksd.transform([rho_uMM[0].imag], broadcast=True)[0]
            rho_iwp[i, w, :] = rho_p
    err = calculate_error(rho_iwp, ref_iwp)
    atol = 1e-18
    assert err < atol


@pytest.mark.rttddft
def test_dipole_moment_from_ksd(ksd_transform, load_ksd,
                                dipole_moment_reference):
    ksd, fdm = load_ksd
    dm_wv = np.empty((len(fdm.freq_w), 3), dtype=complex)
    dm_wv[:] = np.nan + 1j * np.nan
    rho_wp = ksd_transform[0]
    for w in range(len(fdm.freq_w)):
        dm_v = ksd.get_dipole_moment([rho_wp[w]])
        dm_wv[w, :] = dm_v

    ref_wv = dipole_moment_reference
    err = calculate_error(dm_wv, ref_wv)
    atol = 1e-7
    assert err < atol


def get_density_fdm(ksd, fdm, kind):
    assert kind in ['dmat', 'ksd']
    rho_wg = fdm.dmat.density.finegd.empty(len(fdm.freq_w), dtype=complex)
    rho_wg[:] = np.nan + 1j * np.nan
    for w in range(len(fdm.freq_w)):
        rho_uMM = fdm.FReDrho_wuMM[w]
        if kind == 'dmat':
            rho_g = fdm.dmat.get_density([rho_uMM[0].real]) \
                + 1j * fdm.dmat.get_density([rho_uMM[0].imag])
        elif kind == 'ksd':
            rho_up = ksd.transform(rho_uMM, broadcast=True)
            rho_g = ksd.get_density(fdm.dmat.wfs, [rho_up[0].real]) \
                + 1j * ksd.get_density(fdm.dmat.wfs, [rho_up[0].imag])
        rho_wg[w, :] = rho_g
    return rho_wg


@pytest.fixture(scope='module')
@only_on_master(world, broadcast=broadcast)
def density_reference(ksd_reference):
    ksd, fdm = ksd_reference
    dmat_rho_wg = get_density_fdm(ksd, fdm, 'dmat')
    ksd_rho_wg = get_density_fdm(ksd, fdm, 'ksd')
    return dict(dmat=dmat_rho_wg, ksd=ksd_rho_wg)


@pytest.mark.rttddft
def test_ksd_vs_dmat_density(density_reference):
    ref_wg = density_reference['dmat']
    rho_wg = density_reference['ksd']
    err = calculate_error(rho_wg, ref_wg)
    atol = 2e-10
    assert err < atol


@pytest.fixture(scope='module')
def density(load_ksd):
    ksd, fdm = load_ksd
    if ksd.ksl.using_blacs:
        pytest.xfail('Scalapack is not supported')
    dmat_rho_wg = get_density_fdm(ksd, fdm, 'dmat')
    ksd_rho_wg = get_density_fdm(ksd, fdm, 'ksd')
    return dict(dmat=dmat_rho_wg, ksd=ksd_rho_wg)


@pytest.mark.rttddft
@pytest.mark.parametrize('kind', ['ksd', 'dmat'])
def test_density(kind, density, load_ksd, density_reference):
    ksd, fdm = load_ksd
    ref_wg = density_reference[kind]
    rho_wg = fdm.dmat.density.finegd.collect(density[kind])
    err = calculate_error(rho_wg, ref_wg)
    atol = 3e-19
    assert err < atol


@pytest.mark.rttddft
@pytest.mark.parametrize('kind', ['ksd', 'dmat'])
def test_dipole_moment_from_density(kind, density, load_ksd,
                                    dipole_moment_reference):
    ksd, fdm = load_ksd
    rho_wg = density[kind]
    dm_wv = np.empty((len(fdm.freq_w), 3), dtype=complex)
    dm_wv[:] = np.nan + 1j * np.nan
    for w in range(len(fdm.freq_w)):
        dm_v = ksd.density.finegd.calculate_dipole_moment(rho_wg[w])
        dm_wv[w, :] = dm_v

    ref_wv = dipole_moment_reference
    err = calculate_error(dm_wv, ref_wv)
    atol = 5e-7
    assert err < atol


@pytest.mark.rttddft
@only_on_master(world)
def test_read_ksd(ksd_reference):
    # Build a KohnShamDecomposition object from the calculator
    ksd, _ = ksd_reference

    # Now save it and read it without the calculator
    ksd.write('ksd_save.ulm')
    ksd_read = KohnShamDecomposition(filename='ksd_save.ulm')

    np.testing.assert_equal(ksd.atoms, ksd_read.atoms)

    for attr in ['S_uMM', 'C0_unM', 'eig_un', 'occ_un', 'C0S_unM']:
        ref = getattr(ksd, attr)
        test = getattr(ksd_read, attr)

        np.testing.assert_almost_equal(ref, test)


@pytest.fixture(scope='module')
@only_on_master(world)
@workdir('spinpol', mkdir=True)
def initialize_system_spinpol(nacl_spin):
    comm = serial_comm
    calculate_time_propagation(nacl_spin,
                               kick=np.ones(3) * 1e-5,
                               communicator=comm,
                               do_fdm=True)


@pytest.mark.rttddft
def test_spinpol_dipole_moment(initialize_system, initialize_system_spinpol,
                               module_tmp_path):
    # The test system has even number of electrons and is non-magnetic
    # so spin-paired and spin-polarized calculation should give same result
    check_txt_data(module_tmp_path / 'dm.dat',
                   module_tmp_path / 'spinpol' / 'dm.dat',
                   atol=1.0001e-12)


@pytest.mark.rttddft
@pytest.mark.parametrize('parallel', parallel_i)
def test_spinpol_propagation(initialize_system_spinpol, module_tmp_path,
                             parallel, in_tmp_dir, gpw_files):
    ref_path = module_tmp_path / 'spinpol'
    calculate_time_propagation(gpw_files['nacl_spin'],
                               kick=np.ones(3) * 1e-5,
                               parallel=parallel)
    check_wfs(ref_path / 'wf.ulm', 'wf.ulm', atol=1e-12)
