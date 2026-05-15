import numpy as np

from gpaw.mpi import world, broadcast_float
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.wfwriter import WaveFunctionWriter, WaveFunctionReader
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.frequencydensitymatrix import FrequencyDensityMatrix
from gpaw.tddft.folding import frequencies
from gpaw.utilities import compiled_with_sl


def parallel_options(*, include_kpt=False, fix_sl_auto=False):
    """Generate different parallelization options"""
    parallel_i = []
    for sl_auto in [False, True]:
        if not compiled_with_sl() and sl_auto:
            continue
        for band in [1, 2]:
            for kpt in [1, 2] if include_kpt else [1]:
                if world.size < band * kpt:
                    continue
                parallel = {'sl_auto': sl_auto, 'band': band, 'kpt': kpt}

                if fix_sl_auto and world.size == 1 and parallel['sl_auto']:
                    # Choose BLACS grid manually as the one given by sl_auto
                    # doesn't work well for the small test system and 1 process
                    del parallel['sl_auto']
                    parallel['sl_default'] = (1, 1, 8)

                parallel_i.append(parallel)
    return parallel_i


def calculate_time_propagation(gs_fpath, *, kick,
                               communicator=world, parallel={},
                               do_fdm=False):
    td_calc = LCAOTDDFT(gs_fpath,
                        communicator=communicator,
                        parallel=parallel,
                        txt='td.out')
    if do_fdm:
        dmat = DensityMatrix(td_calc)
        ffreqs = frequencies(range(0, 31, 5), 'Gauss', 0.1)
        fdm = FrequencyDensityMatrix(td_calc, dmat, frequencies=ffreqs)
    DipoleMomentWriter(td_calc, 'dm.dat')
    WaveFunctionWriter(td_calc, 'wf.ulm')
    td_calc.absorption_kick(kick)
    td_calc.propagate(20, 3)
    if do_fdm:
        fdm.write('fdm.ulm')

    communicator.barrier()

    if do_fdm:
        return fdm


def calculate_error(a, ref_a):
    if world.rank == 0:
        err = np.abs(a - ref_a).max()
        print()
        print('ERR', err)
    else:
        err = np.nan
    err = broadcast_float(err, world)
    return err


def check_txt_data(ref_fpath, data_fpath, atol):
    world.barrier()
    ref = np.loadtxt(ref_fpath, encoding='utf-8')
    data = np.loadtxt(data_fpath, encoding='utf-8')
    err = calculate_error(data, ref)
    print('err', err, atol)
    assert err < atol


def check_wfs(wf_ref_fpath, wf_fpath, atol=1e-12):
    wfr_ref = WaveFunctionReader(wf_ref_fpath)
    wfr = WaveFunctionReader(wf_fpath)
    assert len(wfr) == len(wfr_ref)
    for i in range(1, len(wfr)):
        ref = wfr_ref[i].wave_functions.coefficients
        coeff = wfr[i].wave_functions.coefficients
        err = calculate_error(coeff, ref)
        assert err < atol, f'error at i={i}'


def copy_and_cut_file(src, dst, *, cut_lines=0):
    with open(src, 'r', encoding='utf-8') as fd:
        lines = fd.readlines()
        if cut_lines > 0:
            lines = lines[:-cut_lines]

    with open(dst, 'w', encoding='utf-8') as fd:
        for line in lines:
            fd.write(line)
