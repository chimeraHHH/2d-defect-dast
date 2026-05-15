import numpy as np
import pytest
from gpaw import GPAW
from gpaw.response import ResponseContext, ResponseGroundStateAdapter
from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.modelinteraction import initialize_w_model
from gpaw.response.chi0 import Chi0Calculator
from gpaw.wannier90 import Wannier90
import os
from gpaw.mpi import world, serial_comm
from subprocess import PIPE, run


def out():
    result = run('wannier90.x --version',
                 stdout=PIPE,
                 stderr=PIPE,
                 universal_newlines=True,
                 shell=True)
    return result.stdout


@pytest.mark.old_gpaw_only
@pytest.mark.parametrize('symm', [True, False])
@pytest.mark.response
@pytest.mark.skipif(': 3.' not in out(),
                    reason="requires at least Wannier90 version 3.0")
def test_w(in_tmp_dir, gpw_files, symm):

    if not symm and world.size < 2:
        pytest.skip('Skip nosymm test in serial')

    if symm:
        gpwfile = gpw_files['gaas_pw']
    else:
        gpwfile = gpw_files['gaas_pw_nosym']

    calc = GPAW(gpwfile, communicator=serial_comm)
    seed = 'GaAs'

    # Wannier90 only works in serial
    if world.rank == 0:
        w90 = Wannier90(calc, orbitals_ai=[[], [0, 1, 2, 3]],
                        bands=range(4),
                        seed=seed)
        w90.write_input(num_iter=100,
                        plot=False,
                        write_u_matrices=True)
        w90.write_wavefunctions()
        os.system('wannier90.x -pp ' + seed)
        w90.write_projections()
        w90.write_eigenvalues()
        w90.write_overlaps()
        os.system('wannier90.x ' + seed)

    world.barrier()

    omega = np.array([0])
    kwargs = dict(hilbert=False, ecut=30, intraband=False)
    gs = ResponseGroundStateAdapter.from_gpw_file(gpwfile)
    context = ResponseContext('test.log')
    wd = FrequencyDescriptor.from_array_or_dict(omega)
    chi0calc = Chi0Calculator(gs, context, wd=wd, **kwargs)
    Wm = initialize_w_model(chi0calc)
    w, Wwann = Wm.calc_in_Wannier(chi0calc, Uwan_mnk=seed, bandrange=[0, 4])
    check_W(Wwann)
    assert np.allclose(w, omega)

    # test block parallelization
    if world.size % 2 == 0 and symm:
        omega = np.array([0, 1])
        wd = FrequencyDescriptor.from_array_or_dict(omega)
        chi0calc = Chi0Calculator(gs, context, wd=wd, nblocks=2, **kwargs)
        Wm = initialize_w_model(chi0calc)
        w, Wwann = Wm.calc_in_Wannier(chi0calc, Uwan_mnk=seed,
                                      bandrange=[0, 4])
        check_W(Wwann)
        print(omega, w)
        assert np.allclose(w, omega)


def check_W(Wwann):
    assert Wwann[0, 0, 0, 0, 0] == pytest.approx(2.478, abs=0.003)
    assert Wwann[0, 1, 1, 1, 1] == pytest.approx(1.681, abs=0.003)
    assert Wwann[0, 2, 2, 2, 2] == pytest.approx(1.681, abs=0.003)
    assert Wwann[0, 3, 3, 3, 3] == pytest.approx(1.681, abs=0.003)
    assert Wwann[0, 3, 3, 0, 0] == pytest.approx(0.861, abs=0.003)
    assert Wwann[0, 3, 0, 3, 0].real == pytest.approx(1.757, abs=0.003)
    assert np.abs(Wwann[0, 3, 0, 3, 0].imag) < 0.005
