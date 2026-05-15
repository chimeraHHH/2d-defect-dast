import time
import pytest
import numpy as np

from ase import Atom, Atoms, io
from ase.parallel import parprint, paropen
from ase.units import Ha

from gpaw import GPAW
from gpaw.mpi import world
from gpaw.lrtddft import LrTDDFT
from gpaw.lrtddft.excited_state import ExcitedState


def get_H2(calculator=None):
    """Define H2 and set calculator if given"""
    R = 0.7  # approx. experimental bond length
    a = 3.0
    c = 4.0
    H2 = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
                Atom('H', (a / 2, a / 2, (c + R) / 2))],
               cell=(a, a, c))

    if calculator is not None:
        H2.calc = calculator

    return H2


def get_H3(calculator=None):
    R = 0.87  # approx. bond length
    a = 4.0
    c = 3.0
    H3 = Atoms('H3', positions=[[0, 0, 0], [R, 0, 0],
                                [R / 2, R / 2 * np.sqrt(3), 0]],
               cell=(a, a, c))
    H3.center()

    if calculator is not None:
        H3.calc = calculator

    return H3


@pytest.mark.lrtddft
def test_split(in_tmp_dir):
    fname = 'exlst.out'
    calc = GPAW(mode='fd', xc='PBE', h=0.25, nbands=3, txt=fname)
    exlst = LrTDDFT(calc, txt=fname)
    exst = ExcitedState(exlst, 0, txt=fname)
    H2 = get_H2(exst)
    H2.get_potential_energy()

    n = world.size
    exst.split(n)
    H2.get_potential_energy()

    if world.rank == 0:
        with open(fname) as f:
            string = f.read()
            assert 'Total number of cores used: {0}'.format(n) in string
            assert 'Total number of cores used: 1' in string


@pytest.mark.lrtddft
def test_lrtddft_excited_state():
    txt = None

    calc = GPAW(mode='fd', xc='PBE', h=0.25, nbands=3, spinpol=False, txt=txt)
    H2 = get_H2(calc)

    xc = 'LDA'
    lr = LrTDDFT(calc, xc=xc)

    # excited state with forces
    accuracy = 0.015
    exst = ExcitedState(lr, 0, d=0.01,
                        parallel=2)

    t0 = time.time()
    parprint("########### first call to forces --> calculate")
    forces = exst.get_forces(H2)
    parprint("time used:", time.time() - t0)
    for c in range(2):
        assert forces[0, c] == pytest.approx(0.0, abs=accuracy)
        assert forces[1, c] == pytest.approx(0.0, abs=accuracy)
    assert forces[0, 2] + forces[1, 2] == pytest.approx(0.0, abs=accuracy)

    parprint("########### second call to potential energy --> just return")
    t0 = time.time()
    E = exst.get_potential_energy()
    parprint("E=", E)
    parprint("time used:", time.time() - t0)
    t0 = time.time()
    E = exst.get_potential_energy(H2)
    parprint("E=", E)
    parprint("time used:", time.time() - t0)

    parprint("########### second call to forces --> just return")
    t0 = time.time()
    exst.get_forces()
    parprint("time used:", time.time() - t0)
    t0 = time.time()
    exst.get_forces(H2)
    parprint("time used:", time.time() - t0)

    parprint("###########  moved atoms, call to forces --> calculate")
    p = H2.get_positions()
    p[1, 1] += 0.1
    H2.set_positions(p)

    t0 = time.time()
    exst.get_forces(H2)
    parprint("time used:", time.time() - t0)


@pytest.mark.lrtddft
def test_io(in_tmp_dir):
    """Test output and input from files"""
    calc = GPAW(mode='fd', xc='PBE', h=0.25, nbands=3, txt=None)
    exlst = LrTDDFT(calc, txt=None)
    exst = ExcitedState(exlst, 0, txt=None)
    H2 = get_H2(exst)

    parprint('----------- calculate')
    E1 = H2.get_potential_energy()
    E0 = exst.calculator.get_potential_energy()
    dE1 = exlst[0].energy * Ha
    assert E1 == pytest.approx(E0 + dE1, 1.e-5)

    parprint('----------- write trajectory')
    ftraj = 'H2exst.traj'
    F = H2.get_forces()
    traj = io.Trajectory(ftraj, 'w')
    traj.write(H2)

    parprint('----------- write')
    fname = 'exst_test_io'
    parprint('----', exst.get_potential_energy())
    exst.write(fname)
    world.barrier()

    parprint('----------- read')
    exst = ExcitedState.read(fname, txt=None)
    E1 = exst.get_potential_energy()
    parprint('-----', exst.get_potential_energy(), E0 + dE1)
    assert E1 == pytest.approx(E0 + dE1, 1.e-5)

    parprint('----------- read trajectory')
    atoms = io.read(ftraj)
    assert atoms.get_potential_energy() == pytest.approx(E1, 1.e-5)
    assert atoms.get_forces() == pytest.approx(F, 1.e-5)


@pytest.mark.lrtddft
def test_log(in_tmp_dir):
    fname = 'ex0_silent.out'
    calc = GPAW(mode='fd', xc='PBE', h=0.25, nbands=5, txt=None)
    calc.calculate(get_H2(calc))
    exlst = LrTDDFT(calc, restrict={'eps': 0.4, 'jend': 3}, txt=None)
    exst = ExcitedState(exlst, 0, txt=fname)
    del calc
    del exlst
    del exst
    world.barrier()

    if world.rank == 0:
        with open(fname) as f:
            string = f.read()
            assert 'ExcitedState' in string
            assert '  ___ ___ ___ _ _ _' not in string
            assert 'Linear response TDDFT calculation' not in string

    fname = 'ex0_split.out'
    calc = GPAW(mode='fd', xc='PBE', h=0.25, nbands=5, txt=fname)
    calc.calculate(get_H2(calc))
    exlst = LrTDDFT(calc, restrict={'eps': 0.4, 'jend': 3}, log=calc.log)
    exst = ExcitedState(exlst, 0, log=exlst.log, parallel=2)
    exst.get_forces()
    del calc
    del exlst
    del exst

    if world.rank == 0:
        with paropen(fname) as f:
            string = f.read()
            assert 'ExcitedState' in string
            if world.size == 1:
                # one eq + 6 * 2 displacements + one eq. = 14 calculations
                n = 14
            else:
                # we see only half of the calculations in parallel
                # one eq + 3 * 2 displacements + one eq. = 8 calculations
                n = 8
            assert string.count('Converged after') == n
            assert string.count('Kohn-Sham single transitions') == n
            assert string.count('Linear response TDDFT calculation') == n


@pytest.mark.lrtddft
def test_forces():
    """Test whether force calculation works"""
    calc = GPAW(mode='fd', xc='PBE', h=0.25, nbands=3, txt=None)
    exlst = LrTDDFT(calc)
    exst = ExcitedState(exlst, 0)
    H2 = get_H2(exst)

    parprint('---------------- serial')

    forces = H2.get_forces()
    accuracy = 1.e-3
    # forces in x and y direction should be 0
    assert forces[:, :2] == pytest.approx(
        np.zeros_like(forces[:, :2]), abs=accuracy)

    # forces in z direction should be opposite
    assert -forces[0, 2] == pytest.approx(forces[1, 2], abs=accuracy)

    # next call should give back the stored forces
    forces1 = exst.get_forces(H2)
    assert (forces1 == forces).all()

    # test parallel
    if world.size > 1:
        parprint('---------------- parallel', world.size)
        exstp = ExcitedState(exlst, 0, parallel=2)
        forcesp = exstp.get_forces(H2)
        assert forcesp == pytest.approx(forces, abs=0.001)


@pytest.mark.lrtddft
def test_unequal_parralel_work():
    """Test whether parallel force calculation works for three atoms"""
    if world.size == 1:
        return

    calc = GPAW(mode='fd', xc='PBE', charge=1, h=0.25, nbands=3, txt=None)
    exlst = LrTDDFT(calc, txt=None)
    H3 = get_H3()

    parprint('---------------- serial')
    exst = ExcitedState(exlst, 0, txt=None)
    forces = exst.get_forces(H3)

    parprint('---------------- parallel', world.size)
    exst = ExcitedState(exlst, 0, parallel=2, txt=None)
    H3 = get_H3(exst)
    forcesp = H3.get_forces()

    assert forcesp == pytest.approx(forces, abs=0.01)
