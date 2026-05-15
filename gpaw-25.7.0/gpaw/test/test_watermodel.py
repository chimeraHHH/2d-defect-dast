import time

import ase.units as units
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.tip3p import TIP3P, angleHOH, rOH
from ase.constraints import FixBondLengths
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md import Langevin as Langevin0

from gpaw.utilities.watermodel import FixBondLengthsWaterModel, TIP3PWaterModel


def Langevin(*args, **kwargs):
    try:
        return Langevin0(*args, **kwargs)
    except TypeError:
        kT = kwargs.pop('temperature_K') * units.kB
        return Langevin0(*args, **kwargs, temperature=kT)


@pytest.mark.slow
def test_watermodel(in_tmp_dir):
    NSTEPS = 600
    SCALE = 200

    cutoff = 4.0

    # Set up water box at 20 deg C density
    x = angleHOH * np.pi / 180 / 2
    pos = [[0, 0, 0],
           [0, rOH * np.cos(x), rOH * np.sin(x)],
           [0, rOH * np.cos(x), -rOH * np.sin(x)]]
    atoms = Atoms('OH2', positions=pos)

    vol = ((18.01528 / 6.022140857e23) / (0.9982 / 1e24))**(1 / 3.)
    atoms.set_cell((vol, vol, vol))
    atoms.center()

    N = 4
    atoms = atoms.repeat((N, N, N))
    atoms.set_pbc(True)

    pairs = [(3 * i + j, 3 * i + (j + 1) % 3)
             for i in range(len(atoms) // 3)
             for j in [0, 1, 2]]

    # Create atoms object with old constraints for reference
    atoms_ref = atoms.copy()
    atoms_ref.constraints = FixBondLengths(pairs)

    # RATTLE-type constraints on O-H1, O-H2, H1-H2.
    atoms.constraints = FixBondLengthsWaterModel(pairs)

    atoms.calc = TIP3PWaterModel(rc=cutoff)
    atoms_ref.calc = TIP3P(rc=cutoff)

    rng = np.random.RandomState(123)
    md = Langevin(atoms, 1 * units.fs, temperature_K=300,
                  rng=rng,
                  friction=0.01, logfile='C.log')
    traj = Trajectory('C.traj', 'w', atoms)
    md.attach(traj.write, interval=1)

    start = time.time()
    with md:
        md.run(NSTEPS)
    end = time.time()
    Cversion = end - start
    print("%d steps of C-MD took %.3fs (%.0f ms/step)" % (
        NSTEPS, Cversion,
        Cversion / NSTEPS * 1000))
    traj.close()

    rng = np.random.RandomState(123)
    md_ref = Langevin(atoms_ref, 1 * units.fs, temperature_K=300,
                      rng=rng,
                      friction=0.01, logfile='ref.log')
    traj_ref = Trajectory('ref.traj', 'w', atoms_ref)
    md_ref.attach(traj_ref.write, interval=1)
    start = time.time()
    with md_ref:
        md_ref.run(NSTEPS / SCALE)
    end = time.time()
    Pyversion = (end - start) * SCALE
    print("%d steps of Py-MD took %.3fs (%.0f ms/step)" % (
        NSTEPS / SCALE,
        Pyversion / SCALE, Pyversion / NSTEPS * 1000))
    traj_ref.close()

    # Compare trajectories
    images = read('C.traj@:')
    images_ref = read('ref.traj@:')
    for img1, img2 in zip(images, images_ref):
        norm = np.linalg.norm(img1.get_positions() - img2.get_positions())
        print(norm)
        assert norm < 1e-11

    print("Speedup", Pyversion / Cversion)
