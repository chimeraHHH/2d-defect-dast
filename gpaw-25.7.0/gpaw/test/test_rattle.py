import numpy as np
from ase.calculators.tip3p import TIP3P, rOH, angleHOH
from ase.optimize import FIRE
from ase.constraints import FixBondLengths
from gpaw.utilities.watermodel import FixBondLengthsWaterModel
from gpaw.utilities.watermodel import TIP3PWaterModel as TIP3Pc
from ase.data import s22

# check that all combinations of watermodel implementations
# and constraints give the same results


def test_rattle(in_tmp_dir):
    pairs = [(3 * i + j, 3 * i + (j + 1) % 3)
             for i in range(2)
             for j in [0, 1, 2]]

    ct = 0
    distances = np.zeros(4)
    for calc in [TIP3P(), TIP3Pc()]:
        for fix in [FixBondLengths, FixBondLengthsWaterModel]:
            atoms = s22.create_s22_system('Water_dimer')

            for m in [0, 3]:
                atoms.set_angle(m + 1, m, m + 2, angleHOH)
                atoms.set_distance(m, m + 1, rOH, fix=0)
                atoms.set_distance(m, m + 2, rOH, fix=0)
                atoms.set_angle(m + 1, m, m + 2, angleHOH)

            atoms.set_constraint(fix(pairs))

            atoms.calc = calc

            with FIRE(atoms, logfile='test.log') as opt:
                opt.run(0.05)
            p = atoms.get_positions()
            d = np.linalg.norm(p[0] - p[3])
            distances[ct] = d
            ct += 1

    assert (np.diff(distances) < 1e-6).all()
