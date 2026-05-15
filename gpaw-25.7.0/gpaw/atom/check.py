import optparse
import traceback
from pathlib import Path
from typing import Any

import ase.db
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.optimize import BFGS
from gpaw import GPAW, PW, KohnShamConvergenceError

cutoffs = [200, 250, 300, 400, 500, 600, 700, 800, 1500]


def check(con, name: str, lcao=True):
    params: dict[str, Any] = dict(xc='PBE',
                                  symmetry='off')

    if '.' in name:
        symbol, _, setup = name.partition('.')
        params['setups'] = setup
    else:
        symbol = name

    for h in [0.16, 0.18, 0.2]:
        a = 16 * h
        atoms = Atoms(symbol, cell=(a, a, 2 * a), pbc=True)
        atoms.calc = GPAW(mode='fd',
                          h=h,
                          txt=f'{name}-eggbox-{h:.2f}.txt',
                          **params)
        energies = []
        try:
            for i in range(4):
                energies.append(atoms.get_potential_energy())
                atoms.positions += h / 6
        except KohnShamConvergenceError:
            continue
        atoms.positions -= h / 6
        eegg = np.ptp(energies)
        con.write(atoms, name=name, test='eggbox', eegg=eegg, h=h)

    a = 4.0
    atoms = Atoms(symbol, cell=(a, a, a), pbc=True)
    for ecut in cutoffs:
        atoms.calc = GPAW(mode=PW(ecut),
                          txt=f'{name}-pw-{ecut:04}.txt',
                          **params)
        try:
            atoms.get_potential_energy()
        except KohnShamConvergenceError:
            continue
        con.write(atoms, name=name, test='pw1', ecut=ecut)

    for g in [20, 24, 28]:
        atoms.calc = GPAW(mode='fd',
                          gpts=(g, g, g),
                          txt=f'{name}-fd-{g}.txt',
                          **params)
        try:
            atoms.get_potential_energy()
        except KohnShamConvergenceError:
            continue
        con.write(atoms, name=name, test='fd1', gpts=g)

    if lcao:
        for g in [20, 24, 28]:
            id = con.reserve(name=name, test='lcao1', gpts=g)
            if id is None:
                continue
            atoms.calc = GPAW(gpts=(g, g, g),
                              mode='lcao', basis='dzp',
                              txt=f'{name}-lcao-{g}.txt',
                              **params)
            atoms.get_potential_energy()
            con.write(atoms, name=name, test='lcao1', gpts=g)
            del con[id]

    Z = atomic_numbers[symbol]
    d = 2 * covalent_radii[Z]
    atoms = Atoms(symbol * 2, cell=(a, a, 2 * a), pbc=True,
                  positions=[(0, 0, 0), (0, 0, d)])
    for ecut in cutoffs:
        atoms.calc = GPAW(mode=PW(ecut),
                          txt=f'{name}2-pw-{ecut:04}.txt',
                          **params)
        try:
            atoms.get_potential_energy()
        except KohnShamConvergenceError:
            continue
        con.write(atoms, name=name, test='pw2', ecut=ecut)

    if 0:
        id = con.reserve(name=name, test='relax')
        if id is not None:
            atoms.calc = GPAW(mode=PW(1500),
                              txt=f'{name}2-relax.txt',
                              **params)
            BFGS(atoms).run(fmax=0.02)
            con.write(atoms, name=name, test='relax')
            del con[id]

    for g in [20, 24, 28]:
        atoms.calc = GPAW(mode='fd',
                          gpts=(g, g, 2 * g),
                          txt=f'{name}2-fd-{g}.txt',
                          **params)
        try:
            atoms.get_potential_energy()
        except KohnShamConvergenceError:
            continue
        con.write(atoms, name=name, test='fd2', gpts=g)

    if lcao:
        for g in [20, 24, 28]:
            id = con.reserve(name=name, test='lcao2', gpts=g)
            if id is None:
                continue
            atoms.calc = GPAW(gpts=(g, g, 2 * g),
                              mode='lcao', basis='dzp',
                              txt=f'{name}2-lcao-{g}.txt',
                              **params)
            atoms.get_potential_energy()
            con.write(atoms, name=name, test='lcao2', gpts=g)
            del con[id]


def solve(energies, de):
    for i1 in range(len(energies) - 3, -1, -1):
        if energies[i1] > de:
            break
    c1 = cutoffs[i1]
    c2 = cutoffs[i1 + 1]
    # a * exp(-b * i)
    e1 = energies[i1]
    e2 = energies[i1 + 1]
    b = np.log(e1 / e2) / (c2 - c1)
    a = e1 * np.exp(b * c1)
    return np.log(a / de) / b


def summary(con, name):
    eegg = [row.get('eegg', np.nan)
            for row in con.select(name=name, test='eggbox', sort='h')]
    ecut = np.array([row.energy for row in con.select(name=name,
                                                      test='pw1',
                                                      sort='ecut')])
    ecut2 = np.array([row.get('energy', np.nan)
                      for row in con.select(name=name, test='pw2',
                                            sort='ecut')])
    eg = np.array([row.energy for row in con.select(name=name,
                                                    test='fd1', sort='gpts')])
    eg2 = np.array([row.get('energy', np.nan)
                    for row in con.select(name=name, test='fd2', sort='gpts')])
    eL = np.array([row.energy for row in con.select(name=name,
                                                    test='lcao1',
                                                    sort='gpts')])
    eL2 = np.array([row.get('energy', np.nan)
                    for row in con.select(name=name,
                                          test='lcao2',
                                          sort='gpts')])

    decut = ecut - 0.5 * ecut2
    energies = abs(ecut - ecut[-1])
    denergies = abs(decut - decut[-1])
    assert len(energies) == len(cutoffs)

    eg -= ecut[-1]
    eg2 -= ecut2[-1]
    deg = eg - 0.5 * eg2

    eL -= ecut[-1]
    eL2 -= ecut2[-1]
    deL = eL - 0.5 * eL2

    return (energies, denergies,
            abs(eg), abs(deg), abs(eL), abs(deL),
            eegg)


all_names = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Na.1',
    'Mg', 'Mg.2', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti',
    'V', 'V.5', 'Cr', 'Mn', 'Mn.7', 'Fe', 'Co', 'Ni', 'Ni.10', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Nb.5',
    'Mo', 'Mo.6', 'Ru', 'Ru.8', 'Rh', 'Rh.9', 'Pd', 'Pd.10', 'Ag', 'Ag.11',
    'Cd', 'In', 'Sn', 'Sb', 'Te', 'Te.16', 'I', 'Xe', 'Cs', 'Ba', 'Hf',
    'Ta', 'Ta.5', 'W', 'W.6', 'Re', 'Os', 'Os.8', 'Ir', 'Ir.9', 'Pt',
    'Pt.10', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Rn']

new_names = [
    'Cr.14',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']

all_names += new_names


def main():
    parser = optparse.OptionParser(usage='python -m gpaw.atom.check '
                                   '[options] symbol ...',
                                   description='Check dataset.')
    parser.add_option('-s', '--summary', action='store_true')
    parser.add_option('-v', '--verbose', action='store_true')
    parser.add_option('-p', '--plot', action='store_true')
    parser.add_option('-l', '--lcao', action='store_true')
    parser.add_option('-d', '--database', default='check.db')
    parser.add_option('--datasets', default='.')
    parser.add_option('-e', '--energy-difference', type=float, default=0.01)
    opts, names = parser.parse_args()
    if not names:
        names = [Path.cwd().name]
    con = ase.db.connect(opts.database)
    if opts.datasets:
        from gpaw import setup_paths
        setup_paths[:0] = opts.datasets.split(',')
    if opts.summary:
        for name in names:
            try:
                E, dE, eegg, ecut, decut, eg, deg, eL, deL = summary(
                    con, name, opts.energy_difference)
            except Exception as ex:
                if opts.verbose:
                    print(name)
                    traceback.print_exc()
                else:
                    print('{} {}: {}'.format(name,
                                             ex.__class__.__name__, ex))
            else:
                print('{0:5} {1:6.1f} {2:6.1f} {2:6.1f}'
                      .format(name, ecut, decut),
                      ''.join(f'{e:7.4f}' for e in eegg),
                      ''.join(f'{e:7.3f}' for e in eg),
                      ''.join(f'{e:7.3f}' for e in deg),
                      ''.join(f'{e:7.3f}' for e in eL),
                      ''.join(f'{e:7.3f}' for e in deL))
    if opts.plot:
        for name in names:
            E, dE, eegg, ecut, decut, eg, deg, eL, deL = summary(
                con, name, opts.energy_difference)
            import matplotlib.pyplot as plt
            plt.semilogy(cutoffs[:-1], E[:-1])
            plt.semilogy(cutoffs[:-1], dE[:-1])
            plt.semilogy([solve(E, de) for de in eg], eg, 's')
            plt.semilogy([solve(dE, de) for de in deg], deg, 'o')
            plt.semilogy([solve(E, de) for de in eL], eL, '+')
            plt.semilogy([solve(dE, de) for de in deL], deL, 'x')
        plt.show()
    else:
        for name in names:
            check(con, name, opts.lcao)


if __name__ == '__main__':
    main()
