from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path
from time import time
from typing import Any, TypeVar, Callable, TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.units import Bohr, Ha
from gpaw.calculator import GPAW as OldGPAW
from gpaw.mpi import world
from gpaw.new.ase_interface import GPAW as NewGPAW

if TYPE_CHECKING:
    T = TypeVar('T')
    PickFunc = Callable[[list[T]], list[T]]


def main(args: str | list[str] = None) -> int:
    if isinstance(args, str):
        args = args.split()

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--repeat')
    parser.add_argument('-p', '--pbc')
    parser.add_argument('-v', '--vacuum')
    parser.add_argument('-M', '--magmoms')
    parser.add_argument('-k', '--kpts')
    parser.add_argument('-m', '--mode')
    parser.add_argument('-c', '--code')
    parser.add_argument('-n', '--ncores')
    parser.add_argument('-s', '--use-symmetry')
    parser.add_argument('-S', '--spin-polarized')
    parser.add_argument('-x', '--complex')
    parser.add_argument('-i', '--ignore-cache', action='store_true')
    parser.add_argument('-o', '--stdout', action='store_true')
    parser.add_argument('--pickle')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--fuzz', action='store_true')
    parser.add_argument('system', nargs='*')
    args = parser.parse_intermixed_args(args)

    if args.pickle:
        pckl_file = Path(args.pickle)
        atoms, params, result_file = pickle.loads(pckl_file.read_bytes())
        run2(atoms, params, result_file)
        return 0

    many = args.all or args.fuzz

    if many:
        system_names = args.system or list(systems)
        args.repeat = args.repeat or '1x1x1,2x1x1'
        args.vacuum = args.vacuum or '0.0,4.0'
        args.pbc = args.pbc or '0,1'
        args.mode = args.mode or 'pw,lcao,fd'
        args.code = args.code or 'new,old'
        args.ncores = args.ncores or '1,2,3,4'
        args.kpts = args.kpts or '2.0,3.0'
        args.use_symmetry = args.use_symmetry or '1,0'
        args.complex = args.complex or '0,1'
    else:
        system_names = args.system
        args.repeat = args.repeat or '1x1x1'
        args.vacuum = args.vacuum or '0.0'
        args.pbc = args.pbc or '0'
        args.mode = args.mode or 'pw'
        args.code = args.code or 'new'
        args.ncores = args.ncores or '1'
        args.kpts = args.kpts or '2.0'
        args.use_symmetry = args.use_symmetry or '1'
        args.complex = args.complex or '0'

    if world.size > 1:
        args.ncores = str(world.size)

    repeat_all = [[int(r) for r in rrr.split('x')]
                  for rrr in args.repeat.split(',')]
    vacuum_all = [float(v) for v in args.vacuum.split(',')]
    pbc_all = [bool(int(p)) for p in args.pbc.split(',')]

    magmoms = None if args.magmoms is None else [
        float(m) for m in args.magmoms.split(',')]

    mode_all = args.mode.split(',')
    kpts_all = [[int(k) for k in kpt.split(',')] if ',' in kpt else
                float(kpt)
                for kpt in args.kpts.split(',')]

    code_all = args.code.split(',')
    ncores_all = [int(c) for c in args.ncores.split(',')]
    use_symmetry_all = [bool(int(s)) for s in args.use_symmetry.split(',')]
    complex_all = [bool(int(s)) for s in args.complex.split(',')]

    # spinpol

    if args.fuzz:
        def pick(choises):
            return [random.choice(choises)]
    else:
        def pick(choises):
            return choises

    count = 0
    calculations = {}
    ok = True

    while ok:
        for atoms, atag in create_systems(system_names,
                                          repeat_all,
                                          vacuum_all,
                                          pbc_all,
                                          magmoms,
                                          pick):
            for params, ptag in create_parameters(mode_all,
                                                  kpts_all,
                                                  pick):
                tag = atag + ' ' + ptag

                for extra, xtag in create_extra_parameters(code_all,
                                                           ncores_all,
                                                           use_symmetry_all,
                                                           complex_all,
                                                           pick):
                    params2 = {**params, **extra}
                    result = run(atoms,
                                 params2,
                                 tag + ' ' + xtag,
                                 args.ignore_cache,
                                 args.stdout)
                    ok = check(tag, result, calculations)
                    count += 1
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break

        if not args.fuzz:
            break

    return int(not ok)


def run(atoms: Atoms,
        params: dict[str, Any],
        tag: str,
        ignore_cache: bool = False,
        use_stdout: bool = False) -> dict[str, Any]:
    params = params.copy()
    name, things = tag.split(' ', 1)
    print(f'{name:3} {things}:', end='')
    tag = tag.replace(' ', '')
    folder = Path('fuzz')
    if not folder.is_dir():
        folder.mkdir()
    result_file = folder / f'{tag}.json'
    if not use_stdout:
        params['txt'] = str(result_file.with_suffix('.txt'))
    if not result_file.is_file() or ignore_cache:
        print(' ...', end='', flush=True)
        ncores = params.pop('ncores')
        if ncores == world.size:
            result = run2(atoms, params, result_file)
        else:
            pckl_file = result_file.with_suffix('.pckl')
            pckl_file.write_bytes(pickle.dumps((atoms, params, result_file)))
            args = ['mpiexec', '-np', str(ncores),
                    sys.executable, '-m', 'gpaw.test.fuzz',
                    '--pickle', str(pckl_file)]
            extra = os.environ.get('GPAW_MPI_OPTIONS')
            if extra:
                args[1:1] = extra.split()
            subprocess.run(args, check=True, env=os.environ)
            result, _ = json.loads(result_file.read_text())
            pckl_file.unlink()
    else:
        print('    ', end='')
        result, _ = json.loads(result_file.read_text())
    print(f' {result["energy"]:14.6f} eV, {result["time"]:9.3f} s')
    return result


def run2(atoms: Atoms,
         params: dict[str, Any],
         result_file: Path) -> dict[str, Any]:
    params = params.copy()

    code = params.pop('code')
    if code[0] == 'n':
        if params.pop('dtype', None) == complex:
            params['mode']['force_complex_dtype'] = True
        calc = NewGPAW(**params)
    else:
        calc = OldGPAW(**params)
    atoms.calc = calc

    t1 = time()
    energy = atoms.get_potential_energy()
    try:
        forces = atoms.get_forces()
    except NotImplementedError:
        forces = None

    t2 = time()

    result = {'time': t2 - t1,
              'energy': energy,
              'forces': None if forces is None else forces.tolist()}

    gpw_file = result_file.with_suffix('.gpw')
    calc.write(gpw_file, mode='all')

    dft = NewGPAW(gpw_file).dft

    energy2 = dft.results['energy'] * Ha
    assert abs(energy2 - energy) < 1e-13, (energy2, energy)

    if forces is not None:
        forces2 = dft.results['forces'] * Ha / Bohr
        assert abs(forces2 - forces).max() < 1e-14

    # ibz_index = atoms.calc.wfs.kd.bz2ibz_k[p.kpt]
    # eigs = atoms.calc.get_eigenvalues(ibz_index, p.spin)

    if world.rank == 0:
        if 'dtype' in params:
            params['dtype'] = 'complex'
        result_file.write_text(json.dumps([result, params], indent=2))

    atoms.calc = None

    return result


def check(tag: str,
          result: dict[str, Any],
          calculations: dict[str, dict[str, Any]]) -> bool:
    if tag not in calculations:
        calculations[tag] = result
        return True

    result0 = calculations[tag]
    e0 = result0['energy']
    f0 = result0['forces']
    e = result['energy']
    f = result['forces']
    error = e - e0
    if abs(error) > 0.0005:
        print('Energy error:', e, e0, error)
        return False
    if f0 is None:
        if f is not None:
            calculations[tag]['forces'] = f
        return True
    if f is not None:
        error = abs(np.array(f) - f0).max()
        if error > 0.001:
            print('Force error:', error)
            return False
    return True


def create_systems(system_names: list[str],
                   repeats: list[list[int]],
                   vacuums: list[float],
                   pbcs: list[bool],
                   magmoms: list[float] | None,
                   pick: PickFunc) -> tuple[Atoms, str]:
    for name in pick(system_names):
        atoms = systems[name]
        for repeat in pick(repeats):
            if any(not p and r > 1 for p, r in zip(atoms.pbc, repeat)):
                continue
            ratoms = atoms.repeat(repeat)
            for vacuum in pick(vacuums):
                if vacuum:
                    vatoms = ratoms.copy()
                    axes = [a for a, p in enumerate(atoms.pbc) if not p]
                    if axes:
                        vatoms.center(vacuum=vacuum, axis=axes)
                    else:
                        continue
                else:
                    vatoms = ratoms
                for pbc in pick(pbcs):
                    if pbc:
                        if vatoms.pbc.all():
                            continue
                        patoms = vatoms.copy()
                        patoms.pbc = pbc
                    else:
                        patoms = vatoms

                    if magmoms is not None:
                        patoms.set_initial_magnetic_moments(
                            magmoms * (len(patoms) // len(magmoms)))

                    tag = (f'{name} '
                           f'-r{"x".join(str(r) for r in repeat)} '
                           f'-v{vacuum:.1f} '
                           f'-p{int(pbc)}')
                    yield patoms, tag


def create_parameters(modes: list[str],
                      kpts_all: list[float | list[int]],
                      pick: PickFunc) -> tuple[dict[str, Any], str]:
    for mode in pick(modes):
        for kpt in pick(kpts_all):
            if isinstance(kpt, float):
                kpts = {'density': kpt}
                ktag = f'-k{kpt:.1f}'
            else:
                kpts = kpt
                ktag = f'-k{"x".join(str(k) for k in kpt)}'
            yield {'eigensolver': 'davidson' if mode == 'pw' else None,
                   'mode': mode,
                   'kpts': kpts}, f'-m{mode} {ktag}'


def create_extra_parameters(codes: list[str],
                            ncores_all: list[int],
                            symmetry_all: list[bool],
                            complex_all: list[bool],
                            pick: PickFunc) -> dict[str, Any]:
    for code in pick(codes):
        for ncores in pick(ncores_all):
            params = {'code': code,
                      'ncores': ncores}
            for use_symm in pick(symmetry_all):
                if not use_symm:
                    sparams = {**params, 'symmetry': 'off'}
                else:
                    sparams = params
                for force_complex_dtype in pick(complex_all):
                    if force_complex_dtype:
                        sparams['dtype'] = complex
                    yield (sparams,
                           (f'-c{code} -n{ncores} -s{int(use_symm)} '
                            f'-x{int(force_complex_dtype)}'))


systems = {}


def system(func):
    systems[func.__name__] = func()
    return func


@system
def h():
    atoms = Atoms('H', magmoms=[1])
    atoms.center(vacuum=2.0)
    return atoms


@system
def h2():
    atoms = Atoms('H2', [(0, 0, 0), (0, 0.75, 0)])
    atoms.center(vacuum=2.0)
    return atoms


@system
def si():
    atoms = bulk('Si', a=5.4)
    return atoms


@system
def fe():
    atoms = bulk('Fe')
    atoms.set_initial_magnetic_moments([2.3])
    return atoms


@system
def li():
    L = 5.0
    atoms = Atoms('Li', cell=[L, L, 1.5], pbc=(0, 0, 1))
    atoms.center()
    return atoms


if __name__ == '__main__':
    raise SystemExit(main())
