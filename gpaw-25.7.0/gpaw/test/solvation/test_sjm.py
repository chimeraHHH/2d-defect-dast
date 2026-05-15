import pytest
from ase.build import fcc111
from gpaw import FermiDirac
from gpaw.mpi import size
from gpaw.new.ase_interface import GPAW
from gpaw.new.sjm import SJM
from gpaw.solvation import (EffectivePotentialCavity, GradientSurface,
                            LinearDielectric, SurfaceInteraction)
from gpaw.solvation.sjm import SJM as OldSJM
from gpaw.solvation.sjm import SJMPower12Potential


@pytest.mark.parametrize('mode', ['pw', 'fd'])
def test_sjm(gpaw_new, in_tmp_dir, mode):
    if mode == 'pw':
        pytest.skip('Not working at the moment!')
    if not gpaw_new and size > 1:
        pytest.skip('https://gitlab.com/gpaw/gpaw/-/issues/1381')
    if not gpaw_new and mode == 'pw':
        pytest.skip('Not implemented')
    # Solvent parameters
    u0 = 0.180  # eV
    epsinf = 78.36  # dielectric constant of water at 298 K
    gamma = 0.00114843767916  # 18.4*1e-3 * Pascal* m
    T = 298.15  # K

    # Structure is created
    atoms = fcc111('Au', size=(1, 1, 3))
    atoms.cell[2][2] = 15
    atoms.translate([0, 0, 6 - min(atoms.positions[:, 2])])

    # SJM parameters
    potential = 4.5
    tol = 0.02
    sj = {'target_potential': potential,
          'excess_electrons': -0.045,
          'jelliumregion': {'top': 14.5},
          'tol': tol}

    convergence = {
        'energy': 0.05 / 8.,
        'density': 1e-4,
        'eigenstates': 1e-4}

    params = dict(
        mode=mode,
        kpts=(2, 2, 1),
        xc='PBE',
        convergence=convergence,
        occupations=FermiDirac(0.1),
        txt=f'{gpaw_new}-{mode}.txt')

    solvation = dict(
        cavity=EffectivePotentialCavity(
            effective_potential=SJMPower12Potential(u0=u0,
                                                    unsolv_backside=False),
            temperature=T,
            surface_calculator=GradientSurface()),
        dielectric=LinearDielectric(epsinf=epsinf),
        interactions=[SurfaceInteraction(surface_tension=gamma)])

    if not gpaw_new:
        atoms.calc = OldSJM(**params, sj=sj, **solvation)
        atoms.get_potential_energy()
        pot = atoms.calc.get_electrode_potential()
    else:
        atoms.calc = GPAW(
            **params,
            environment=SJM(**sj, **solvation))
        atoms.get_potential_energy()
        pot = -atoms.calc.get_fermi_level()

    assert abs(pot - potential) < tol

    atoms.write('Au.traj')

    atoms.calc.write(f'Au-{gpaw_new}-{mode}.gpw')
    if gpaw_new:
        calc = GPAW(f'Au-{gpaw_new}-{mode}.gpw')
        print(atoms.calc.environment)
        print(calc.environment)

    if 0:  # gpaw_new:
        import matplotlib.pyplot as plt
        import numpy as np
        x, y = np.array(atoms.calc.environment.jellium.history).T
        plt.plot(x, y)
        plt.show()
    if 0:
        v = atoms.calc.get_electrostatic_potential()
        import matplotlib.pyplot as plt
        import numpy as np
        plt.plot(np.linspace(0, atoms.cell[2, 2], v.shape[2], 0), v[0, 0])
        plt.show()


if __name__ == '__main__':
    import sys
    test_sjm(sys.argv[1] == 'new', None, sys.argv[2])
