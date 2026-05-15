from gpaw import GPAW, restart
from ase.build import bcc111
from gpaw.poisson import PoissonSolver
import numpy as np


def test_poisson_metallic_poisson(in_tmp_dir):
    slab = bcc111('Na', (1, 1, 2), vacuum=8)
    electrodes = ['both', 'single']
    charge = 0.05

    for electrode in electrodes:
        slab.calc = GPAW(
            mode='fd',
            xc='LDA', h=0.22,
            txt='metallic.txt',
            charge=charge,
            convergence={'density': 1e-1,
                         'energy': 1e-1,
                         'eigenstates': 1e-1},
            kpts=(2, 2, 1),
            poissonsolver=PoissonSolver(metallic_electrodes=electrode))

        _ = slab.get_potential_energy()
        phi0 = slab.calc.get_electrostatic_potential()
        assert np.all(abs(phi0[:, :, 0]) < 1e-10)
        if electrode == 'both':
            print(phi0[:, :, 0])
            print(phi0[:, :, 1])
            # The last zero boundary condition is implicit, so extrapolate
            d = phi0[:, :, -1] - phi0[:, :, -2]
            assert np.all(abs(phi0[:, :, -1] + d) < 1e-5)

        slab.calc.write('%s.gpw' % electrode)

        atoms, calc = restart('%s.gpw' % electrode, txt='restart.txt')
        phi02 = calc.get_electrostatic_potential()
        assert np.all(abs(phi02 - phi0) < 1e-10)
