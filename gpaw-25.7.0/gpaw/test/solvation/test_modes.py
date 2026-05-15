import numpy as np
import pytest
from ase import Atoms
from gpaw import FermiDirac
from gpaw.new.ase_interface import GPAW
from gpaw.new.sjm import SJM
from gpaw.solvation.sjm import SJM as OldSJM


@pytest.mark.parametrize('mode', ['pw', 'fd'])
def test_h(gpaw_new, mode, in_tmp_dir):
    if mode == 'pw' and not gpaw_new:
        pytest.skip('PW-mode not implemented for old GPAW')

    a = 1.4
    atoms = Atoms('H', cell=[a, a, 11.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 4.0

    k = 2
    params = dict(
        mode=mode,
        kpts=(k, k, 1),
        occupations=FermiDirac(0.2))
    sjm = {'target_potential': 7.5,
           'excess_electrons': -0.0045,
           'jelliumregion': {'top': 9.0, 'bottom': 7.0},
           'tol': 0.01}
    solvation = dict(cavity=NoCavity(), dielectric=Vacuum(), interactions=[])

    if gpaw_new:
        atoms.calc = GPAW(environment=SJM(**sjm, **solvation), **params)
        atoms.get_potential_energy()
        pot = -atoms.calc.get_fermi_level()
    else:
        atoms.calc = OldSJM(**solvation, sj=sjm, **params)
        atoms.get_potential_energy()
        pot = atoms.calc.get_electrode_potential()

    assert pot == pytest.approx(7.5, abs=0.01)

    atoms.write('h.traj')
    if gpaw_new:
        atoms.calc.write('h.gpw')

        def hook(dct):
            return SJM(**sjm, **solvation)

        GPAW('h.gpw', object_hooks={'environment': hook})

    if 0:
        v = atoms.calc.get_electrostatic_potential()
        import matplotlib.pyplot as plt
        plt.plot(np.linspace(0, 11, v.shape[2], 0), v[0, 0])
        plt.show()


class NoCavity:
    depends_on_el_density = False

    def set_grid_descriptor(self, gd):
        pass

    def allocate(self):
        pass

    def update_atoms(self, atoms, log):
        pass

    def update(self, atoms, density):
        pass

    def communicate_vol_surf(self, comm):
        pass

    def summary(self, log):
        pass

    def todict(self):
        return {}


class Vacuum:
    def set_grid_descriptor(self, gd):
        self.eps_gradeps = [gd.zeros() for _ in range(4)]
        self.eps_gradeps[0][:] = 1.0

    def allocate(self):
        pass

    def todict(self):
        return {}


if __name__ == '__main__':
    import sys
    test_h(int(sys.argv[1]), sys.argv[2], 1)
