import pytest
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr

from gpaw import GPAW
from gpaw.external import ConstantElectricField
from gpaw.external import static_polarizability


@pytest.mark.old_gpaw_only
def test_ext_potential_constant_e_field(in_tmp_dir):
    """A proton in an electric field."""
    h = Atoms('H')
    h.center(vacuum=2.5)
    h.calc = GPAW(mode='fd',
                  external=ConstantElectricField(1.0),  # 1 V / Ang
                  charge=1,
                  txt='h.txt')
    e = h.get_potential_energy()
    f1 = h.get_forces()[0, 2]
    h[0].z += 0.001
    de = h.get_potential_energy() - e
    f2 = -de / 0.001
    print(f1, f2)
    assert abs(f1 - 1) < 1e-4
    assert abs(f2 - 1) < 5e-3

    # Check writing and reading:
    h.calc.write('h')
    vext = GPAW('h', txt=None).hamiltonian.vext
    assert abs(vext.field_v[2] - 1.0 * Bohr / Hartree) < 1e-13


@pytest.mark.old_gpaw_only
def test_polarizability(in_tmp_dir):
    H2 = Atoms('H2', positions=[(0, 0, 0), (0.7, 0, 0)])
    H2.center(vacuum=2.5)
    H2.calc = GPAW(mode='fd', symmetry={'point_group': False})

    strength = 0.1  # V/Ang
    alpha_cc = static_polarizability(H2, strength)

    # make sure no external potential is left over
    assert H2.calc.parameters.external is None

    assert alpha_cc.shape == (3, 3)
    assert alpha_cc == pytest.approx(
        np.diag([6.48529231e-02, 4.61303856e-2, 4.61303856e-2]))

    # displace positions and make sure that you can still
    # get the energy
    H2[1].position[0] -= 0.001
    H2.get_potential_energy()
