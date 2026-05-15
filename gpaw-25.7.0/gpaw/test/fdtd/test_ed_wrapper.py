import numpy as np
import pytest
from ase import Atoms
from gpaw.fdtd.poisson_fdtd import QSFDTD
from gpaw.fdtd.polarizable_material import (PermittivityPlus,
                                            PolarizableMaterial,
                                            PolarizableSphere)
from gpaw.mpi import world


@pytest.mark.old_gpaw_only
def test_fdtd_ed_wrapper(in_tmp_dir):
    # This test does the same calculation as ed.py, but using
    # QSFDTD wrapper instead

    # Accuracy
    energy_eps = 0.0005

    # Whole simulation cell (Angstroms)
    cell = [20, 20, 30]

    # Quantum subsystem
    atom_center = np.array([10.0, 10.0, 20.0])
    atoms = Atoms('Na2', [atom_center + [0.0, 0.0, -1.50],
                          atom_center + [0.0, 0.0, +1.50]])

    # Classical subsystem
    sphere_center = np.array([10.0, 10.0, 10.0])
    classical_material = PolarizableMaterial()
    classical_material.add_component(
        PolarizableSphere(
            permittivity=PermittivityPlus(data=[[1.20, 0.20, 25.0]]),
            center=sphere_center,
            radius=5.0))

    # Wrap calculators
    qsfdtd = QSFDTD(classical_material=classical_material,
                    atoms=atoms,
                    cells=(cell, 2.50),
                    spacings=[1.60, 0.40],
                    remove_moments=(1, 4),
                    communicator=world)

    # Run
    qsfdtd.ground_state('gs.gpw',
                        mode='fd',
                        eigensolver='cg',
                        nbands=-1,
                        convergence={'energy': energy_eps},
                        symmetry={'point_group': False})
    assert qsfdtd.energy == pytest.approx(
        -0.631881, abs=energy_eps * qsfdtd.gs_calc.get_number_of_electrons())
    qsfdtd.time_propagation('gs.gpw', kick_strength=[0.000, 0.000, 0.001],
                            time_step=10, iterations=5,
                            dipole_moment_file='dm.dat', restart_file='td.gpw')
    qsfdtd.time_propagation('td.gpw', kick_strength=None, time_step=10,
                            iterations=5, dipole_moment_file='dm.dat')

    # Test
    ref_cl_dipole_moment = [5.25374117e-14, 5.75811267e-14, 3.08349334e-02]
    ref_qm_dipole_moment = [1.78620337e-11, -1.57782578e-11, 5.21368300e-01]

    tol = 1e-4
    cl_dm = qsfdtd.td_calc.hamiltonian.poisson.get_classical_dipole_moment()
    qm_dm = qsfdtd.td_calc.hamiltonian.poisson.get_quantum_dipole_moment()
    assert cl_dm == pytest.approx(ref_cl_dipole_moment, abs=tol)
    assert qm_dm == pytest.approx(ref_qm_dipole_moment, abs=tol)
