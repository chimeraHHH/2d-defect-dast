import numpy as np
import pytest
from gpaw import GPAW
from gpaw.inducedfield.inducedfield_base import BaseInducedField
from gpaw.inducedfield.inducedfield_fdtd import (
    FDTDInducedField, calculate_hybrid_induced_field)
from gpaw.inducedfield.inducedfield_tddft import TDDFTInducedField
from gpaw.tddft import TDDFT, DipoleMomentWriter


@pytest.mark.old_gpaw_only
def test_fdtd_ed_inducedfield(in_tmp_dir, gpw_files):
    do_print_values = 0  # Use this for printing the reference values

    # Accuracy
    energy_eps = 0.0005
    poisson_eps = 1e-12
    density_eps = 1e-6

    # load gpw file
    gs_calc = GPAW(gpw_files['na2_isolated'])
    energy = gs_calc.get_potential_energy()

    # Test ground state
    assert energy == pytest.approx(
        -0.631881,
        abs=energy_eps * gs_calc.get_number_of_electrons())

    # Test floating point arithmetic errors
    assert gs_calc.hamiltonian.poisson.shift_indices_1 == pytest.approx(
        [4, 4, 10], abs=0)
    assert gs_calc.hamiltonian.poisson.shift_indices_2 == pytest.approx(
        [8, 8, 16], abs=0)

    # Initialize TDDFT and FDTD
    kick = [0.0, 0.0, 1.0e-3]
    time_step = 10.0
    iterations = 10

    td_calc = TDDFT(gpw_files['na2_isolated'])
    DipoleMomentWriter(td_calc, 'dm.dat')

    # Attach InducedFields to the calculation
    frequencies = [2.05, 4.0]
    width = 0.15
    cl_ind = FDTDInducedField(paw=td_calc,
                              frequencies=frequencies,
                              width=width)
    qm_ind = TDDFTInducedField(paw=td_calc,
                               frequencies=frequencies,
                               width=width)

    # Propagate TDDFT and FDTD
    td_calc.absorption_kick(kick_strength=kick)
    td_calc.propagate(time_step, iterations // 2)
    td_calc.write('td.gpw', 'all')
    cl_ind.write('cl.ind')
    qm_ind.write('qm.ind')

    # Restart
    td_calc = TDDFT('td.gpw')
    DipoleMomentWriter(td_calc, 'dm.dat')
    cl_ind = FDTDInducedField(filename='cl.ind',
                              paw=td_calc)
    qm_ind = TDDFTInducedField(filename='qm.ind',
                               paw=td_calc)
    td_calc.propagate(time_step, iterations // 2)
    td_calc.write('td.gpw', 'all')
    cl_ind.write('cl.ind')
    qm_ind.write('qm.ind')

    # Test
    ref_cl_dipole_moment = [5.25374117e-14, 5.75811267e-14, 3.08349334e-02]
    ref_qm_dipole_moment = [1.78620337e-11, -1.57782578e-11, 5.21368300e-01]

    tol = 1e-4
    assert td_calc.hamiltonian.poisson.get_classical_dipole_moment() == (
        pytest.approx(ref_cl_dipole_moment, abs=tol))
    assert td_calc.hamiltonian.poisson.get_quantum_dipole_moment() == (
        pytest.approx(ref_qm_dipole_moment, abs=tol))

    # Calculate induced fields
    td_calc = TDDFT('td.gpw')

    # Classical subsystem
    cl_ind = FDTDInducedField(filename='cl.ind', paw=td_calc)
    cl_ind.calculate_induced_field(gridrefinement=2)
    cl_ind.write('cl_field.ind', mode='all')

    # Quantum subsystem
    qm_ind = TDDFTInducedField(filename='qm.ind', paw=td_calc)
    qm_ind.calculate_induced_field(gridrefinement=2)
    qm_ind.write('qm_field.ind', mode='all')

    # Total system, interpolate/extrapolate to a grid with spacing h
    tot_ind = calculate_hybrid_induced_field(cl_ind, qm_ind, h=0.4)
    tot_ind.write('tot_field.ind', mode='all')

    # Test induced fields
    ref_values = [72404.467117024149,
                  0.520770766296,
                  0.520770766299,
                  0.830247064075,
                  72416.234345610734,
                  0.517294132489,
                  0.517294132492,
                  0.824704513888,
                  2451.767847927681,
                  0.088037476748,
                  0.088037476316,
                  0.123334033914,
                  2454.462292798476,
                  0.087537484422,
                  0.087537483971,
                  0.122592730690,
                  76582.089818637178,
                  0.589941751987,
                  0.589941751804,
                  0.869526245360,
                  76592.175846021099,
                  0.586223386358,
                  0.586223386102,
                  0.864478308364]

    for fname in ['cl_field.ind', 'qm_field.ind', 'tot_field.ind']:
        ind = BaseInducedField(filename=fname,
                               readmode='field')
        # Estimate tolerance (worst case error accumulation)
        tol = (iterations * ind.fieldgd.integrate(ind.fieldgd.zeros() + 1.0) *
               max(density_eps, np.sqrt(poisson_eps)))
        if do_print_values:
            print('tol = %.12f' % tol)
        for w in range(len(frequencies)):
            val = ind.fieldgd.integrate(ind.Ffe_wg[w])
            assert val == pytest.approx(ref_values.pop(0), abs=tol)
            for v in range(3):
                val = ind.fieldgd.integrate(np.abs(ind.Fef_wvg[w][v]))
                assert val == pytest.approx(ref_values.pop(0), abs=tol)
