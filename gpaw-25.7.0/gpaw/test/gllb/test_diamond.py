"""This calculation has the following structure.

1) Calculate the ground state of diamond.
2) Calculate the band structure of diamond in order to obtain
   accurate KS band gap.
3) Calculate the potential discontinuity using the ground state calculator
   and the accurate band gap.
4) Apply the discontinuity to CBM using the band structure calculator.

Compare to reference.
"""
import pytest
from ase.build import bulk
from gpaw import GPAW, Davidson, Mixer
from gpaw.mpi import world
from gpaw.calculator import DeprecatedParameterWarning


@pytest.mark.gllb
@pytest.mark.libxc
@pytest.mark.parametrize('deprecated_syntax', [False, True])
def test_gllb_diamond(in_tmp_dir, deprecated_syntax):
    xc = 'GLLBSC'
    KS_gap_ref = 4.180237125868162
    QP_gap_ref = 5.469387490357182
    KS_gap_direct_ref = 5.682571428480924
    QP_gap_direct_ref = 7.452331561454596
    dxc_ave_ref = 1.33769712051036
    # M. Kuisma et. al, https://doi.org/10.1103/PhysRevB.82.115106
    #     C: KS gap 4.14 eV, QP gap 5.41eV, expt. 5.48 eV

    # Calculate ground state
    atoms = bulk('C', 'diamond', a=3.567)
    # We want sufficiently many grid points that the calculator
    # can use wfs.world for the finegd, to test that part of the code.
    calc = GPAW(mode='fd',
                h=0.2,
                kpts=(4, 4, 4),
                xc=xc,
                nbands=8,
                mixer=Mixer(0.5, 5, 10.0),
                parallel=dict(domain=min(world.size, 2), band=1),
                eigensolver=Davidson(niter=2),
                txt='gs.out')
    atoms.calc = calc
    atoms.get_potential_energy()

    if deprecated_syntax:
        with pytest.warns(DeprecationWarning):
            # Calculate the discontinuity directly
            response = calc.hamiltonian.xc.xcs['RESPONSE']
            response.calculate_delta_xc()
            KS_gap, dxc = response.calculate_delta_xc_perturbation()
    else:
        # Calculate the discontinuity directly
        homo, lumo = calc.get_homo_lumo()
        response = calc.hamiltonian.xc.response
        dxc_pot = response.calculate_discontinuity_potential(homo, lumo)
        KS_gap, dxc = response.calculate_discontinuity(dxc_pot)

    QP_gap = KS_gap + dxc
    print('KS gap direct', KS_gap)
    print('QP gap direct', QP_gap)
    assert KS_gap == pytest.approx(KS_gap_direct_ref, abs=1e-4)
    assert QP_gap == pytest.approx(QP_gap_direct_ref, abs=1e-4)

    if deprecated_syntax:
        with pytest.warns(DeprecatedParameterWarning):
            from gpaw import restart

            calc.write('gs.gpw')
            # Calculate accurate KS-band gap from band structure
            atoms, bs_calc = restart('gs.gpw',
                                     fixdensity=True,
                                     kpts={'path': 'GX', 'npoints': 12},
                                     symmetry='off',
                                     nbands=8,
                                     convergence={'bands': 6},
                                     eigensolver=Davidson(niter=4),
                                     txt='bs.out')
            atoms.get_potential_energy()

        # Get the accurate KS-band gap
        homolumo = bs_calc.get_homo_lumo()
        homo, lumo = homolumo
    else:
        # Calculate accurate KS-band gap from band structure
        bs_calc = calc.fixed_density(kpts={'path': 'GX', 'npoints': 12},
                                     symmetry='off',
                                     nbands=8,
                                     convergence={'bands': 6},
                                     eigensolver=Davidson(niter=4),
                                     txt='bs.out')

        # Get the accurate KS-band gap
        homo, lumo = bs_calc.get_homo_lumo()

    if deprecated_syntax:
        with pytest.warns(DeprecationWarning):
            from ase.units import Ha
            from gpaw.xc.gllb.c_response import ResponsePotential

            # Calculate the discontinuity potential with accurate band gap
            response = calc.hamiltonian.xc.xcs['RESPONSE']
            response.calculate_delta_xc(homolumo=homolumo / Ha)

            # Calculate the discontinuity using the band structure calculator
            bs_response = bs_calc.hamiltonian.xc.xcs['RESPONSE']
            # Copy response potential to avoid recalculation
            pot = ResponsePotential(response,
                                    response.Dxc_vt_sG,
                                    None,
                                    response.Dxc_D_asp,
                                    response.Dxc_Dresp_asp)
            pot.redistribute(bs_response)
            bs_response.Dxc_vt_sG = pot.vt_sG
            bs_response.Dxc_D_asp = pot.D_asp
            bs_response.Dxc_Dresp_asp = pot.Dresp_asp
            KS_gap, dxc = bs_response.calculate_delta_xc_perturbation()
    else:
        # Calculate the discontinuity potential with accurate band gap
        response = calc.hamiltonian.xc.response
        dxc_pot = response.calculate_discontinuity_potential(homo, lumo)

        # Calculate the discontinuity using the band structure calculator
        bs_response = bs_calc.hamiltonian.xc.response
        KS_gap, dxc = bs_response.calculate_discontinuity(dxc_pot)

        # Test alternative (testing) implementation
        if world.size == 1:
            with pytest.warns(Warning):
                dxc_ave = bs_response.calculate_discontinuity_from_average(
                    dxc_pot, 0, True)
            assert dxc_ave == pytest.approx(dxc_ave_ref, abs=1e-4)

    QP_gap = KS_gap + dxc
    print('KS gap', KS_gap)
    print('QP gap', QP_gap)
    assert KS_gap == pytest.approx(lumo - homo, abs=1e-10)
    assert KS_gap == pytest.approx(KS_gap_ref, abs=1e-4)
    assert QP_gap == pytest.approx(QP_gap_ref, abs=1e-4)
