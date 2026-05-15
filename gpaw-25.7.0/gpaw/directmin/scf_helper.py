import warnings
import numpy as np

from ase.units import Ha

from gpaw.directmin.tools import (sort_orbitals_according_to_energies,
                                  get_n_occ)


def do_if_converged(eigensolver_name, wfs, ham, dens, log):
    name = eigensolver_name
    if name == 'etdm-lcao' or name == 'etdm-fdpw':
        occ_name = getattr(wfs.occupations, 'name', None)
        solver = wfs.eigensolver
        if hasattr(solver, 'dm_helper'):
            func_name = solver.dm_helper.func.name
        elif hasattr(solver, 'odd'):
            func_name = solver.odd.name
        sic_calc = 'SIC' in func_name
    else:
        return

    if hasattr(solver, 'e_sic'):
        e_sic = solver.e_sic
    else:
        e_sic = 0.0

    if hasattr(solver, 'constraints'):
        constraints = solver.constraints
    else:
        constraints = None

    if eigensolver_name == 'etdm-lcao':
        with ((wfs.timer('Get canonical representation'))):
            for kpt in wfs.kpt_u:
                solver.dm_helper.update_to_canonical_orbitals(
                    wfs, ham, kpt, False, False)

        log('\nOccupied states converged after'
            ' {:d} e/g evaluations'.format(solver.eg_count))

    elif eigensolver_name == 'etdm-fdpw':
        solver.choose_optimal_orbitals(wfs)
        niter1 = solver.eg_count
        niter2 = 0
        niter3 = 0

        iloop1 = solver.iloop is not None
        iloop2 = solver.outer_iloop is not None
        if iloop1:
            niter2 = solver.total_eg_count_iloop
        if iloop2:
            niter3 = solver.total_eg_count_outer_iloop

        if iloop1 and iloop2:
            log(
                '\nOccupied states converged after'
                ' {:d} KS and {:d} SIC e/g '
                'evaluations'.format(niter3,
                                     niter2 + niter3))
        elif not iloop1 and iloop2:
            log(
                '\nOccupied states converged after'
                ' {:d} e/g evaluations'.format(niter3))
        elif iloop1 and not iloop2:
            log(
                '\nOccupied states converged after'
                ' {:d} KS and {:d} SIC e/g '
                'evaluations'.format(niter1, niter2))
        else:
            log(
                '\nOccupied states converged after'
                ' {:d} e/g evaluations'.format(niter1))
        if solver.converge_unocc:
            log('Converge unoccupied states:')
            max_er = wfs.eigensolver.error
            max_er *= Ha ** 2 / wfs.nvalence
            solver.run_unocc(ham, wfs, dens, max_er, log)
        else:
            log('Unoccupied states are not converged.')
        solver.initialized = False

        rewrite_psi = True
        if sic_calc:
            rewrite_psi = False

        solver.get_canonical_representation(ham, wfs, rewrite_psi)

    if occ_name == 'mom':
        check_mom_no_update_of_occupations(wfs)

    solver.update_ks_energy(ham, wfs, dens)
    ham.get_energy(0.0, wfs, kin_en_using_band=False, e_sic=e_sic)
    sort_orbitals_according_to_energies(ham, wfs, constraints)

    if eigensolver_name == 'etdm-lcao':
        solver.set_ref_orbitals_and_a_vec(wfs)

    if occ_name == 'mom':
        not_update = not wfs.occupations.update_numbers
        if not_update:
            wfs.occupations.numbers = solver.initial_occupation_numbers


def check_eigensolver_state(eigensolver_name, wfs, ham, dens, log):
    solver = wfs.eigensolver
    name = eigensolver_name
    if name == 'etdm-lcao' or name == 'etdm-fdpw':
        solver.eg_count = 0
        solver.globaliters = 0

        if hasattr(solver, 'iloop'):
            if solver.iloop is not None:
                solver.iloop.total_eg_count = 0
        if hasattr(solver, 'outer_iloop'):
            if solver.outer_iloop is not None:
                solver.outer_iloop.total_eg_count = 0

        solver.check_assertions(wfs, dens)
        if (hasattr(solver, 'dm_helper') and solver.dm_helper is None) \
                or not solver.initialized:
            solver.initialize_dm_helper(wfs, ham, dens, log)


def check_mom_no_update_of_occupations(wfs):
    f_sn = wfs.occupations.update_occupations()
    for kpt in wfs.kpt_u:
        k = wfs.kd.nibzkpts * kpt.s + kpt.q
        n_occ, occupied = get_n_occ(kpt)
        if n_occ != 0.0 and np.min(f_sn[k][:n_occ]) == 0:
            warnings.warn('MOM has detected variational collapse '
                          'after getting canonical orbitals. Check '
                          'that the orbitals are consistent with the '
                          'initial guess.')
