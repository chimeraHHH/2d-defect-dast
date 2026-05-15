import numpy as np
from gpaw.mpi import world
from gpaw.berryphase import polarization_phase, ionic_phase
from ase.parallel import paropen, parprint
from ase.io.jsonio import write_json, read_json
from pathlib import Path


def born_charges_wf(atoms, calc, delta=0.01, cleanup=False,
                    ionic_only=False, out='born_charges.json'):

    # generate displacement dictionary
    disps_av = _all_disp(atoms, delta)

    # carry out polarization phase calculation
    # for each displacement
    phases_c = {}
    for dlabel in disps_av:
        ia, iv, sign, delta = disps_av[dlabel]
        atoms_d = displace_atom(atoms, ia, iv, sign, delta)
        check_distance_to_non_pbc_boundary(atoms_d)

        if not ionic_only:

            # proper polarization phase calculation
            gpw_wfs = Path(dlabel + '.gpw')
            berryname = Path(dlabel + '_berry-phases.json')
            if not berryname.is_file():
                if not gpw_wfs.is_file():

                    # run calculations
                    atoms_d.calc = calc
                    assert is_symmetry_off(atoms_d.calc), 'Set symmetry off'
                    atoms_d.get_potential_energy()

                    # write wavefunctions
                    atoms_d.calc.write(gpw_wfs, 'all')

                # dict with entries phase_c, electronic_phase_c
                # atomic_phase_c, dipole_moment_c
                phase_c = polarization_phase(gpw_wfs, comm=world)

                # only master rank should write
                with paropen(berryname, 'w') as fd:
                    write_json(fd, phase_c)

            else:
                # all ranks can read
                with open(berryname, 'r') as fd:
                    phase_c = read_json(fd)

            if cleanup:
                if berryname.is_file():
                    # remove gpw file
                    if world.rank == 0:
                        gpw_wfs.unlink()
        else:
            # only atomic contribution considered
            # for unexpensive testing only
            phase_c = ionic_phase(atoms_d)

        phases_c[dlabel] = phase_c['phase_c']

    results = born_charges(atoms, disps_av, phases_c, check=(not ionic_only))
    with paropen(out, 'w') as fd:
        write_json(fd, results)

    return results


def is_symmetry_off(calc):
    params = calc.parameters
    if calc.old:
        if 'symmetry' in params:
            return params['symmetry'] == 'off'
        else:
            return False
    else:
        # new:
        return (not params.symmetry.point_group and
                not params.symmetry.time_reversal)


def born_charges(atoms, disps_av, phases_c, check=True):

    natoms = len(atoms)
    cell_cv = atoms.get_cell()
    vol = abs(np.linalg.det(cell_cv))
    sym_a = atoms.get_chemical_symbols()

    ndisp = len(disps_av)
    parprint('Not using symmetry: ndisp:', ndisp)

    # obtain phi(dr) map
    phi_ascv = np.zeros((natoms, 2, 3, 3), float)
    for dlabel in disps_av:
        ia, iv, sign, delta = disps_av[dlabel]
        isign = [None, 1, 0][sign]
        phi_ascv[ia, isign, :, iv] = phases_c[dlabel]

    # calculate dphi / dr
    # exploit +- displacement
    dphi_acv = phi_ascv[:, 1] - phi_ascv[:, 0]
    # mod 2 pi
    mod_acv = np.round(dphi_acv / (2 * np.pi)) * 2 * np.pi
    dphi_acv -= mod_acv
    # transform to cartesian
    dphi_avv = np.array([np.dot(dphi_cv.T, cell_cv).T for dphi_cv in dphi_acv])
    dphi_dr_avv = dphi_avv / (2.0 * delta)

    # calculate polarization change and born charges
    dP_dr_avv = dphi_dr_avv / (2 * np.pi * vol)
    Z_avv = dP_dr_avv * vol

    if check:
        # check acoustic sum rule: sum_a Z_aij = 0 for all i,j
        asr_vv = np.sum(Z_avv, axis=0)
        asr_dev = np.abs(asr_vv).max() / natoms
        assert asr_dev < 1e-1, f'Acoustic sum rule violated: {asr_vv}'

        # correct to match acoustic sum rule
        Z_avv -= asr_vv[None, :, :] / natoms

    results = {'Z_avv': Z_avv, 'sym_a': sym_a}

    return results


def _cartesian_label(ia, iv, sign):
    """Generate name from (ia, iv, sign).
    ia ... atomic_index
    iv ... cartesian_index
    sign ... +-
    """

    sym_v = 'xyz'[iv]
    sym_s = ' +-'[sign]
    return f'{ia}{sym_v}{sym_s}'


def _all_avs(atoms):
    """Generate ia, iv, sign for all displacements."""
    for ia in range(len(atoms)):
        for iv in range(3):
            for sign in [-1, 1]:
                yield (ia, iv, sign)


def _all_disp(atoms, delta):
    all_disp = {}
    for dd, avs in enumerate(_all_avs(atoms)):
        dd = int(dd)
        lavs = _cartesian_label(*avs)
        label = f'disp_{dd:03d}_' + lavs
        all_disp[label] = (*avs, delta)
    return all_disp


def displace_atom(atoms, ia, iv, sign, delta):
    new_atoms = atoms.copy()
    pos_av = new_atoms.get_positions()
    pos_av[ia, iv] += sign * delta
    new_atoms.set_positions(pos_av)
    return new_atoms


def check_distance_to_non_pbc_boundary(atoms, eps=1):
    dist_a = distance_to_non_pbc_boundary(atoms)
    if dist_a is not None and np.any(dist_a < eps):
        raise AtomsTooCloseToBoundary(
            'The atoms are too close to a non-pbc boundary '
            'which creates problems when using a dipole correction. '
            f'Please center the atoms in the unit-cell. Distances: {dist_a}.'
        )


def distance_to_non_pbc_boundary(atoms):
    pbc_c = atoms.get_pbc()
    if pbc_c.all():
        return None
    cell_cv = atoms.get_cell()
    pos_ac = atoms.get_scaled_positions()
    pos_ac -= np.round(pos_ac)
    posnonpbc_av = np.dot(pos_ac[:, ~pbc_c], cell_cv[~pbc_c])
    dist_to_cell_edge_a = np.linalg.norm(posnonpbc_av, axis=1)
    return dist_to_cell_edge_a


class AtomsTooCloseToBoundary(Exception):
    pass
