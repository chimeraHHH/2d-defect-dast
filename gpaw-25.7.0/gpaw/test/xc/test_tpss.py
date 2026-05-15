import pytest
from ase.build import molecule
from gpaw import GPAW, Davidson, Mixer, PoissonSolver
from gpaw.utilities.tools import split_formula


@pytest.mark.mgga
def test_tpss(in_tmp_dir):
    cell = [6., 6., 7.]

    # Reference from J. Chem. Phys. Vol 120 No. 15, 15 April 2004, page 6898
    tpss_de = {
        'Li2': 22.5}
    tpss_old = {
        'Li2': 20.9}

    exp_bonds_dE = {
        'Li2': (2.673, 24.4)}

    systems = ['Li2']

    # Add atoms
    for formula in systems:
        temp = split_formula(formula)
        for atom in temp:
            if atom not in systems:
                systems.append(atom)
    energies = {}

    # Calculate energies
    for formula in systems:
        loa = molecule(formula)
        loa.set_cell(cell)
        loa.center()
        calc = GPAW(mode='fd',
                    h=0.3,
                    eigensolver=Davidson(8),
                    parallel=dict(kpt=1),
                    mixer=Mixer(0.5, 5),
                    nbands=-2,
                    poissonsolver=PoissonSolver('fd', relax='GS'),
                    xc='oldPBE',
                    txt=formula + '.txt')
        if len(loa) == 1:
            calc = calc.new(hund=True)
        else:
            pos = loa.get_positions()
            pos[1, :] = pos[0, :] + [0.0, 0.0, exp_bonds_dE[formula][0]]
            loa.set_positions(pos)
            loa.center()
        loa.calc = calc
        energy = loa.get_potential_energy()
        diff = calc.get_xc_difference('TPSS')
        energies[formula] = (energy, energy + diff)
        print(formula, energy, energy + diff)

    # calculate atomization energies
    print('formula\tGPAW\tRef\tGPAW-Ref\tGPAW-exp')
    mae_ref, mae_exp, mae_pbe, count = 0.0, 0.0, 0.0, 0
    for formula in tpss_de.keys():
        atoms_formula = split_formula(formula)
        de_tpss = -1.0 * energies[formula][1]
        de_pbe = -1.0 * energies[formula][0]
        for atom_formula in atoms_formula:
            de_tpss += energies[atom_formula][1]
            de_pbe += energies[atom_formula][0]

        de_tpss *= 627.5 / 27.211
        de_pbe *= 627.5 / 27.211
        mae_ref += abs(de_tpss - tpss_de[formula])
        mae_exp += abs(de_tpss - exp_bonds_dE[formula][1])
        mae_pbe += abs(de_pbe - exp_bonds_dE[formula][1])
        count += 1
        out = ("%s\t%.1f\t%.1f\t%.1f\t%.1f kcal/mol" %
               (formula, de_tpss, tpss_de[formula],
                de_tpss - tpss_de[formula],
                de_tpss - exp_bonds_dE[formula][1]))
        print(out)

        # comparison to gpaw revision 5450 version value
        # in kcal/mol (note the grid:0.3 Ang)
        assert de_tpss == pytest.approx(tpss_old[formula], abs=0.15)
