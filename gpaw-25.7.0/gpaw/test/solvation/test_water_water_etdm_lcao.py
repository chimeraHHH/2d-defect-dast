from gpaw import GPAW
from gpaw.utilities.adjust_cell import adjust_cell
import pytest
from ase.build import molecule
from ase.units import mol, kcal
from gpaw.directmin.etdm_lcao import LCAOETDM
from gpaw.solvation import SolvationGPAW, get_HW14_water_kwargs


@pytest.mark.old_gpaw_only
def test_solvation_water_water_etdm_lcao():
    SKIP_VAC_CALC = True

    h = 0.24
    vac = 4.0
    convergence = {
        'energy': 0.05 / 8.,
        'density': 10.,
        'eigenstates': 10.,
    }

    atoms = molecule('H2O')
    adjust_cell(atoms, vac, h)

    if not SKIP_VAC_CALC:
        atoms.calc = GPAW(mode='lcao', xc='PBE', h=h, basis='dzp',
                          occupations={'name': 'fixed-uniform'},
                          eigensolver='etdm-lcao',
                          mixer={'backend': 'no-mixing'},
                          nbands='nao', symmetry='off',
                          convergence=convergence)
        Evac = atoms.get_potential_energy()
        print(Evac)
    else:
        # h=0.24, vac=4.0, setups: 0.9.20000, convergence: only energy 0.05 / 8
        Evac = -12.68228003345474

    atoms.calc = SolvationGPAW(mode='lcao', xc='PBE', h=h, basis='dzp',
                               occupations={'name': 'fixed-uniform'},
                               eigensolver=LCAOETDM(
                                   linesearch_algo={'name': 'max-step'}),
                               mixer={'backend': 'no-mixing'},
                               nbands='nao', symmetry='off',
                               convergence=convergence,
                               **get_HW14_water_kwargs())
    Ewater = atoms.get_potential_energy()
    Eelwater = atoms.calc.get_electrostatic_energy()
    Esurfwater = atoms.calc.get_solvation_interaction_energy('surf')
    DGSol = (Ewater - Evac) / (kcal / mol)
    print('Delta Gsol: %s kcal / mol' % DGSol)

    assert DGSol == pytest.approx(-6.3, abs=2.)
    assert Ewater == pytest.approx(Eelwater + Esurfwater, abs=1e-14)
