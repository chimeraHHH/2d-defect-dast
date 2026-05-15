import pytest
from ase.build import graphene_nanoribbon, molecule
from gpaw import GPAW
from gpaw.lcao.local_orbitals import LocalOrbitals
from gpaw.mpi import world

pytestmark = pytest.mark.skipif(world.size > 1,
                                reason='world.size > 1')


@pytest.mark.old_gpaw_only
def test_gamma_point_calculation():
    atoms = molecule('C6H6', vacuum=2)

    calc = GPAW(mode='lcao',
                xc='LDA',
                basis='szp(dzp)',
                txt=None)
    atoms.calc = calc
    atoms.get_potential_energy()

    los = LocalOrbitals(calc)

    los.subdiagonalize('C', groupby='energy')
    group_energies = sorted(los.groups.keys())

    # Assert minimal model contains only pz-LOs
    los.take_model(minimal=True)
    assert los.indices == los.groups[group_energies[3]]
    assert len(los.indices) == 6

    # Assert extended model also contains +2 d-LOs
    los.take_model(minimal=False)
    assert los.indices == (
        los.groups[group_energies[3]]
        + los.groups[group_energies[-3]]
        + los.groups[group_energies[-1]])
    assert len(los.indices) == (6 * 3)


@pytest.mark.old_gpaw_only
def test_k_point_calculation():
    atoms = graphene_nanoribbon(2, 1, type='zigzag', saturated=True,
                                C_H=1.1, C_C=1.4, vacuum=2)

    calc = GPAW(mode='lcao',
                xc='LDA',
                basis='szp(dzp)',
                txt=None,
                kpts={'size': (1, 1, 11),
                      'gamma': True},
                symmetry={'point_group': False,
                          'time_reversal': True})
    atoms.calc = calc
    atoms.get_potential_energy()

    los = LocalOrbitals(calc)

    los.subdiagonalize('C', groupby='symmetry')
    group_energies = sorted(los.groups.keys())

    # Assert minimal model contains only pz-LOs
    los.take_model(minimal=True)
    assert los.indices == los.groups[group_energies[3]]
    assert len(los.indices) == 4

    # Assert extended model also contains d-LOs
    los.take_model(minimal=False)
    assert los.indices == (
        los.groups[group_energies[3]]
        + los.groups[group_energies[-3]]
        + los.groups[group_energies[-1]])
    assert len(los.indices) == (4 * 3)
