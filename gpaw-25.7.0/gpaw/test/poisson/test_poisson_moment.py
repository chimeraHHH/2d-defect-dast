import numpy as np
import pytest

from ase.units import Bohr
from gpaw.poisson import PoissonSolver, NoInteractionPoissonSolver
from gpaw.poisson_moment import MomentCorrectionPoissonSolver, MomentCorrection
from gpaw.poisson_extravacuum import ExtraVacuumPoissonSolver
from gpaw.grid_descriptor import GridDescriptor


@pytest.mark.parametrize('moment_corrections, expected_len', [
    (None, 0),
    ([], 0),
    (4, 1),
    (9, 1),
    ([dict(moms=range(4), center=np.array([1, 3, 5]))], 1),
    ([dict(moms=range(4), center=np.array([5, 3, 5])),
     dict(moms=range(4), center=np.array([7, 5, 3]))], 2)
])
def test_defaults(moment_corrections, expected_len):
    poisson_ref = NoInteractionPoissonSolver()
    poisson = MomentCorrectionPoissonSolver(
        poissonsolver=poisson_ref,
        moment_corrections=moment_corrections)

    assert isinstance(poisson.moment_corrections, list), \
        poisson.moment_corrections
    assert len(poisson.moment_corrections) == expected_len
    assert all([isinstance(mom, MomentCorrection)
                for mom in poisson.moment_corrections])


@pytest.mark.parametrize('moment_corrections', [
    None,
    [],
])
def test_description_empty(moment_corrections):
    poisson_ref = NoInteractionPoissonSolver()
    poisson = MomentCorrectionPoissonSolver(
        poissonsolver=poisson_ref,
        moment_corrections=moment_corrections)

    desc = poisson.get_description()
    desc_ref = poisson_ref.get_description()

    assert isinstance(desc, str)
    assert isinstance(desc_ref, str)
    assert desc_ref in desc
    assert '0 moment corrections' in desc


@pytest.mark.parametrize('moment_corrections, expected_strings', [
    (4, ['1 moment corrections', 'center', 'range(0, 4)']),
    (9, ['1 moment corrections', 'center', 'range(0, 9)']),
    ([dict(moms=range(4), center=np.array([1, 1, 1]))],
        ['1 moment corrections', '[1.00, 1.00, 1.00]', 'range(0, 4)']),
    ([dict(moms=[1, 2, 3], center=np.array([1, 1, 1]))],
        ['1 moment corrections', '[1.00, 1.00, 1.00]', 'range(1, 4)']),
    ([dict(moms=[0, 2, 3], center=np.array([1, 1, 1]))],
        ['1 moment corrections', '[1.00, 1.00, 1.00]', '(0, 2, 3)']),
    ([dict(moms=range(4), center=np.array([2, 3, 4])),
      dict(moms=range(4), center=np.array([7.4, 3.1, 0.1]))],
        ['2 moment corrections', '[2.00, 3.00, 4.00]',
         '[7.40, 3.10, 0.10]', 'range(0, 4)']),
])
def test_description(moment_corrections, expected_strings):
    poisson_ref = NoInteractionPoissonSolver()
    poisson = MomentCorrectionPoissonSolver(
        poissonsolver=poisson_ref,
        moment_corrections=moment_corrections)

    desc = poisson.get_description()
    desc_ref = poisson_ref.get_description()

    assert isinstance(desc, str)
    assert isinstance(desc_ref, str)

    # Make sure that the description starts with the description of the wrapped
    # solver
    assert desc.startswith(desc_ref)

    # and follows with the moments
    desc_rem = desc[len(desc_ref):]
    for expected_str in expected_strings:
        assert expected_str in desc_rem, \
            f'"{expected_str}" not in "{desc_rem}"'


@pytest.mark.parametrize('moment_corrections, expected_string', [
    ([], 'no corrections'),
    (4, 'array([0, 1, 2, 3]) @ None'),
    (9, 'array([0, 1, 2, 3, 4, 5, 6, 7, 8]) @ None'),
    ([dict(moms=range(4), center=np.array([1., 1., 1.]))],
        'array([0, 1, 2, 3]) @ array([1., 1., 1.])'),
    ([dict(moms=[1, 2, 3], center=np.array([1., 1., 1.]))],
        'array([1, 2, 3]) @ array([1., 1., 1.])'),
    ([dict(moms=[0, 2, 3], center=np.array([1., 1., 1.]))],
        'array([0, 2, 3]) @ array([1., 1., 1.])'),
    ([dict(moms=range(4), center=np.array([2, 3, 4])),
      dict(moms=range(4), center=np.array([7.4, 3.1, 0.1]))],
        '2 corrections'),
])
def test_repr(moment_corrections, expected_string):
    poisson_ref = NoInteractionPoissonSolver()
    poisson = MomentCorrectionPoissonSolver(
        poissonsolver=poisson_ref,
        moment_corrections=moment_corrections)

    rep = repr(poisson)
    expected_repr = f'MomentCorrectionPoissonSolver ({expected_string})'

    assert isinstance(rep, str)
    assert rep == expected_repr, f'{rep} not equal to {expected_repr}'


@pytest.fixture
def gd():
    N_c = (16, 16, 3 * 16)
    cell_cv = (1, 1, 3)
    gd = GridDescriptor(N_c, cell_cv, False)

    return gd


@pytest.mark.parametrize('moment_corrections', [
    4,
    9,
    [dict(moms=range(4), center=np.array([1, 1, 1]))],
    [dict(moms=range(4), center=np.array([2, 3, 4])),
     dict(moms=range(4), center=np.array([7.4, 3.1, 0.1]))],
])
def test_write(gd, moment_corrections):
    poisson_ref = PoissonSolver()
    poisson_ref.set_grid_descriptor(gd)

    poisson = MomentCorrectionPoissonSolver(
        poissonsolver=poisson_ref,
        moment_corrections=moment_corrections)
    poisson.set_grid_descriptor(gd)

    from gpaw.io import Writer
    from gpaw.mpi import world
    filename = '/dev/null'

    # By using the Writer we check that everything is JSON serializable
    writer = Writer(filename, world)
    writer.child('poisson').write(**poisson.todict())
    writer.close()


@pytest.fixture
def rho_g(gd):
    # Construct model density
    coord_vg = gd.get_grid_point_coordinates()
    z_g = coord_vg[2, :]
    rho_g = gd.zeros()
    for z0 in [1, 2]:
        rho_g += 10 * (z_g - z0) * \
            np.exp(-20 * np.sum((coord_vg.T - np.array([.5, .5, z0])).T**2,
                                axis=0))

    return rho_g


@pytest.fixture
def poisson_solve(gd, rho_g):

    def _poisson_solve(poisson):
        poisson.set_grid_descriptor(gd)
        phi_g = gd.zeros()
        poisson.solve(phi_g, rho_g)

        return phi_g

    return _poisson_solve


@pytest.fixture
def compare(gd, tolerance, cmp_begin):
    # Some test cases compare in only a small region of space
    if cmp_begin is None:
        slice = None
    else:
        Ng_c = gd.get_size_of_global_array()
        cmp_end = 1 - cmp_begin
        idx_c = [np.arange(int(N * cmp_begin), int(N * cmp_end)) for N in Ng_c]
        slice = np.ix_(*idx_c)

    def _compare(phi1_g, phi2_g):
        big_phi1_g = gd.collect(phi1_g)
        big_phi2_g = gd.collect(phi2_g)
        if gd.comm.rank == 0:
            if slice is not None:
                big_phi1_g = big_phi1_g[slice]
                big_phi2_g = big_phi2_g[slice]
            assert np.max(np.absolute(big_phi1_g - big_phi2_g)) == (
                pytest.approx(0.0, abs=tolerance))

    return _compare


@pytest.fixture
def poisson_ref(gd, ref):
    poisson_default = PoissonSolver()
    if ref == 'default':
        # Get reference from default poissonsolver
        # Using the default solver the potential is forced to zero at the box
        # boundries. The potential thus has the wrong shape near the boundries
        # but is nearly right in the center of the box
        return poisson_default
    elif ref == 'extravac':
        # Get reference from extravacuum solver
        # With 4 times extra vacuum the potential is well converged everywhere
        poisson_extravac = ExtraVacuumPoissonSolver(
            gpts=4 * gd.N_c,
            poissonsolver_large=poisson_default)
        return poisson_extravac
    else:
        raise ValueError(f'No such ref {ref}')


@pytest.mark.parametrize('ref, moment_corrections, tolerance, cmp_begin', [
    # MomentCorrectionPoissonSolver without any moment corrections should be
    # exactly as the underlying solver
    ('default', None, 0.0, None),
    # It should also be possible to chain default+extravacuum+moment correction
    # With moment_correction=None the MomentCorrection solver doesn't actually
    # do anything, so the potential should be identical to the extra vacuum
    # reference
    ('extravac', None, 0.0, None),
    # Test moment_corrections=int
    # The moment correction is applied to the center of the cell. This is not
    # enough to have a converged potential near the edges
    # The closer we are to the center the better though
    ('default', 4, 3.5e-2, 0.25),
    ('default', 4, 2.5e-2, 0.40),
    # Test moment_corrections=list
    # Remember that the solver expects Ångström units and we have specified
    # the grid in Bohr
    # This should give a well converged potential everywhere, that we can
    # compare to the reference extravacuum potential
    ('extravac',
     [{'moms': range(4), 'center': np.array([.5, .5, 1]) * Bohr},
      {'moms': range(4), 'center': np.array([.5, .5, 2]) * Bohr}],
     3e-3, None),
    # It should be possible to chain default+extravacuum+moment correction
    # As the potential is already well converged, there should be little change
    ('extravac',
     [{'moms': range(4), 'center': np.array([.5, .5, 1]) * Bohr},
      {'moms': range(4), 'center': np.array([.5, .5, 2]) * Bohr}],
     5e-4, None),
])
def test_poisson_moment_correction(gd, rho_g, poisson_solve,
                                   compare, poisson_ref,
                                   ref, moment_corrections,
                                   tolerance, cmp_begin):
    # Solve for the potential using the reference solver
    phiref_g = poisson_solve(poisson_ref)

    # Create a MomentCorrectionPoissonSolver and solve for the potential
    poisson = MomentCorrectionPoissonSolver(poissonsolver=poisson_ref,
                                            moment_corrections=None)
    phi_g = poisson_solve(poisson)

    # Test the MomentCorrectionPoissonSolver
    compare(phi_g, phiref_g)
