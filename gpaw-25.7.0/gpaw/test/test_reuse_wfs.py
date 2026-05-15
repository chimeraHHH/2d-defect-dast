import pytest
from ase.build import bulk

from gpaw import GPAW, Mixer
from gpaw.convergence_criteria import Eigenstates


class MyConvergenceCriterion(Eigenstates):
    def __init__(self, tol):
        super().__init__(tol)
        self.history = []

    def get_error(self, context):
        value = super().get_error(context)
        self.history.append(value)
        return value


def run(atoms, method, kwargs):
    conv_tol = 1e-9
    conv = MyConvergenceCriterion(conv_tol)

    kwargs = {
        'convergence': {'custom': [conv]},
        'experimental': {'reuse_wfs_method': method},
        **kwargs}

    calc = GPAW(**kwargs)

    atoms.calc = calc
    with pytest.warns(UserWarning,
                      match='Custom convergence criterion'):
        E1 = atoms.get_potential_energy()
    assert conv.history[-1] < conv_tol
    niter1 = len(conv.history)
    del conv.history[:]

    atoms.rattle(stdev=0.0001)

    if method is None and not calc.old:
        calc.dft.ibzwfs.move_wave_functions = lambda *args: None

    E2 = atoms.get_potential_energy()
    niter2 = len(conv.history)
    reuse_error = conv.history[0]

    # If the change in energy is exactly or suspiciously close to zero, it's
    # because nothing was done at all (something was cached but shouldn't
    # have been)
    delta_e = abs(E2 - E1)
    assert delta_e > 1e-6, delta_e
    return niter1, niter2, reuse_error


@pytest.mark.parametrize('mode, reuse_type, max_reuse_error', [
    ('pw', 'paw', 1e-5),
    ('pw', None, 1e-4),
    ('fd', 'paw', 1e-4),
    ('fd', None, 1e-3)])
def test_reuse_wfs(mode, reuse_type, max_reuse_error):
    """Check that wavefunctions are meaningfully reused.

    For a different modes and parameters, this test asserts that the
    initial wavefunction error in the second scf step is below a
    certain threshold, indicating that we are doing better than if
    we started from scratch."""

    atoms = bulk('Si')
    atoms.rattle(stdev=0.01, seed=17)  # Break symmetry

    kwargs = dict(
        mode=mode,
        kpts=(2, 2, 2),
        xc='PBE',
        mixer=Mixer(0.4, 5, 20.0))

    niter1, niter2, reuse_error = run(
        atoms, reuse_type, kwargs)

    # It should at the very least be faster to do the second step:
    assert niter2 < niter1
    assert reuse_error < max_reuse_error


# @pytest.mark.old_gpaw_only
def test_reuse_sg15(sg15_hydrogen):
    """Test wfs reuse with sg15.

    As of writing this test, the sg15 pseudopotentials have no pseudo
    partial waves, and therefore, reusing them should have no effect."""
    from ase.build import molecule
    atoms = molecule('H2', vacuum=2.0)

    # If we do not rattle, we will get broken symmetry error:
    atoms.rattle(stdev=.001)

    kwargs = dict(
        mode='pw',
        setups={'H': sg15_hydrogen},
        xc='PBE')

    niter1, niter2, reuse_error = run(atoms, 'paw', kwargs)
    assert niter2 < niter1
    assert reuse_error < 1e-5
