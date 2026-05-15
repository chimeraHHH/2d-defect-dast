# General modules
import pytest
import numpy as np

# Script modules
from ase.build import bulk

from gpaw import GPAW, PW, FermiDirac
from gpaw.response import ResponseGroundStateAdapter
from gpaw.response.chi0 import Chi0Calculator, get_frequency_descriptor


@pytest.mark.response
@pytest.mark.serial
def test_si_update_consistency(in_tmp_dir):
    """Test that we get consistent results, when calculating
    chi0 in one or multiple calls to update_chi0."""

    # ---------- Inputs ---------- #

    # Ground state calculation
    xc = 'LDA'
    a = 5.431
    pw = 200
    nbands = 8
    kpts = 4
    occw = 0.05

    # Response calculation
    q_c = np.array([0., 0., 0.])
    intermediate_m = 5

    # ---------- Script ---------- #

    # Ground state calculation
    atoms = bulk('Si', 'diamond', a=a)
    atoms.center()
    calc = GPAW(mode=PW(pw),
                nbands=nbands,
                kpts=(kpts, kpts, kpts),
                parallel={'domain': 1},
                occupations=FermiDirac(width=occw),
                xc=xc)

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('Si', 'all')

    # Response calculation
    gs = ResponseGroundStateAdapter.from_gpw_file('Si')
    chi0 = Chi0Calculator(gs, wd=get_frequency_descriptor(gs=gs),
                          hilbert=True, intraband=False)

    chi0_full = chi0.create_chi0(q_c)
    chi0_steps = chi0.create_chi0(q_c)
    spins = range(gs.nspins)
    # Add chi0 contribution from all the unoccupied bands
    chi0.update_chi0(chi0_full, m1=gs.nocc1, m2=nbands, spins=spins)
    # Add chi0 contribution from *some* of the unoccupied bands
    chi0.update_chi0(chi0_steps, m1=gs.nocc1, m2=intermediate_m, spins=spins)
    # Add chi0 contribution from the remaining unoccupied bands
    chi0.update_chi0(chi0_steps, m1=intermediate_m, m2=nbands, spins=spins)

    # Compare the output chi0 body
    assert chi0_steps.chi0_WgG == pytest.approx(chi0_full.chi0_WgG,
                                                abs=1e-8, rel=1e-6)
