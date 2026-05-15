import numpy as np
import pytest

from ase.parallel import world, parprint
from ase.units import Bohr
from gpaw import GPAW
from gpaw.lcao.dipoletransition import get_dipole_transitions
from gpaw.lrtddft.kssingle import KSSingles


@pytest.mark.old_gpaw_only
def test_dipole_transition(gpw_files, tmp_path_factory):
    """Check dipole matrix-elements for Li."""
    calc = GPAW(gpw_files['bcc_li_lcao'], parallel=dict(sl_auto=True))
    # Initialize calculator if necessary
    if not hasattr(calc.wfs, 'C_nM'):
        calc.wfs.set_positions
        calc.initialize_positions(calc.atoms)
    from gpaw.kohnsham_layouts import BlacsOrbitalLayouts
    isblacs = isinstance(calc.wfs.ksl, BlacsOrbitalLayouts)  # XXX
    print(calc.wfs.ksl.using_blacs, isblacs)
    dip_skvnm = get_dipole_transitions(calc.wfs)
    parprint("Dipole moments calculated")
    assert dip_skvnm.shape == (1, 4, 3, 4, 4)
    dip_kvnm = dip_skvnm[0] * Bohr

    print(world.rank, dip_kvnm[:, 0, 0, 3])

    # check symmetry: abs(d[i,j]) == abs(d[j,i])
    for k in range(4):
        for v in range(3):
            dip_kvnm[k, v].T == pytest.approx(dip_kvnm[k, v])

    # Check numerical value of a few elements - signs might change!
    assert 0.0824 == pytest.approx(abs(dip_kvnm[0, 0, 0, 1]), abs=1e-4)
    assert abs(-0.0781 + 0.0282j) == pytest.approx(abs(dip_kvnm[1, 2, 0, 3]),
                                                   abs=1e-4)

    calc = GPAW(gpw_files['bcc_li_fd'])
    # compare to lrtddft implementation
    kss = KSSingles()
    atoms = calc.atoms
    atoms.calc = calc
    kss.calculate(calc.atoms, 0)
    lrrefv = []
    for ex in kss:
        print(ex)
        lrrefv.append(-1. * ex.muv * Bohr)
    lrrefv = np.array(lrrefv)

    # some printout for manual inspection, if wanted
    parprint("               lrtddft(fd)(v)        raman(v)")
    f = "{} {:+.4f}    {:+.4f}"
    # At gamma at three excited states are degenerate, so order is a problem
    parprint(f.format('k=0, 0->1 (x)', lrrefv[0, 0], dip_kvnm[0, 0, 0, 1]))
    parprint(f.format('k=0, 0->1 (y)', lrrefv[0, 1], dip_kvnm[0, 1, 0, 1]))
    parprint(f.format('k=0, 0->1 (z)', lrrefv[0, 2], dip_kvnm[0, 2, 0, 1]))
    parprint(f.format('k=0, 0->2 (x)', lrrefv[1, 0], dip_kvnm[0, 0, 0, 2]))
    parprint(f.format('k=0, 0->2 (y)', lrrefv[1, 1], dip_kvnm[0, 1, 0, 2]))
    parprint(f.format('k=0, 0->2 (z)', lrrefv[1, 2], dip_kvnm[0, 2, 0, 2]))
    parprint(f.format('k=0, 0->3 (x)', lrrefv[2, 0], dip_kvnm[0, 0, 0, 3]))
    parprint(f.format('k=0, 0->3 (y)', lrrefv[2, 1], dip_kvnm[0, 1, 0, 3]))
    parprint(f.format('k=0, 0->3 (z)', lrrefv[2, 2], dip_kvnm[0, 2, 0, 3]))
    parprint("")
    # At 2nd kpoint 2nd and 3rd excited state almost degenerate in LCAO,
    # order might be different from FD mode
    parprint(f.format('k=1, 0->1 (x)', lrrefv[3, 0], dip_kvnm[1, 0, 0, 1]))
    parprint(f.format('k=1, 0->1 (y)', lrrefv[3, 1], dip_kvnm[1, 1, 0, 1]))
    parprint(f.format('k=1, 0->1 (z)', lrrefv[3, 2], dip_kvnm[1, 2, 0, 1]))

    parprint(f.format('k=1, 0->2 (x)', lrrefv[4, 0], dip_kvnm[1, 0, 0, 2]))
    parprint(f.format('k=1, 0->2 (y)', lrrefv[4, 1], dip_kvnm[1, 1, 0, 2]))
    parprint(f.format('k=1, 0->2 (z)', lrrefv[4, 2], dip_kvnm[1, 2, 0, 2]))
    parprint(f.format('k=1, 0->3 (x)', lrrefv[5, 0], dip_kvnm[1, 0, 0, 3]))
    parprint(f.format('k=1, 0->3 (y)', lrrefv[5, 1], dip_kvnm[1, 1, 0, 3]))
    parprint(f.format('k=1, 0->3 (z)', lrrefv[5, 2], dip_kvnm[1, 2, 0, 3]))
    parprint("")
    # At the third k-point first and second excited state are degenerate
    parprint(f.format('k=2, 0->3 (x)', lrrefv[8, 0], dip_kvnm[2, 0, 0, 3]))
    parprint(f.format('k=2, 0->3 (y)', lrrefv[8, 1], dip_kvnm[2, 1, 0, 3]))
    parprint(f.format('k=2, 0->3 (z)', lrrefv[8, 2], dip_kvnm[2, 2, 0, 3]))
    # 4th k-point not included KSSingles, probably all bands above Fermi level
