"""check if fhc can be disabled for libxc >= 7.x.x (needed for mggas)."""
import pytest
import gpaw.cgpaw as cgpaw
from ase.build import molecule
from gpaw import GPAW
from gpaw.utilities.adjust_cell import adjust_cell

vacuum = 4.0
h = 0.3


@pytest.mark.mgga
@pytest.mark.libxc
def test_mgga_lxc_fhc():
    libxc_version = getattr(cgpaw, 'libxc_version', '2.x.y')
    if int(libxc_version.split('.')[0]) < 7:
        from unittest import SkipTest
        raise SkipTest
    cluster = molecule('CO')
    adjust_cell(cluster, border=vacuum, h=h)
    calc = GPAW(xc='MGGA_X_TPSS+MGGA_C_TPSS',
                mode='fd',
                h=h,
                maxiter=14,
                convergence={
                    'energy': 0.5,
                    'density': 1.0e-1,
                    'eigenstates': 4.0e-1})
    cluster.calc = calc
    cluster.get_potential_energy()
