import pytest
from gpaw.utilities import compiled_with_libvdwxc

from ase.build import bulk

from gpaw import GPAW, Davidson, Mixer, PW
from gpaw.xc.libvdwxc import vdw_mbeef

from gpaw.test import gen

pytestmark = pytest.mark.skipif(not compiled_with_libvdwxc(),
                                reason='not compiled_with_libvdwxc()')


@pytest.mark.old_gpaw_only
@pytest.mark.mgga
def test_vdw_libvdwxc_mbeef():
    setup = gen('Si', xcname='PBEsol')

    system = bulk('Si')
    calc = GPAW(mode=PW(200), xc=vdw_mbeef(),
                kpts=(2, 2, 2),
                nbands=4,
                convergence=dict(density=1e-6),
                mixer=Mixer(1.0),
                eigensolver=Davidson(4),
                setups={'Si': setup})
    system.calc = calc
    e = system.get_potential_energy()
    ref = -60.53662105617721
    err = abs(e - ref)
    print('e=%r ref=%r err=%r' % (e, ref, err))
    # It would be reasonable to put 1e-6 as tolerance,
    # but the value changes by 4e-4 depending on libxc version.
    # See https://gitlab.com/gpaw/gpaw/issues/161 .
    assert err < 1e-3, err
