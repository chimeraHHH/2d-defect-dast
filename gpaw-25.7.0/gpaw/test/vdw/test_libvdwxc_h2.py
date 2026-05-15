import pytest
from gpaw.utilities import compiled_with_libvdwxc
from ase.build import molecule
from gpaw import GPAW, Mixer, Davidson, PW
from gpaw.xc.libvdwxc import vdw_df
from gpaw.mpi import world

pytestmark = pytest.mark.skipif(not compiled_with_libvdwxc(),
                                reason='not compiled_with_libvdwxc()')


@pytest.mark.old_gpaw_only
def test_vdw_libvdwxc_h2(in_tmp_dir):
    system = molecule('H2')
    system.center(vacuum=1.0)
    system.pbc = 1

    def calculate(mode, kwargs1):
        kwargs = dict(mode=mode,
                      basis='sz(dzp)',
                      eigensolver=Davidson(4) if mode != 'lcao' else None,
                      xc=vdw_df(),
                      h=0.25,
                      convergence=dict(energy=1e-6),
                      mixer=Mixer(0.5, 5, 10.))
        kwargs.update(kwargs1)
        calc = GPAW(**kwargs)
        system.calc = calc
        system.get_potential_energy()
        return calc

    fddomainpar = min(2, world.size)

    for mode in ['fd', 'pw', 'lcao']:
        kwargs = {}
        if mode == 'pw':
            kwargs['mode'] = PW(250)
        else:
            kwargs['parallel'] = {'domain': fddomainpar}
        calc = calculate(mode, kwargs)

    E1 = calc.get_potential_energy()
    calc.write('dump.libvdwxc.gpw')
    _ = GPAW('dump.libvdwxc.gpw', txt='restart.txt',
             parallel={'domain': fddomainpar})
    system2 = calc.get_atoms()

    # Verify same energy after restart
    E2 = system2.get_potential_energy()
    assert abs(E2 - E1) < 1e-14  # Should be exact

    # Trigger recaclulation of essentially same system
    system2.positions[0, 0] += 1e-13
    print('reconverge')
    E3 = system2.get_potential_energy()
    err2 = abs(E3 - E2)
    print('error', err2)
    assert err2 < 1e-6  # Around SCF precision
