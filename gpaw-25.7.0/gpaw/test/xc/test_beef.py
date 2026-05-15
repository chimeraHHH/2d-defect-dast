import warnings
import pytest
import numpy as np
from ase.build import bulk
from ase.dft.bee import BEEFEnsemble, readbee
from gpaw import GPAW, Mixer, PW
from gpaw.test import gen
from gpaw.mpi import world
import gpaw.cgpaw as cgpaw


@pytest.mark.mgga
@pytest.mark.libxc
@pytest.mark.slow
@pytest.mark.parametrize('xc', ['mBEEF', 'BEEF-vdW', 'mBEEF-vdW'])
def test_beef(in_tmp_dir, xc, gpaw_new):
    if xc[0] == 'm':
        assert cgpaw.lxcXCFuncNum('MGGA_X_MBEEF') is not None

    results = {'mBEEF': (5.449, 0.056),
               'BEEF-vdW': (5.484, 0.071),
               'mBEEF-vdW': (5.426, 0.025)}

    kwargs = dict()
    if xc == 'mBEEF-vdW':
        kwargs['setups'] = dict(Si=gen('Si', xcname='PBEsol'))

    E = []
    V = []
    for a in np.linspace(5.4, 5.5, 3):
        si = bulk('Si', a=a)
        si.calc = GPAW(txt='Si-' + xc + '.txt',
                       mixer=Mixer(0.8, 7, 50.0),
                       xc=xc,
                       kpts=[2, 2, 2],
                       mode=PW(200),
                       **kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='gpaw.xc.libxc')
            E.append(si.get_potential_energy())
        ens = BEEFEnsemble(si.calc, verbose=False)
        ens.get_ensemble_energies(200)
        ens.write('Si-{}-{:.3f}'.format(xc, a))
        V.append(si.get_volume())
    p = np.polyfit(V, E, 2)
    v0 = np.roots(np.polyder(p))[0]
    a = (v0 * 4)**(1 / 3)

    a0, da0 = results[xc]

    assert abs(a - a0) < 0.002, (xc, a, a0)

    if world.rank == 0:
        E = []
        for a in np.linspace(5.4, 5.5, 3):
            e = readbee('Si-{}-{:.3f}'.format(xc, a))
            E.append(e)

        A = []
        for energies in np.array(E).T:
            p = np.polyfit(V, energies, 2)
            assert p[0] > 0, (V, E, p)
            v0 = np.roots(np.polyder(p))[0]
            A.append((v0 * 4)**(1 / 3))

        A = np.array(A)
        a = A.mean()
        da = A.std()

        print('a(ref) = {:.3f} +- {:.3f}'.format(a0, da0))
        print('a      = {:.3f} +- {:.3f}'.format(a, da))
        assert abs(a - a0) < 0.01
        assert abs(da - da0) < 0.01
