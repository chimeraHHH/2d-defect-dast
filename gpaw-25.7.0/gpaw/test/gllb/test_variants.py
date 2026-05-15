import numpy as np
import pytest
from ase.build import bulk
from gpaw import GPAW
from gpaw.test import gen


def run(xc):
    setup_name = {'GLLBLDA': 'LDA', 'GLLBPBE': 'PBE', 'GLLBM': 'GLLB'}.get(xc,
                                                                           xc)
    if setup_name in ['GLLB', 'GLLBC', 'GLLBCP86', 'GLLBNORESP']:
        gen('Si', xcname=setup_name, write_xml=True)

    atoms = bulk('Si')
    calc = GPAW(mode='lcao',
                basis='sz(dzp)',
                h=0.3,
                nbands=8,
                xc=xc,
                convergence={'density': 1e-5},
                kpts={'size': (1, 1, 1), 'gamma': True},
                txt=f'{xc}.out')
    atoms.calc = calc
    atoms.get_potential_energy()
    eig_n = calc.get_eigenvalues(kpt=0)
    return eig_n


@pytest.mark.gllb
@pytest.mark.libxc
@pytest.mark.parametrize('xc', ['GLLBLDA', 'GLLBPBE'])
def test_wrappers(xc, in_tmp_dir, add_cwd_to_setup_paths):
    eig_n = run(xc)

    # Check values against regular xc
    ref_eig_n = run({'GLLBLDA': 'LDA', 'GLLBPBE': 'PBE'}[xc])
    assert np.allclose(eig_n, ref_eig_n, rtol=0, atol=1e-8), \
        f'{xc} error = {np.max(np.abs(eig_n - ref_eig_n))}'


refs = {'GLLB':
        [-7.39405717, 5.01549354, 5.01549354, 5.01636595, 7.96095802,
         7.96095802, 7.96694721, 8.88669699],
        'GLLBM':
        [-5.74729132, 6.72606263, 6.72626409, 6.72626409, 9.58974119,
         9.58974119, 9.59459313, 10.37720714],
        'GLLBC':
        [-8.69990692, 3.74307763, 3.74449328, 3.74449328, 6.62701764,
         6.62701764, 6.63089065, 7.63596163],
        'GLLBCP86':
        [-8.85224195, 3.61988450, 3.62088667, 3.62088667, 6.49494071,
         6.49494071, 6.49896814, 7.43709444],
        'GLLBNORESP':
        [-11.08041529, 1.29983637, 1.30174632, 1.30174632, 3.22508733,
         3.95890694, 3.95890694, 3.96210188],
        'GLLBSC':
        [-8.30692283, 4.08893353, 4.08969201, 4.08969201, 7.08814258,
         7.08814258, 7.09267773, 8.05448801],
        'GLLBSCM':
        [-6.64285287, 5.81684766, 5.81869048, 5.81869048, 8.73473982,
         8.73473982, 8.73812616, 9.56188223]
        }


@pytest.mark.gllb
@pytest.mark.libxc
@pytest.mark.parametrize('xc', ['GLLB', 'GLLBM', 'GLLBC',
                                'GLLBCP86', 'GLLBNORESP',
                                'GLLBSC', 'GLLBSCM'])
def test_eigenvalues(xc, in_tmp_dir, add_cwd_to_setup_paths):
    eig_n = run(xc)
    ref_eig_n = refs[xc]
    assert np.allclose(eig_n, ref_eig_n, rtol=0, atol=2e-5), \
        f'{xc} error = {np.max(np.abs(eig_n - ref_eig_n))}'
