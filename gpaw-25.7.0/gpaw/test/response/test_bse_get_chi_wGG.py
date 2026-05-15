import numpy as np
from ase.build import bulk
import pytest
from gpaw import GPAW, FermiDirac
from gpaw.response.bse import BSE


@pytest.mark.response
@pytest.mark.parametrize('tda', [False, True])
def test_bse_plus(tda, in_tmp_dir, scalapack):
    calc = GPAW(mode='pw',
                kpts={'size': (2, 2, 2), 'gamma': True},
                occupations=FermiDirac(0.01),
                nbands=8,
                symmetry='off',
                convergence={'bands': -4, 'density': 1e-7,
                             'eigenstates': 1e-10})

    a = 5.431
    atoms = bulk('Si', 'diamond', a=a)
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write('Si.gpw', 'all')

    val = range(4) if tda else range(8)
    cond = range(4, 8) if tda else range(8)
    bse = BSE('Si.gpw',
              ecut=20,
              valence_bands=val,
              conduction_bands=cond,
              eshift=0,
              mode='BSE',
              nbands=8,
              q_c=[0.0, 0.0, 0.0])

    chi_irr_bse = bse.get_chi_wGG(eta=0.2, optical=True, irreducible=True,
                                  w_w=np.array([-3, 0, 6]))
    ref = [(-0.12319305784052169 - 0.005900101520066767j),
           (-3.519763508433997e-10 - 0.035759806607911705j),
           (3.031540803184087e-05 - 0.0004896329314266086j)]

    ref_ntda = [(-0.12062692823607513 - 0.007967378306545791j),
                (1.2221039158990135e-06 - 0.0395577620626965j),
                (-4.0745632474313676e-05 - 0.0005604860798409453j)]

    for i, r in enumerate(ref if tda else ref_ntda):
        assert np.allclose(chi_irr_bse[i, i, i + 1], r)
