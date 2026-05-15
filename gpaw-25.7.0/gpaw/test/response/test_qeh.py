import pytest
import numpy as np

from gpaw.response.df import DielectricFunction
from ase.parallel import world
from ase.units import Hartree


def dielectric(calc, domega, omega2, rate=0.0,
               ecut=20, nblocks=1, cyl_pw=False):
    if cyl_pw:
        from gpaw.response.qpd import SingleCylQPWDescriptor
        ecut = {
            'class': SingleCylQPWDescriptor,
            'kwargs': {'ecut_xy': ecut / Hartree,
                       'ecut_z': ecut / 2 / Hartree}
        }
    diel = DielectricFunction(calc=calc,
                              frequencies={'type': 'nonlinear',
                                           'omegamax': 10,
                                           'domega0': domega,
                                           'omega2': omega2},
                              nblocks=nblocks,
                              ecut=ecut,
                              rate=rate,
                              truncation='2D')
    return diel


@pytest.mark.dielectricfunction
@pytest.mark.serial
@pytest.mark.response
def test_basics(in_tmp_dir, gpw_files):
    pytest.importorskip('qeh')
    from gpaw.response.qeh import QEHChiCalc

    df = dielectric(gpw_files['graphene_pw'], 0.1, 0.5,
                    ecut=10, rate=0.01, cyl_pw=False)

    chicalc = QEHChiCalc(df)

    assert len(chicalc.get_q_grid(q_max=0.6)) == 3
    assert len(chicalc.get_q_grid(q_max=2.6)) == 6
    assert len(chicalc.get_q_grid(q_max=2.6)[0].P_rv) == 2
    assert len(chicalc.get_z_grid()) == 30

    q_q = chicalc.get_q_grid(q_max=0.6)
    chi_wGG, G_Gv, wblocks = chicalc.get_chi_wGG(qpoint=q_q[2])

    assert chi_wGG[0, 0, 0] == pytest.approx(-3.134762463291029e-10
                                             + 3.407232927207498e-27j)
    assert chi_wGG[3, 2, 1] == pytest.approx(-2.69008628970302e-10
                                             - 6.74306768078481e-11j)
    assert chi_wGG.shape[1] == len(G_Gv)

    df2 = dielectric(gpw_files['graphene_pw'], 0.1, 0.5,
                     ecut=10, rate=0.01, cyl_pw=True)

    chicalc = QEHChiCalc(df2)
    chi2_wGG, G2_Gv, wblocks = chicalc.get_chi_wGG(qpoint=q_q[2])

    assert chi2_wGG[0, 0, 0] == pytest.approx(chi_wGG[0, 0, 0])
    G1 = np.argmin(np.linalg.norm(G_Gv[None, 1] - G2_Gv, axis=1))
    G2 = np.argmin(np.linalg.norm(G_Gv[None, 2] - G2_Gv, axis=1))
    assert chi2_wGG[3, G2, G1] == pytest.approx(chi_wGG[3, 2, 1])
    assert chi2_wGG.shape[1] == len(G2_Gv)


@pytest.mark.skipif(world.size == 1, reason='Features already tested '
                    'in serial in test_basics')
@pytest.mark.skipif(world.size > 6, reason='Parallelization for '
                    'small test-system broken for many cores')
@pytest.mark.dielectricfunction
@pytest.mark.response
def test_qeh_parallel(in_tmp_dir, gpw_files):
    pytest.importorskip('qeh')
    from gpaw.response.qeh import QEHChiCalc

    df = dielectric(gpw_files['mos2_pw'], 0.05, 0.5, nblocks=world.size,
                    cyl_pw=False)
    chicalc = QEHChiCalc(df)

    q_q = chicalc.get_q_grid(q_max=0.6)
    chi_wGG, G_Gv, wblocks = chicalc.get_chi_wGG(qpoint=q_q[2])
    chi_wGG = wblocks.all_gather(chi_wGG)

    df2 = dielectric(gpw_files['mos2_pw'], 0.05, 0.5, nblocks=world.size,
                     cyl_pw=True)
    chicalc = QEHChiCalc(df2)
    chi2_wGG, G2_Gv, wblocks = chicalc.get_chi_wGG(qpoint=q_q[2])
    chi2_wGG = wblocks.all_gather(chi2_wGG)

    if world.rank == 0:
        assert chi_wGG.shape[0] == 23
        assert chi_wGG[0, 0, 0] == pytest.approx(-0.004795395871467905
                                                 + 1.8961244666811292e-19j)
        assert chi_wGG[3, 2, 1] == pytest.approx(0.004216141281855623
                                                 + 6.949418840278167e-05j)
        assert chi_wGG.shape[1] == len(G_Gv)

        assert chi2_wGG[0, 0, 0] == pytest.approx(chi_wGG[0, 0, 0], rel=1e-2)
        G1 = np.argmin(np.linalg.norm(G_Gv[None, 1] - G2_Gv, axis=1))
        G2 = np.argmin(np.linalg.norm(G_Gv[None, 2] - G2_Gv, axis=1))
        assert chi2_wGG[3, G2, G1] == pytest.approx(chi_wGG[3, 2, 1], rel=1e-2)
        assert chi2_wGG.shape[1] == len(G2_Gv)
