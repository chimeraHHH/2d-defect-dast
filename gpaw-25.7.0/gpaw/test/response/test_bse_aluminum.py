import pytest
import numpy as np
from gpaw.response.df import DielectricFunction
from gpaw.response.bse import BSE, read_spectrum
from gpaw.test import findpeak


@pytest.mark.dielectricfunction
@pytest.mark.response
def test_response_bse_aluminum(in_tmp_dir, gpw_files):
    df = 1
    bse = 1
    check_spectrum = 1

    q_c = np.array([0.25, 0.0, 0.0])
    w_w = np.linspace(0, 24, 241)
    eta = 0.2
    ecut = 50
    if bse:
        bse = BSE(gpw_files['bse_al'],
                  valence_bands=range(4),
                  conduction_bands=range(4),
                  mode='RPA',
                  nbands=4,
                  q_c=q_c,
                  ecut=ecut,
                  )
        bse.get_eels_spectrum(filename='bse_eels.csv',
                              w_w=w_w,
                              eta=eta)
        omega_w, bse_w = read_spectrum('bse_eels.csv')

    if df:
        df = DielectricFunction(calc=gpw_files['bse_al'],
                                frequencies=w_w,
                                eta=eta,
                                ecut=ecut,
                                hilbert=False)
        df_w = df.get_eels_spectrum(q_c=q_c, filename=None)[1]

    if check_spectrum:
        assert w_w == pytest.approx(omega_w)
        w_ = 15.1423
        I_ = 25.4359
        wbse, Ibse = findpeak(w_w, bse_w)
        wdf, Idf = findpeak(w_w, df_w)
        assert wbse == pytest.approx(w_, abs=0.01)
        assert wdf == pytest.approx(w_, abs=0.01)
        assert Ibse == pytest.approx(I_, abs=0.1)
        assert Idf == pytest.approx(I_, abs=0.1)
