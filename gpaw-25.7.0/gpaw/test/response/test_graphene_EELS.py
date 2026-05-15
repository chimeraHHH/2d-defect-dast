import numpy as np
import pytest

from gpaw.response.df import DielectricFunction
from gpaw.test import findpeak


@pytest.mark.dielectricfunction
@pytest.mark.response
def test_graphene_EELS(in_tmp_dir, gpw_files):
    # Test values
    q_qc = [
        [0, 0, 0],
        [1 / 3, 0, 0],
    ]
    refs_q = [
        # peak frequency and intensity (rows), w.o./w. local-field corr. (cols)
        np.array([[7.57, 7.66], [3.78, 4.01]]),
        np.array([[5.86, 6.29], [0.99, 0.48]]),
    ]
    # Calculation
    dfcalc = DielectricFunction(gpw_files['graphene_pw'], truncation='2D',
                                frequencies=np.linspace(0., 9., 141),
                                eta=0.3, ecut=150, rate='eta', hilbert=False)
    for q_c, refs in zip(q_qc, refs_q):
        epsinv = dfcalc.get_inverse_dielectric_function(q_c=q_c)
        omega_w, eels0_w, eels_w = epsinv.eels_spectrum().arrays

        # import matplotlib.pyplot as plt
        # plt.plot(omega_w, eels0_w)
        # plt.plot(omega_w, eels_w)
        # plt.show()

        # Test against reference
        e0, I0 = findpeak(omega_w, eels0_w)
        e, I = findpeak(omega_w, eels_w)
        assert np.array([e0, e]) == pytest.approx(refs[0, :], abs=0.05)
        assert np.array([I0, I]) == pytest.approx(refs[1, :], abs=0.02)
