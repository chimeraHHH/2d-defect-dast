import io
import pytest

from gpaw import GPAW
import numpy as np

import numpy.testing as npt
from gpaw.io.logger import GPAWLogger
from gpaw.wavefunctions.base import eigenvalue_string
from gpaw.test.sic._utils import (mk_arr_from_str,
                                  extract_lagrange_section,
                                  MockWorld)
from gpaw.mpi import rank


@pytest.mark.old_gpaw_only
@pytest.mark.sic
def test_gmf_lcaosic(in_tmp_dir, gpw_files):
    """
    test Perdew-Zunger Self-Interaction
    Correction  in LCAO mode using DirectMin
    :param in_tmp_dir:
    :return:
    """
    calc = GPAW(gpw_files['h2o_gmf_lcaosic'])
    H2O = calc.atoms
    H2O.calc = calc
    e = H2O.get_potential_energy()
    f = H2O.get_forces()

    f_num = np.array([[-8.01206297e+00, -1.51553367e+01, 3.60670227e-03],
                      [1.42287594e+01, -9.81724693e-01, -5.09333905e-04],
                      [-4.92299436e+00, 1.55306540e+01, 2.12438557e-03]])

    numeric = False
    if numeric:
        from gpaw.test import calculate_numerical_forces
        f_num = calculate_numerical_forces(H2O, 0.001)
        print('Numerical forces')
        print(f_num)
        print(f - f_num, np.abs(f - f_num).max())

    assert e == pytest.approx(-2.007241, abs=1.0e-3)
    assert f == pytest.approx(f_num, abs=0.75)

    if rank == 0:
        logger = GPAWLogger(MockWorld(rank=0))
        string_io = io.StringIO()
        logger.fd = string_io
        calc.wfs.summary_func(logger)
        lstr = extract_lagrange_section(string_io.getvalue())

        expect_lagrange_str = """\
        Band         L_ii   Occupancy   Band      L_ii   Occupancy
           0    -19.72305    1.00000    0    -23.07192    1.00000
           1    -18.87889    1.00000    1    -22.32765    1.00000
           2    -16.46949    1.00000    2    -18.76539    1.00000
           3    -12.38574    1.00000    3    -18.76537    1.00000
           4     -9.12084    0.00000    4      2.49357    0.00000
           5      4.16510    0.00000    5      4.81118    0.00000
        """
        expect_eigen_str = """\
        Band  Eigenvalues  Occupancy  Eigenvalues  Occupancy
           0    -32.37420    1.00000    -32.74320    1.00000
           1    -17.73904    1.00000    -18.76757    1.00000
           2    -15.58035    1.00000    -15.92320    1.00000
           3     -1.76357    1.00000    -15.49637    1.00000
           4     -9.12084    0.00000      2.48462    0.00000
           5      4.16510    0.00000      4.82013    0.00000
        """

        npt.assert_allclose(
            mk_arr_from_str(expect_lagrange_str, 6),
            mk_arr_from_str(lstr, 6),
            atol=0.3,
        )

        npt.assert_allclose(
            mk_arr_from_str(expect_eigen_str, 5),
            mk_arr_from_str(eigenvalue_string(calc.wfs), 5, skip_rows=1),
            atol=0.3,
        )
