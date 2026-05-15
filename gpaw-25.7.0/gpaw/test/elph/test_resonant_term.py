import numpy as np
import pytest

from gpaw.elph import ResonantRamanCalculator


def make_hermitian(mat):
    n = mat.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            mat[j, i] = mat[i, j].conj()


@pytest.mark.serial
def test_resonant_term():
    rt = ResonantRamanCalculator.resonant_term

    f_n = np.array([1.0, 0.0, 0.0])
    f_vc = np.outer(f_n, 1.0 - f_n)
    assert f_vc[0] == pytest.approx([0.0, 1.0, 1.0])
    print(f_vc)

    E_n = np.array([-1.0, 0.0, 1.0])
    E_vc = np.zeros((3, 3), dtype=complex) + 1j * 0.1
    for n in range(3):
        E_vc[n] += (E_n - E_n[n])
    assert E_vc[0] == pytest.approx([0.0 + 1j * 0.1,
                                     1.0 + 1j * 0.1,
                                     2.0 + 1j * 0.1])

    nn = np.zeros((3, 3), dtype=complex)
    nn[0, 0] = 1.0
    nn[1, 1] = 2.0
    nn[2, 2] = 3.0
    nn[0, 1] = 0.5 + 0.1j
    nn[0, 2] = 0.25 + 0.1j
    nn[1, 2] = 0.125 + 0.1j
    make_hermitian(nn)
    # print(nn)
    mom_dnn = np.zeros((2, 3, 3), dtype=complex)
    mom_dnn[0] = nn
    mom_dnn[1] = 2. * nn

    elph_lnn = np.zeros((2, 3, 3), dtype=complex)
    elph_lnn[1] = 3. * nn

    wph_w = np.array([0.0, 0.2])

    term = rt(f_vc, E_vc, mom_dnn, elph_lnn, 0, 3, 1.0, wph_w)
    # we'll want to check whether this is the expect number
    assert term[1] == pytest.approx(-29.768613861386154 - 64.0461386138614j)

    # complement polarisations need to yield the same result
    term2 = rt(f_vc, E_vc, mom_dnn[[1, 0], :, :], elph_lnn, 0, 3, 1.0, wph_w)
    assert term[1] == pytest.approx(term2[1])
