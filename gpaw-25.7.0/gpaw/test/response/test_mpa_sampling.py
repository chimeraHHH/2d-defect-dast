import pytest
from gpaw.response.mpa_sampling import mpa_frequency_sampling


@pytest.mark.response
def test_mpa_sampling_errors():
    with pytest.raises(ValueError):
        mpa_frequency_sampling(2, wrange=[0, 0], varpi=1,
                               eta0=0.1, eta_rest=0.1, parallel_lines=3,
                               alpha=1)
    with pytest.raises(ValueError):
        mpa_frequency_sampling(2, wrange=[0, 0], varpi=1,
                               eta0=0.1, eta_rest=0.1, parallel_lines=1,
                               alpha=1)


@pytest.mark.response
def test_mpa_sampling_1pole():
    # print("npol=1, parallel_lines=1, w1=0.1j, w2=1j, alpha=1:")
    with pytest.raises(AssertionError):
        mpa_frequency_sampling(1, wrange=[0, 0], varpi=1,
                               eta0=0.1, eta_rest=0.1, parallel_lines=1,
                               alpha=1)

    # print("npol=1, parallel_lines=2, w1=0.1j, w2=1j, alpha=0:")
    w_grid = mpa_frequency_sampling(1, wrange=[0, 0], varpi=1,
                                    eta0=0.1, eta_rest=0.1, parallel_lines=2,
                                    alpha=0)
    assert w_grid == pytest.approx([0. + 0.1j, 0. + 1.j])


@pytest.mark.response
def test_mpa_sampling_2poles():
    w_grid = mpa_frequency_sampling(2, wrange=[0, 1], varpi=1,
                                    eta0=0.1, eta_rest=0.1, parallel_lines=2,
                                    alpha=1)
    assert w_grid == pytest.approx([0. + 0.1j, 1. + 0.1j, 0 + 1j, 1 + 1j])

    # print("npol=2, parallel_lines=1, w1=0+1j, w2=2+1j, alpha=0:")
    w_grid = mpa_frequency_sampling(2, wrange=[0, 2], varpi=1,
                                    eta0=0.01, eta_rest=0.1, parallel_lines=1,
                                    alpha=0)
    assert w_grid == pytest.approx([0. + 1.j, 2 / 3. + 1.j, 4 / 3. + 1.j,
                                    2. + 1.j])

    # print("npol=2, parallel_lines=2, w1=0+1j, w2=2+1j, alpha=1:")
    w_grid = mpa_frequency_sampling(2, wrange=[0, 2], varpi=1,
                                    eta0=0.01, eta_rest=0.1)
    assert w_grid == pytest.approx([0. + 0.01j, 2. + 0.1j,
                                    0. + 1.j, 2. + 1.j])


@pytest.mark.response
def test_mpa_sampling_multiple_poles():
    # print("npol=3, parallel_lines=2, w1=0+1j, w2=2+1j, alpha=1:")
    w_grid = mpa_frequency_sampling(3, wrange=[0, 2], varpi=1,
                                    eta0=0.01, eta_rest=0.1)
    assert w_grid == pytest.approx([0. + 0.01j, 1 + 0.1j, 2. + 0.1j,
                                    0. + 1.j, 1 + 1.j, 2. + 1.j])

    # print("npol=4, parallel_lines=2, w1=0+1j, w2=2+1j, alpha=1:")
    w_grid = mpa_frequency_sampling(4, wrange=[0, 2], varpi=1,
                                    eta0=0.01, eta_rest=0.1)
    assert w_grid == pytest.approx([0. + 0.01j, 0.5 + 0.1j, 1. + 0.1j,
                                    2. + 0.1j, 0. + 1.j, 0.5 + 1.j, 1. + 1.j,
                                    2. + 1.j])

    # print("npol=5, parallel_lines=2, w1=0+1j, w2=2+1j, alpha=1:")
    w_grid = mpa_frequency_sampling(5, wrange=[0, 2], varpi=1,
                                    eta0=0.01, eta_rest=0.1)
    assert w_grid == pytest.approx([0. + 0.01j, 0.25 + 0.1j, 0.5 + 0.1j,
                                    1. + 0.1j, 2. + 0.1j, 0. + 1.j, 0.25 + 1.j,
                                    0.5 + 1.j, 1. + 1.j, 2. + 1.j])

    # print("npol=6, parallel_lines=2, w1=0+1j, w2=2+1j, alpha=1:")
    w_grid = mpa_frequency_sampling(6, wrange=[0, 2], varpi=1,
                                    eta0=0.01, eta_rest=0.1)
    assert w_grid == pytest.approx([0. + 0.01j, 0.25 + 0.1j, 0.5 + 0.1j,
                                    1. + 0.1j, 1.5 + 0.1j, 2. + 0.1j, 0. + 1.j,
                                    0.25 + 1.j, 0.5 + 1.j, 1. + 1.j, 1.5 + 1.j,
                                    2. + 1.j])
