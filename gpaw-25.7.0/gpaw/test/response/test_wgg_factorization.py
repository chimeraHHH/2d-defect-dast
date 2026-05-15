from gpaw.response.wgg import choose_parallelization


def test_parallelizations():
    assert choose_parallelization(131, 1455, 160) == (10, 4, 4)
    assert choose_parallelization(470, 10000, 1) == (1, 1, 1)
    assert choose_parallelization(1, 10000, 256) == (1, 16, 16)
    assert choose_parallelization(470, 10000, 256) == (16, 4, 4)
