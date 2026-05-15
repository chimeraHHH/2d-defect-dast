from gpaw.cgpaw import GG_shuffle
import numpy as np
import time


def test_GG_shuffle(rng):
    N = 1000
    G_G = np.arange(N, dtype=np.int32)
    rng.shuffle(G_G)
    A_GG = np.zeros((N, N), dtype=np.complex128)
    B_GG = np.zeros((N, N), dtype=np.complex128)
    A_GG[:] = rng.random((N, N))
    A_GG[:] += 1j * rng.random((N, N))

    start = time.perf_counter()
    GG_shuffle(G_G, 1, A_GG, B_GG)
    Cversion = time.perf_counter() - start

    start = time.perf_counter()
    B2_GG = A_GG.copy()[:, G_G][G_G]
    numpymagic = time.perf_counter() - start

    print('Speedup ', numpymagic / Cversion)
    assert np.allclose(B_GG, B2_GG)
    B_GG[:] = 0.0
    GG_shuffle(G_G, -1, A_GG, B_GG)
    B2_GG = A_GG.copy()[:, G_G][G_G].T
    assert np.allclose(B_GG, B2_GG)
