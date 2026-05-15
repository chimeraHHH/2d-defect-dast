import time

import pytest

import gpaw.fftw as fftw

N = 1
# n1, n2, n3 = 32, 28, 128
n1, n2, n3 = 8, 6, 12


def check(Plan, flags, input, output, sign):
    t0 = time.time()
    plan = Plan(input, output, sign, flags)
    t1 = time.time()
    t = 0.0
    for i in range(N):
        input[:] = 1.3
        t2 = time.time()
        plan.execute()
        t3 = time.time()
        t += t3 - t2
    return t1 - t0, t / N


@pytest.mark.skipif(not fftw.have_fftw(), reason='No FFTW')
def test_fft():
    a1 = fftw.empty((n1, n2, n3), complex)
    a2 = fftw.empty((n1, n2, n3), complex)
    b = fftw.empty((n1, n2, n3 // 2 + 1), complex)
    c1 = b.view(dtype=float)[:, :, :n3]
    c2 = fftw.empty((n1, n2, n3 // 2), complex).view(dtype=float)
    for input, output, sign in [
        (a1, a1, -1),
        (a1, a2, -1),
        (b, c1, 1),
        (b, c2, 1),
        (c1, b, -1),
        (c2, b, -1)]:
        for Plan, flags in [(fftw.NumpyFFTPlan, 117),
                            (fftw.FFTWPlan, fftw.ESTIMATE),
                            (fftw.FFTWPlan, fftw.MEASURE),
                            (fftw.FFTWPlan, fftw.PATIENT),
                            (fftw.FFTWPlan, fftw.EXHAUSTIVE)]:
            tplan, tfft = check(Plan, flags, input, output, sign)
            print(('%-12s %3d %10.6f %10.6f' %
                  (Plan.__name__, flags, tplan, tfft)))
