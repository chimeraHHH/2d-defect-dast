import pytest
from gpaw.grid_descriptor import GridDescriptor
from gpaw.pw.descriptor import PWDescriptor
from gpaw.mpi import world


@pytest.mark.ci
def test_pw_interpol(rng):
    def test(gd1, gd2, pd1, pd2, R1, R2):
        a1 = gd1.zeros(dtype=pd1.dtype)
        a1[R1] = 1
        a2 = pd1.interpolate(a1, pd2)[0]
        x = a2[R2]

        a2 = gd2.zeros(dtype=pd2.dtype)
        a2[R2] = 1
        y = pd2.restrict(a2, pd1)[0][R1] * a2.size / a1.size

        assert x == pytest.approx(y, abs=1e-9)
        return x

    if world.size == 1:
        for size1, size2 in [
            [(3, 3, 3), (8, 8, 8)],
            [(4, 4, 4), (9, 9, 9)],
            [(2, 4, 4), (5, 9, 9)],
            [(2, 3, 4), (5, 6, 9)],
            [(2, 3, 4), (5, 6, 8)],
            [(4, 4, 4), (8, 8, 8)],
            [(2, 4, 4), (4, 8, 8)],
            [(2, 4, 2), (4, 8, 4)]]:
            print(size1, size2)
            gd1 = GridDescriptor(size1, size1)
            gd2 = GridDescriptor(size2, size1)
            pd1 = PWDescriptor(1, gd1, complex)
            pd2 = PWDescriptor(1, gd2, complex)
            pd1r = PWDescriptor(1, gd1)
            pd2r = PWDescriptor(1, gd2)
            for R1, R2 in [[(0, 0, 0), (0, 0, 0)],
                           [(0, 0, 0), (0, 0, 1)]]:
                x = test(gd1, gd2, pd1, pd2, R1, R2)
                y = test(gd1, gd2, pd1r, pd2r, R1, R2)
                assert x == pytest.approx(y, abs=1e-9)

            a1 = rng.random(size1)
            a2 = pd1r.interpolate(a1, pd2r)[0]
            c2 = pd1.interpolate(a1 + 0.0j, pd2)[0]
            d2 = pd1.interpolate(a1 * 1.0j, pd2)[0]
            assert abs(c2.imag).max() == pytest.approx(0, abs=1e-14)
            assert abs(d2.real).max() == pytest.approx(0, abs=1e-14)
            assert gd1.integrate(a1) == pytest.approx(gd2.integrate(a2),
                                                      abs=1e-13)
            assert abs(c2 - a2).max() == pytest.approx(0, abs=1e-14)
            assert abs(d2 - a2 * 1.0j).max() == pytest.approx(0, abs=1e-14)

            a1 = pd2r.restrict(a2, pd1r)[0]
            c1 = pd2.restrict(a2 + 0.0j, pd1)[0]
            d1 = pd2.restrict(a2 * 1.0j, pd1)[0]
            assert gd1.integrate(a1) == pytest.approx(gd2.integrate(a2),
                                                      abs=1e-13)
            assert abs(c1 - a1).max() == pytest.approx(0, abs=1e-14)
            assert abs(d1 - a1 * 1.0j).max() == pytest.approx(0, abs=1e-14)
