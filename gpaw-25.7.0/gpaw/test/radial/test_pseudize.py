import numpy as np
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor


def test_smoothness():
    """Make sure pseudize_smooth() produces the most smooth function."""
    rgd = EquidistantRadialGridDescriptor(0.05, 200)
    r = rgd.r_g

    a = 3.0
    l = 0
    ecut = 10
    Gcut = (2 * ecut)**0.5
    gc = 20

    f0 = np.exp(-a * r**2)
    f1, _ = rgd.pseudize_normalized(f0, gc, l, 4)
    f2, _ = rgd.pseudize(f0, gc, l, 4)
    f3, _ = rgd.pseudize_smooth(f0, gc, l, 4, Gcut)

    # Calculate weight of large wave vectors (G > Gcut):
    weights = []
    for f in [f0, f1, f2, f3]:
        x, ft = rgd.fft(f * r, l=l)
        weights.append(((x * ft)[x > Gcut]**2).sum())

    print(weights)
    assert (np.diff(weights) < 0).all()
