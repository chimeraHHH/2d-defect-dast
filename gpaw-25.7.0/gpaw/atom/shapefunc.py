import numpy as np
from .radialgd import RadialGridDescriptor


def shape_functions(rgd: RadialGridDescriptor,
                    type: str,
                    rc: float,
                    lmax: int) -> np.ndarray:
    """Shape functions for compensation charges."""
    g_lg = rgd.zeros(lmax + 1)
    r_g = rgd.r_g

    if type == 'gauss':
        g_lg[0] = 4 / rc**3 / np.sqrt(np.pi) * np.exp(-(r_g / rc)**2)
        for l in range(1, lmax + 1):
            g_lg[l] = 2.0 / (2 * l + 1) / rc**2 * r_g * g_lg[l - 1]
    elif type == 'sinc':
        g_lg[0] = np.sinc(r_g / rc)**2
        g_lg[0, rgd.ceil(rc):] = 0.0
        for l in range(1, lmax + 1):
            g_lg[l] = r_g * g_lg[l - 1]
    elif type == 'bessel':
        from scipy.special import spherical_jn as jn
        roots = [[3.141592653589793, 6.283185307179586],
                 [4.493409457909095, 7.7252518369375],
                 [5.76345919689455, 9.095011330476355]]
        for l in range(lmax + 1):
            q1, q2 = (x0 / rc for x0 in roots[l])
            alpha = -q1 / q2 * jn(l, q1 * rc, True) / jn(l, q2 * rc, True)
            g_lg[l] = jn(l, q1 * r_g) + alpha * jn(l, q2 * r_g)
        g_lg[:, rgd.ceil(rc):] = 0.0
    else:
        1 / 0

    for l in range(lmax + 1):
        g_lg[l] /= rgd.integrate(g_lg[l], l) / (4 * np.pi)

    return g_lg


if __name__ == '__main__':
    from .radialgd import EquidistantRadialGridDescriptor as RGD
    from scipy.special import spherical_jn as jn
    from scipy.optimize import root
    import matplotlib.pyplot as plt

    r = np.linspace(0, 1.2, 200)

    if 0:
        # Find roots of spherical Bessel functions:
        for l in range(3):
            for i, x0 in enumerate([3, 6]):
                result = root(lambda x: jn(l, x), x0 + 1.5 * l)
                x0 = result['x']
                print(l, i, x0.item())
                plt.plot(r, jn(l, r * x0), label=str(x0))
        plt.legend()
        plt.show()

    rgd = RGD(0.01, 120)
    for l in range(3):
        rc = 0.3
        for type in ['gauss', 'sinc', 'bessel']:
            g_lg = shape_functions(rgd, type, rc, l)
            plt.plot(rgd.r_g, g_lg[l], label=type)  # type: ignore
            rc = 1.0

        plt.legend()
        plt.show()
