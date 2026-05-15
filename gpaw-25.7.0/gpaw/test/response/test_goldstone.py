import pytest
from functools import partial

import numpy as np

from gpaw.response.goldstone import find_root


def parabolic_function(lambd: float, *, root: float):
    """Define a parabolic function f(λ) with a root nearby λ=1.

    Defining,

    f(λ) = aλ² + bλ + c

    we need the function to be monotonically decreasing in the range
    λ∊]0.1, 10[. With parametrization

    (a, b) = (1/4, -6)

    the minimum lies at λ = -b/(2a) = 12 and the lower root at
    λ_r = -(b+d)/(2a) = 12 - d/(2a).

    Solving for c;

    d = ⎷(b²-4ac) = 2a (12 - λ_r)

    c = b² - (12 - λ_r)²/4
    """
    assert 0.1 < root < 10.
    a = 1 / 4.
    b = -6.
    c = b**2. - (12. - root)**2. / 4.
    return a * lambd**2. + b * lambd + c


@pytest.mark.response
@pytest.mark.parametrize('target', np.linspace(0.4, 2.4, 51))
def test_find_root(target):
    fnct = partial(parabolic_function, root=target)

    def is_converged(value):
        return 0. <= value < 1e-7

    root = find_root(fnct, is_converged)
    assert root == pytest.approx(target)
    assert is_converged(fnct(root))
