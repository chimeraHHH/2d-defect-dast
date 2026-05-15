from itertools import product

import pytest
import numpy as np

from gpaw.response.pw_parallelization import block_partition
from gpaw.response.integrators import (TetrahedronIntegrator, Integrand,
                                       HilbertTetrahedron, Domain)
from gpaw.response.frequencies import FrequencyGridDescriptor

from gpaw.response import ResponseContext


class MyIntegrand(Integrand):
    def matrix_element(self, point):
        return np.array([[1.]], complex)

    def eigenvalues(self, point):
        return np.array([(point.kpt_c**2).sum()**0.5], float)


@pytest.mark.tetrahedron
@pytest.mark.response
def test_tetrahedron_integrator():
    cell_cv = np.eye(3)
    context = ResponseContext()
    integrator = TetrahedronIntegrator(
        cell_cv, context, *block_partition(context.comm, nblocks=1))
    x_g = np.linspace(-1, 1, 30)
    x_gc = np.array([comb for comb in product(*([x_g] * 3))])

    # XXX we now hardcode "spins" as [0] but the previous API
    # could do any extra *args.
    #
    # After refactoring it should again be possible to do any args,
    # so this test isn't forced to specify the "spins"
    domain = Domain(x_gc, [0])
    out_wxx = np.zeros((1, 1, 1), complex)
    integrator.integrate(task=HilbertTetrahedron(integrator.blockcomm),
                         domain=domain,
                         integrand=MyIntegrand(),
                         wd=FrequencyGridDescriptor([-1.0]),
                         out_wxx=out_wxx)

    assert abs(out_wxx[0, 0, 0] - 4 * np.pi) < 1e-2
    # equal(out_wxx[0, 0, 0], 4 * np.pi, 1e-2,
    #       msg='Integrated area of unit sphere is not 4 * pi')
