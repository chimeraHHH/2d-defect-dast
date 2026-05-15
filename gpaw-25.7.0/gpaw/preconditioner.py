import gpaw.cgpaw as cgpaw
import numpy as np

from gpaw import debug
from gpaw.fd_operators import Laplace
from gpaw.transformers import Transformer


class Preconditioner:
    def __init__(self, gd0, kin0, dtype=float, block=1, xp=np):
        gd1 = gd0.coarsen()
        gd2 = gd1.coarsen()
        self.kin0 = kin0
        self.kin1 = Laplace(gd1, -0.5, 1, dtype, xp=xp)
        self.kin2 = Laplace(gd2, -0.5, 1, dtype, xp=xp)
        self.scratch0 = gd0.zeros((2, block), dtype, False, xp=xp)
        self.scratch1 = gd1.zeros((3, block), dtype, False, xp=xp)
        self.scratch2 = gd2.zeros((3, block), dtype, False, xp=xp)
        self.step = 0.66666666 / kin0.get_diagonal_element()

        self.restrictor_object0 = Transformer(gd0, gd1, 1, dtype, xp=xp)
        self.restrictor_object1 = Transformer(gd1, gd2, 1, dtype, xp=xp)
        self.interpolator_object2 = Transformer(gd2, gd1, 1, dtype, xp=xp)
        self.interpolator_object1 = Transformer(gd1, gd0, 1, dtype, xp=xp)
        self.restrictor0 = self.restrictor_object0.apply
        self.restrictor1 = self.restrictor_object1.apply
        self.interpolator2 = self.interpolator_object2.apply
        self.interpolator1 = self.interpolator_object1.apply
        self.use_c_precond = True
        self.xp = xp

    def calculate_kinetic_energy(self, psit_xG, kpt):
        return None

    def __call__(self, residuals, kpt, ekin=None, out=None):
        if residuals.ndim == 3:
            if out is None:
                return self.__call__(residuals[np.newaxis], kpt)[0]
            return self.__call__(residuals[np.newaxis], kpt,
                                 out=out[np.newaxis])[0]

        nb = len(residuals)  # number of bands
        nb0 = self.scratch0.shape[1]
        if nb > nb0:
            assert out is not None
            for n1 in range(0, nb, nb0):
                self(residuals[n1:n1 + nb0], kpt, out=out[n1:n1 + nb0])
            return out

        phases = kpt.phase_cd
        step = self.step
        if out is None:
            d0, q0 = self.scratch0[:, :nb]
        else:
            d0 = out
            q0 = self.scratch0[0, :nb]
        r1, d1, q1 = self.scratch1[:, :nb]
        r2, d2, q2 = self.scratch2[:, :nb]
        if self.use_c_precond and self.xp is np:
            transformers = [self.restrictor_object0.transformer,
                            self.restrictor_object1.transformer,
                            self.interpolator_object1.transformer,
                            self.interpolator_object2.transformer]
            if debug:
                # Unwrap wrapper:
                transformers = [getattr(t, 'transformer', t)
                                for t in transformers]
            cgpaw.fd_precond(*transformers,
                             self.kin0.operator, self.kin1.operator,
                             self.kin2.operator,
                             d0, q0, r1, d1, q1, r2, d2, q2,
                             residuals, -residuals, step, phases)
            return d0
        self.restrictor0(-residuals, r1, phases)
        d1[:] = 4 * step * r1
        self.kin1.apply(d1, q1, phases)
        q1 -= r1
        self.restrictor1(q1, r2, phases)
        d2 = 16 * step * r2
        self.kin2.apply(d2, q2, phases)
        q2 -= r2
        d2 -= 16 * step * q2
        self.interpolator2(d2, q1, phases)
        d1 -= q1
        self.kin1.apply(d1, q1, phases)
        q1 -= r1
        d1 -= 4 * step * q1
        self.interpolator1(-d1, d0, phases)
        self.kin0.apply(d0, q0, phases)
        q0 -= residuals
        d0 -= step * q0
        d0 *= -1.0
        return d0
