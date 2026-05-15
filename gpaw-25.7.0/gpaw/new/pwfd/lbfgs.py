import numpy as np
from gpaw.new import zips


class Vector:
    def __init__(self, x_unX):
        self.x_unX = x_unX

    def copy(self):
        return Vector([x_nX.copy() for x_nX in self.x_unX])

    def __iter__(self):
        yield from self.x_unX

    def zeros(self):
        y_unX = []
        for x_nX in self:
            y_nX = x_nX.new()
            y_nX.data[:] = 0.0
            y_unX.append(y_nX)
        return Vector(y_unX)

    def __mul__(self, other):
        y_unX = self.copy()
        for y_nX in y_unX:
            y_nX.data *= other
        return y_unX

    def __matmul__(self, other):
        z = 0.0
        for x_nX, y_nX in zips(self, other):
            for x_X, y_X in zips(x_nX, y_nX):
                z += x_X.integrate(y_X)
        return 2 * z.real

    def __sub__(self, other):
        x_unX = self.copy()
        for x_nX, y_nX in zips(x_unX, other):
            x_nX.data -= y_nX.data
        return x_unX


class LBFGS:
    def __init__(self,
                 memory=3):
        self.memory = memory
        self.iters = 0
        self.kp = None
        self.p = None
        self.k = None
        self.s_k = {i: None for i in range(memory)}
        self.y_k = {i: None for i in range(memory)}
        self.rho_k = np.zeros(shape=memory)
        self.kp = {}
        self.p = 0
        self.k = 0
        self.stable = True

    def update(self,
               x_k1,
               g_k1):
        x_k1 = Vector(x_k1)
        g_k1 = Vector(g_k1)
        self.iters += 1

        if self.k == 0:
            self.kp[self.k] = self.p
            self.x_k = x_k1.copy()
            self.g_k = g_k1
            self.s_k[self.kp[self.k]] = g_k1.zeros()
            self.y_k[self.kp[self.k]] = g_k1.zeros()
            self.k += 1
            self.p += 1
            self.kp[self.k] = self.p
            return (g_k1 * -1.0).x_unX

        if self.p == self.memory:
            self.p = 0
            self.kp[self.k] = self.p

        s_k = self.s_k
        x_k = self.x_k
        y_k = self.y_k
        g_k = self.g_k

        x_k1 = x_k1.copy()

        rho_k = self.rho_k

        kp = self.kp
        k = self.k
        m = self.memory

        s_k[kp[k]] = x_k1 - x_k
        y_k[kp[k]] = g_k1 - g_k
        dot_ys = y_k[kp[k]] @ s_k[kp[k]]

        if abs(dot_ys) > 1.0e-15:
            rho_k[kp[k]] = 1.0 / dot_ys
        else:
            rho_k[kp[k]] = 1.0e15

        if dot_ys < 0.0:
            self.stable = False

        q = g_k1.copy()

        alpha = np.zeros(np.minimum(k + 1, m))
        j = np.maximum(-1, k - m)

        for i in range(k, j, -1):
            dot_sq = s_k[kp[i]] @ q
            alpha[kp[i]] = rho_k[kp[i]] * dot_sq
            q = q - y_k[kp[i]] * alpha[kp[i]]

        t = k
        dot_yy = y_k[kp[t]] @ y_k[kp[t]]

        if abs(dot_yy) > 1.0e-15:
            r = q * (1.0 / (rho_k[kp[t]] * dot_yy))
        else:
            r = q * 1.0e15

        for i in range(np.maximum(0, k - m + 1), k + 1):
            dot_yr = y_k[kp[i]] @ r
            beta = rho_k[kp[i]] * dot_yr
            r = r - s_k[kp[i]] * (beta - alpha[kp[i]])

        # save this step:
        self.x_k = x_k1.copy()
        self.g_k = g_k1.copy()
        self.k += 1
        self.p += 1
        self.kp[self.k] = self.p

        return (r * -1.0).x_unX
