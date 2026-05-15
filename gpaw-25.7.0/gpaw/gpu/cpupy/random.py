import numpy as np


def default_rng(seed):
    return RNG(np.random.default_rng(seed))


class RNG:
    def __init__(self, rng):
        self.rng = rng

    def random(self, shape=None, out=None, dtype=float):
        if shape:
            return self.rng.random(shape, out=out._data, dtype=dtype)
        else:
            return self.rng.random(out=out._data, dtype=dtype)
