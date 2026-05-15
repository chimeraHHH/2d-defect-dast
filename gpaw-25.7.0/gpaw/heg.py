"""Analytic expressions of the homogeneous electron gas (HEG).

The goal of this module is to centralize all the standard expressions
needed around GPAW for doing HEG things.  Please move appropriate
code here whenever relevant."""

import numpy as np
from functools import cached_property


class HEG:
    def __init__(self, rs):
        self._rs = rs

    @property
    def rs(self):
        return self._rs

    @cached_property
    def qF(self):
        return (9.0 * np.pi / 4.0)**(1.0 / 3.0) / self.rs
