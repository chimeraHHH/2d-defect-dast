"""
Python wrapper for FFTW3 library
================================

.. autoclass:: FFTPlans

"""
from __future__ import annotations

import weakref
from types import ModuleType

import numpy as np
from scipy.fft import fftn, ifftn, irfftn, rfftn

import gpaw.cgpaw as cgpaw
from gpaw.utilities import as_complex_dtype, as_real_dtype
from gpaw.new.c import pw_insert_gpu
from gpaw.new import trace
from gpaw.typing import Array1D, Array3D, DTypeLike, IntVector
from gpaw.gpu import is_hip

ESTIMATE = 64
MEASURE = 0
PATIENT = 32
EXHAUSTIVE = 8

_plan_cache: dict[tuple, weakref.ReferenceType] = {}


def have_fftw() -> bool:
    """Did we compile with FFTW?"""
    return hasattr(cgpaw, 'FFTWPlan')


def check_fft_size(n: int, factors=[2, 3, 5, 7]) -> bool:
    """Check if n is an efficient fft size.

    Efficient means that n can be factored into small primes (2, 3, 5, 7).

    >>> check_fft_size(17)
    False
    >>> check_fft_size(18)
    True
    """

    if n == 1:
        return True
    for x in factors:
        if n % x == 0:
            return check_fft_size(n // x, factors)
    return False


def get_efficient_fft_size(N: int, n=1, factors=[2, 3, 5, 7]) -> int:
    """Return smallest efficient fft size.

    Must be greater than or equal to N and divisible by n.

    >>> get_efficient_fft_size(17)
    18
    """
    N = -(-N // n) * n
    while not check_fft_size(N, factors):
        N += n
    return N


def empty(shape, dtype=float):
    """numpy.empty() equivalent with 16 byte alignment."""
    assert np.issubdtype(dtype, np.complexfloating)

    real_dtype = as_real_dtype(dtype)
    complex_dtype = as_complex_dtype(dtype)

    N = np.prod(shape)
    a = np.empty(2 * N + 16 // real_dtype.itemsize - 1, real_dtype)
    offset = (a.ctypes.data % 16) // real_dtype.itemsize
    a = a[offset:2 * N + offset].view(complex_dtype)
    a.shape = shape
    return a


def create_plans(size_c: IntVector,
                 dtype: DTypeLike,
                 flags: int = MEASURE,
                 xp: ModuleType = np) -> FFTPlans:
    """Create plan-objects for FFT and inverse FFT."""
    key = (tuple(size_c), dtype, flags, xp)
    # Look up weakref to plan:
    if key in _plan_cache:
        plan = _plan_cache[key]()
        # Check if plan is still "alive":
        if plan is not None:
            return plan
    # Create new plan:
    if xp is not np:
        plan = CuPyFFTPlans(size_c, dtype)
    elif have_fftw():
        plan = FFTWPlans(size_c, dtype, flags)
    else:
        plan = NumpyFFTPlans(size_c, dtype)
    _plan_cache[key] = weakref.ref(plan)
    return plan


class FFTPlans:
    def __init__(self,
                 size_c: IntVector,
                 dtype: DTypeLike,
                 empty=empty):
        self.shape: tuple[int, ...]

        if np.issubdtype(dtype, np.floating):
            self.shape = (size_c[0], size_c[1], size_c[2] // 2 + 1)
            self.tmp_Q = empty(self.shape, as_complex_dtype(dtype))
            self.tmp_R = self.tmp_Q.view(dtype)[:, :, :size_c[2]]
        else:
            self.shape = tuple(size_c)
            self.tmp_Q = empty(size_c, dtype)
            self.tmp_R = self.tmp_Q

    def fft(self) -> None:
        """Do FFT from ``tmp_R`` to ``tmp_Q``.

        >>> plans = create_plans([4, 1, 1], float)
        >>> plans.tmp_R[:, 0, 0] = [1, 0, 1, 0]
        >>> plans.fft()
        >>> plans.tmp_Q[:, 0, 0]
        array([2.+0.j, 0.+0.j, 2.+0.j, 0.+0.j])
        """
        raise NotImplementedError

    def ifft(self) -> None:
        """Do inverse FFT from ``tmp_Q`` to ``tmp_R``.

        >>> plans = create_plans([4, 1, 1], complex)
        >>> plans.tmp_Q[:, 0, 0] = [0, 1j, 0, 0]
        >>> plans.ifft()
        >>> plans.tmp_R[:, 0, 0]
        array([ 0.+1.j, -1.+0.j,  0.-1.j,  1.+0.j])
        """
        raise NotImplementedError

    def ifft_sphere(self, coef_G, pw, out_R):
        if coef_G is None:
            out_R.scatter_from(None)
            return
        pw.paste(coef_G, self.tmp_Q)

        if np.issubdtype(pw.dtype, np.floating):
            t = self.tmp_Q[:, :, 0]
            n, m = (s // 2 - 1 for s in out_R.desc.size_c[:2])
            t[0, -m:] = t[0, m:0:-1].conj()
            t[n:0:-1, -m:] = t[-n:, m:0:-1].conj()
            t[-n:, -m:] = t[n:0:-1, m:0:-1].conj()
            t[-n:, 0] = t[n:0:-1, 0].conj()
        self.ifft()
        out_R.scatter_from(self.tmp_R)

    def fft_sphere(self, in_R, pw):
        self.tmp_R[:] = in_R.data
        self.fft()
        coefs = pw.cut(self.tmp_Q) * (1 / self.tmp_R.size)
        return coefs


class FFTWPlans(FFTPlans):
    """FFTW3 3d transforms."""
    def __init__(self, size_c, dtype, flags=MEASURE):
        if not have_fftw():
            raise ImportError('Not compiled with FFTW.')
        super().__init__(size_c, dtype)
        self._fftplan = cgpaw.FFTWPlan(self.tmp_R, self.tmp_Q, -1, flags)
        self._ifftplan = cgpaw.FFTWPlan(self.tmp_Q, self.tmp_R, 1, flags)

    def fft(self):
        cgpaw.FFTWExecute(self._fftplan)

    def ifft(self):
        cgpaw.FFTWExecute(self._ifftplan)

    def __del__(self):
        # Attributes will not exist if execution stops during FFTW planning
        if hasattr(self, '_fftplan'):
            cgpaw.FFTWDestroy(self._fftplan)
        if hasattr(self, '_ifftplan'):
            cgpaw.FFTWDestroy(self._ifftplan)


class NumpyFFTPlans(FFTPlans):
    """Numpy fallback."""
    def fft(self):
        if np.issubdtype(self.tmp_R.dtype, np.floating):
            self.tmp_Q[:] = rfftn(self.tmp_R, overwrite_x=True)
        else:
            self.tmp_Q[:] = fftn(self.tmp_R, overwrite_x=True)

    def ifft(self):
        if np.issubdtype(self.tmp_R.dtype, np.floating):
            self.tmp_R[:] = irfftn(self.tmp_Q, self.tmp_R.shape,
                                   norm='forward', overwrite_x=True)
        else:
            self.tmp_R[:] = ifftn(self.tmp_Q, self.tmp_R.shape,
                                  norm='forward', overwrite_x=True)


def rfftn_patch(tmp_R):
    from gpaw.gpu import cupyx
    return cupyx.scipy.fft.fftn(tmp_R)[:, :, :tmp_R.shape[-1] // 2 + 1]


def irfftn_patch(B, shape):
    from gpaw.gpu import cupyx
    import cupy as xp
    A = xp.empty(shape, dtype=complex)
    A[:, :, :B.shape[2]] = B
    inv_ind1 = -xp.arange(B.shape[0])[:, None, None]
    inv_ind2 = -xp.arange(B.shape[1])[None, :, None]
    inv_ind3 = -xp.arange(1, B.shape[2])[None, None, :]
    A[:, :, -(B.shape[2] - 1):] = B[inv_ind1, inv_ind2, inv_ind3].conj()
    return cupyx.scipy.fft.ifftn(A).real


class CuPyFFTPlans(FFTPlans):
    def __init__(self,
                 size_c: IntVector,
                 dtype: DTypeLike):
        from gpaw.core import PWDesc
        from gpaw.gpu import cupy as cp
        self.dtype = dtype
        super().__init__(size_c, dtype, empty=cp.empty)
        self.Q_G_cache: dict[PWDesc, Array1D] = {}

    @trace(gpu=True)
    def fft(self):
        from gpaw.gpu import cupyx
        if self.tmp_R.dtype == float:
            if is_hip:
                self.tmp_Q[:] = rfftn_patch(self.tmp_R)
            else:
                self.tmp_Q[:] = cupyx.scipy.fft.rfftn(self.tmp_R)
        else:
            self.tmp_Q[:] = cupyx.scipy.fft.fftn(self.tmp_R)

    @trace(gpu=True)
    def ifft(self):
        from gpaw.gpu import cupyx
        if self.tmp_R.dtype == float:
            if is_hip:
                self.tmp_R[:] = irfftn_patch(self.tmp_Q, self.tmp_R.shape) \
                    * self.tmp_R.size
            else:
                self.tmp_R[:] = cupyx.scipy.fft.irfftn(
                    self.tmp_Q, self.tmp_R.shape,
                    norm='forward',
                    overwrite_x=True)
        else:
            self.tmp_R[:] = cupyx.scipy.fft.ifftn(
                self.tmp_Q, self.tmp_R.shape,
                norm='forward',
                overwrite_x=True)

    def indices(self, pw):
        from gpaw.gpu import cupy as cp
        Q_G = self.Q_G_cache.get(pw)
        if Q_G is None:
            Q_G = cp.asarray(pw.indices(self.shape))
            self.Q_G_cache[pw] = Q_G
        return Q_G

    @trace
    def ifft_sphere(self, coef_G, pw, out_R):
        from gpaw.gpu import cupyx

        if coef_G is None:
            out_R.scatter_from(None)
            return

        if out_R.desc.comm.size == 1:
            array_R = out_R.data
        else:
            array_R = self.tmp_R
        array_Q = self.tmp_Q

        array_Q[:] = 0.0
        Q_G = self.indices(pw)

        assert np.issubdtype(array_Q.dtype, np.complexfloating)
        assert np.issubdtype(coef_G.dtype, np.complexfloating)
        pw_insert_gpu(coef_G,
                      Q_G,
                      1.0,
                      array_Q.ravel(),
                      *out_R.desc.size_c)

        if np.issubdtype(self.dtype, np.complexfloating):
            array_R[:] = cupyx.scipy.fft.ifftn(
                array_Q, array_Q.shape,
                norm='forward', overwrite_x=True)
        else:
            if is_hip:
                array_R[:] = irfftn_patch(array_Q, out_R.desc.global_shape())\
                    * array_R.size
            else:
                array_R[:] = cupyx.scipy.fft.irfftn(
                    array_Q, out_R.desc.global_shape(),
                    norm='forward', overwrite_x=True)

        if out_R.desc.comm.size > 1:
            out_R.scatter_from(array_R)

    @trace
    def fft_sphere(self, in_R, pw):
        from gpaw.gpu import cupyx
        if np.issubdtype(self.dtype, np.complexfloating):
            out_Q = cupyx.scipy.fft.fftn(in_R)
        else:
            if is_hip:
                out_Q = rfftn_patch(in_R)
            else:
                # CuPy bug? rfftn fails on non-aligned arrays
                # To that end, make a copy. However, display a warning.
                if in_R.data.ptr % 16:
                    in_R = in_R.copy()
                    from warnings import warn
                    warn('Circumventing GPU array alignment problem '
                         'with copy at rfftn.')
                out_Q = cupyx.scipy.fft.rfftn(in_R)

        Q_G = self.indices(pw)
        coef_G = out_Q.ravel()[Q_G] * (1 / in_R.size)
        return coef_G


# The rest of this file will be removed in the future ...

def check_fftw_inputs(in_R, out_R):
    for arr in in_R, out_R:
        # Note: Arrays not necessarily contiguous due to 16-byte alignment
        assert arr.ndim == 3  # We can perhaps relax this requirement
        assert arr.dtype == float or arr.dtype == complex

    if in_R.dtype == out_R.dtype == complex:
        assert in_R.shape == out_R.shape
    else:
        # One real and one complex:
        R, C = (in_R, out_R) if in_R.dtype == float else (out_R, in_R)
        assert C.dtype == complex
        assert R.shape[:2] == C.shape[:2]
        assert C.shape[2] == 1 + R.shape[2] // 2


class FFTPlan:
    """FFT 3d transform."""
    def __init__(self,
                 in_R: Array3D,
                 out_R: Array3D,
                 sign: int,
                 flags: int = MEASURE):
        check_fftw_inputs(in_R, out_R)
        self.in_R = in_R
        self.out_R = out_R
        self.sign = sign
        self.flags = flags

    def execute(self) -> None:
        raise NotImplementedError


class FFTWPlan(FFTPlan):
    """FFTW3 3d transform."""
    def __init__(self, in_R, out_R, sign, flags=MEASURE):
        if not have_fftw():
            raise ImportError('Not compiled with FFTW.')
        self._ptr = cgpaw.FFTWPlan(in_R, out_R, sign, flags)
        FFTPlan.__init__(self, in_R, out_R, sign, flags)

    def execute(self):
        cgpaw.FFTWExecute(self._ptr)

    def __del__(self):
        if getattr(self, '_ptr', None):
            cgpaw.FFTWDestroy(self._ptr)
        self._ptr = None


class NumpyFFTPlan(FFTPlan):
    """Numpy fallback."""
    def execute(self):
        if self.in_R.dtype == float:
            self.out_R[:] = np.fft.rfftn(self.in_R)
        elif self.out_R.dtype == float:
            self.out_R[:] = np.fft.irfftn(self.in_R,
                                          self.out_R.shape,
                                          [0, 1, 2])
            self.out_R *= self.out_R.size
        elif self.sign == 1:
            self.out_R[:] = np.fft.ifftn(self.in_R,
                                         self.out_R.shape,
                                         [0, 1, 2])
            self.out_R *= self.out_R.size
        else:
            self.out_R[:] = np.fft.fftn(self.in_R)


def create_plan(in_R: Array3D,
                out_R: Array3D,
                sign: int,
                flags: int = MEASURE) -> FFTPlan:
    if have_fftw():
        return FFTWPlan(in_R, out_R, sign, flags)
    return NumpyFFTPlan(in_R, out_R, sign, flags)
