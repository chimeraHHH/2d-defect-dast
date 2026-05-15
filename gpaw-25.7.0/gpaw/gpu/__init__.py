from __future__ import annotations
import contextlib
from time import time
from typing import TYPE_CHECKING
from types import ModuleType
from collections.abc import Iterable
from gpaw.new.timer import trace

import numpy as np

from gpaw.cgpaw import have_magma

cupy_is_fake = True
"""True if :mod:`cupy` has been replaced by ``gpaw.gpu.cpupy``"""

is_hip = False
"""True if we are using HIP"""

device_id = None
"""Device id"""


def gpu_gemm(*args, **kwargs):
    raise NotImplementedError('gpu_gemm: You are not using GPAW with GPUs.')


if TYPE_CHECKING:
    import gpaw.gpu.cpupy as cupy
    import gpaw.gpu.cpupyx as cupyx
else:
    try:
        import gpaw.cgpaw as cgpaw
        if not hasattr(cgpaw, 'gpaw_gpu_init'):
            raise ImportError

        import cupy
        # Cupy gemm wrapper (does extra copying):
        # from cupy import cublas
        # gpu_gemm = trace(gpu=True)(cublas.gemm)  # noqa: F811

        # Homerolled gemm wrapper and helper functions:
        from cupy.cublas import (_get_scalar_ptr, _trans_to_cublas_op,
                                 _change_order_if_necessary, device)
        from cupy_backends.cuda.libs import cublas as _cublas

        def _decide_ld_and_trans(a, trans):
            ld = None
            if a._c_contiguous:
                ld = a.shape[1]
                trans = 1 - trans
            elif a._f_contiguous:
                ld = a.shape[0]
            elif a.strides[-1] == a.dtype.itemsize:
                # Semi C-contiguous (sliced along second dim)
                ld = a.strides[-2] // a.strides[-1]
                trans = 1 - trans
            return ld, trans

        @trace(gpu=True)
        def gpu_gemm(  # noqa: F811
                transa, transb, a, b, out=None, alpha=1.0, beta=0.0):
            """Computes out = alpha * op(a) @ op(b) + beta * out

            op(a) = a if transa is 'N', op(a) = a.T if transa is 'T',
            op(a) = a.T.conj() if transa is 'H'.
            op(b) = b if transb is 'N', op(b) = b.T if transb is 'T',
            op(b) = b.T.conj() if transb is 'H'.

            This is pretty much a copy of the code from cupy.cublas.gemm,
            with _decide_ld_and_trans modified to not be afraid of
            hermitian transposes and a preference to C-contiguous arrays.
            """
            assert a.ndim == b.ndim == 2
            assert a.dtype == b.dtype
            dtype = a.dtype.char
            if dtype == 'f':
                func = _cublas.sgemm
            elif dtype == 'd':
                func = _cublas.dgemm
            elif dtype == 'F':
                func = _cublas.cgemm
            elif dtype == 'D':
                func = _cublas.zgemm
            else:
                raise TypeError('invalid dtype')

            transa = _trans_to_cublas_op(transa)
            transb = _trans_to_cublas_op(transb)
            if transa == _cublas.CUBLAS_OP_N:
                m, k = a.shape
            else:
                k, m = a.shape
            if transb == _cublas.CUBLAS_OP_N:
                n = b.shape[1]
                assert b.shape[0] == k
            else:
                n = b.shape[0]
                assert b.shape[1] == k
            if out is None:
                out = cupy.empty((m, n), dtype=dtype, order='C')
                beta = 0.0
            else:
                assert out.ndim == 2
                assert out.shape == (m, n)
                assert out.dtype == dtype

            alpha, alpha_ptr = _get_scalar_ptr(alpha, a.dtype)
            beta, beta_ptr = _get_scalar_ptr(beta, a.dtype)
            handle = device.get_cublas_handle()
            orig_mode = _cublas.getPointerMode(handle)
            if isinstance(alpha, cupy.ndarray) or \
               isinstance(beta, cupy.ndarray):
                if not isinstance(alpha, cupy.ndarray):
                    alpha = cupy.array(alpha)
                    alpha_ptr = alpha.data.ptr
                if not isinstance(beta, cupy.ndarray):
                    beta = cupy.array(beta)
                    beta_ptr = beta.data.ptr
                _cublas.setPointerMode(handle,
                                       _cublas.CUBLAS_POINTER_MODE_DEVICE)
            else:
                _cublas.setPointerMode(handle,
                                       _cublas.CUBLAS_POINTER_MODE_HOST)

            lda, transa = _decide_ld_and_trans(a, transa)
            ldb, transb = _decide_ld_and_trans(b, transb)
            if not (lda is None or ldb is None):
                if out._c_contiguous:
                    # Computes out.T = alpha * b.T @ a.T + beta * out.T
                    try:
                        func(handle, 1 - transb, 1 - transa, n, m, k,
                             alpha_ptr, b.data.ptr, ldb, a.data.ptr, lda,
                             beta_ptr, out.data.ptr, n)
                    finally:
                        _cublas.setPointerMode(handle, orig_mode)
                    return out
                elif out._f_contiguous:
                    try:
                        func(handle, transa, transb, m, n, k, alpha_ptr,
                             a.data.ptr, lda, b.data.ptr, ldb, beta_ptr,
                             out.data.ptr, m)
                    finally:
                        _cublas.setPointerMode(handle, orig_mode)
                    return out
                elif out.strides[-1] == out.dtype.itemsize:
                    # Semi C-contiguous (sliced along second dim)
                    # Computes out.T = alpha * b.T @ a.T + beta * out.T
                    try:
                        ld_out = out.strides[-2] // out.strides[-1]
                        func(handle, 1 - transb, 1 - transa, n, m, k,
                             alpha_ptr, b.data.ptr, ldb, a.data.ptr, lda,
                             beta_ptr, out.data.ptr, ld_out)
                    finally:
                        _cublas.setPointerMode(handle, orig_mode)
                    return out

            # Backup plan with copies
            a, lda = _change_order_if_necessary(a, lda)
            b, ldb = _change_order_if_necessary(b, ldb)
            c = out
            if not out._f_contiguous:
                c = out.copy(order='F')
            try:
                func(handle, transa, transb, m, n, k, alpha_ptr, a.data.ptr,
                     lda, b.data.ptr, ldb, beta_ptr, c.data.ptr, m)
            finally:
                _cublas.setPointerMode(handle, orig_mode)
            if not out._f_contiguous:
                cupy._core.elementwise_copy(c, out)
            return out

        import cupyx
        from cupy.cuda import runtime
        numpy2 = np.__version__.split('.')[0] == '2'

        def fftshift_patch(x, axes=None):
            x = cupy.asarray(x)
            if axes is None:
                axes = list(range(x.ndim))
            elif not isinstance(axes, Iterable):
                axes = (axes,)
            return cupy.roll(x, [x.shape[axis] // 2 for axis in axes], axes)

        def ifftshift_patch(x, axes=None):
            x = cupy.asarray(x)
            if axes is None:
                axes = list(range(x.ndim))
            elif not isinstance(axes, Iterable):
                axes = (axes,)
            return cupy.roll(x, [-(x.shape[axis] // 2) for axis in axes], axes)

        if numpy2:
            cupy.fft.fftshift = fftshift_patch
            cupy.fft.ifftshift = ifftshift_patch

        is_hip = runtime.is_hip
        cupy_is_fake = False

        # Check the number of devices
        # Do not fail when calling `gpaw info` on a login node without GPUs
        try:
            device_count = runtime.getDeviceCount()
        except runtime.CUDARuntimeError as e:
            # Likely no device present
            if 'ErrorNoDevice' not in str(e):
                # Raise error in case of some other error
                raise e
            device_count = 0

        if device_count > 0:
            # select GPU device (round-robin based on MPI rank)
            # if not set, all MPI ranks will use the same default device
            from gpaw.mpi import rank
            runtime.setDevice(rank % device_count)

            # initialise C parameters and memory buffers
            import gpaw.cgpaw as cgpaw
            cgpaw.gpaw_gpu_init()

            # Generate a device id
            import os
            nodename = os.uname()[1]
            bus_id = runtime.deviceGetPCIBusId(runtime.getDevice())
            device_id = f'{nodename}:{bus_id}'

    except ImportError:
        import gpaw.gpu.cpupy as cupy
        import gpaw.gpu.cpupyx as cupyx
        from gpaw.gpu.cpupy.cublas import gemm as gpu_gemm  # noqa


__all__ = ['cupy', 'cupyx', 'as_xp', 'as_np', 'synchronize']


def synchronize():
    if not cupy_is_fake:
        cupy.cuda.get_current_stream().synchronize()


def as_np(array: np.ndarray | cupy.ndarray) -> np.ndarray:
    """Transfer array to CPU (if not already there).

    Parameters
    ==========
    array:
        Numpy or CuPy array.
    """
    if isinstance(array, np.ndarray):
        return array
    return cupy.asnumpy(array)


def as_xp(array, xp):
    """Transfer array to CPU or GPU (if not already there).

    Parameters
    ==========
    array:
        Numpy or CuPy array.
    xp:
        :mod:`numpy` or :mod:`cupy`.
    """
    if xp is np:
        if isinstance(array, np.ndarray):
            return array
        return cupy.asnumpy(array)
    if isinstance(array, np.ndarray):
        return cupy.asarray(array)
    1 / 0
    return array


def einsum(subscripts, *operands, out):
    if isinstance(out, np.ndarray):
        np.einsum(subscripts, *operands, out=out)
    else:
        out[:] = cupy.einsum(subscripts, *operands)


@trace(gpu=True)
def cupy_eigh(a: cupy.ndarray, UPLO: str) -> tuple[cupy.ndarray, cupy.ndarray]:
    """Wrapper for ``eigh()``.

    Usually CUDA > MAGMA > HIP, so we try to choose the best one.
    HIP native solver is questionably slow so for now do it on the CPU if
    MAGMA is not available.
    """
    from scipy.linalg import eigh
    if not is_hip:
        return cupy.linalg.eigh(a, UPLO=UPLO)

    elif have_magma and a.ndim == 2 and a.shape[0] > 128:
        # import here to avoid circular import.
        # magma needs cupy (possibly fake),
        # which must be imported from this file
        from gpaw.new.magma import eigh_magma_gpu

        return eigh_magma_gpu(a, UPLO)

    else:
        # fallback to CPU
        eigs, evals = eigh(cupy.asnumpy(a),
                           lower=(UPLO == 'L'),
                           check_finite=False)

    return cupy.asarray(eigs), cupy.asarray(evals)


class XP:
    """Class for adding xp attribute (numpy or cupy).

    Also implements pickling which will not work out of the box
    because a module can't be pickled.
    """
    def __init__(self, xp: ModuleType):
        self.xp = xp

    def __getstate__(self):
        state = self.__dict__.copy()
        assert self.xp is np
        del state['xp']
        return state

    def __setstate__(self, state):
        state['xp'] = np
        self.__dict__.update(state)


@contextlib.contextmanager
def T():
    t1 = time()
    yield
    synchronize()
    t2 = time()
    print(f'{(t2 - t1) * 1e9:_.3f} ns')
