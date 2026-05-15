#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "../gpu.h"
#include "../gpu-complex.h"

#ifndef GPU_USE_COMPLEX
#  define BLOCK_X  128
#  define MAX_BLOCKS  (65535)
#endif

__global__ void Zgpu(elmenwise_mul_add_kernelx)(
        int n, const double* a, const Tgpu* b, Tgpu *c)
{
    int i = blockIdx.x * BLOCK_X + threadIdx.x;
    while (i < n) {
        IADD(c[i], MULDT(a[i], b[i]));
        i += gridDim.x * BLOCK_X;
    }
}

__global__ void Zgpu(multi_elmenwise_mul_add_kernel1x)(
        int n, const double* a, const Tgpu* b, Tgpu *c)
{
    int i = blockIdx.x * BLOCK_X + threadIdx.x;
    int k = blockIdx.y;
    a += k * n;
    c += k * n;
    while (i < n) {
        IADD(c[i], MULDT(a[i], b[i]));
        i += gridDim.x * BLOCK_X;
    }
}

__global__ void Zgpu(multi_elmenwise_mul_add_kernel2x)(
        int n, const double* a, const Tgpu* b, Tgpu *c)
{
    int i = blockIdx.x * BLOCK_X + threadIdx.x;
    int k = blockIdx.y;
    b += k * n;
    c += k * n;
    while (i < n) {
        IADD(c[i], MULDT(a[i], b[i]));
        i += gridDim.x * BLOCK_X;
    }
}

__global__ void Zgpu(ax2py_kernel)(int n, double a, const Tgpu* x,
                                   double* y)
{
    int i = blockIdx.x * BLOCK_X + threadIdx.x;
    while (i < n) {
        y[i] += a * (REAL(x[i]) * REAL(x[i]) + IMAG(x[i]) * IMAG(x[i]));
        i += gridDim.x * BLOCK_X;
    }
}

__global__ void Zgpu(csign_kernel)(int n, Tgpu* x)
{
    int i = blockIdx.x * BLOCK_X + threadIdx.x;
    while (i < n) {
        x[i] = NEG(x[i]);
        i += gridDim.x * BLOCK_X;
    }
}

__global__ void Zgpu(multi_ax2py_kernel)(int n, int nvec, double *a,
                                         const Tgpu* x, double* y)
{
    int i = blockIdx.x * BLOCK_X + threadIdx.x;
    for (int k=0; k < nvec; k++) {
        int ii = i;
        while (ii < n) {
            y[ii] += a[k] * (REAL(x[ii]) * REAL(x[ii])
                             + IMAG(x[ii]) * IMAG(x[ii]));
            ii += gridDim.x * BLOCK_X;
        }
        x += n;
    }
}

#ifndef GPU_USE_COMPLEX
#define GPU_USE_COMPLEX
#include "linalg.cpp"

__global__ void elmenwise_mul_add_kernelzz(
        int n, const gpuDoubleComplex* a, const gpuDoubleComplex* b,
        gpuDoubleComplex* c)
{
    int i = blockIdx.x * BLOCK_X + threadIdx.x;
    while (i < n) {
        c[i] = gpuCadd(c[i], gpuCmul(a[i], b[i]));
        i += gridDim.x * BLOCK_X;
    }
}

__global__ void multi_elmenwise_mul_add_kernel1zz(
        int n, const gpuDoubleComplex* a, const gpuDoubleComplex* b,
        gpuDoubleComplex* c)
{
    int i = blockIdx.x * BLOCK_X + threadIdx.x;
    int k = blockIdx.y;
    a += k * n;
    c += k * n;
    while (i < n) {
        c[i] = gpuCadd(c[i], gpuCmul(a[i], b[i]));
        i += gridDim.x * BLOCK_X;
    }
}

__global__ void multi_elmenwise_mul_add_kernel2zz(
        int n, const gpuDoubleComplex* a, const gpuDoubleComplex* b,
        gpuDoubleComplex* c)
{
    int i = blockIdx.x * BLOCK_X + threadIdx.x;
    int k = blockIdx.y;
    b += k * n;
    c += k * n;
    while (i < n) {
        c[i] = gpuCadd(c[i], gpuCmul(a[i], b[i]));
        i += gridDim.x * BLOCK_X;
    }
}

extern "C"
PyObject* elementwise_multiply_add_gpu(PyObject *self, PyObject *args)
{
    void *x_gpu;
    void *y_gpu;
    void *c_gpu;
    PyObject *a_shape;
    PyArray_Descr *a_type, *y_type;

    if (!PyArg_ParseTuple(args, "nOOnOn", &x_gpu, &a_shape, &a_type,
                          &y_gpu, &y_type, &c_gpu))
        return NULL;

    int n = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
    Py_ssize_t nd = PyTuple_Size(a_shape);
    for (int d=1; d < nd; d++)
        n *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, d));

    int gridx = MIN(MAX((n + BLOCK_X - 1) / BLOCK_X, 1), MAX_BLOCKS);

    dim3 dimBlock(BLOCK_X, 1);
    dim3 dimGrid(gridx, 1);
    if (a_type->type_num == NPY_DOUBLE) {
        if (y_type->type_num == NPY_DOUBLE) {
            gpuLaunchKernel(
                    elmenwise_mul_add_kernelx, dimGrid, dimBlock, 0, 0,
                    n, (double*) x_gpu, (double*) y_gpu, (double*) c_gpu);
        } else {
            gpuLaunchKernel(
                    elmenwise_mul_add_kernelxz, dimGrid, dimBlock, 0, 0,
                    n, (double*) x_gpu, (gpuDoubleComplex*) y_gpu,
                    (gpuDoubleComplex*) c_gpu);
        }
    } else {
        if (y_type->type_num == NPY_DOUBLE) {
            gpuLaunchKernel(
                    elmenwise_mul_add_kernelxz, dimGrid, dimBlock, 0, 0,
                    n, (double*) y_gpu, (gpuDoubleComplex*) x_gpu,
                    (gpuDoubleComplex*) c_gpu);
        } else {
            gpuLaunchKernel(
                    elmenwise_mul_add_kernelzz, dimGrid, dimBlock, 0, 0,
                    n, (gpuDoubleComplex*) x_gpu, (gpuDoubleComplex*) y_gpu,
                    (gpuDoubleComplex*) c_gpu);
        }
    }
    gpuCheckLastError();
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}

extern "C"
PyObject* multi_elementwise_multiply_add_gpu(PyObject *self,
                                             PyObject *args)
{
    void *x_gpu;
    void *y_gpu;
    void *c_gpu;
    PyObject *x_shape, *y_shape, *shape;
    PyArray_Descr *x_type, *y_type;

    if (!PyArg_ParseTuple(args, "nOOnOOn", &x_gpu, &x_shape, &x_type,
                          &y_gpu, &y_shape, &y_type, &c_gpu))
        return NULL;

    Py_ssize_t x_nd = PyTuple_Size(x_shape);
    Py_ssize_t y_nd = PyTuple_Size(y_shape);

    shape = (x_nd > y_nd) ? x_shape : y_shape;

    int n = (int) PyLong_AsLong(PyTuple_GetItem(shape, 1));
    Py_ssize_t nd=PyTuple_Size(shape);
    for (int d=2; d < nd; d++)
        n *= (int) PyLong_AsLong(PyTuple_GetItem(shape, d));

    int nvec = (int) PyLong_AsLong(PyTuple_GetItem(shape, 0));

    int gridx = MIN(MAX((n + BLOCK_X - 1) / BLOCK_X, 1), MAX_BLOCKS);

    dim3 dimBlock(BLOCK_X, 1);
    dim3 dimGrid(gridx, nvec);

    if (x_type->type_num == NPY_DOUBLE) {
        if (y_type->type_num == NPY_DOUBLE) {
            if (x_nd > y_nd) {
                gpuLaunchKernel(
                        multi_elmenwise_mul_add_kernel1x,
                        dimGrid, dimBlock, 0, 0,
                        n, (double*) x_gpu, (double*) y_gpu, (double*) c_gpu);
            } else {
                gpuLaunchKernel(
                        multi_elmenwise_mul_add_kernel2x,
                        dimGrid, dimBlock, 0, 0,
                        n, (double*) x_gpu, (double*) y_gpu, (double*) c_gpu);
            }
        } else {
            if (x_nd > y_nd) {
                gpuLaunchKernel(
                        multi_elmenwise_mul_add_kernel1xz,
                        dimGrid, dimBlock, 0, 0,
                        n, (double*) x_gpu, (gpuDoubleComplex*) y_gpu,
                        (gpuDoubleComplex*) c_gpu);
            } else {
                gpuLaunchKernel(
                        multi_elmenwise_mul_add_kernel2xz,
                        dimGrid, dimBlock, 0, 0,
                        n, (double*) x_gpu, (gpuDoubleComplex*) y_gpu,
                        (gpuDoubleComplex*) c_gpu);
            }
        }
    } else {
        if (y_type->type_num == NPY_DOUBLE) {
            if (y_nd > x_nd) {
                gpuLaunchKernel(
                        multi_elmenwise_mul_add_kernel1xz,
                        dimGrid, dimBlock, 0, 0,
                        n, (double*) y_gpu, (gpuDoubleComplex*) x_gpu,
                        (gpuDoubleComplex*) c_gpu);
            } else {
                gpuLaunchKernel(
                        multi_elmenwise_mul_add_kernel2xz,
                        dimGrid, dimBlock, 0, 0,
                        n, (double*) y_gpu, (gpuDoubleComplex*) x_gpu,
                        (gpuDoubleComplex*) c_gpu);
            }
        } else {
            if (x_nd > y_nd) {
                gpuLaunchKernel(
                        multi_elmenwise_mul_add_kernel1zz,
                        dimGrid, dimBlock, 0, 0,
                        n, (gpuDoubleComplex*) x_gpu,
                        (gpuDoubleComplex*) y_gpu, (gpuDoubleComplex*) c_gpu);
            } else {
                gpuLaunchKernel(
                        multi_elmenwise_mul_add_kernel2zz,
                        dimGrid, dimBlock, 0, 0,
                        n, (gpuDoubleComplex*) x_gpu,
                        (gpuDoubleComplex*) y_gpu, (gpuDoubleComplex*) c_gpu);
            }
        }
    }
    gpuCheckLastError();
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}

extern "C"
PyObject* ax2py_gpu(PyObject *self, PyObject *args)
{
    double alpha;
    void *x_gpu;
    void *y_gpu;
    PyObject *x_shape, y_shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "dnOnOO", &alpha, &x_gpu, &x_shape,
                          &y_gpu, &y_shape, &type))
        return NULL;

    int n = (int) PyLong_AsLong(PyTuple_GetItem(x_shape, 0));
    Py_ssize_t nd = PyTuple_Size(x_shape);
    for (int d=1; d < nd; d++)
        n *= (int) PyLong_AsLong(PyTuple_GetItem(x_shape, d));

    int gridx = MIN(MAX((n + BLOCK_X - 1) / BLOCK_X, 1), MAX_BLOCKS);

    dim3 dimBlock(BLOCK_X, 1);
    dim3 dimGrid(gridx, 1);
    if (type->type_num == NPY_DOUBLE) {
        gpuLaunchKernel(
                ax2py_kernel, dimGrid, dimBlock, 0, 0,
                n, alpha, (double*) x_gpu, (double*) y_gpu);
    } else {
        gpuLaunchKernel(
                ax2py_kernelz, dimGrid, dimBlock, 0, 0,
                n, alpha, (Tgpu*) x_gpu, (double*) y_gpu);
    }
    gpuCheckLastError();
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}

extern "C"
PyObject* csign_gpu(PyObject *self, PyObject *args)
{
    void *x_gpu;
    PyObject *x_shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "nOO", &x_gpu, &x_shape, &type))
        return NULL;

    int n = (int) PyLong_AsLong(PyTuple_GetItem(x_shape, 0));
    Py_ssize_t nd = PyTuple_Size(x_shape);
    for (int d=1; d < nd; d++)
        n *= (int) PyLong_AsLong(PyTuple_GetItem(x_shape, d));

    int gridx = MIN(MAX((n + BLOCK_X - 1) / BLOCK_X, 1), MAX_BLOCKS);

    dim3 dimBlock(BLOCK_X, 1);
    dim3 dimGrid(gridx, 1);
    if (type->type_num == NPY_DOUBLE) {
        gpuLaunchKernel(
                csign_kernel, dimGrid, dimBlock, 0, 0,
                n, (double*) x_gpu);
    } else {
        gpuLaunchKernel(
                csign_kernelz, dimGrid, dimBlock, 0, 0,
                n, (Tgpu*) x_gpu);
    }
    gpuCheckLastError();
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}

extern "C"
PyObject* multi_ax2py_gpu(PyObject *self, PyObject *args)
{
    void *alpha_gpu;
    void *x_gpu;
    void *y_gpu;
    PyObject *x_shape, *y_shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "nnOnOO", &alpha_gpu, &x_gpu, &x_shape,
                          &y_gpu, &y_shape, &type))
        return NULL;

    int n = (int) PyLong_AsLong(PyTuple_GetItem(x_shape, 1));
    Py_ssize_t nd = PyTuple_Size(x_shape);
    for (int d=2; d < nd; d++)
        n *= (int) PyLong_AsLong(PyTuple_GetItem(x_shape, d));

    int nvec = (int) PyLong_AsLong(PyTuple_GetItem(x_shape, 0));

    if (type->type_num == NPY_DOUBLE) {
        double *alpha = (double*) alpha_gpu;
        int gridx = MIN(MAX((n + BLOCK_X - 1) / BLOCK_X, 1), MAX_BLOCKS);
        int gridy = 1;

        dim3 dimBlock(BLOCK_X, 1);
        dim3 dimGrid(gridx, gridy);

        gpuLaunchKernel(
                multi_ax2py_kernel, dimGrid, dimBlock, 0, 0,
                n, nvec, alpha, (double*) x_gpu, (double*) y_gpu);
    } else {
        double *alpha = (double*) alpha_gpu;
        int gridx = MIN(MAX((n + BLOCK_X - 1) / BLOCK_X, 1), MAX_BLOCKS);
        int gridy = 1;

        dim3 dimBlock(BLOCK_X, 1);
        dim3 dimGrid(gridx, gridy);
        gpuLaunchKernel(
                multi_ax2py_kernelz, dimGrid, dimBlock, 0, 0,
                n, nvec, alpha, (gpuDoubleComplex*) x_gpu, (double*) y_gpu);
    }
    gpuCheckLastError();
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}
#endif
