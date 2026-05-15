#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <complex.h>
#include <sys/types.h>
#include <sys/time.h>

#include "../gpu-complex.h"

#ifndef GPU_USE_COMPLEX
#  define MBLAS_BLOCK_X  (128)
#  define MAX_BLOCKS     (65535)
#  define MIN_BLOCKS     (MAX_BLOCKS)
#endif

#define MAPNAME(f) Zgpu(f ## _dotu)
#define MAPFUNC(a,b) MULTT((a), (b))
#include "reduce.cpp"
#undef MAPNAME
#undef MAPFUNC

#define MAPNAME(f) Zgpu(f ## _dotc)
#define MAPFUNC(a,b) MULTT(CONJ(a), (b))
#include "reduce.cpp"
#undef MAPNAME
#undef MAPFUNC


__global__ void Zgpu(multi_scal_kernel)(int n, const Tgpu *alpha, Tgpu *a)
{
    int i = blockIdx.x * MBLAS_BLOCK_X + threadIdx.x;
    int k = blockIdx.y;

    a += n * k;

    while (i < n) {
        a[i] = MULTT(a[i], alpha[k]);
        i += gridDim.x * MBLAS_BLOCK_X;
    }
}

__global__ void Zgpu(multi_axpy_kernel)(int n, const Tgpu *alpha,
                                             const Tgpu *a, Tgpu *b)
{
    int k = blockIdx.y;
    int i = blockIdx.x * MBLAS_BLOCK_X + threadIdx.x;

    a += n * k;
    b += n * k;
    while (i < n) {
        IADD(b[i], MULTT(a[i], alpha[k]));
        i += gridDim.x * MBLAS_BLOCK_X;
    }
}


#ifndef GPU_USE_COMPLEX
#define GPU_USE_COMPLEX
#include "mblas.cpp"

extern "C"
PyObject* multi_scal_gpu(PyObject *self, PyObject *args)
{
    void *alpha_gpu;
    void *x_gpu;
    PyObject *x_shape;
    PyArray_Descr *type, *a_type;

    if (!PyArg_ParseTuple(args, "nOnOO", &alpha_gpu, &a_type,
                          &x_gpu, &x_shape, &type))
        return NULL;

    int n = (int) PyLong_AsLong(PyTuple_GetItem(x_shape, 1));
    Py_ssize_t nd = PyTuple_Size(x_shape);
    for (int d=2; d < nd; d++) {
        n *= (int) PyLong_AsLong(PyTuple_GetItem(x_shape, d));
    }
    int nvec = (int) PyLong_AsLong(PyTuple_GetItem(x_shape, 0));

    if (type->type_num == NPY_DOUBLE) {
        int gridx = MIN(MAX((n + MBLAS_BLOCK_X - 1) / MBLAS_BLOCK_X, 1),
                        MAX_BLOCKS);
        int gridy = nvec;
        dim3 dimBlock(MBLAS_BLOCK_X, 1);
        dim3 dimGrid(gridx, gridy);

        gpuLaunchKernel(
                multi_scal_kernel, dimGrid, dimBlock, 0, 0,
                n, (double *) alpha_gpu, (double*) x_gpu);
    } else if (a_type->type_num == NPY_DOUBLE) {
        double *alpha = (double*) (alpha_gpu);
        int gridx = MIN(MAX((2 * n + MBLAS_BLOCK_X - 1)
                             / MBLAS_BLOCK_X, 1),
                        MAX_BLOCKS);
        int gridy = nvec;
        dim3 dimBlock(MBLAS_BLOCK_X, 1);
        dim3 dimGrid(gridx, gridy);

        gpuLaunchKernel(
                multi_scal_kernel, dimGrid, dimBlock, 0, 0,
                2 * n, alpha, (double *) x_gpu);
    } else {
        gpuDoubleComplex *alpha = (gpuDoubleComplex*) (alpha_gpu);
        int gridx = MIN(MAX((n + MBLAS_BLOCK_X - 1) / MBLAS_BLOCK_X, 1),
                        MAX_BLOCKS);
        int gridy = nvec;
        dim3 dimBlock(MBLAS_BLOCK_X, 1);
        dim3 dimGrid(gridx, gridy);

        gpuLaunchKernel(
                multi_scal_kernelz, dimGrid, dimBlock, 0, 0,
                n, alpha, (gpuDoubleComplex*) x_gpu);
    }
    gpuCheckLastError();
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}

extern "C"
PyObject* multi_axpy_gpu(PyObject *self, PyObject *args)
{
    void *alpha_gpu;
    void *x_gpu;
    void *y_gpu;
    PyObject *x_shape, *y_shape;
    PyArray_Descr *type, *a_type;

    if (!PyArg_ParseTuple(args, "nOnOnOO", &alpha_gpu, &a_type,
                          &x_gpu, &x_shape, &y_gpu, &y_shape, &type))
        return NULL;

    int n = (int) PyLong_AsLong(PyTuple_GetItem(x_shape, 1));
    Py_ssize_t nd = PyTuple_Size(x_shape);
    for (int d=2; d < nd; d++) {
        n *= (int) PyLong_AsLong(PyTuple_GetItem(x_shape, d));
    }
    int nvec = (int) PyLong_AsLong(PyTuple_GetItem(x_shape, 0));
    if (type->type_num == NPY_DOUBLE) {
        double *alpha = (double*) alpha_gpu;
        int gridx = MIN(MAX((n + MBLAS_BLOCK_X - 1) / MBLAS_BLOCK_X, 1),
                        MAX_BLOCKS);
        int gridy = nvec;
        dim3 dimBlock(MBLAS_BLOCK_X, 1);
        dim3 dimGrid(gridx, gridy);

        gpuLaunchKernel(
                multi_axpy_kernel, dimGrid, dimBlock, 0, 0,
                n, alpha, (double*) x_gpu, (double*) y_gpu);
    } else  if (a_type->type_num == NPY_DOUBLE) {
        double *alpha = (double*) alpha_gpu;
        int gridx = MIN(MAX((2 * n + MBLAS_BLOCK_X - 1)
                            / MBLAS_BLOCK_X, 1),
                        MAX_BLOCKS);
        int gridy = nvec;
        dim3 dimBlock(MBLAS_BLOCK_X, 1);
        dim3 dimGrid(gridx, gridy);

        gpuLaunchKernel(
                multi_axpy_kernel, dimGrid, dimBlock, 0, 0,
                2 * n, alpha, (double*) x_gpu, (double*) y_gpu);
    } else {
        gpuDoubleComplex *alpha = (gpuDoubleComplex*) alpha_gpu;
        int gridx = MIN(MAX((n + MBLAS_BLOCK_X - 1) / MBLAS_BLOCK_X, 1),
                        MAX_BLOCKS);
        int gridy = nvec;
        dim3 dimBlock(MBLAS_BLOCK_X, 1);
        dim3 dimGrid(gridx, gridy);

        gpuLaunchKernel(
                multi_axpy_kernelz, dimGrid, dimBlock, 0, 0,
                n, alpha, (gpuDoubleComplex*) x_gpu,
                (gpuDoubleComplex*) y_gpu);
    }
    gpuCheckLastError();
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}

extern "C"
PyObject* multi_dotu_gpu(PyObject *self, PyObject *args)
{
    void *a_gpu;
    void *b_gpu;
    void *res_gpu;

    PyObject *a_shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "nOnOn", &a_gpu, &a_shape, &b_gpu,
                          &type, &res_gpu))
        return NULL;

    int n = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 1));
    for (int i=2; i < PyTuple_Size(a_shape); i++) {
        n *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, i));
    }
    int nvec = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
    if (type->type_num == NPY_DOUBLE) {
        double *result = (double *) res_gpu;
        reducemap_dotu((double*) a_gpu, (double*) b_gpu, result, n,
                       nvec);
    } else {
        gpuDoubleComplex *result = (gpuDoubleComplex *) res_gpu;
        reducemap_dotuz((gpuDoubleComplex *) a_gpu,
                        (gpuDoubleComplex *) b_gpu,
                        result, n, nvec);
    }
    gpuCheckLastError();
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}

extern "C"
PyObject* multi_dotc_gpu(PyObject *self, PyObject *args)
{
    void *a_gpu;
    void *b_gpu;
    void *res_gpu;

    PyObject *a_shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "nOnOn", &a_gpu, &a_shape, &b_gpu,
                          &type, &res_gpu))
        return NULL;

    int n = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 1));
    Py_ssize_t nd=PyTuple_Size(a_shape);
    for (int i = 2; i < nd; i++) {
        n *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, i));
    }
    int nvec = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
    if (type->type_num == NPY_DOUBLE) {
        double *result = (double *) res_gpu;
        reducemap_dotc((double*) a_gpu, (double*) b_gpu, result, n,
                       nvec);
    } else {
        gpuDoubleComplex *result = (gpuDoubleComplex *) res_gpu;
        reducemap_dotcz((gpuDoubleComplex*) a_gpu,
                        (gpuDoubleComplex*) b_gpu,
                        result, n, nvec);
    }
    gpuCheckLastError();
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}
#endif
