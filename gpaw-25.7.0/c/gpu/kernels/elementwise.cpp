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

/*
 * GPU kernel for axpbyz, i.e. z[i] = a * x[i] + b * y[i]
 */
__global__ void axpbyz_kernel(double a, double *x, double b, double *y,
                              double *z, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        z[tid] = a * x[tid] + b * y[tid];
    }
}

/*
 * GPU kernel for axpbz, i.e. z[i] = a * x[i] + b
 */
__global__ void axpbz_kernel(double a, double *x, double b,
                             double *z, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        z[tid] = a * x[tid] + b;
    }
}

/*
 * GPU kernel for axpbyz, i.e. z[i] = a * x[i] + b * y[i],
 * on complex numbers.
 */
__global__ void axpbyz_kernelz(double a, gpuDoubleComplex *x,
                               double b, gpuDoubleComplex *y,
                               gpuDoubleComplex *z, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        (z[tid]).x = a * gpuCreal(x[tid]) + b * gpuCreal(y[tid]);
        (z[tid]).y = a * gpuCimag(x[tid]) + b * gpuCimag(y[tid]);
    }
}

/*
 * GPU kernel for axpbz, i.e. z[i] = a * x[i] + b, on complex numbers.
 */
__global__ void axpbz_kernelz(double a, gpuDoubleComplex *x, double b,
                              gpuDoubleComplex *z, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        (z[tid]).x = a * gpuCreal(x[tid]) + b;
        (z[tid]).y = a * gpuCimag(x[tid]) + b;
    }
}

/*
 * GPU kernel to fill an array of doubles with a given value.
 */
__global__ void fill_kernel(double a, double *z, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        z[tid] = a;
    }
}

/*
 * GPU kernel to fill an array of complex numbers with a given
 * complex value.
 */
__global__ void fill_kernelz(double real, double imag, gpuDoubleComplex *z,
                             int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (; tid < n; tid += stride) {
        (z[tid]).x = real;
        (z[tid]).y = imag;
    }
}

/*
 * GPU version of axpbyz, i.e. z[i] = a * x[i] + b * y[i]
 *
 * Arguments:
 *   a, x, b, y, z -- (as above)
 *   shape         -- shape of the arrays
 *   type          -- datatype of elements in the arrays
 */
extern "C"
PyObject* axpbyz_gpu(PyObject *self, PyObject *args)
{
    double a, b;
    void *x;
    void *y;
    void *z;
    PyObject *shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "dndnnOO", &a, &x, &b, &y, &z, &shape,
                          &type))
        return NULL;

    int n = 1;
    Py_ssize_t nd = PyTuple_Size(shape);
    for (int d=0; d < nd; d++)
        n *= (int) PyLong_AsLong(PyTuple_GetItem(shape, d));

    int gridx = MIN(MAX((n + BLOCK_X - 1) / BLOCK_X, 1), MAX_BLOCKS);

    dim3 dimBlock(BLOCK_X, 1);
    dim3 dimGrid(gridx, 1);
    if (type->type_num == NPY_DOUBLE) {
        gpuLaunchKernel(axpbyz_kernel, dimGrid, dimBlock, 0, 0,
                        a, (double*) x, b, (double*) y, (double *) z, n);

    } else {
        gpuLaunchKernel(axpbyz_kernelz, dimGrid, dimBlock, 0, 0,
                        a, (gpuDoubleComplex*) x, b, (gpuDoubleComplex*) y,
                        (gpuDoubleComplex*) z, n);
    }
    gpuCheckLastError();
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}

/*
 * GPU version of axpbz, i.e. z[i] = a * x[i] + b
 *
 * Arguments:
 *   a, x, b, z -- (as above)
 *   shape      -- shape of the arrays
 *   type       -- datatype of elements in the arrays
 */
extern "C"
PyObject* axpbz_gpu(PyObject *self, PyObject *args)
{
    double a, b;
    void *x;
    void *z;
    PyObject *shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "dndnOO", &a, &x, &b, &z, &shape,
                          &type))
        return NULL;

    int n = 1;
    Py_ssize_t nd = PyTuple_Size(shape);
    for (int d=0; d < nd; d++)
        n *= (int) PyLong_AsLong(PyTuple_GetItem(shape, d));

    int gridx = MIN(MAX((n + BLOCK_X - 1) / BLOCK_X, 1), MAX_BLOCKS);

    dim3 dimBlock(BLOCK_X, 1);
    dim3 dimGrid(gridx, 1);
    if (type->type_num == NPY_DOUBLE) {
        gpuLaunchKernel(axpbz_kernel, dimGrid, dimBlock, 0, 0,
                        a, (double*) x, b, (double *) z, n);

    } else {
        gpuLaunchKernel(axpbz_kernelz, dimGrid, dimBlock, 0, 0,
                        a, (gpuDoubleComplex*) x, b, (gpuDoubleComplex*) z, n);
    }
    gpuCheckLastError();
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}

/*
 * Fill a GPU array with a given value, i.e. x[i] = value
 *
 * Arguments:
 *   x, value -- (as above)
 *   shape    -- shape of the arrays
 *   type     -- datatype of elements in the arrays
 */
extern "C"
PyObject* fill_gpu(PyObject *self, PyObject *args)
{
    PyObject *value;
    void *x;
    PyObject *shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "OnOO", &value, &x, &shape, &type))
        return NULL;

    double real;
    double imag;
    if (PyComplex_Check(value)) {
        Py_complex c;
        c = PyComplex_AsCComplex(value);
        real = c.real;
        imag = c.imag;
    } else {
        real = PyFloat_AsDouble(value);
        imag = 0.0;
    }

    int n = 1;
    Py_ssize_t nd = PyTuple_Size(shape);
    for (int d=0; d < nd; d++)
        n *= (int) PyLong_AsLong(PyTuple_GetItem(shape, d));

    int gridx = MIN(MAX((n + BLOCK_X - 1) / BLOCK_X, 1), MAX_BLOCKS);

    dim3 dimBlock(BLOCK_X, 1);
    dim3 dimGrid(gridx, 1);
    if (type->type_num == NPY_DOUBLE) {
        gpuLaunchKernel(fill_kernel, dimGrid, dimBlock, 0, 0,
                        real, (double*) x, n);

    } else {
        gpuLaunchKernel(fill_kernelz, dimGrid, dimBlock, 0, 0,
                        real, imag, (gpuDoubleComplex*) x, n);
    }
    gpuCheckLastError();
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}
