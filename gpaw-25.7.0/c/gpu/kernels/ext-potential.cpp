#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "../gpu.h"
#include "../gpu-complex.h"

#ifndef GPU_USE_COMPLEX
#define BLOCK_SIZEX 32
#define BLOCK_SIZEY 8
#define XDIV 4
#endif


__global__ void Zgpu(add_linear_field_kernel)(
        const Tgpu *a, const int3 c_sizea, Tgpu *b, const int3 c_n,
        const int3 c_beg, const double3 strength, int blocks)
{
    int xx = gridDim.x / XDIV;
    int yy = gridDim.y / blocks;

    int blocksi = blockIdx.y / yy;
    int i1bl = blockIdx.y - yy * blocksi;
    int i1tid = threadIdx.y;
    int i1 = i1bl * BLOCK_SIZEY + i1tid;

    int xind = blockIdx.x / xx;
    int i2bl = blockIdx.x - xind * xx;
    int i2 = i2bl * BLOCK_SIZEX + threadIdx.x;

    int xlen = (c_n.x + XDIV - 1) / XDIV;
    int xstart = xind * xlen;
    int xend = MIN(xstart + xlen, c_n.x);

    b += c_sizea.x * c_sizea.y * c_sizea.z * blocksi;
    a += c_sizea.x * c_sizea.y * c_sizea.z * blocksi;

    b += i2 + i1 * c_sizea.z + xstart * c_sizea.y * c_sizea.z;
    a += i2 + i1 * c_sizea.z + xstart * c_sizea.y * c_sizea.z;

    double yz = (i1 + c_beg.y) * strength.y + (i2 + c_beg.z) * strength.z;
    for (int i0=xstart; i0 < xend; i0++) {
        if ((i2 < c_n.z) && (i1 < c_n.y)) {
            IADD(b[0], MULDT(((i0 + c_beg.x) * strength.x + yz), a[0]));
        }
        b += c_sizea.y * c_sizea.z;
        a += c_sizea.y * c_sizea.z;
    }
}

#ifndef GPU_USE_COMPLEX
#define GPU_USE_COMPLEX
#include "ext-potential.cpp"

extern "C"
PyObject* add_linear_field_gpu(PyObject *self, PyObject *args)
{
    void *a_gpu;
    void *b_gpu;
    PyObject *shape;
    PyArrayObject *c_ni, *c_begi, *c_vi, *strengthi;
    PyArray_Descr *type;
    int blocks=1;

    int3 hc_sizea, hc_n, hc_beg;
    double3 h_strength;

    if (!PyArg_ParseTuple(args, "nOOnOOOO", &a_gpu, &shape, &type,
                          &b_gpu, &c_ni, &c_begi, &c_vi, &strengthi))
        return NULL;

    int nd = PyTuple_Size(shape);
    if (nd == 4)
        blocks = (int) PyLong_AsLong(PyTuple_GetItem(shape, 0));

    hc_sizea.x = (int) PyLong_AsLong(PyTuple_GetItem(shape, nd-3+0));
    hc_sizea.y = (int) PyLong_AsLong(PyTuple_GetItem(shape, nd-3+1));
    hc_sizea.z = (int) PyLong_AsLong(PyTuple_GetItem(shape, nd-3+2));

    hc_n.x = ((long*) PyArray_DATA(c_ni))[0];
    hc_n.y = ((long*) PyArray_DATA(c_ni))[1];
    hc_n.z = ((long*) PyArray_DATA(c_ni))[2];

    hc_beg.x = ((long*) PyArray_DATA(c_begi))[0];
    hc_beg.y = ((long*) PyArray_DATA(c_begi))[1];
    hc_beg.z = ((long*) PyArray_DATA(c_begi))[2];

    h_strength.x = ((double*) PyArray_DATA(strengthi))[0]
                 * ((double*) PyArray_DATA(c_vi))[0+0*3];
    h_strength.y = ((double*) PyArray_DATA(strengthi))[1]
                 * ((double*) PyArray_DATA(c_vi))[1+1*3];
    h_strength.z = ((double*) PyArray_DATA(strengthi))[2]
                 * ((double*) PyArray_DATA(c_vi))[2+2*3];

    int gridy = blocks * ((hc_n.y + BLOCK_SIZEY - 1) / BLOCK_SIZEY);
    int gridx = XDIV * ((hc_n.z + BLOCK_SIZEX - 1) / BLOCK_SIZEX);

    dim3 dimBlock(BLOCK_SIZEX, BLOCK_SIZEY);
    dim3 dimGrid(gridx, gridy);

    if (type->type_num == NPY_DOUBLE) {
        gpuLaunchKernel(add_linear_field_kernel,
                        dimGrid, dimBlock, 0, 0,
                        (double*) a_gpu, hc_sizea,
                        (double*) b_gpu, hc_n, hc_beg,
                        h_strength, blocks);
    } else {
        gpuLaunchKernel(add_linear_field_kernelz,
                        dimGrid, dimBlock, 0, 0,
                        (gpuDoubleComplex*) a_gpu, hc_sizea,
                        (gpuDoubleComplex*) b_gpu, hc_n, hc_beg,
                        h_strength, blocks);
    }
    gpuCheckLastError();
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}
#endif
