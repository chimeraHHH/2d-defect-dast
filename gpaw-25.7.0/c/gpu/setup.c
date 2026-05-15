#include "../extensions.h"
#include "gpu.h"
#include "gpu-complex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void bc_init_buffers_gpu();
void blas_init_gpu();
void transformer_init_buffers_gpu();
void operator_init_buffers_gpu();
void reduce_init_buffers_gpu();
void lfc_reduce_init_buffers_gpu();
void bc_dealloc_gpu(int force);
void transformer_dealloc_gpu(int force);
void operator_dealloc_gpu(int force);
void reduce_dealloc_gpu();
void lfc_reduce_dealloc_gpu();


PyObject* gpaw_gpu_init(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    bc_init_buffers_gpu();
    transformer_init_buffers_gpu();
    operator_init_buffers_gpu();
    reduce_init_buffers_gpu();
    lfc_reduce_init_buffers_gpu();
    blas_init_gpu();

    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}

PyObject* gpaw_gpu_delete(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    reduce_dealloc_gpu();
    lfc_reduce_dealloc_gpu();
    bc_dealloc_gpu(1);
    transformer_dealloc_gpu(1);
    operator_dealloc_gpu(1);

    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}
