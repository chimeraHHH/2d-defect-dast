#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <pthread.h>

#include "../extensions.h"
#define __TRANSFORMERS_C
#include "../transformers.h"
#undef __TRANSFORMERS_C
#include "bmgs.h"
#include "gpu.h"

static double *transformer_buf_gpu = NULL;
static double *transformer_buf16_gpu = NULL;
static int transformer_buf_size = 0;
static int transformer_init_count = 0;

/*
 * Increment reference count to register a new tranformer object.
 */
void transformer_init_gpu(TransformerObject *self)
{
    transformer_init_count++;
}

/*
 * Ensure buffer is allocated and is big enough. Reallocate only if
 * size has increased.
 */
void transformer_init_buffers(TransformerObject *self, int blocks)
{
    const boundary_conditions* bc = self->bc;
    const int* size2 = bc->size2;
    int ng2 = (bc->ndouble * size2[0] * size2[1] * size2[2]) * blocks;

    if (ng2 > transformer_buf_size) {
        gpuFree(transformer_buf_gpu);
        gpuCheckLastError();
        gpuMalloc(&transformer_buf_gpu, sizeof(double) * ng2);
        gpuFree(transformer_buf16_gpu);
        gpuCheckLastError();
        gpuMalloc(&transformer_buf16_gpu, sizeof(double) * ng2 * 16);
        transformer_buf_size = ng2;
    }
}

/*
 * Reset reference count and unset buffer.
 */
void transformer_init_buffers_gpu()
{
    transformer_buf_gpu = NULL;
    transformer_buf16_gpu = NULL;
    transformer_buf_size = 0;
    transformer_init_count = 0;
}

/*
 * Deallocate buffer or decrease reference count.
 *
 * arguments:
 *   (int) force -- if true, force deallocation
 */
void transformer_dealloc_gpu(int force)
{
    if (force)
        transformer_init_count = 1;

    if (transformer_init_count == 1) {
        gpuFree(transformer_buf_gpu);
        gpuCheckLastError();
        transformer_init_buffers_gpu();
        return;
    }
    if (transformer_init_count > 0)
        transformer_init_count--;
}

/*
 * Run the interpolate and restrict algorithm (see transapply_worker()
 * in ../transformers.c) on the GPU.
 */
static void _transformer_apply_gpu(TransformerObject* self,
                                   const double *in, double *out,
                                   int nin, int blocks, bool real,
                                   const double_complex *ph, bool stencil)
{
    boundary_conditions* bc = self->bc;
    const int* size1 = bc->size1;
    int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
    int out_ng = bc->ndouble * self->size_out[0] * self->size_out[1]
               * self->size_out[2];

    int mpi_size = 1;
    if ((bc->maxsend || bc->maxrecv) && bc->comm != MPI_COMM_NULL)
        MPI_Comm_size(bc->comm, &mpi_size);

    MPI_Request recvreq[3][2];
    MPI_Request sendreq[3][2];

    transformer_init_buffers(self, blocks);

    double* buf = transformer_buf_gpu;
    double* buf16 = transformer_buf16_gpu;

    /* use stencil version, if no optimised kernel available */
    if (self->k != 2) {
        stencil = 1;
    }

    for (int n = 0; n < nin; n += blocks) {
        const double* in2 = in + n * ng;
        double* out2 = out + n * out_ng;
        int myblocks = MIN(blocks, nin - n);

        bc_unpack_paste_gpu(bc, in2, buf, recvreq, 0, myblocks);
        for (int i=0; i < 3; i++) {
            bc_unpack_gpu(bc, buf, i, recvreq, sendreq[i],
                          ph + 2 * i, 0, myblocks);
        }
        if (self->interpolate) {
            if (stencil) {
                if (real) {
                    bmgs_interpolate_stencil_gpu(self->k, self->skip, buf,
                                                 bc->size2, out2,
                                                 self->size_out, buf16,
                                                 myblocks);
                } else {
                    bmgs_interpolate_stencil_gpuz(self->k, self->skip,
                                                  (gpuDoubleComplex*) (buf),
                                                  bc->size2,
                                                  (gpuDoubleComplex*) (out2),
                                                  self->size_out,
                                                  (gpuDoubleComplex*) (buf16),
                                                  myblocks);
                }
            } else {
                if (real) {
                    bmgs_interpolate_gpu(self->k, self->skip, buf,
                                         bc->size2, out2, self->size_out,
                                         myblocks);
                } else {
                    bmgs_interpolate_gpuz(self->k, self->skip,
                                          (gpuDoubleComplex*) (buf),
                                          bc->size2,
                                          (gpuDoubleComplex*) (out2),
                                          self->size_out, myblocks);
                }
            }
        } else {
            if (stencil) {
                if (real) {
                    bmgs_restrict_stencil_gpu(self->k, buf, bc->size2,
                                              out2, self->size_out, buf16,
                                              myblocks);
                } else {
                    bmgs_restrict_stencil_gpuz(self->k,
                                               (gpuDoubleComplex*) (buf),
                                               bc->size2,
                                               (gpuDoubleComplex*) (out2),
                                               self->size_out,
                                               (gpuDoubleComplex*) (buf16),
                                               myblocks);
                }
            } else {
                if (real) {
                    bmgs_restrict_gpu(self->k, buf, bc->size2,
                                      out2, self->size_out, myblocks);
                } else {
                    bmgs_restrict_gpuz(self->k,
                                       (gpuDoubleComplex*) (buf),
                                       bc->size2,
                                       (gpuDoubleComplex*) (out2),
                                       self->size_out, myblocks);
                }
            }
        }
    }
}

/*
 * Python interface for the GPU version of the interpolate and restrict
 * algorithm (similar to Transformer_apply() for CPUs).
 *
 * arguments:
 *   input_gpu  -- pointer to device memory (GPUArray.gpudata)
 *   output_gpu -- pointer to device memory (GPUArray.gpudata)
 *   shape      -- shape of the array (tuple)
 *   type       -- datatype of array elements
 *   phases     -- phase (complex) (ignored if type is NPY_DOUBLE)
 *   stencil    -- use stencil version of interpolate functions
 */
PyObject* Transformer_apply_gpu(TransformerObject *self, PyObject *args)
{
    PyArrayObject* phases = 0;
    void *input_gpu;
    void *output_gpu;
    PyObject *shape;
    PyArray_Descr *type;
    int stencil = 0;

    if (!PyArg_ParseTuple(args, "nnOO|Oi", &input_gpu, &output_gpu, &shape,
                          &type, &phases, &stencil))
        return NULL;

    int nin = 1;
    if (PyTuple_Size(shape) == 4)
        nin = (int) PyLong_AsLong(PyTuple_GetItem(shape, 0));

    const double* in = (double*) input_gpu;
    double* out = (double*) output_gpu;

    bool real = (type->type_num == NPY_DOUBLE);
    const double_complex* ph = (real ? 0 : COMPLEXP(phases));

    boundary_conditions* bc = self->bc;
    int mpi_size = 1;
    if ((bc->maxsend || bc->maxrecv) && bc->comm != MPI_COMM_NULL)
        MPI_Comm_size(bc->comm, &mpi_size);

    int blocks = MAX(1, MIN(nin, MIN((GPU_BLOCKS_MIN) * mpi_size,
                                     (GPU_BLOCKS_MAX) / bc->ndouble)));

    _transformer_apply_gpu(self, in, out, nin, blocks, real, ph, stencil);

    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}
