/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2008  CAMd
 *  Copyright (C) 2005-2020  CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#include <Python.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "extensions.h"
#include "bc.h"
#include "mympi.h"
#include "bmgs/bmgs.h"
#include "threading.h"

#define __TRANSFORMERS_C
#include "transformers.h"
#undef __TRANSFORMERS_C

#ifdef GPAW_GPU
#include "gpu/gpu.h"
#include "gpu/bmgs.h"

PyObject* Transformer_apply_gpu(TransformerObject *self, PyObject *args);
#endif

static void Transformer_dealloc(TransformerObject *self)
{
#ifdef GPAW_GPU
  if (self->use_gpu) {
    transformer_dealloc_gpu(0);
    bc_dealloc_gpu(0);
  }
#endif
  free(self->bc);
  PyObject_DEL(self);
}

// The actual computation routine for interpolation and restriction
// operations. The routine is used also in C-preconditioner
void transapply_worker(TransformerObject *self, int chunksize, int start,
		       int end, int thread_id, int nthreads, 
		       const double* in, double* out,
		       bool real, const double_complex* ph)
{
  boundary_conditions* bc = self->bc;
  const int* size1 = bc->size1;
  const int* size2 = bc->size2;
  int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
  int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];

  double* sendbuf = GPAW_MALLOC(double, bc->maxsend * chunksize);
  double* recvbuf = GPAW_MALLOC(double, bc->maxrecv * chunksize);
  double* buf = GPAW_MALLOC(double, ng2 * chunksize);
  int buf2size = ng2;
  if (self->interpolate)
    buf2size *= 16;
  else
    buf2size /= 2;
  double* buf2 = GPAW_MALLOC(double, buf2size * chunksize);
  MPI_Request recvreq[2];
  MPI_Request sendreq[2];

  const double* my_in;
  double* my_out;
  
  int out_ng = bc->ndouble * self->size_out[0] * self->size_out[1]
               * self->size_out[2];

  for (int n = start; n < end; n += chunksize)
    {
      if (n + chunksize >= end && chunksize > 1)
        chunksize = end - n;
      my_in = in + n * ng;
      my_out = out + n * out_ng;
      for (int i = 0; i < 3; i++)
        {
          bc_unpack1(bc, my_in, buf, i,
                     recvreq, sendreq,
                     recvbuf, sendbuf, ph + 2 * i,
                     thread_id, 1);
          bc_unpack2(bc, buf, i,
                     recvreq, sendreq, recvbuf, 1);
        }
      for (int m = 0; m < chunksize; m++) {
        if (real)
          {
            if (self->interpolate)
              bmgs_interpolate(self->k, self->skip, buf + m * ng2, bc->size2,
                               my_out + m * out_ng, buf2 + m * buf2size);
            else
              bmgs_restrict(self->k, buf + m * ng2, bc->size2,
                            my_out + m * out_ng, buf2 + m * buf2size);
          }
        else
          {
            if (self->interpolate)
              bmgs_interpolatez(self->k, self->skip, (double_complex*)(buf + m * ng2),
                                bc->size2, (double_complex*)(my_out + m * out_ng),
                                (double_complex*)(buf2 + m * buf2size));
            else
              bmgs_restrictz(self->k, (double_complex*)(buf + m * ng2),
                             bc->size2, (double_complex*)(my_out + m * out_ng),
                             (double_complex*)(buf2 + m * buf2size));
          }
      }
    }
  free(buf2);
  free(buf);
  free(recvbuf);
  free(sendbuf);
}


static PyObject* Transformer_apply(TransformerObject *self, PyObject *args)
{
  PyArrayObject* input;
  PyArrayObject* output;
  PyArrayObject* phases = 0;
  if (!PyArg_ParseTuple(args, "OO|O", &input, &output, &phases))
    return NULL;

  int nin = 1;
  if (PyArray_NDIM(input) == 4)
    nin = PyArray_DIMS(input)[0];

  boundary_conditions* bc = self->bc;
  
  const double* in = DOUBLEP(input);
  double* out = DOUBLEP(output);
  bool real = (PyArray_DESCR(input)->type_num == NPY_DOUBLE);
  const double_complex* ph = (real ? 0 : COMPLEXP(phases));

  int chunksize = 1;
  if (getenv("GPAW_MPI_OPTIMAL_MSG_SIZE") != NULL)
    {
      int opt_msg_size = atoi(getenv("GPAW_MPI_OPTIMAL_MSG_SIZE"));
      if (bc->maxsend > 0 )
          chunksize = opt_msg_size * 1024 / (bc->maxsend / 2 * 
                                             (2 - (int)real) * sizeof(double));
      chunksize = (chunksize > 0) ? chunksize : 1;
      chunksize = (chunksize < nin) ? chunksize : nin;
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
  int thread_id = 0;
  int nthreads = 1;
  int start, end;
#ifdef _OPENMP
  thread_id = omp_get_thread_num();
  nthreads = omp_get_num_threads();
#endif
  SHARE_WORK(nin, nthreads, thread_id, &start, &end);
  transapply_worker(self, chunksize, start, end, thread_id, nthreads, 
		    in, out, real, ph);
  } // omp parallel for 
  Py_RETURN_NONE;
}

static PyObject * Transformer_get_async_sizes(TransformerObject *self, PyObject *args)
{
  if (!PyArg_ParseTuple(args, ""))
    return NULL;

#ifdef GPAW_ASYNC
  return Py_BuildValue("(ii)", 1, GPAW_ASYNC_D);
#else
  return Py_BuildValue("(ii)", 0, GPAW_ASYNC_D);
#endif
}

static PyMethodDef Transformer_Methods[] = {
    {"apply", (PyCFunction)Transformer_apply, METH_VARARGS, NULL},
#ifdef GPAW_GPU
    {"apply_gpu", (PyCFunction)Transformer_apply_gpu,
        METH_VARARGS, NULL},
#endif
    {"get_async_sizes",
     (PyCFunction)Transformer_get_async_sizes, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

PyTypeObject TransformerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "Transformer",
    sizeof(TransformerObject),
    0,
    (destructor)Transformer_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Transformer object",
    0, 0, 0, 0, 0, 0,
    Transformer_Methods
};

PyObject * NewTransformerObject(PyObject *obj, PyObject *args)
{
  PyArrayObject* size_in;
  PyArrayObject* size_out;
  int k;
  PyArrayObject* paddings;
  PyArrayObject* npaddings;
  PyArrayObject* skip;
  PyArrayObject* neighbors;
  int real;
  PyObject* comm_obj;
  int interpolate;
  int use_gpu = 0;
  if (!PyArg_ParseTuple(args, "OOiOOOOiOi|i",
                        &size_in, &size_out, &k, &paddings, &npaddings, &skip,
                        &neighbors, &real, &comm_obj,
                        &interpolate, &use_gpu))
    return NULL;

  TransformerObject* self = PyObject_NEW(TransformerObject, &TransformerType);
  if (self == NULL)
    return NULL;

  self->k = k;
  self->interpolate = interpolate;

  MPI_Comm comm = MPI_COMM_NULL;
  if (comm_obj != Py_None)
    comm = ((MPIObject*)comm_obj)->comm;

  const long (*nb)[2] = (const long (*)[2])LONGP(neighbors);
  const long (*pad)[2] = (const long (*)[2])LONGP(paddings);
  const long (*npad)[2] = (const long (*)[2])LONGP(npaddings);
  const long (*skp)[2] = (const long (*)[2])LONGP(skip);
  self->bc = bc_init(LONGP(size_in), pad, npad, nb, comm, real, 0);

  for (int c = 0; c < 3; c++)
      self->size_out[c] = LONGP(size_out)[c];

  for (int c = 0; c < 3; c++)
    for (int d = 0; d < 2; d++)
      self->skip[c][d] = (int)skp[c][d];

#ifdef GPAW_GPU
  self->use_gpu = use_gpu;
  if (self->use_gpu) {
    transformer_init_gpu(self);
  }
#endif
  return (PyObject*)self;
}
