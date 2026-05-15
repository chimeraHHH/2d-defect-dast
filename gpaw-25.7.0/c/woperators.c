/*  This file (woperators.c) is a modified copy of operators.c
 *  with added support for nonlocal operator weights.
 *  The original copyright note of operators.c follows:
 *  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2008  CAMd
 *  Copyright (C) 2005-2020  CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include "extensions.h"
#include "bc.h"
#include "mympi.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#include "threading.h"

#ifdef GPAW_ASYNC
  #define GPAW_ASYNC3 3
  #define GPAW_ASYNC2 2
#else
  #define GPAW_ASYNC3 1
  #define GPAW_ASYNC2 1
#endif

typedef struct
{
  PyObject_HEAD
  int nweights;
  const double** weights;
  bmgsstencil* stencils;
  boundary_conditions* bc;
  MPI_Request recvreq[2];
  MPI_Request sendreq[2];
} WOperatorObject;

static void WOperator_dealloc(WOperatorObject *self)
{
  free(self->bc);
  for (int i = 0; i < self->nweights; i++)
    {
      free(self->stencils[i].coefs);
      free(self->stencils[i].offsets);
    }
  free(self->stencils);
  free(self->weights);
  PyObject_DEL(self);
}


static PyObject * WOperator_relax(WOperatorObject *self,
                                 PyObject *args)
{
  int relax_method;
  PyArrayObject* func;
  PyArrayObject* source;
  int nrelax;
  double w = 1.0;
  if (!PyArg_ParseTuple(args, "iOOi|d", &relax_method, &func, &source,
                        &nrelax, &w))
    return NULL;

  const boundary_conditions* bc = self->bc;

  double* fun = DOUBLEP(func);
  const double* src = DOUBLEP(source);
  const double_complex* ph;

  const int* size2 = bc->size2;
  double* buf = (double*) GPAW_MALLOC(double, size2[0] * size2[1] * size2[2] *
                                      bc->ndouble);
  double* sendbuf = (double*) GPAW_MALLOC(double, bc->maxsend);
  double* recvbuf = (double*) GPAW_MALLOC(double, bc->maxrecv);
  const double** weights = (const double**) GPAW_MALLOC(double*, self->nweights);

  ph = 0;

  for (int n = 0; n < nrelax; n++ )
    {
      for (int i = 0; i < 3; i++)
        {
          bc_unpack1(bc, fun, buf, i,
               self->recvreq, self->sendreq,
               recvbuf, sendbuf, ph + 2 * i, 0, 1);
          bc_unpack2(bc, buf, i,
               self->recvreq, self->sendreq, recvbuf, 1);
        }
      for (int iw = 0; iw < self->nweights; iw++)
        weights[iw] = self->weights[iw];
      bmgs_wrelax(relax_method, self->nweights, self->stencils, weights, buf, fun, src, w);
    }
  free(weights);
  free(recvbuf);
  free(sendbuf);
  free(buf);
  Py_RETURN_NONE;
}


//Plain worker
void wapply_worker(WOperatorObject *self, int chunksize, int start,
		  int end, int thread_id, int nthreads,
		  const double* in, double* out,
		  bool real, const double_complex* ph)
{
  boundary_conditions* bc = self->bc;
  const int* size1 = bc->size1;
  const int* size2 = bc->size2;
  int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
  int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];

  MPI_Request recvreq[2];
  MPI_Request sendreq[2];

  const double* my_in;
  double* my_out;

  double* sendbuf = (double*) GPAW_MALLOC(double, bc->maxsend * chunksize);
  double* recvbuf = (double*) GPAW_MALLOC(double, bc->maxrecv * chunksize);
  double* buf = (double*) GPAW_MALLOC(double, ng2 * chunksize);
  const double** weights = (const double**) GPAW_MALLOC(double*, self->nweights);

  for (int n = start; n < end; n += chunksize)
    {
      if (n + chunksize >= end && chunksize > 1)
        chunksize = end - n;
      my_in = in + n * ng;
      my_out = out + n * ng;
      for (int i = 0; i < 3; i++)
        {
          bc_unpack1(bc, my_in, buf, i,
                     recvreq, sendreq,
                     recvbuf, sendbuf, ph + 2 * i,
                     thread_id, chunksize);
          bc_unpack2(bc, buf, i, recvreq, sendreq, recvbuf, chunksize);
        }
      for (int m = 0; m < chunksize; m++)
        {
          for (int iw = 0; iw < self->nweights; iw++)
            weights[iw] = self->weights[iw] + m * ng2;
          if (real)
            bmgs_wfd(self->nweights, self->stencils, weights,
                     buf + m * ng2, my_out + m * ng);
          else
            bmgs_wfdz(self->nweights, self->stencils, weights,
                      (const double_complex*) (buf + m * ng2),
                      (double_complex*) (my_out + m * ng));
        }
    }
  free(weights);
  free(buf);
  free(recvbuf);
  free(sendbuf);
}


//Double buffering async worker
void wapply_worker_cfd(WOperatorObject *self, int chunksize, int chunkinc, 
      int start, int end, int thread_id, int nthreads,
		  const double* in, double* out,
		  bool real, const double_complex* ph)
{
  if (start >= end)
    return;
  boundary_conditions* bc = self->bc;
  const int* size1 = bc->size1;
  const int* size2 = bc->size2;
  int ng = bc->ndouble * size1[0] * size1[1] * size1[2];
  int ng2 = bc->ndouble * size2[0] * size2[1] * size2[2];
  
  MPI_Request recvreq[2 * GPAW_ASYNC3 * GPAW_ASYNC2];
  MPI_Request sendreq[2 * GPAW_ASYNC3 * GPAW_ASYNC2];

  double* sendbuf = (double*) GPAW_MALLOC(double, bc->maxsend * chunksize
                                          * GPAW_ASYNC3 * GPAW_ASYNC2);
  double* recvbuf = (double*) GPAW_MALLOC(double, bc->maxrecv * chunksize
                                * GPAW_ASYNC3 * GPAW_ASYNC2);
  double* buf = (double*) GPAW_MALLOC(double, ng2 * chunksize * GPAW_ASYNC2);
  const double** weights = (const double**) GPAW_MALLOC(double*, self->nweights);

if ((end - start) < chunksize)
    chunksize = end - start;

  int chunk = chunkinc;
  if (chunk > chunksize)
    chunk = chunksize;

  int odd = 0;
  const double* my_in = in + start * ng;
  double* my_out;
  for (int i = 0; i < 3; i++)
    bc_unpack1(bc, my_in, buf + odd * ng2 * chunksize, i,
               recvreq + odd * 2 + i * 4, sendreq + odd * 2 + i * 4,
               recvbuf + odd * bc->maxrecv * chunksize + i * bc->maxrecv * chunksize * GPAW_ASYNC2,
               sendbuf + odd * bc->maxsend * chunksize + i * bc->maxsend * chunksize * GPAW_ASYNC2, ph + 2 * i,
               thread_id, chunk);
  odd = odd ^ 1;
  int last_chunk = chunk;
  for (int n = start+chunk; n < end; n += chunk)
    {
      last_chunk += chunkinc;
      if (last_chunk > chunksize)
        last_chunk = chunksize;

      if (n + last_chunk >= end && last_chunk > 1)
        last_chunk = end - n;
      my_in = in + n * ng;
      my_out = out + (n-chunk) * ng;
      for (int i = 0; i < 3; i++)
        {
          bc_unpack1(bc, my_in, buf + odd * ng2 * chunksize, i,
                     recvreq + odd * 2 + i * 4, sendreq + odd * 2 + i * 4,
                     recvbuf + odd * bc->maxrecv * chunksize + i * bc->maxrecv * chunksize * GPAW_ASYNC2,
                     sendbuf + odd * bc->maxsend * chunksize + i * bc->maxsend * chunksize * GPAW_ASYNC2, ph + 2 * i,
                     thread_id, last_chunk);
        }
      odd = odd ^ 1;
      for (int i = 0; i < 3; i++)
        {
          bc_unpack2(bc, buf + odd * ng2 * chunksize, i,
                     recvreq + odd * 2 + i * 4, sendreq + odd * 2 + i * 4,
                     recvbuf + odd * bc->maxrecv * chunksize + i * bc->maxrecv * chunksize * GPAW_ASYNC2, chunk);
        }
      for (int m = 0; m < chunk; m++)
        {
          for (int iw = 0; iw < self->nweights; iw++)
            weights[iw] = self->weights[iw] + m * ng2 + odd * ng2 * chunksize;
          if (real)
            bmgs_wfd(self->nweights, self->stencils, weights,
                     buf + m * ng2 + odd * ng2 * chunksize,
                     my_out + m * ng);
          else
            bmgs_wfdz(self->nweights, self->stencils, weights,
                      (const double_complex*) (buf + m * ng2 + odd * ng2 * chunksize),
                      (double_complex*) (my_out + m * ng));
        }
      chunk = last_chunk;
    }

  odd = odd ^ 1;
  my_out = out + (end-last_chunk) * ng;
  for (int i = 0; i < 3; i++)
    {
      bc_unpack2(bc, buf + odd * ng2 * chunksize, i,
                 recvreq + odd * 2 + i * 4, sendreq + odd * 2 + i * 4,
                 recvbuf + odd * bc->maxrecv * chunksize + i * bc->maxrecv * chunksize * GPAW_ASYNC2, last_chunk);
    }
  for (int m = 0; m < last_chunk; m++)
    {
      for (int iw = 0; iw < self->nweights; iw++)
        weights[iw] = self->weights[iw] + m * ng2 + odd * ng2 * chunksize;
      if (real)
        bmgs_wfd(self->nweights, self->stencils, weights,
                 buf + m * ng2 + odd * ng2 * chunksize,
                 my_out + m * ng);
      else
        bmgs_wfdz(self->nweights, self->stencils, weights,
                  (const double_complex*) (buf + m * ng2 + odd * ng2 * chunksize),
                  (double_complex*) (out + m * ng));
    }

  free(weights);
  free(buf);
  free(recvbuf);
  free(sendbuf);
}

static PyObject * WOperator_apply(WOperatorObject *self,
                                 PyObject *args)
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
  const double_complex* ph;

  bool real = (PyArray_DESCR(input)->type_num == NPY_DOUBLE);

  if (real)
    ph = 0;
  else
    ph = COMPLEXP(phases);

  int chunksize = 1;
  if (getenv("GPAW_MPI_OPTIMAL_MSG_SIZE") != NULL)
    {
      int opt_msg_size = atoi(getenv("GPAW_MPI_OPTIMAL_MSG_SIZE"));
      if (bc->maxsend > 0 )
          chunksize = opt_msg_size * 1024 / (bc->maxsend / 2 * (2 - (int)real) *
                                             sizeof(double));
      chunksize = (chunksize > 0) ? chunksize : 1;
      chunksize = (chunksize < nin) ? chunksize : nin;
    }

  int chunkinc = chunksize;
  if (getenv("GPAW_CHUNK_INC") != NULL)
    chunkinc = atoi(getenv("GPAW_CHUNK_INC"));

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

#ifndef GPAW_ASYNC
  if (1)
#else
  if (bc->cfd == 0)
#endif
    {
      wapply_worker(self, chunksize, start, end, thread_id, nthreads,
	       in, out, real, ph);
    }
  else
    {
      wapply_worker_cfd(self, chunksize, chunkinc, start, end, thread_id, nthreads,
	      in, out, real, ph);
    }
}
  Py_RETURN_NONE;
}


static PyObject * WOperator_get_diagonal_element(WOperatorObject *self,
                                              PyObject *args)
{
  if (!PyArg_ParseTuple(args, ""))
    return NULL;

  const double** weights = (const double**) GPAW_MALLOC(double*, self->nweights);
  for (int iw = 0; iw < self->nweights; iw++)
    weights[iw] = self->weights[iw];
  const int n0 = self->stencils[0].n[0];
  const int n1 = self->stencils[0].n[1];
  const int n2 = self->stencils[0].n[2];

  double d = 0.0;
  for (int i0 = 0; i0 < n0; i0++)
    {
      for (int i1 = 0; i1 < n1; i1++)
        {
          for (int i2 = 0; i2 < n2; i2++)
            {
              double coef = 0.0;
              for (int iw = 0; iw < self->nweights; iw++)
                {
                  coef += weights[iw][0] * self->stencils[iw].coefs[0];
                  weights[iw]++;
                }
              if (coef < 0)
                coef = -coef;
              if (coef > d)
                d = coef;
            }
        }
    }

  free(weights);

  return Py_BuildValue("d", d);
}

static PyObject * WOperator_get_async_sizes(WOperatorObject *self, PyObject *args)
{
  if (!PyArg_ParseTuple(args, ""))
    return NULL;

#ifdef GPAW_ASYNC
  return Py_BuildValue("(iii)", 1, GPAW_ASYNC2, GPAW_ASYNC3);
#else
  return Py_BuildValue("(iii)", 0, GPAW_ASYNC2, GPAW_ASYNC3);
#endif
}

static PyMethodDef WOperator_Methods[] = {
    {"apply",
     (PyCFunction)WOperator_apply, METH_VARARGS, NULL},
    {"relax",
     (PyCFunction)WOperator_relax, METH_VARARGS, NULL},
    {"get_diagonal_element",
     (PyCFunction)WOperator_get_diagonal_element, METH_VARARGS, NULL},
    {"get_async_sizes",
     (PyCFunction)WOperator_get_async_sizes, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}

};


PyTypeObject WOperatorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "WOperator",
    sizeof(WOperatorObject),
    0,
    (destructor)WOperator_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "FDW-operator object",
    0, 0, 0, 0, 0, 0,
    WOperator_Methods
};

PyObject* NewWOperatorObject(PyObject *obj, PyObject *args)
{
  PyObject* coefs_list;
  PyArrayObject* coefs;
  PyObject* offsets_list;
  PyArrayObject* offsets;
  PyObject* weights_list;
  PyArrayObject* weights;
  PyArrayObject* size;
  PyArrayObject* neighbors;
  int real;
  PyObject* comm_obj;
  int cfd;
  int range;
  int nweights;

  if (!PyArg_ParseTuple(args, "iO!O!O!OiOiOi",
                        &nweights,
                        &PyList_Type, &weights_list,
                        &PyList_Type, &coefs_list,
                        &PyList_Type, &offsets_list,
                        &size,
                        &range,
                        &neighbors, &real, &comm_obj, &cfd))
    return NULL;

  WOperatorObject *self = PyObject_NEW(WOperatorObject, &WOperatorType);
  if (self == NULL)
    return NULL;

  self->stencils = (bmgsstencil*) GPAW_MALLOC(bmgsstencil, nweights);
  self->weights = (const double**) GPAW_MALLOC(double*, nweights);
  self->nweights = nweights;

  for (int iw = 0; iw < nweights; iw++)
    {
      coefs = (PyArrayObject*) PyList_GetItem(coefs_list, iw);
      offsets = (PyArrayObject*) PyList_GetItem(offsets_list, iw);
      weights = (PyArrayObject*) PyList_GetItem(weights_list, iw);
      self->stencils[iw] = bmgs_stencil(PyArray_DIMS(coefs)[0], DOUBLEP(coefs),
                                        LONGP(offsets), range, LONGP(size));
      self->weights[iw] = DOUBLEP(weights);
    }

  const long (*nb)[2] = (const long (*)[2])LONGP(neighbors);
  const long padding[3][2] = {{range, range},
                             {range, range},
                             {range, range}};

  MPI_Comm comm = MPI_COMM_NULL;
  if (comm_obj != Py_None)
    comm = ((MPIObject*)comm_obj)->comm;

  self->bc = bc_init(LONGP(size), padding, padding, nb, comm, real, cfd);

  return (PyObject*) self;
}
