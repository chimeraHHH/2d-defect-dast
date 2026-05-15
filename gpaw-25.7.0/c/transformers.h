#ifndef __TRANSFORMERS_H
#define __TRANSFORMERS_H

/*  Copyright (C) 2009-2012  CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#include "bc.h"

#ifdef GPAW_ASYNC
  #define GPAW_ASYNC_D 3
#else
  #define GPAW_ASYNC_D 1
#endif

#ifdef __TRANSFORMERS_C
typedef struct
{
  PyObject_HEAD
  boundary_conditions* bc;
  int p;
  int k;
  bool interpolate;
  MPI_Request recvreq[2];
  MPI_Request sendreq[2];
  int skip[3][2];
  int size_out[3];          /* Size of the output grid */
#ifdef GPAW_GPU
  int use_gpu;
#endif
} TransformerObject;
#else
// Provide an opaque type for routines outside transformers.c 
struct _TransformerObject;
typedef struct _TransformerObject TransformerObject;

#endif

#ifdef GPAW_GPU
void transformer_init_gpu(TransformerObject *self);
void transformer_dealloc_gpu(int force);
#endif

void transapply_worker(TransformerObject *self, int chunksize, int start,
		  int end, int thread_id, int nthreads,
		  const double* in, double* out,
		  bool real, const double_complex* ph);
#endif
