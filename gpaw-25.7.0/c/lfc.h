/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2009  CAMd
 *  Copyright (C) 2010,2020  CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#ifndef LFC_H
#define LFC_H

#include <Python.h>

typedef struct
{
  const double* A_gm;  // function values
  int nm;              // number of functions (2*l+1)
  int M;               // global number of first function
  int W;               // volume number
} LFVolume;

#ifdef GPAW_GPU
#include "gpu/gpu-align.h"
#include "gpu/gpu-complex.h"

typedef struct ALIGN(16)
{
  double *A_gm;
  int len_A_gm;
  int nm;              // number of functions (2*l+1)
  int M;               // global number of first function
  int W;               // volume number

  int nB;
  int *GB1;
  int *nGBcum;
  gpuDoubleComplex *phase_k;
} LFVolume_gpu;
#endif

typedef struct 
{
  PyObject_HEAD
  double dv;                 // volume per grid point
  int nW;                    // number of volumes
  int nB;                    // number of boundary points
  int nimax;                 // maximum number of current volumes
  double* work_gm;           // work space
  LFVolume* volume_W;        // pointers to volumes
  LFVolume** volume_i;       // pointers to volumes at current grid point
  int* G_B;                  // boundary grid points
  int* W_B;                  // volume numbers
  int* i_W;                  // mapping from all volumes to current volumes
  int* ngm_W;                // number of grid points per volume
  bool bloch_boundary_conditions;  // Gamma-point calculation?
  complex double* phase_kW;  // phase factors: exp(ik.R)
  complex double* phase_i;   // phase factors for current volumes

#ifdef GPAW_GPU
  int use_gpu;
  LFVolume_gpu *volume_W_gpu;
  LFVolume_gpu *volume_W_gpu_host;
  int nB_gpu;                    // number of boundary points
  int* G_B1_gpu;                  // boundary grid points
  int* G_B2_gpu;                  // boundary grid points
  int max_len_A_gm;
  int max_nG;
  gpuDoubleComplex *phase_i_gpu;
  int max_k;
  LFVolume_gpu **volume_i_gpu;
  int *A_gm_i_gpu;
  int *ni_gpu;

  int Mcount;
  int *volume_WMi_gpu;
  int *WMi_gpu;
  int WMimax;
#endif
} LFCObject;


#define GRID_LOOP_START(lfc, k, thread_id)                         \
{                                                                  \
  const int* G_B = lfc->G_B;                                       \
  const int* W_B = lfc->W_B;                                       \
  int* i_W = lfc->i_W + thread_id * lfc->nW;                       \
  complex double* phase_i = lfc->phase_i + thread_id * lfc->nimax; \
  LFVolume **volume_i = lfc->volume_i + thread_id * lfc->nimax;    \
  LFVolume *volume_W = lfc->volume_W + thread_id * lfc->nW;        \
  const double complex* phase_W = lfc->phase_kW + k * lfc->nW;     \
  int Ga = 0;                                                      \
  int ni = 0;                                                      \
  for (int B = 0; B < lfc->nB; B++)                                \
    {                                                              \
      int Gb = G_B[B];                                             \
      int nG = Gb - Ga;                                            \
      if (nG > 0)                                                  \
        {

#define GRID_LOOP_STOP(lfc, k, thread_id)                          \
          for (int i = 0; i < ni; i++)                             \
            volume_i[i]->A_gm += nG * volume_i[i]->nm;             \
        }                                                          \
      int Wnew = W_B[B];                                           \
      if (Wnew >= 0)                                               \
        {                                                          \
          /* Entering new sphere. Add the new volume to
             the head of the list of current volumes. */           \
          volume_i[ni] = &volume_W[Wnew];                          \
          if (k >= 0)                                              \
            phase_i[ni] = phase_W[Wnew];                           \
          i_W[Wnew] = ni;                                          \
          ni++;                                                    \
        }                                                          \
      else                                                         \
        {                                                          \
          /* Leaving sphere. Remove the volume from the list
             of current volumes. */                                \
          int Wold = -1 - Wnew;                                    \
          int iold = i_W[Wold];                                    \
          ni--;                                                    \
          volume_i[iold] = volume_i[ni];                           \
          if (k >= 0)                                              \
            phase_i[iold] = phase_i[ni];                           \
          int Wlast = volume_i[iold]->W;                           \
          i_W[Wlast] = iold;                                       \
        }                                                          \
      Ga = Gb;                                                     \
    }                                                              \
  /* Restore function value pointers to the initial state. */      \
  for (int W = 0; W < lfc->nW; W++)                                \
    volume_W[W].A_gm -= lfc->ngm_W[W];                             \
}

#endif
