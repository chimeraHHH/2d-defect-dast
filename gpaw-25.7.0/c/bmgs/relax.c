/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2005,2010  CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#include "bmgs.h"

void
bmgs_relax(const int relax_method, const bmgsstencil* s,
           double* restrict a, double* restrict b, const double* src, 
           const double w)
{
  if (relax_method == 1) {
    // Coefficient needed multiple times later
    const double coef = 1.0/s->coefs[0];

    a += (s->j[0] + s->j[1] + s->j[2]) / 2;

    /* Weighted Gauss-Seidel relaxation for the equation "operator" b = src
       a contains the temporary array holding also the boundary values. */
    for (int i0 = 0; i0 < s->n[0]; i0++) {
      for (int i1 = 0; i1 < s->n[1]; i1++) {
#ifdef _OPENMP
#pragma omp simd
#endif
        for (int i2 = 0; i2 < s->n[2]; i2++) {
          int i = i2 + i1 * s->n[2] + i0 * s->n[1] * s->n[2];
          int j = i2
                + i1 * (s->n[2] + s->j[2])
                + i0 * (s->n[1] * (s->n[2] + s->j[2]) + s->j[1]);
          double x = 0.0;

          for (int c = 1; c < s->ncoefs; c++)
            x += a[j + s->offsets[c]] * s->coefs[c];
          x = (src[i] - x) * coef;
          b[i] = x;
          a[j] = x;
        }
      }
    }
  }
  else {
    a += (s->j[0] + s->j[1] + s->j[2]) / 2;

    /* Weighted Jacobi relaxation for the equation "operator" b = src
       a contains the temporariry array holding also the boundary values. */
    #pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < s->n[0]; i0++) {
      for (int i1 = 0; i1 < s->n[1]; i1++) {
#ifdef _OPENMP
#pragma omp simd
#endif
        for (int i2 = 0; i2 < s->n[2]; i2++) {
          int i = i2 + i1 * s->n[2] + i0 * s->n[1] * s->n[2];
          int j = i2
                + i1 * (s->n[2] + s->j[2])
                + i0 * (s->n[1] * (s->n[2] + s->j[2]) + s->j[1]);
          double x = 0.0;

          for (int c = 1; c < s->ncoefs; c++)
            x += a[j + s->offsets[c]] * s->coefs[c];
          b[i] = (1.0 - w) * b[i] + w * (src[i] - x) / s->coefs[0];
        }
      }
    }
  }
}
