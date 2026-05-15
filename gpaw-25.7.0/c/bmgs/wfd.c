/*  This file (wfd.c) is a modified copy of fd.c
 *  with added support for nonlocal operator weights.
 *  The original copyright note of fd.c follows:
 *  Copyright (C) 2003-2007  CAMP
 *  Please see the accompanying LICENSE file for further information. */

#include "../extensions.h"
#include "bmgs.h"

void Z(bmgs_wfd)(int nweights, const bmgsstencil* stencils, const double** weights, const T* a, T* b)
{
  a += (stencils[0].j[0] + stencils[0].j[1] + stencils[0].j[2]) / 2;

  const int n0 = stencils[0].n[0];
  const int n1 = stencils[0].n[1];
  const int n2 = stencils[0].n[2];
  const int j1 = stencils[0].j[1];
  const int j2 = stencils[0].j[2];

  for (int i0 = 0; i0 < n0; i0++)
    {
      const T* aa = a + i0 * (j1 + n1 * (j2 + n2));
      T* bb = b + i0 * n1 * n2;
      for (int i1 = 0; i1 < n1; i1++)
        {
          for (int i2 = 0; i2 < n2; i2++)
            {
              T x = 0.0;
              for (int iw = 0; iw < nweights; iw++)
                {
                  const bmgsstencil* s = &(stencils[iw]);
                  T tmp = 0.0;
                  for (int c = 0; c < s->ncoefs; c++)
                    tmp += aa[s->offsets[c]] * s->coefs[c];
                  tmp *= weights[iw][0];
                  x += tmp;
                  weights[iw]++;
                }
              *bb++ = x;
              aa++;
            }
          aa += j2;
        }
    }
}
