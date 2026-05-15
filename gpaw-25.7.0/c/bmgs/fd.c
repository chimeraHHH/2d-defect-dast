/*  Copyright (C) 2003-2007  CAMP
    Copyright (C) 2010  CSC - IT Center for Science Ltd.

 *  Please see the accompanying LICENSE file for further information. */

#include "../extensions.h"
#include "bmgs.h"

void
Z(bmgs_fd)(const bmgsstencil* s, const T* a, T* b)
{
    /* Skip the leading halo area. */
    a += (s->j[0] + s->j[1] + s->j[2]) / 2;

    for (int i0 = 0; i0 < s->n[0]; i0++) {
        for (int i1 = 0; i1 < s->n[1]; i1++) {
#ifdef _OPENMP
#pragma omp simd
#endif                
            for (int i2 = 0; i2 < s->n[2]; i2++) {
                int i = i2
                      + i1 * (s->j[2] + s->n[2])
                      + i0 * (s->j[1] + s->n[1] * (s->j[2] + s->n[2]));
                int j = i2 + i1 * s->n[2] + i0 * s->n[1] * s->n[2];
                T x = 0.0;

                for (int c = 0; c < s->ncoefs; c++)
                    x += a[i + s->offsets[c]] * s->coefs[c];
                b[j] = x;
            }
        }
    }
}
