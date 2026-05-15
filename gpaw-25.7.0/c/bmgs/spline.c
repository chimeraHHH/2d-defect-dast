/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2008  CAMd
 *  Please see the accompanying LICENSE file for further information. */

#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "bmgs.h"


bmgsspline bmgs_spline(int l, double dr, int nbins, double* f)
{
  double c = 3.0 / (dr * dr);
  double* f2 = (double*)malloc((nbins + 1) * sizeof(double));
  assert(f2 != NULL);
  double* u = (double*)malloc(nbins * sizeof(double));
  assert(u != NULL);
  f2[0] = -0.5;
  u[0] = (f[1] - f[0]) * c;
  for (int b = 1; b < nbins; b++)
    {
      double p = 0.5 * f2[b - 1] + 2.0;
      f2[b] = -0.5 / p;
      u[b] = ((f[b + 1] - 2.0 * f[b] + f[b - 1]) * c - 0.5 * u[b - 1]) / p;
    }
  f2[nbins] = ((f[nbins - 1] * c - 0.5 * u[nbins - 1]) /
               (0.5 * f2[nbins - 1] + 1.0));
  for (int b = nbins - 1; b >= 0; b--)
    f2[b] = f2[b] * f2[b + 1] + u[b];
  double* data = (double*)malloc(4 * (nbins + 1) * sizeof(double));
  assert(data != NULL);
  bmgsspline spline = {l, dr, nbins, data};
  for (int b = 0; b < nbins; b++)
    {
      *data++ = f[b];
      *data++ = (f[b + 1] - f[b]) / dr - (f2[b] / 3 + f2[b + 1] / 6) * dr;
      *data++ = 0.5 * f2[b];
      *data++ = (f2[b + 1] - f2[b]) / (6 * dr);
    }
  data[0] = 0.0;
  data[1] = 0.0;
  data[2] = 0.0;
  data[3] = 0.0;
  free(u);
  free(f2);
  return spline;
}


double bmgs_splinevalue(const bmgsspline* spline, double r)
{
  int b = r / spline->dr;
  if (b >= spline->nbins)
    return 0.0;
  double u = r - b * spline->dr;
  double* s = spline->data + 4 * b;
  return  s[0] + u * (s[1] + u * (s[2] + u * s[3]));
}


void bmgs_get_value_and_derivative(const bmgsspline* spline, double r,
                                   double *f, double *dfdr)
{
  int b = r / spline->dr;
  if (b >= spline->nbins)
    {
      *f = 0.0;
      *dfdr = 0.0;
      return;
    }
  double u = r - b * spline->dr;
  double* s = spline->data + 4 * b;
  *f = s[0] + u * (s[1] + u * (s[2] + u * s[3]));
  *dfdr = s[1] + u * (2.0 * s[2] + u * 3.0 * s[3]);
}


void bmgs_deletespline(bmgsspline* spline)
{
  free(spline->data);
}
