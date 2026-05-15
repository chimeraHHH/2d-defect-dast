/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2009  CAMd
 *  Please see the accompanying LICENSE file for further information. */

#include "extensions.h"
#include "bmgs/bmgs.h"
#include "spline.h"
#include <complex.h>


//                    +-----------n
//  +----m   +----m   | +----c+m  |
//  |    |   |    |   | |    |    |
//  |  b | = |  v | * | |  a |    |
//  |    |   |    |   | |    |    |
//  0----+   0----+   | c----+    |
//                    |           |
//                    0-----------+
void cut(const double* a, const int n[3], const int c[3],
         const double* v,
         double* b, const int m[3])
{
  a += c[2] + (c[1] + c[0] * n[1]) * n[2];
  for (int i0 = 0; i0 < m[0]; i0++)
    {
      for (int i1 = 0; i1 < m[1]; i1++)
        {
          for (int i2 = 0; i2 < m[2]; i2++)
            b[i2] = v[i2] * a[i2];
          a += n[2];
          b += m[2];
          v += m[2];
        }
      a += n[2] * (n[1] - m[1]);
    }
}


PyObject *tci_overlap(PyObject *self, PyObject *args)
{
    /*
    Calculate two-center integral overlaps:

             --       --          l      _
      X   =  >  s (r) >  G       r  Y   (r)
       LL'   --  l    --  LL'L''     L''
              l       L''

    or derivatives

    / dX \       ^ --        --        l     _
    | -- |    =  R >   s'(r) > G      r Y   (r)
    \ dR /LL'      --   l    -- LL'L''   L''
                    l        L''
                                           l  _
                   --       --        / d r Y(r) \
                 + >  s (r) > G       | -----    |    ,
                   --  l    -- LL'L'' \   dR     /L''
                    l       L''
                                                        ^
    where dR denotes movement of one of the centers and R is a unit vector
    parallel to the displacement vector r.

    Without derivatives, Rhat_c_obj, drLYdR_Lc_obj, and dxdR_cmi_obj must still
    be numpy arrays but are otherwise ignored (may have size 0).

    With derivatives, x_mi_obj can be likewise ignored.

    */

    int la, lb;
    PyArrayObject *G_LLL_obj;
    PyObject *spline_l;
    double r;

    PyArrayObject *rlY_L_obj, *x_mi_obj;
    int is_derivative;
    PyArrayObject *Rhat_c_obj, *drlYdR_Lc_obj, *dxdR_cmi_obj;

    if (!PyArg_ParseTuple(args, "iiOOdOOiOOO", &la, &lb, &G_LLL_obj, &spline_l,
                          &r, &rlY_L_obj, &x_mi_obj,
                          &is_derivative,
                          &Rhat_c_obj, &drlYdR_Lc_obj,
                          &dxdR_cmi_obj))
        return NULL;


    SplineObject *spline_obj;
    bmgsspline *spline;

    double *x_mi = (double *) PyArray_DATA(x_mi_obj);
    double *G_LLL = (double *) PyArray_DATA(G_LLL_obj);
    double *rlY_L = (double *) PyArray_DATA(rlY_L_obj);

    double *Rhat_c = (double *) PyArray_DATA(Rhat_c_obj);
    double *drlYdR_Lc = (double *) PyArray_DATA(drlYdR_Lc_obj);
    double *dxdR_cmi = (double *) PyArray_DATA(dxdR_cmi_obj);

    int Lastart = la * la;
    int Lbstart = lb * lb;

    int l = (la + lb) % 2;
    int nsplines = PyList_Size(spline_l);
    int ispline;

    int itemsize = PyArray_ITEMSIZE(G_LLL_obj);
    npy_intp *strides = PyArray_STRIDES(G_LLL_obj);
    npy_intp *xstrides = PyArray_STRIDES(x_mi_obj);
    int stride0 = strides[0] / itemsize;
    int stride1 = strides[1] / itemsize;
    int xstride = xstrides[0] / itemsize;

    G_LLL += Lastart * stride0 + Lbstart * stride1;

    for(ispline=0; ispline < nsplines; ispline++, l+=2) {
        int Lstart = l * l;
        spline_obj = (SplineObject*)PyList_GET_ITEM(spline_l, ispline);
        spline = &spline_obj->spline;
        double s, dsdr;
        if(is_derivative) {
            bmgs_get_value_and_derivative(spline, r, &s, &dsdr);
        } else {
            s = bmgs_splinevalue(spline, r);
        }

        if(fabs(s) < 1e-10) {
            continue;
        }

        int nm1 = 2 * la + 1;
        int nm2 = 2 * lb + 1;

        int m1, m2, L;
        int nL = 2 * l + 1;
        double srlY_L[2 * l + 1];  // Variable but very small alloc on stack
        for(L=0; L < nL; L++) {
            srlY_L[L] = s * rlY_L[Lstart + L];
        }

        if(!is_derivative) {
            for(m1=0; m1 < nm1; m1++) {
                for(m2=0; m2 < nm2; m2++) {
                    double x = 0.0;
                    for(L=0; L < nL; L++) {
                        x += G_LLL[stride0 * m1 + stride1 * m2 + Lstart + L] * srlY_L[L];
                    }
                    x_mi[m1 * xstride + m2] += x;
                }
            }
            continue;
        }

        // Derivative only
        int c;

        npy_intp *dxdRstrides = PyArray_STRIDES(dxdR_cmi_obj);
        int dxdRstride0 = dxdRstrides[0] / itemsize;
        int dxdRstride1 = dxdRstrides[1] / itemsize;

        double dsdr_Rhat_c[3];
        for(c=0; c < 3; c++) {
            dsdr_Rhat_c[c] = dsdr * Rhat_c[c];
        }

        double s_drlYdR_Lc[nL * 3];
        for(L=0; L < nL; L++) {
            for(c=0; c < 3; c++) {
                s_drlYdR_Lc[L * 3 + c] = s * drlYdR_Lc[(Lstart + L) * 3 + c];
            }
        }

        // This loop can probably be written a lot better, but it turns out
        // it is so fast that we need not worry for a long time.
        for(m1=0; m1 < nm1; m1++) {
            for(m2=0; m2 < nm2; m2++) {
                double GrlY_mi = 0.0;
                for(L=0; L < nL; L++) {
                    GrlY_mi += G_LLL[stride0 * m1 + stride1 * m2 + Lstart + L] * rlY_L[Lstart + L];
                }
                for(c=0; c < 3; c++) {
                    double derivative = 0.0;
                    derivative += dsdr_Rhat_c[c] * GrlY_mi;
                    for(L=0; L < nL; L++) {
                        derivative += G_LLL[stride0 * m1 + stride1 * m2 + Lstart + L] * s_drlYdR_Lc[L * 3 + c];
                    }
                    dxdR_cmi[dxdRstride0 * c + dxdRstride1 * m1 + m2] += derivative;
                }
            }
        }
    }

    Py_RETURN_NONE;
}
