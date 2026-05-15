/*  Copyright (C) 2010-2011 CAMd
 *  Please see the accompanying LICENSE file for further information. */
#include "extensions.h"



PyObject* GG_shuffle(PyObject *self, PyObject *args)
{
    PyArrayObject* G_G_obj;
    int sign;
    PyArrayObject* A_GG_obj;
    PyArrayObject* B_GG_obj;

    // def GG_shuffle(G_G:int32 array, sign:int, A_GG:complex128 array, B_GG:complex128 array)
    if (!PyArg_ParseTuple(args, "OiOO",
                          &G_G_obj, &sign, &A_GG_obj, &B_GG_obj))
        return NULL;


    int nG = PyArray_DIMS(G_G_obj)[0];
    // Check dimensions
    if ((nG != PyArray_DIMS(B_GG_obj)[0]) ||
        (nG != PyArray_DIMS(B_GG_obj)[1]) ||
        (nG != PyArray_DIMS(A_GG_obj)[0]) ||
        (nG != PyArray_DIMS(A_GG_obj)[1]))
     {
         PyErr_SetString(PyExc_TypeError, "Unmatched dimensions at GG_shuffle.");
         return NULL;
     }

    // Check input types
    if ((PyArray_TYPE(B_GG_obj) != NPY_COMPLEX128) ||
        (PyArray_TYPE(A_GG_obj) != NPY_COMPLEX128))
    {
         PyErr_SetString(PyExc_TypeError, "Expected complex arrays.");
         return NULL;
    }

    if (PyArray_TYPE(G_G_obj) != NPY_INT)
    {
         PyErr_SetString(PyExc_TypeError, "G_G expected to be an integer array.");
         return NULL;
    }

    if (!PyArray_IS_C_CONTIGUOUS(B_GG_obj))
    {
        PyErr_SetString(PyExc_TypeError, "B_GG need to be c-contiguous.");
        return NULL;
    }

    if (!((sign == 1) || (sign == -1)))
    {
        PyErr_SetString(PyExc_TypeError, "Sign must be 1 or -1.");
        return NULL;
    }

    int* G0_G = (int*)malloc(nG * sizeof(int));
    int* G1_G = (int*)malloc(nG * sizeof(int));

    npy_int32* G_G = (npy_int32*)PyArray_DATA(G_G_obj);

    int stride0 = PyArray_STRIDES(A_GG_obj)[0];
    int stride1 = PyArray_STRIDES(A_GG_obj)[1];
    for (int G=0; G < nG; G++)
    {
        if (sign==1)
        {
            G0_G[G] = G_G[G] * stride0;
            G1_G[G] = G_G[G] * stride1;
        }
        else  // Transpose
        {
            G0_G[G] = G_G[G] * stride1;
            G1_G[G] = G_G[G] * stride0;
        }
    }

    double complex* A_GG = (double complex*)PyArray_DATA(A_GG_obj);
    double complex* B_GG = (double complex*)PyArray_DATA(B_GG_obj);

    for (int G0=0; G0<nG; G0++)
    {
        int take0 = G0_G[G0];
        for (int G1=0; G1<nG; G1++)
        {
            int take1 = G1_G[G1];
            // Instead of numpy magic, we do some C magic.
            char* ptr = (char*)A_GG + take0 + take1;
            double complex* value_ptr = (double_complex*) ptr;
            *(B_GG++) += *value_ptr;
        }
    }

    free(G0_G);
    free(G1_G);
    Py_RETURN_NONE;
}

//
// Apply symmetry operation op_cc to a and add result to b:
//
//     =T_       _
//   b(U g) += a(g),
//
// where:
//
//   =                         _T
//   U     = op_cc[c1, c2] and g = (g0, g1, g2).
//    c1,c2
//
PyObject* symmetrize(PyObject *self, PyObject *args)
{
    PyArrayObject* a_g_obj;
    PyArrayObject* b_g_obj;
    PyArrayObject* op_cc_obj;
    PyArrayObject* offset_c_obj;

    if (!PyArg_ParseTuple(args, "OOOO",
                          &a_g_obj, &b_g_obj, &op_cc_obj, &offset_c_obj))
        return NULL;

    const long* C = (const long*)PyArray_DATA(op_cc_obj);
    const long* o_c = (const long*)PyArray_DATA(offset_c_obj);
    int ng0 = PyArray_DIMS(a_g_obj)[0];
    int ng1 = PyArray_DIMS(a_g_obj)[1];
    int ng2 = PyArray_DIMS(a_g_obj)[2];
    int Ng0 = ng0 + o_c[0];
    int Ng1 = ng1 + o_c[1];
    int Ng2 = ng2 + o_c[2];

    const double* a_g = (const double*)PyArray_DATA(a_g_obj);
    double* b_g = (double*)PyArray_DATA(b_g_obj);
    for (int g0 = o_c[0]; g0 < Ng0; g0++)
        for (int g1 = o_c[1]; g1 < Ng1; g1++)
            for (int g2 = o_c[2]; g2 < Ng2; g2++) {
                int p0 = ((C[0] * g0 + C[3] * g1 + C[6] * g2) %
                          Ng0 + Ng0) % Ng0;
                int p1 = ((C[1] * g0 + C[4] * g1 + C[7] * g2) %
                          Ng1 + Ng1) % Ng1;
                int p2 = ((C[2] * g0 + C[5] * g1 + C[8] * g2) %
                          Ng2 + Ng2) % Ng2;
                b_g[((p0 - o_c[0]) * ng1 +
                     (p1 - o_c[1])) * ng2 +
                    p2 - o_c[2]] += *a_g++;
            }

    Py_RETURN_NONE;
}

PyObject* symmetrize_ft(PyObject *self, PyObject *args)
{
    PyArrayObject* a_g_obj;
    PyArrayObject* b_g_obj;
    PyArrayObject* op_cc_obj;
    PyArrayObject* t_c_obj;
    PyArrayObject* offset_c_obj;

    if (!PyArg_ParseTuple(args, "OOOOO",
                          &a_g_obj, &b_g_obj, &op_cc_obj, &t_c_obj,
                          &offset_c_obj))
        return NULL;

    const long* t_c = (const long*)PyArray_DATA(t_c_obj);
    const long* C = (const long*)PyArray_DATA(op_cc_obj);
    const long* o_c = (const long*)PyArray_DATA(offset_c_obj);

    int ng0 = PyArray_DIMS(a_g_obj)[0];
    int ng1 = PyArray_DIMS(a_g_obj)[1];
    int ng2 = PyArray_DIMS(a_g_obj)[2];
    int Ng0 = ng0 + o_c[0];
    int Ng1 = ng1 + o_c[1];
    int Ng2 = ng2 + o_c[2];

    const double* a_g = (const double*)PyArray_DATA(a_g_obj);
    double* b_g = (double*)PyArray_DATA(b_g_obj);
    for (int g0 = o_c[0]; g0 < Ng0; g0++)
        for (int g1 = o_c[1]; g1 < Ng1; g1++)
            for (int g2 = o_c[2]; g2 < Ng2; g2++) {
                int p0 = ((C[0] * g0 + C[3] * g1 + C[6] * g2 - t_c[0]) %
                          Ng0 + Ng0) % Ng0;
                int p1 = ((C[1] * g0 + C[4] * g1 + C[7] * g2 - t_c[1]) %
                          Ng1 + Ng1) % Ng1;
                int p2 = ((C[2] * g0 + C[5] * g1 + C[8] * g2 - t_c[2]) %
                          Ng2 + Ng2) % Ng2;
                b_g[((p0 - o_c[0]) * ng1 +
                     (p1 - o_c[1])) * ng2 +
                    p2 - o_c[2]] += *a_g++;
            }

    Py_RETURN_NONE;
}

PyObject* symmetrize_wavefunction(PyObject *self, PyObject *args)
{
    PyArrayObject* a_g_obj;
    PyArrayObject* b_g_obj;
    PyArrayObject* op_cc_obj;
    PyArrayObject* kpt0_obj;
    PyArrayObject* kpt1_obj;

    if (!PyArg_ParseTuple(args, "OOOOO", &a_g_obj, &b_g_obj, &op_cc_obj, &kpt0_obj, &kpt1_obj))
        return NULL;

    const long* C = (const long*)PyArray_DATA(op_cc_obj);
    const double* kpt0 = (const double*) PyArray_DATA(kpt0_obj);
    const double* kpt1 = (const double*) PyArray_DATA(kpt1_obj);
    int ng0 = PyArray_DIMS(a_g_obj)[0];
    int ng1 = PyArray_DIMS(a_g_obj)[1];
    int ng2 = PyArray_DIMS(a_g_obj)[2];

    const double complex* a_g = (const double complex*)PyArray_DATA(a_g_obj);
    double complex* b_g = (double complex*)PyArray_DATA(b_g_obj);

    for (int g0 = 0; g0 < ng0; g0++)
        for (int g1 = 0; g1 < ng1; g1++)
            for (int g2 = 0; g2 < ng2; g2++) {
              int p0 = ((C[0] * g0 + C[3] * g1 + C[6] * g2) % ng0 + ng0) % ng0;
              int p1 = ((C[1] * g0 + C[4] * g1 + C[7] * g2) % ng1 + ng1) % ng1;
              int p2 = ((C[2] * g0 + C[5] * g1 + C[8] * g2) % ng2 + ng2) % ng2;

              double complex phase = cexp(I * 2. * M_PI *
                                          (kpt1[0]/ng0*p0 +
                                           kpt1[1]/ng1*p1 +
                                           kpt1[2]/ng2*p2 -
                                           kpt0[0]/ng0*g0 -
                                           kpt0[1]/ng1*g1 -
                                           kpt0[2]/ng2*g2));
              b_g[(p0 * ng1 + p1) * ng2 + p2] += (*a_g * phase);
              a_g++;
            }

    Py_RETURN_NONE;
}

PyObject* symmetrize_return_index(PyObject *self, PyObject *args)
{
    PyArrayObject* a_g_obj;
    PyArrayObject* b_g_obj;
    PyArrayObject* op_cc_obj;
    PyArrayObject* kpt0_obj;
    PyArrayObject* kpt1_obj;

    if (!PyArg_ParseTuple(args, "OOOOO", &a_g_obj, &b_g_obj, &op_cc_obj, &kpt0_obj, &kpt1_obj))
        return NULL;

    const long* C = (const long*)PyArray_DATA(op_cc_obj);
    const double* kpt0 = (const double*) PyArray_DATA(kpt0_obj);
    const double* kpt1 = (const double*) PyArray_DATA(kpt1_obj);

    int ng0 = PyArray_DIMS(a_g_obj)[0];
    int ng1 = PyArray_DIMS(a_g_obj)[1];
    int ng2 = PyArray_DIMS(a_g_obj)[2];

    unsigned long* a_g = (unsigned long*)PyArray_DATA(a_g_obj);
    double complex* b_g = (double complex*)PyArray_DATA(b_g_obj);

    for (int g0 = 0; g0 < ng0; g0++)
        for (int g1 = 0; g1 < ng1; g1++)
            for (int g2 = 0; g2 < ng2; g2++) {
              int p0 = ((C[0] * g0 + C[3] * g1 + C[6] * g2) % ng0 + ng0) % ng0;
              int p1 = ((C[1] * g0 + C[4] * g1 + C[7] * g2) % ng1 + ng1) % ng1;
              int p2 = ((C[2] * g0 + C[5] * g1 + C[8] * g2) % ng2 + ng2) % ng2;

              double complex phase = cexp(I * 2. * M_PI *
                                          (kpt1[0]/ng0*p0 +
                                           kpt1[1]/ng1*p1 +
                                           kpt1[2]/ng2*p2 -
                                           kpt0[0]/ng0*g0 -
                                           kpt0[1]/ng1*g1 -
                                           kpt0[2]/ng2*g2));
              *a_g++ = (p0 * ng1 + p1) * ng2 + p2;
              *b_g++ = phase;
            }

    Py_RETURN_NONE;
}

PyObject* symmetrize_with_index(PyObject *self, PyObject *args)
{
    PyArrayObject* a_g_obj;
    PyArrayObject* b_g_obj;
    PyArrayObject* index_g_obj;
    PyArrayObject* phase_g_obj;

    if (!PyArg_ParseTuple(args, "OOOO", &a_g_obj, &b_g_obj, &index_g_obj, &phase_g_obj))
        return NULL;

    int ng0 = PyArray_DIMS(a_g_obj)[0];
    int ng1 = PyArray_DIMS(a_g_obj)[1];
    int ng2 = PyArray_DIMS(a_g_obj)[2];

    const unsigned long* index_g = (const unsigned long*)PyArray_DATA(index_g_obj);
    const double complex* phase_g = (const double complex*)PyArray_DATA(phase_g_obj);
    const double complex* a_g = (const double complex*)PyArray_DATA(a_g_obj);
    double complex* b_g = (double complex*)PyArray_DATA(b_g_obj);


    for (int g0 = 0; g0 < ng0; g0++)
        for (int g1 = 0; g1 < ng1; g1++)
            for (int g2 = 0; g2 < ng2; g2++) {
              b_g[*index_g] += (*a_g * *phase_g);
              a_g++;
              phase_g++;
              index_g++;
            }

    Py_RETURN_NONE;
}

PyObject* map_k_points(PyObject *self, PyObject *args)
{
    PyArrayObject* bzk_kc_obj;
    PyArrayObject* U_scc_obj;
    double tol;
    PyArrayObject* bz2bz_ks_obj;
    int ka, kb;

    if (!PyArg_ParseTuple(args, "OOdOii", &bzk_kc_obj, &U_scc_obj,
                           &tol, &bz2bz_ks_obj, &ka, &kb))
        return NULL;

    const long* U_scc = (const long*)PyArray_DATA(U_scc_obj);
    const double* bzk_kc = (const double*)PyArray_DATA(bzk_kc_obj);
    long* bz2bz_ks = (long*)PyArray_DATA(bz2bz_ks_obj);

    int nbzkpts = PyArray_DIMS(bzk_kc_obj)[0];
    int nsym = PyArray_DIMS(U_scc_obj)[0];

    for (int k1 = ka; k1 < kb; k1++) {
        const double* q = bzk_kc + k1 * 3;
         for (int s = 0; s < nsym; s++) {
             const long* U = U_scc + s * 9;
             double q0 = U[0] * q[0] + U[1] * q[1] + U[2] * q[2];
             double q1 = U[3] * q[0] + U[4] * q[1] + U[5] * q[2];
             double q2 = U[6] * q[0] + U[7] * q[1] + U[8] * q[2];
             for (int k2 = 0; k2 < nbzkpts; k2++) {
                 double p0 = q0 - bzk_kc[k2 * 3];
                 if (fabs(p0 - round(p0)) > tol)
                     continue;
                 double p1 = q1 - bzk_kc[k2 * 3 + 1];
                 if (fabs(p1 - round(p1)) > tol)
                     continue;
                 double p2 = q2 - bzk_kc[k2 * 3 + 2];
                 if (fabs(p2 - round(p2)) > tol)
                     continue;
                 bz2bz_ks[k1 * nsym + s] = k2;
                 break;
             }
         }
    }
    Py_RETURN_NONE;
}
