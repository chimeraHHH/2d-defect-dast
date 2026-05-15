#include "extensions.h"

// Evaluate the multipole expansion
// x_GG: output, the multipole exansion evaluated at omega
// dx_GG: output, the derivative of the multipole expansion evaluated at omega
// omega: frequency to evaluate the expansion at
// f: occupation number
// omegat_nGG: The positions of poles for each GG
// W_nGG: The residue of poles for each GG
// eta: extra broadening
// factor: total prefactor to multiply the result
PyObject* evaluate_mpa_poly(PyObject *self, PyObject *args)
{
    PyArrayObject* x_GG_obj;
    PyArrayObject* dx_GG_obj;
    double omega;
    double f;
    PyArrayObject* omegat_nGG_obj;
    PyArrayObject* W_nGG_obj;
    double eta;
    double factor;

    // evaluate_mpa_poly(x2_GG, dx2_GG, omega, f, omegat_nGG, W_nGG, self.eta, self.factor)
    if (!PyArg_ParseTuple(args, "OOddOOdd",
                          &x_GG_obj, &dx_GG_obj, &omega, &f, &omegat_nGG_obj, &W_nGG_obj, &eta, &factor))
        return NULL;


    if (PyArray_NDIM(W_nGG_obj) != 3)
    {
         PyErr_SetString(PyExc_TypeError, "W_nGG should be 3 dimensional");
         return NULL;
    }

    if (PyArray_NDIM(omegat_nGG_obj) != 3)
    {
         PyErr_SetString(PyExc_TypeError, "omegat_nGG should be 3 dimensional");
         return NULL;
    }

    int np = PyArray_DIMS(omegat_nGG_obj)[0];
    int nG1 = PyArray_DIMS(omegat_nGG_obj)[1];
    int nG2 = PyArray_DIMS(omegat_nGG_obj)[2];

    // Check dimensions
    if ((np != PyArray_DIMS(W_nGG_obj)[0]) ||
        (nG1 != PyArray_DIMS(W_nGG_obj)[1]) ||
        (nG2 != PyArray_DIMS(W_nGG_obj)[2]))
     {
         PyErr_SetString(PyExc_TypeError, "Unmatched dimensions between omegat_nnG and W_nGG.");
         return NULL;
     }

    // Check input types
    if ((PyArray_TYPE(omegat_nGG_obj) != NPY_COMPLEX128) ||
        (PyArray_TYPE(W_nGG_obj) != NPY_COMPLEX128))
    {
         PyErr_SetString(PyExc_TypeError, "Expected complex arrays for omegat_nGG and W_nGG.");
         return NULL;
    }

    if (PyArray_TYPE(x_GG_obj) != NPY_COMPLEX128)
    {
         PyErr_SetString(PyExc_TypeError, "x_GG expected to be a complex array.");
         return NULL;
    }

    if (!PyArray_IS_C_CONTIGUOUS(x_GG_obj)
        || !PyArray_IS_C_CONTIGUOUS(W_nGG_obj)
        || !PyArray_IS_C_CONTIGUOUS(omegat_nGG_obj))
    {
        PyErr_SetString(PyExc_TypeError, "Arrays need to be c-contiguous.");
        return NULL;
    }
    
    double complex* x_GG = (double complex*)PyArray_DATA(x_GG_obj);
    double complex* dx_GG = (double complex*)PyArray_DATA(dx_GG_obj);
    double complex* omegat_nGG = (double complex*)PyArray_DATA(omegat_nGG_obj);
    double complex* W_nGG = (double complex*)PyArray_DATA(W_nGG_obj);

    double complex omega_eta_m = omega - eta * I;
    double complex omega_eta_p = omega + eta * I;

    if (f<0 || f>1)
    {
        PyErr_SetString(PyExc_TypeError, "Occupation out of bounds.");
        return NULL;
    }

    // Optimizations for things being close to one, or close to zero
    // such that only one branch is evaluated
    if (f > 1 - 1e-10)
    {
        for (int G1=0; G1<nG1; G1++)
        {
            for (int G2=0; G2<nG2; G2++)
            {
                double complex result = 0;
                double complex dresult = 0;
                for (int p=0; p<np; p++)
                {
                    int index = G2 + G1 * nG2 + p * nG1 * nG2;
                    double complex omegat = omegat_nGG[index];
                    double complex x1 = 1.0 / (omega_eta_m + omegat);
                    result += x1 * W_nGG[index];
                    dresult -= x1 * x1 * W_nGG[index];
                }
                *x_GG++ = result * 2 * factor;
                *dx_GG++ = dresult * 2 * factor;
            }
        }
    }
    else if (f < 1e-10)
    {
        for (int G1=0; G1<nG1; G1++)
        {
            for (int G2=0; G2<nG2; G2++)
            {
                double complex result = 0;
                double complex dresult = 0;
                for (int p=0; p<np; p++)
                {
                    int index = G2 + G1 * nG2 + p * nG1 * nG2;
                    double complex omegat = omegat_nGG[index];
                    double complex x2 = 1.0 / (omega_eta_p - omegat);
                    result += x2 * W_nGG[index];
                    dresult -= x2 * x2 * W_nGG[index];
                }
                *x_GG++ = result * (2 * factor);
                *dx_GG++ = dresult * 2 * factor;
            }
        }
    }
    else
    {
        for (int G1=0; G1<nG1; G1++)
        {
            for (int G2=0; G2<nG2; G2++)
            {
                double complex result = 0;
                double complex dresult = 0;
                for (int p=0; p<np; p++)
                {
                    int index = G2 + G1 * nG2 + p * nG1 * nG2;
                    double complex omegat = omegat_nGG[index];
                    double complex x1 = f / (omega_eta_m + omegat);
                    double complex x2 = (1.0 - f) / (omega_eta_p - omegat);
                    result += (x1 + x2) * W_nGG[index];
                    dresult -= (x1 * x1 + x2 * x2) * W_nGG[index];
                }
                *x_GG++ = result * (2 * factor);
                *dx_GG++ = dresult * (2 * factor);
            }
        }
    }
    Py_RETURN_NONE;
}


