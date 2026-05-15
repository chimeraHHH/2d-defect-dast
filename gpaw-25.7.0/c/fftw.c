#ifdef GPAW_WITH_FFTW
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <fftw3.h>

/* Create plan and return pointer to plan as a string */
PyObject * FFTWPlan(PyObject *self, PyObject *args)
{
    PyArrayObject* in;
    PyArrayObject* out;
    int sign;
    unsigned int flags;
    if (!PyArg_ParseTuple(args, "OOiI",
                          &in, &out, &sign, &flags))
        return NULL;

    fftw_plan* plan = (fftw_plan*)malloc(sizeof(fftw_plan));

    int ndim = PyArray_NDIM(in);
    int dims_in[ndim];
    int dims_out[ndim];
    int i;

    void *indata = PyArray_DATA(in);
    void *outdata = PyArray_DATA(out);

    for(i=0; i < ndim; i++) {
        dims_in[i] = (int)PyArray_DIMS(in)[i];
        dims_out[i] = (int)PyArray_DIMS(out)[i];
    }

    if (PyArray_DESCR(in)->type_num == NPY_DOUBLE) {
        *plan = fftw_plan_dft_r2c(ndim, dims_in,
                                  (double *)indata,
                                  (fftw_complex *)outdata,
                                  flags);
    } else if (PyArray_DESCR(out)->type_num == NPY_DOUBLE) {
        *plan = fftw_plan_dft_c2r(ndim, dims_out,
                                  (fftw_complex *)indata,
                                  (double *)outdata,
                                  flags);
    } else {
        *plan = fftw_plan_dft(ndim, dims_out,
                              (fftw_complex *)indata,
                              (fftw_complex *)outdata,
                              sign, flags);
    }

    return Py_BuildValue("y#", plan, (Py_ssize_t)sizeof(fftw_plan*));
}


PyObject * FFTWExecute(PyObject *self, PyObject *args)
{
    fftw_plan* plan;
    Py_ssize_t n;
    if (!PyArg_ParseTuple(args, "y#", &plan, &n))
        return NULL;
    fftw_execute(*plan);
    Py_RETURN_NONE;
}


PyObject * FFTWDestroy(PyObject *self, PyObject *args)
{
    fftw_plan* plan;
    Py_ssize_t n;
    if (!PyArg_ParseTuple(args, "y#", &plan, &n))
        return NULL;
    fftw_destroy_plan(*plan);
    Py_RETURN_NONE;
}

#endif // GPAW_WITH_FFTW
