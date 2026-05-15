#ifdef GPAW_WITH_MAGMA

#include "extensions.h"
#include "array.h"

#include <assert.h>
#include <string.h>

#include "magma_gpaw.h"

// Wrappers for MAGMA CPU-eigensolvers. These input and output Numpy arrays

PyObject* eigh_magma_dsyevd(PyObject* self, PyObject* args)
{
    PyObject *in_matrix;
    char* in_uplo;

    if (!PyArg_ParseTuple(args, "Os", &in_matrix, &in_uplo))
    {
        return NULL;
    }

    if (!PyArray_Check(in_matrix))
    {
        PyErr_SetString(PyExc_TypeError, "Input must be a numpy array");
        return NULL;
    }

    // Must be symmetric, real, double precision matrix
    assert(Array_NDIM(in_matrix) == 2);
    assert(Array_DIM(in_matrix, 0) == Array_DIM(in_matrix, 1));
    assert(Array_ITEMSIZE(in_matrix) == sizeof(double));

    const size_t n = Array_DIM(in_matrix, 0);

    PyObject *eigvals = PyArray_SimpleNew(1, (npy_intp[]){n}, NPY_DOUBLE);
    PyObject* eigvects = PyArray_SimpleNew(2, PyArray_DIMS((PyArrayObject*)in_matrix), NPY_DOUBLE);

    assert(eigvals != NULL);
    assert(eigvects != NULL);

    const magma_vec_t jobz = MagmaVec; // always compute eigenvectors
    const magma_int_t lda = n;
    const magma_uplo_t uplo = get_magma_uplo(in_uplo);

    // Copy the input matrix because MAGMA will override it with eigenvectors.
    // So we can use the eigenvector buffer both as a work copy and as output.
    double* dA = Array_DATA(eigvects);
    memcpy(dA, Array_DATA(in_matrix), n*n*sizeof(double));

    dsyevd_workgroup workgroup = {};

    // Query optimal workgroup sizes
    double work_temp;
    magma_int_t iwork_temp;
    magma_int_t status;
    magma_dsyevd(
        jobz,
        uplo,
        n,
        NULL,
        lda,
        NULL,
        &work_temp,
        -1,
        &iwork_temp,
        -1,
        &status
    );

    assert(status == 0 && "magma_dsyevd query failed");
    workgroup.lwork = (magma_int_t) work_temp;
    workgroup.liwork = iwork_temp;

    workgroup.work = malloc(workgroup.lwork * sizeof(double));
    workgroup.iwork = malloc(workgroup.liwork * sizeof(magma_int_t));

    magma_dsyevd(
        jobz,
        uplo,
        n,
        dA,
        lda,
        Array_DATA(eigvals),
        workgroup.work,
        workgroup.lwork,
        workgroup.iwork,
        workgroup.liwork,
        &status
    );

    assert(status >= 0 && "Invalid input to MAGMA solver");
    if (status > 0)
    {
        PyErr_WarnEx(PyExc_RuntimeWarning,
            "MAGMA eigensolver failed to converge",
            1
        );
    }

    free(workgroup.work);
    free(workgroup.iwork);

    // Eigenvectors (dA) were already filled in by MAGMA

    PyObject* result = PyTuple_Pack(2, eigvals, eigvects);

    Py_DECREF(eigvals);
    Py_DECREF(eigvects);

    return result;
}

PyObject* eigh_magma_zheevd(PyObject* self, PyObject* args)
{
    PyObject *in_matrix;
    char* in_uplo;

    if (!PyArg_ParseTuple(args, "Os", &in_matrix, &in_uplo))
    {
        return NULL;
    }

    if (!PyArray_Check(in_matrix))
    {
        PyErr_SetString(PyExc_TypeError, "Input must be a numpy array");
        return NULL;
    }

    // Must be symmetric, complex double-precision matrix
    assert(Array_NDIM(in_matrix) == 2);
    assert(Array_DIM(in_matrix, 0) == Array_DIM(in_matrix, 1));
    assert(Array_ITEMSIZE(in_matrix) == 2*sizeof(double));

    const size_t n = Array_DIM(in_matrix, 0);

    PyObject *eigvals = PyArray_SimpleNew(1, (npy_intp[]){n}, NPY_DOUBLE);
    PyObject* eigvects = PyArray_SimpleNew(2, PyArray_DIMS((PyArrayObject*)in_matrix), NPY_CDOUBLE);

    assert(eigvals != NULL);
    assert(eigvects != NULL);

    const magma_vec_t jobz = MagmaVec;
    const magma_int_t lda = n;
    const magma_uplo_t uplo = get_magma_uplo(in_uplo);

    magmaDoubleComplex* dA = (magmaDoubleComplex*) Array_DATA(eigvects);
    memcpy(dA, Array_DATA(in_matrix), n*n*sizeof(magmaDoubleComplex));

    zheevd_workgroup workgroup = {};

    // Query
    magmaDoubleComplex work_temp;
    double rwork_temp;
    magma_int_t iwork_temp;
    magma_int_t status;
    magma_zheevd(
        jobz,
        uplo,
        n,
        NULL,
        lda,
        NULL,
        &work_temp,
        -1,
        &rwork_temp,
        -1,
        &iwork_temp,
        -1,
        &status
    );
    assert(status == 0 && "magma_zheevd query failed");

    workgroup.lwork = (magma_int_t) MAGMA_Z_REAL(work_temp);
    workgroup.lrwork = (magma_int_t) rwork_temp;
    workgroup.liwork = iwork_temp;

    workgroup.work = malloc(workgroup.lwork * sizeof(magmaDoubleComplex));
    workgroup.rwork = malloc(workgroup.lrwork * sizeof(double));
    workgroup.iwork = malloc(workgroup.liwork * sizeof(magma_int_t));

    magma_zheevd(
        jobz,
        uplo,
        n,
        dA,
        lda,
        Array_DATA(eigvals),
        workgroup.work,
        workgroup.lwork,
        workgroup.rwork,
        workgroup.lrwork,
        workgroup.iwork,
        workgroup.liwork,
        &status
    );

    assert(status >= 0 && "Invalid input to MAGMA solver");
    if (status > 0)
    {
        PyErr_WarnEx(PyExc_RuntimeWarning,
            "MAGMA eigensolver failed to converge",
            1
        );
    }

    free(workgroup.work);
    free(workgroup.rwork);
    free(workgroup.iwork);

    PyObject* result = PyTuple_Pack(2, eigvals, eigvects);

    Py_DECREF(eigvals);
    Py_DECREF(eigvects);

    return result;

}

#endif
