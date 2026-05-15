#if defined(GPAW_WITH_MAGMA) && defined(GPAW_GPU)

#include "../extensions.h"

// Define magic to enable custom Array_** macros for CUPY arrays
#define GPAW_ARRAY_DISABLE_NUMPY
#define GPAW_ARRAY_ALLOW_CUPY
#include "../array.h"
#undef GPAW_ARRAY_DISABLE_NUMPY

#include "gpu.h"
#include "gpu-complex.h"
#include "../magma_gpaw.h"

#include <assert.h>
#include <string.h>

// CUPY doesn't provide a nice C-interface like Numpy, so need to do tricks.
// We require that the user allocates and passes valid CUPY arrays from the
// Python side for both inputs AND outputs. We parse them here and pass the
// underlying memory pointers to an internal function that does the work, ie.
// calls MAGMA. Output is written to the buffers that were passed from Python.


static magma_int_t _eigh_magma_dsyevd_gpu(int matrix_size, magma_uplo_t uplo,
    double* in_matrix, double* inout_eigvals, double* inout_eigvects)
{
    // Caller is responsible for ensuring that buffers are already correct size

    // Input matrix to MAGMA gets overriden by eigenvectors, so take a copy here.
    gpuMemcpy(
        inout_eigvects, in_matrix,
        matrix_size * matrix_size * sizeof(double),
        gpuMemcpyDeviceToDevice
    );

    const magma_vec_t jobz = MagmaVec; // always compute eigenvectors
    const magma_int_t lda = matrix_size;

    dsyevd_workgroup workgroup = {};
    // Query
    double work_temp;
    magma_int_t iwork_temp;
    magma_int_t status;
    magma_dsyevd_gpu(jobz, uplo, matrix_size, NULL, lda, NULL,
        NULL, lda, &work_temp, -1, &iwork_temp, -1, &status
    );

    assert(status == 0 && "magma_dsyevd_gpu query failed");
    workgroup.lwork = (magma_int_t) work_temp;
    workgroup.liwork = iwork_temp;

    // All buffers apart from the input matrix are in HOST memory.
    // Use MAGMA allocators instead of malloc for possibly better alignment
    MAGMA_CHECK(magma_dmalloc_cpu(&workgroup.work, workgroup.lwork));
    MAGMA_CHECK(magma_imalloc_cpu(&workgroup.iwork, workgroup.liwork));

    double* h_wA;
    double* h_eigvals;
    MAGMA_CHECK(magma_dmalloc_cpu(&h_wA, matrix_size * lda));
    MAGMA_CHECK(magma_dmalloc_cpu(&h_eigvals, matrix_size));

    magma_dsyevd_gpu(jobz, uplo, matrix_size, inout_eigvects, lda,
        h_eigvals, h_wA, lda, workgroup.work, workgroup.lwork,
        workgroup.iwork, workgroup.liwork, &status
    );

    // copy eigenvalues to device output buffer
    gpuMemcpy(inout_eigvals, h_eigvals, matrix_size * sizeof(double), gpuMemcpyHostToDevice);

    MAGMA_CHECK(magma_free_cpu(h_wA));
    MAGMA_CHECK(magma_free_cpu(workgroup.work));
    MAGMA_CHECK(magma_free_cpu(workgroup.iwork));
    MAGMA_CHECK(magma_free_cpu(h_eigvals));

    return status;
}

static magma_int_t _eigh_magma_zheevd_gpu(int matrix_size, magma_uplo_t uplo,
    magmaDoubleComplex* in_matrix, double* inout_eigvals,
    magmaDoubleComplex* inout_eigvects)
{

    gpuMemcpy(
        inout_eigvects, in_matrix,
        matrix_size * matrix_size * sizeof(magmaDoubleComplex),
        gpuMemcpyDeviceToDevice
    );

    const magma_vec_t jobz = MagmaVec;
    const magma_int_t lda = matrix_size;

    zheevd_workgroup workgroup = {};

    // Query
    magmaDoubleComplex work_temp;
    double rwork_temp;
    magma_int_t iwork_temp;
    magma_int_t status;
    magma_zheevd_gpu(jobz, uplo, matrix_size, NULL, lda, NULL,
        NULL, lda, &work_temp, -1, &rwork_temp, -1,
        &iwork_temp, -1, &status
    );

    assert(status == 0 && "magma_zheevd_gpu query failed");
    workgroup.lwork = (magma_int_t) MAGMA_Z_REAL(work_temp);
    workgroup.lrwork = (magma_int_t) rwork_temp;
    workgroup.liwork = iwork_temp;

    MAGMA_CHECK(magma_zmalloc_cpu(&workgroup.work, workgroup.lwork));
    MAGMA_CHECK(magma_dmalloc_cpu(&workgroup.rwork, workgroup.lrwork));
    MAGMA_CHECK(magma_imalloc_cpu(&workgroup.iwork, workgroup.liwork));

    magmaDoubleComplex* h_wA;
    double* h_eigvals;
    MAGMA_CHECK(magma_zmalloc_cpu(&h_wA, matrix_size * lda));
    MAGMA_CHECK(magma_dmalloc_cpu(&h_eigvals, matrix_size));

    magma_zheevd_gpu(jobz, uplo, matrix_size, inout_eigvects, lda,
        h_eigvals, h_wA, lda, workgroup.work, workgroup.lwork,
        workgroup.rwork, workgroup.lrwork, workgroup.iwork, workgroup.liwork,
        &status
    );

    // copy eigenvalues to device output buffer
    gpuMemcpy(inout_eigvals, h_eigvals, matrix_size * sizeof(double), gpuMemcpyHostToDevice);


    MAGMA_CHECK(magma_free_cpu(h_wA));
    MAGMA_CHECK(magma_free_cpu(workgroup.work));
    MAGMA_CHECK(magma_free_cpu(workgroup.rwork));
    MAGMA_CHECK(magma_free_cpu(workgroup.iwork));
    MAGMA_CHECK(magma_free_cpu(h_eigvals));

    return status;

}

// dsyevd: Real symmetric matrix, double precision
PyObject* eigh_magma_dsyevd_gpu(PyObject* self, PyObject* args)
{
    PyObject *in_matrix;
    char* in_uplo;

    // Must be allocated on python side
    PyObject *inout_eigvals;
    PyObject *inout_eigvects;

    if (!PyArg_ParseTuple(args, "OsOO", &in_matrix, &in_uplo, &inout_eigvals,
        &inout_eigvects))
    {
        return NULL;
    }

    assert(Array_NDIM(in_matrix) == 2);
    assert(Array_DIM(in_matrix, 0) == Array_DIM(in_matrix, 1));
    assert(Array_ITEMSIZE(in_matrix) == sizeof(double));

    assert(Array_NDIM(inout_eigvects) == 2);
    assert(Array_DIM(inout_eigvects, 0) == Array_DIM(inout_eigvects, 1));
    assert(Array_ITEMSIZE(inout_eigvects) == sizeof(double));

    assert(Array_NDIM(inout_eigvals) == 1);
    assert(Array_ITEMSIZE(inout_eigvals) == sizeof(double));

    const size_t n = Array_DIM(in_matrix, 0);
    const magma_uplo_t uplo = get_magma_uplo(in_uplo);

    magma_int_t status = _eigh_magma_dsyevd_gpu(
        n,
        uplo,
        Array_DATA(in_matrix),
        Array_DATA(inout_eigvals),
        Array_DATA(inout_eigvects)
    );

    assert(status >= 0 && "Invalid input to MAGMA solver");
    if (status > 0)
    {
        PyErr_WarnEx(PyExc_RuntimeWarning,
            "MAGMA eigensolver failed to converge",
            1
        );
    }

    Py_RETURN_NONE;
}

// zheevd: Complex Hermitian matrix, double precision
PyObject* eigh_magma_zheevd_gpu(PyObject* self, PyObject* args)
{
    PyObject *in_matrix;
    char* in_uplo;

    // Must be allocated on python side
    PyObject *inout_eigvals;
    PyObject *inout_eigvects;

    if (!PyArg_ParseTuple(args, "OsOO", &in_matrix, &in_uplo, &inout_eigvals,
        &inout_eigvects))
    {
        return NULL;
    }

    assert(Array_NDIM(in_matrix) == 2);
    assert(Array_DIM(in_matrix, 0) == Array_DIM(in_matrix, 1));
    assert(Array_ITEMSIZE(in_matrix) == 2*sizeof(double));

    assert(Array_NDIM(inout_eigvects) == 2);
    assert(Array_DIM(inout_eigvects, 0) == Array_DIM(inout_eigvects, 1));
    assert(Array_ITEMSIZE(inout_eigvects) == 2*sizeof(double));

    assert(Array_NDIM(inout_eigvals) == 1);
    assert(Array_ITEMSIZE(inout_eigvals) == sizeof(double));

    const size_t n = Array_DIM(in_matrix, 0);
    const magma_uplo_t uplo = get_magma_uplo(in_uplo);

    magma_int_t status = _eigh_magma_zheevd_gpu(
        n,
        uplo,
        (magmaDoubleComplex*) Array_DATA(in_matrix),
        Array_DATA(inout_eigvals),
        (magmaDoubleComplex*) Array_DATA(inout_eigvects)
    );

    assert(status >= 0 && "Invalid input to MAGMA solver");
    if (status > 0)
    {
        PyErr_WarnEx(PyExc_RuntimeWarning,
            "MAGMA eigensolver failed to converge",
            1
        );
    }

    Py_RETURN_NONE;
}

#endif
