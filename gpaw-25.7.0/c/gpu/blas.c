#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "gpu.h"
#include "gpu-complex.h"


gpublasHandle_t _gpaw_gpublas_handle;


void blas_init_gpu()
{
    gpublasSafeCall(gpublasCreate(&_gpaw_gpublas_handle));
}


static gpublasOperation_t gpublas_operation(int op)
{
    gpublasOperation_t gpu_op;

    if (op == 'N' || op == 'n')
        gpu_op = GPUBLAS_OP_N;
    else if (op == 'T' || op == 't')
        gpu_op = GPUBLAS_OP_T;
    else if (op == 'C' || op == 'c')
        gpu_op = GPUBLAS_OP_C;
    else
        assert(0);
    return gpu_op;
}


PyObject* scal_gpu(PyObject *self, PyObject *args)
{
    Py_complex alpha;

    void *x_gpu;
    PyObject *x_shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "DnOO", &alpha, &x_gpu, &x_shape, &type))
        return NULL;

    int n = (int) PyLong_AsLong(PyTuple_GetItem(x_shape, 0));
    Py_ssize_t nd = PyTuple_Size(x_shape);
    for (int d=1; d < nd; d++)
        n *= (int) PyLong_AsLong(PyTuple_GetItem(x_shape, d));
    int incx = 1;
    if (type->type_num == NPY_DOUBLE) {
        gpublasSafeCall(
                gpublasDscal(_gpaw_gpublas_handle, n, &alpha.real,
                            (double*) x_gpu, incx));
    } else {
        gpublasDoubleComplex alpha_gpu = {alpha.real, alpha.imag};
        gpublasSafeCall(
                gpublasZscal(_gpaw_gpublas_handle, n, &alpha_gpu,
                    (gpublasDoubleComplex*) x_gpu, incx));
    }
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}


static void _mmm_gpu(gpublasOperation_t gpu_opa, gpublasOperation_t gpu_opb,
                     int m, int n, int k,
                     Py_complex alpha, void *a, int lda,
                     void *b, int ldb, Py_complex beta,
                     void *c, int ldc, int real)
{
    if (real) {
        gpublasSafeCall(
                gpublasDgemm(_gpaw_gpublas_handle, gpu_opa, gpu_opb, m, n, k,
                            &(alpha.real), (double*) a, lda, (double*) b, ldb,
                            &(beta.real), (double*) c, ldc));
    } else {
        gpublasDoubleComplex alpha_gpu = {alpha.real, alpha.imag};
        gpublasDoubleComplex beta_gpu = {beta.real, beta.imag};
        gpublasSafeCall(
                gpublasZgemm(_gpaw_gpublas_handle, gpu_opa, gpu_opb, m, n, k,
                            &alpha_gpu, (gpublasDoubleComplex*) a, lda,
                            (gpublasDoubleComplex*) b, ldb,
                            &beta_gpu, (gpublasDoubleComplex*) c, ldc));
    }
}

PyObject* mmm_gpu(PyObject *self, PyObject *args)
{
    Py_complex alpha;
    void *b;
    int ldb;
    int opb;
    void *a;
    int lda;
    int opa;
    Py_complex beta;
    void *c;
    int ldc;
    int bytes;
    int m, n, k;

    if (!PyArg_ParseTuple(args, "DniCniCDniiiii",
                          &alpha, &b, &ldb, &opb, &a, &lda, &opa,
                          &beta, &c, &ldc, &bytes, &m, &n, &k))
        return NULL;

    gpublasOperation_t gpu_opa = gpublas_operation(opa);
    gpublasOperation_t gpu_opb = gpublas_operation(opb);
    int real = (bytes == NPY_SIZEOF_DOUBLE);

    _mmm_gpu(gpu_opa, gpu_opb, m, n, k, alpha, a, lda, b, ldb, beta,
             c, ldc, real);

    Py_RETURN_NONE;
}


static void _gemm_gpu(gpublasOperation_t transa_c,
                      int m, int n, int k,
                      Py_complex alpha, void *a_gpu, int lda,
                      void *b_gpu, int ldb, Py_complex beta,
                      void *c_gpu, int ldc,
                      int real)
{
    _mmm_gpu(transa_c, GPUBLAS_OP_N, m, n, k, alpha, a_gpu, lda,
             b_gpu, ldb, beta, c_gpu, ldc, real);
}


PyObject* gemm_gpu(PyObject *self, PyObject *args)
{
    Py_complex alpha;
    Py_complex beta;

    void *a_gpu;
    void *b_gpu;
    void *c_gpu;
    PyObject *a_shape, *b_shape, *c_shape;
    PyArray_Descr *type;

    int transa = 'n';

    if (!PyArg_ParseTuple(args, "DnOnODnOO|Ci", &alpha, &a_gpu, &a_shape,
                          &b_gpu, &b_shape, &beta, &c_gpu, &c_shape, &type,
                          &transa))
        return NULL;

    int real = 0;
    if (type->type_num == NPY_DOUBLE) {
        real = 1;
    }

    gpublasOperation_t transa_c = gpublas_operation(transa);

    int m, k, lda, ldb, ldc;
    int n = (int) PyLong_AsLong(PyTuple_GetItem(b_shape, 0));
    if (transa == 'n') {
        m = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 1));
        for (int i=2; i < PyTuple_Size(a_shape); i++)
            m *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, i));
        k = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
        lda = m;
        ldb = k;
        ldc = m;
    } else {
        k = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 1));
        for (int i=2; i < PyTuple_Size(a_shape); i++)
            k *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, i));
        m = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
        lda = k;
        ldb = k;
        ldc = m;
    }

    _gemm_gpu(transa_c, m, n, k, alpha, a_gpu, lda, b_gpu, ldb, beta,
              c_gpu, ldc, real);

    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}


PyObject* gemv_gpu(PyObject *self, PyObject *args)
{
    Py_complex alpha;

    void *a_gpu;
    void *x_gpu;
    void *y_gpu;

    Py_complex beta;
    PyObject *a_shape, *x_shape;
    PyArray_Descr *type;

    int trans = 't';
    if (!PyArg_ParseTuple(args, "DnOnODnO|C", &alpha, &a_gpu, &a_shape,
                          &x_gpu, &x_shape, &beta, &y_gpu, &type, &trans))
        return NULL;

    gpublasOperation_t trans_c = gpublas_operation(trans);

    int m, n, lda, incx, incy;
    if (trans == 'n') {
        m = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 1));
        for (int i=2; i < PyTuple_Size(a_shape); i++)
            m *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, i));
        n = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
        lda = m;
    } else {
        n = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
        for (int i=1; i < PyTuple_Size(a_shape) - 1; i++)
            n *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, i));
        m = (int) PyLong_AsLong(
                PyTuple_GetItem(a_shape, PyTuple_Size(a_shape) - 1));
        lda = m;
    }

    incx = 1;
    incy = 1;
    if (type->type_num == NPY_DOUBLE) {
        gpublasSafeCall(
                gpublasDgemv(_gpaw_gpublas_handle, trans_c, m, n,
                            &alpha.real, (double*) a_gpu, lda,
                            (double*) x_gpu, incx,
                            &beta.real, (double*) y_gpu, incy));
    } else {
        gpublasDoubleComplex alpha_gpu = {alpha.real, alpha.imag};
        gpublasDoubleComplex beta_gpu = {beta.real, beta.imag};
        gpublasSafeCall(
                gpublasZgemv(_gpaw_gpublas_handle, trans_c, m, n,
                            &alpha_gpu, (gpublasDoubleComplex*) a_gpu, lda,
                            (gpublasDoubleComplex*) x_gpu, incx,
                            &beta_gpu, (gpublasDoubleComplex*) y_gpu, incy));
    }
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}


PyObject* axpy_gpu(PyObject *self, PyObject *args)
{
    Py_complex alpha;

    void *x_gpu;
    void *y_gpu;
    PyObject *x_shape,*y_shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "DnOnOO", &alpha, &x_gpu, &x_shape,
                          &y_gpu, &y_shape, &type))
        return NULL;

    Py_ssize_t nd = PyTuple_Size(x_shape);
    int n = (int) PyLong_AsLong(PyTuple_GetItem(x_shape, 0));
    for (int d=1; d < nd; d++)
        n *= (int) PyLong_AsLong(PyTuple_GetItem(x_shape, d));
    int incx = 1;
    int incy = 1;
    if (type->type_num == NPY_DOUBLE) {
        gpublasSafeCall(
                gpublasDaxpy(_gpaw_gpublas_handle, n, &alpha.real,
                            (double*) x_gpu, incx,
                            (double*) y_gpu, incy));
    } else {
        gpublasDoubleComplex alpha_gpu = {alpha.real, alpha.imag};
        gpublasSafeCall(
                gpublasZaxpy(_gpaw_gpublas_handle, n, &alpha_gpu,
                            (gpublasDoubleComplex*) x_gpu, incx,
                            (gpublasDoubleComplex*) y_gpu, incy));
    }
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}


static void _rk_gpu(int n, int k,
                    double alpha, void *a_gpu, int lda,
                    double beta, void *c_gpu, int ldc,
                    int real)
{
    if (real) {
        gpublasSafeCall(
                gpublasDsyrk(_gpaw_gpublas_handle,
                    GPUBLAS_FILL_MODE_UPPER, GPUBLAS_OP_T,
                    n, k,
                    &alpha, (double*) a_gpu, lda,
                    &beta, (double*) c_gpu, ldc));
    } else {
        gpublasSafeCall(
                gpublasZherk(_gpaw_gpublas_handle,
                    GPUBLAS_FILL_MODE_UPPER, GPUBLAS_OP_C,
                    n, k,
                    &alpha, (gpublasDoubleComplex*) a_gpu, lda,
                    &beta, (gpublasDoubleComplex*) c_gpu, ldc));
    }
}


PyObject* rk_gpu(PyObject *self, PyObject *args)
{
    double alpha;
    double beta;

    void *a_gpu;
    void *c_gpu;
    PyObject *a_shape, *c_shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "dnOdnOO|i", &alpha, &a_gpu, &a_shape,
                          &beta, &c_gpu, &c_shape, &type))
        return NULL;

    int real = 0;
    if (type->type_num == NPY_DOUBLE) {
        real = 1;
    }

    int n = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
    int k = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 1));
    for (int d=2; d < PyTuple_Size(a_shape); d++)
        k *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, d));
    int ldc = n;
    int lda = k;

    _rk_gpu(n, k, alpha, a_gpu, lda, beta, c_gpu, ldc, real);

    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}


static void _r2k_gpu(int n, int k,
                     Py_complex alpha, void *a_gpu, int lda,
                     void *b_gpu, double beta,
                     void *c_gpu, int ldc, int real)
{
    if (real) {
        gpublasSafeCall(
                gpublasDsyr2k(_gpaw_gpublas_handle,
                    GPUBLAS_FILL_MODE_UPPER, GPUBLAS_OP_T, n, k,
                    &alpha.real, (double*) a_gpu, lda,
                    (double*) b_gpu, lda,
                    &beta, (double*) c_gpu, ldc));
    } else {
        gpublasDoubleComplex alpha_gpu = {alpha.real, alpha.imag};
        gpublasSafeCall(
                gpublasZher2k(_gpaw_gpublas_handle,
                    GPUBLAS_FILL_MODE_UPPER, GPUBLAS_OP_C, n, k,
                    &alpha_gpu, (gpublasDoubleComplex*) a_gpu, lda,
                    (gpublasDoubleComplex*) b_gpu, lda,
                    &beta, (gpublasDoubleComplex*) c_gpu, ldc));
    }
}


PyObject* r2k_gpu(PyObject *self, PyObject *args)
{
    Py_complex alpha;
    double beta;

    void *a_gpu;
    void *b_gpu;
    void *c_gpu;
    PyObject *a_shape, *b_shape, *c_shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "DnOnOdnOO|i", &alpha, &a_gpu, &a_shape,
                          &b_gpu, &b_shape, &beta, &c_gpu, &c_shape,
                          &type))
        return NULL;

    int real = 0;
    if (type->type_num == NPY_DOUBLE) {
        real = 1;
    }

    int n = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
    int k = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 1));
    for (int d=2; d < PyTuple_Size(a_shape); d++)
        k *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, d));
    int ldc = n;
    int lda = k;

    _r2k_gpu(n, k, alpha, a_gpu, lda, b_gpu, beta, c_gpu, ldc, real);

    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}


PyObject* dotc_gpu(PyObject *self, PyObject *args)
{
    void *a_gpu;
    void *b_gpu;

    PyObject *a_shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "nOnO", &a_gpu, &a_shape, &b_gpu, &type))
        return NULL;

    int n = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
    for (int i=1; i < PyTuple_Size(a_shape); i++)
        n *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, i));

    int incx = 1;
    int incy = 1;
    if (type->type_num == NPY_DOUBLE) {
        double result;
        gpublasSafeCall(
                gpublasDdot(_gpaw_gpublas_handle, n,
                           (double*) a_gpu, incx,
                           (double*) b_gpu, incy, &result));
        if (PyErr_Occurred())
            return NULL;
        else
            return PyFloat_FromDouble(result);
    } else {
        gpublasDoubleComplex result;
        gpublasSafeCall(
                gpublasZdotc(_gpaw_gpublas_handle, n,
                            (gpublasDoubleComplex*) a_gpu, incx,
                            (gpublasDoubleComplex*) b_gpu, incy, &result));
        if (PyErr_Occurred())
            return NULL;
        else
            return PyComplex_FromDoubles(result.x,result.y);
    }
}


PyObject* dotu_gpu(PyObject *self, PyObject *args)
{
    void *a_gpu;
    void *b_gpu;

    PyObject *a_shape;
    PyArray_Descr *type;

    if (!PyArg_ParseTuple(args, "nOnO", &a_gpu, &a_shape, &b_gpu, &type))
        return NULL;

    int n = (int) PyLong_AsLong(PyTuple_GetItem(a_shape, 0));
    for (int i=1; i < PyTuple_Size(a_shape); i++)
        n *= (int) PyLong_AsLong(PyTuple_GetItem(a_shape, i));

    int incx = 1;
    int incy = 1;
    if (type->type_num == NPY_DOUBLE) {
        double result;
        gpublasSafeCall(
                gpublasDdot(_gpaw_gpublas_handle, n,
                    (double*) a_gpu, incx,
                    (double*) b_gpu, incy, &result));
        if (PyErr_Occurred())
            return NULL;
        else
            return PyFloat_FromDouble(result);
    } else {
        gpublasDoubleComplex result;
        gpublasSafeCall(
                gpublasZdotu(_gpaw_gpublas_handle, n,
                    (gpublasDoubleComplex*) a_gpu, incx,
                    (gpublasDoubleComplex*) b_gpu, incy, &result));
        if (PyErr_Occurred())
            return NULL;
        else
            return PyComplex_FromDoubles(result.x,result.y);
    }
}
