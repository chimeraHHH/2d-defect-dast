/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2008  CAMd
 *  Copyright (C) 2005-2020  CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information.

 Pure C implementation of preconditioner                           */


#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "extensions.h"
#include "operators.h"
#include "transformers.h"
#include "threading.h"

#define DIMS_SAME(a, b) (PyArray_NDIM(a) == PyArray_NDIM(b) && \
                         memcmp(PyArray_DIMS(a), \
                                PyArray_DIMS(b), \
                                PyArray_NDIM(a) * sizeof(PyArray_DIMS(a)[0])) == 0)

/*
 * Array operations for the preconditioner. BLAS routines are not sensible
 * here since those might be multithreaded. Nested multithreading is not
 * a smart move performance-wise in general unless you really know what
 * you are doing.
 *
 * With a decent compiler there should be no need for optimization by hand.
 */

#define ARRAY_NEGATE(name, type)                                        \
static void                                                             \
name(type *x, int start, int end, int size)                             \
{                                                                       \
    int i;                                                              \
                                                                        \
    x += start * size;                                                  \
    for (i = 0; i < (end - start) * size; i++)                          \
        x[i] = -x[i];                                                   \
}

ARRAY_NEGATE(array_negate,  double)
ARRAY_NEGATE(array_negatez, double complex)

#define ARRAY_SUB(name, type)                                           \
static void                                                             \
name(type *x, const type *y, int start, int end, int size)              \
{                                                                       \
    int i;                                                              \
                                                                        \
    x += start * size;                                                  \
    y += start * size;                                                  \
    for (i = 0; i < (end - start) * size; i++)                          \
        x[i] -= y[i];                                                   \
}

ARRAY_SUB(array_sub,  double)
ARRAY_SUB(array_subz, double complex)

#define ARRAY_SUB_MULT(name, type)                                      \
static void                                                             \
name(type *x, const type *y, double a, int start, int end, int size)    \
{                                                                       \
    int i;                                                              \
                                                                        \
    x += start * size;                                                  \
    y += start * size;                                                  \
    for (i = 0; i < (end - start) * size; i++)                          \
        x[i] -= a * y[i];                                               \
}

ARRAY_SUB_MULT(array_sub_mult,  double)
ARRAY_SUB_MULT(array_sub_multz, double complex)

#define ARRAY_MULTO(name, type)                                         \
static void                                                             \
name(type *x, const type *y, double a, int start, int end, int size)    \
{                                                                       \
    int i;                                                              \
                                                                        \
    x += start * size;                                                  \
    y += start * size;                                                  \
    for (i = 0; i < (end - start) * size; i++)                          \
        x[i] = a * y[i];                                                \
}

ARRAY_MULTO(array_multo,  double)
ARRAY_MULTO(array_multoz, double complex)

/*
 * Implements the computational part of the preconditioner in C. In overall,
 * it is at least somewhat faster than the original Python code. There is less
 * overhead. The performance is even better with multiple threads since
 * synchronizations are largely eliminated and threads can be kept running
 * throughout the computation.
 *
 * This code is a drop-in replacement for the computational Python code.
 * However, this code does not do administrative tasks, such as buffer
 * allocation. Those things are done reliably a lot easier in Python.
 */

PyObject *fd_precond(PyObject *self, PyObject *args)
{
    PyArrayObject *d0, *q0, *r1, *d1, *q1, *r2, *d2, *q2;
    PyArrayObject *residuals, *nresiduals, *phases;
    TransformerObject *rest0, *rest1, *intp1, *intp2;
    OperatorObject *kin0, *kin1, *kin2;
    double step;
    int size0, size1, size2;  /* Grid sizes on different levels */
    int nin;

    if (PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOOOd|O",
                &rest0, &rest1, &intp1, &intp2, &kin0, &kin1, &kin2,
                &d0, &q0, &r1, &d1, &q1, &r2, &d2, &q2,
                &residuals, &nresiduals, &step, &phases) == 0)
        return NULL;

    /* Input and output buffers */
    const double* in; 
    double* out;

    /*
     * Check parameters. Sometimes it tends to save time... It is anything
     * but foolproof, however.
     */

    /*
     * Resolve the number of input grids. Array residuals can contain
     * either a single or multiple grids.
     */
    nin = 1;
    if (PyArray_NDIM(residuals) == 4)
        nin = PyArray_DIMS(residuals)[0];
    else if (PyArray_NDIM(residuals) != 3) {
        PyErr_SetString(PyExc_TypeError, "Bad array dimension.");
        return NULL;
    }
    assert(nin >= 0);  /* Paranoia. */

    /* Calculate the size of a single grid on every level of coarseness. */
    size0 = PyArray_DIMS(d0)[1] * PyArray_DIMS(d0)[2] * PyArray_DIMS(d0)[3];
    size1 = PyArray_DIMS(d1)[1] * PyArray_DIMS(d1)[2] * PyArray_DIMS(d1)[3];
    size2 = PyArray_DIMS(d2)[1] * PyArray_DIMS(d2)[2] * PyArray_DIMS(d2)[3];

    /*
     * Scrutinize the array shapes since the correctness of upcoming
     * computatons depends heavily on them.
     */
    if (PyArray_NDIM(d0) != 4 || PyArray_NDIM(q0) != 4 ||
        PyArray_NDIM(r1) != 4 || PyArray_NDIM(d1) != 4 || PyArray_NDIM(q1) != 4 ||
        PyArray_NDIM(r2) != 4 || PyArray_NDIM(d2) != 4 || PyArray_NDIM(q2) != 4) {
        PyErr_SetString(PyExc_TypeError,
                "Work arrays do not have 4 dimensions.");
        return NULL;
    }
    if (!DIMS_SAME(residuals, nresiduals)) {
        PyErr_SetString(PyExc_TypeError,
          "Arrays residuals and nresiduals do not have the same shape.");
        return NULL;
    }
    if (!DIMS_SAME(d0, q0)) {
        PyErr_SetString(PyExc_TypeError,
          "Arrays d0 and q0 do not have the same shape.");
        return NULL;
    }
    if (!DIMS_SAME(d1, r1) || !DIMS_SAME(d1, q1)) {
        PyErr_SetString(PyExc_TypeError,
          "Arrays d1, q1, r1 do not have the same shape.");
        return NULL;
    }
    if (!DIMS_SAME(d2, r2) || !DIMS_SAME(d2, q2)) {
        PyErr_SetString(PyExc_TypeError,
          "Arrays d2, q2, r2 do not have the same shape.");
        return NULL;
    }
    if ((PyArray_NDIM(residuals) == 3 &&
            memcmp(PyArray_DIMS(residuals), PyArray_DIMS(d0) + 1,
              3 * sizeof(PyArray_DIMS(d0)[0])) != 0) &&
            !DIMS_SAME(residuals, d0)) {
        PyErr_SetString(PyExc_TypeError,
          "Input grid shape does not match with arrays d0 and q0.");
        return NULL;
    }
    if (nin != PyArray_DIMS(d0)[0] ||
            nin != PyArray_DIMS(d1)[0] ||
            nin != PyArray_DIMS(d2)[0]) {
        PyErr_SetString(PyExc_TypeError,
          "Number of input grids does not match with work arrays.");
        return NULL;
    }

    /* Checks for non-strided arrays and contiguous data would be nice. */

    bool real = (PyArray_DESCR(residuals)->type_num == NPY_DOUBLE);

    const double_complex* ph;
    ph = (real != 0) ? NULL : COMPLEXP(phases);

    int chunksize = 1; // Use a single chunk for a while 

    #pragma omp parallel
    {
        int nthreads, thread_id;
        int start, end;

#ifdef _OPENMP
        nthreads = omp_get_num_threads();
        thread_id    = omp_get_thread_num();
#else
        nthreads = 1;
        thread_id    = 0;
#endif

        /* Partition the grids among threads. */
        SHARE_WORK(nin, nthreads, thread_id, &start, &end);

        /* Restrict (-residuals) -> r1. */
        in   = DOUBLEP(nresiduals);
        out  = DOUBLEP(r1);
        transapply_worker(rest0, chunksize, start, end, thread_id, nthreads, 
			  in, out, real, ph );

        /* d1 <- 4 * step * r1 */
        if (real != 0)
            array_multo(DOUBLEP(d1), DOUBLEP(r1), 4 * step, start, end, size1);
        else
            array_multoz(COMPLEXP(d1), COMPLEXP(r1), 4 * step,
                    start, end, size1);

        /* Apply d1 --kin1--> q1. */
        in   = DOUBLEP(d1);
        out  = DOUBLEP(q1);
        apply_worker(kin1, chunksize, start, end, thread_id, nthreads, 
		     in, out, real, ph);

        /* q1 -= r1 */
        if (real != 0)
            array_sub(DOUBLEP(q1), DOUBLEP(r1), start, end, size1);
        else
            array_subz(COMPLEXP(q1), COMPLEXP(r1), start, end, size1);

        /* Restrict q1 -> r2. */
        in   = DOUBLEP(q1);
        out  = DOUBLEP(r2);
        transapply_worker(rest1, chunksize, start, end, thread_id, nthreads, 
			  in, out, real, ph);

        /* d2 <- 16 * step * r2 */
        if (real != 0)
            array_multo(DOUBLEP(d2), DOUBLEP(r2), 16 * step,
                    start, end, size2);
        else
            array_multoz(COMPLEXP(d2), COMPLEXP(r2), 16 * step,
                    start, end, size2);

        /* Apply d2 --kin2--> q2. */
        in   = DOUBLEP(d2);
        out  = DOUBLEP(q2);
        apply_worker(kin2, chunksize, start, end, thread_id, nthreads, 
		 in, out, real, ph);

        /*
         * q2 -= r2
         * d2 -= 16 * step * q2
         */
        if (real != 0) {
            array_sub(DOUBLEP(q2), DOUBLEP(r2), start, end, size2);
            array_sub_mult(DOUBLEP(d2), DOUBLEP(q2), 16 * step,
                    start, end, size2);
        }
        else {
            array_subz(COMPLEXP(q2), COMPLEXP(r2), start, end, size2);
            array_sub_multz(COMPLEXP(d2), COMPLEXP(q2), 16 * step,
                    start, end, size2);
        }

        /* Interpolate d2 -> q1. */
        in   = DOUBLEP(d2);
        out  = DOUBLEP(q1);
        transapply_worker(intp2, chunksize, start, end, thread_id, nthreads, 
			  in, out, real, ph);

        /* d1 -= q1 */
        if (real != 0)
            array_sub(DOUBLEP(d1), DOUBLEP(q1), start, end, size1);
        else
            array_subz(COMPLEXP(d1), COMPLEXP(q1), start, end, size1);

        /* Apply d1 --kin1--> q1. */
        in   = DOUBLEP(d1);
        out  = DOUBLEP(q1);
        apply_worker(kin1, chunksize, start, end, thread_id, nthreads, 
		 in, out, real, ph);

        /*
         * q1 -= r1
         * d1 -= 4 * step * q1
         */
        if (real != 0) {
            array_sub(DOUBLEP(q1), DOUBLEP(r1), start, end, size1);
            array_sub_mult(DOUBLEP(d1), DOUBLEP(q1), 4 * step,
                    start, end, size1);
        }
        else {
            array_subz(COMPLEXP(q1), COMPLEXP(r1), start, end, size1);
            array_sub_multz(COMPLEXP(d1), COMPLEXP(q1), 4 * step,
                    start, end, size1);
        }

        /* Interpolate (-d1) -> d0. Do negation in place. */
        if (real != 0)
            array_negate(DOUBLEP(d1), start, end, size1);
        else
            array_negatez(COMPLEXP(d1), start, end, size1);
        in   = DOUBLEP(d1);
        out  = DOUBLEP(d0);
        transapply_worker(intp1, chunksize, start, end, thread_id, nthreads, 
			  in, out, real, ph);

        /* Apply d0 --kin0--> q0. */
        in   = DOUBLEP(d0);
        out  = DOUBLEP(q0);
        apply_worker(kin0, chunksize, start, end, thread_id, nthreads, 
		 in, out, real, ph);

        /*
         * q0 -= residuals
         * d0 -= step * q0
         * d0 *= -1
         */
        if (real != 0) {
            array_sub(DOUBLEP(q0), DOUBLEP(residuals), start, end, size0);
            array_sub_mult(DOUBLEP(d0), DOUBLEP(q0), step, start, end, size0);
            array_negate(DOUBLEP(d0), start, end, size0);
        }
        else {
            array_subz(COMPLEXP(q0), COMPLEXP(residuals), start, end, size0);
            array_sub_multz(COMPLEXP(d0), COMPLEXP(q0),
                    step, start, end, size0);
            array_negatez(COMPLEXP(d0), start, end, size0);
        }
    }

    /* The return value is in d0. The calling code handles it. */

    Py_RETURN_NONE;
}