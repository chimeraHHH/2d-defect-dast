#if defined(GPAW_WITH_SL) && defined(PARALLEL) && defined(GPAW_WITH_ELPA)

#include "extensions.h"
#include <elpa/elpa.h>
#include <mpi.h>
#include "mympi.h"

elpa_t* unpack_handleptr(PyObject* handle_obj)
{
    elpa_t* elpa = (elpa_t *)PyArray_DATA((PyArrayObject *)handle_obj);
    return elpa;
}

elpa_t unpack_handle(PyObject* handle_obj)
{
    elpa_t* elpa = unpack_handleptr(handle_obj);
    return *elpa;
}

PyObject* checkerr(int err)
{
    if(err != ELPA_OK) {
        const char * errmsg = elpa_strerr(err);
        PyErr_SetString(PyExc_RuntimeError, errmsg);
        return NULL;
    }
    Py_RETURN_NONE;
}

PyObject* pyelpa_version(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
#ifdef ELPA_API_VERSION
    int version = ELPA_API_VERSION;
    return Py_BuildValue("i", version);
#else
    Py_RETURN_NONE;  // This means 'old', e.g. 2018.05.001
#endif
}

PyObject* pyelpa_set(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    char* varname;
    int value;
    if (!PyArg_ParseTuple(args, "Osi",
                          &handle_obj,
                          &varname,
                          &value)) {
        return NULL;
    }
    elpa_t handle = unpack_handle(handle_obj);
    int err;
    elpa_set(handle, varname, value, &err);
    return checkerr(err);
}

PyObject* pyelpa_init(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    // Globally initialize Elpa library if present:
    if (elpa_init(20171201) != ELPA_OK) {
        // What API versions do we support?
        PyErr_SetString(PyExc_RuntimeError, "Elpa >= 20171201 required");
        PyErr_Print();
        return NULL;
    }
    Py_RETURN_NONE;
}

PyObject* pyelpa_uninit(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;

#ifdef ELPA_API_VERSION
    // Newer Elpas define their version but older ones don't.
    int elpa_err;
    elpa_uninit(&elpa_err);
    if (elpa_err != ELPA_OK) {
        PyErr_SetString(PyExc_RuntimeError,
                        "elpa_uninit() failed");
        return NULL;
    }
#else
    elpa_uninit();  // 2018.05.001: no errcode
#endif
    Py_RETURN_NONE;
}

PyObject* pyelpa_allocate(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj))
        return NULL;

    elpa_t *handle = unpack_handleptr(handle_obj);
    int err = 0;
    handle[0] = elpa_allocate(&err);
    return checkerr(err);
}

PyObject* pyelpa_setup(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj))
        return NULL;

    elpa_t handle = unpack_handle(handle_obj);
    int err = elpa_setup(handle);
    return checkerr(err);
}


PyObject* pyelpa_set_comm(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    PyObject *gpaw_comm_obj;

    if(!PyArg_ParseTuple(args, "OO", &handle_obj,
                         &gpaw_comm_obj))
        return NULL;
    elpa_t handle = unpack_handle(handle_obj);
    MPIObject *gpaw_comm = (MPIObject *)gpaw_comm_obj;
    MPI_Comm comm = gpaw_comm->comm;
    int fcomm = MPI_Comm_c2f(comm);
    int err;
    elpa_set(handle, "mpi_comm_parent", fcomm, &err);
    return checkerr(err);
}

PyObject* pyelpa_constants(PyObject *self, PyObject *args)
{
    if(!PyArg_ParseTuple(args, ""))
        return NULL;
    return Py_BuildValue("iii",
                         ELPA_OK,
                         ELPA_SOLVER_1STAGE,
                         ELPA_SOLVER_2STAGE);
}


PyObject* pyelpa_diagonalize(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    PyArrayObject *A_obj, *C_obj, *eps_obj;
    PyObject *is_complex_obj;

    if (!PyArg_ParseTuple(args,
                          "OOOOO",
                          &handle_obj,
                          &A_obj,
                          &C_obj,
                          &eps_obj,
                          &is_complex_obj))
        return NULL;

    elpa_t handle = unpack_handle(handle_obj);

    void *a = (void*)PyArray_DATA(A_obj);
    double *ev = (double*)PyArray_DATA(eps_obj);
    void *q = (void*)PyArray_DATA(C_obj);

    int err;
    if (PyObject_IsTrue(is_complex_obj)) {
        elpa_eigenvectors_double_complex(handle, a, ev, q, &err);
    } else {
        elpa_eigenvectors_double(handle, a, ev, q, &err);
    }
    return checkerr(err);
}

PyObject* pyelpa_general_diagonalize(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    PyArrayObject *A_obj, *S_obj, *C_obj, *eps_obj;
    PyObject *is_complex_obj;
    int is_already_decomposed;

    if (!PyArg_ParseTuple(args,
                          "OOOOOiO",
                          &handle_obj,
                          &A_obj,
                          &S_obj,
                          &C_obj,
                          &eps_obj,
                          &is_already_decomposed,
                          &is_complex_obj))
        return NULL;

    elpa_t handle = unpack_handle(handle_obj);

    int err;
    double *ev = (double *)PyArray_DATA(eps_obj);
    void *a = (void *)PyArray_DATA(A_obj);
    void *b = (void *)PyArray_DATA(S_obj);
    void *q = (void *)PyArray_DATA(C_obj);

    if (PyObject_IsTrue(is_complex_obj)) {
        elpa_generalized_eigenvectors_dc(handle, a, b, ev, q,
                                         is_already_decomposed, &err);

    } else {
        elpa_generalized_eigenvectors_d(handle, a, b, ev, q,
                                        is_already_decomposed, &err);
    }
    return checkerr(err);
}

PyObject *pyelpa_deallocate(PyObject *self, PyObject *args)
{
    PyObject *handle_obj;
    if(!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }

    elpa_t handle = unpack_handle(handle_obj);

#ifdef ELPA_API_VERSION
    int err;
    elpa_deallocate(handle, &err);
    return checkerr(err);
#else
    // This function provides no error checking in older Elpas
    Py_RETURN_NONE;
#endif
}

#endif
