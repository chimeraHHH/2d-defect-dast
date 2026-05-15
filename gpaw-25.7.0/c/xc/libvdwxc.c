#ifdef GPAW_WITH_LIBVDWXC
#include "../extensions.h"

#ifdef PARALLEL
#include <mpi.h>
#include "../mympi.h"
#include <vdwxc_mpi.h>
#else
#include <vdwxc.h>
#endif

// Our heinous plan is to abuse a numpy array so that it will contain a pointer to the vdwxc_data.
// This is because PyCapsules are not there until Python 3.1/2.7.
// This function takes an array and returns the pointer it so outrageously contains.
vdwxc_data* unpack_vdwxc_pointer(PyObject* vdwxc_obj)
{
    vdwxc_data* vdw = (vdwxc_data *)PyArray_DATA((PyArrayObject *)vdwxc_obj);
    return vdw;
}

PyObject* libvdwxc_has(PyObject* self, PyObject* args)
{
    char* name;
    if(!PyArg_ParseTuple(args, "s", &name)) {
            return NULL;
    }
    int val;
    if(strcmp("mpi", name) == 0) {
        val = vdwxc_has_mpi();
    } else if(strcmp("pfft", name) == 0) {
        val = vdwxc_has_pfft();
    } else if(strcmp("spin", name) == 0) {
#ifdef VDWXC_HAS_SPIN
        val = true;
#else
        val = false;
#endif
    } else if(strcmp("stress", name) == 0) {
#ifdef VDWXC_HAS_STRESS
        val = true;
#else
        val = false;
#endif
    } else {
        return NULL;
    }
    PyObject* pyval = val ? Py_True : Py_False;
    Py_INCREF(pyval);
    return pyval;
}

PyObject* libvdwxc_create(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* vdwxc_obj;
    int vdwxc_code;
    int nspins;
    int Nx, Ny, Nz;
    double C00, C10, C20, C01, C11, C21, C02, C12, C22;

    if(!PyArg_ParseTuple(args, "Oii(iii)(ddddddddd)",
                         &vdwxc_obj,
                         &vdwxc_code, // functional identifier
                         &nspins,
                         &Nx, &Ny, &Nz, // number of grid points
                         &C00, &C10, &C20, // 3x3 cell
                         &C01, &C11,&C21,
                         &C02, &C12, &C22)) {
        return NULL;
    }

    vdwxc_data vdw;
    if(nspins == 1) {
        vdw = vdwxc_new(vdwxc_code);
    } else if(nspins == 2) {
#ifdef VDWXC_HAS_SPIN
        vdw = vdwxc_new_spin(vdwxc_code);
#else
        PyErr_SetString(PyExc_ImportError, "this version of libvdwxc has no spin support");
        return NULL;
#endif
    } else {
        PyErr_SetString(PyExc_ValueError, "nspins must be 1 or 2");
        return NULL;
    }
    vdwxc_data* vdwxc_ptr = unpack_vdwxc_pointer(vdwxc_obj);
    vdwxc_ptr[0] = vdw;
    vdwxc_set_unit_cell(vdw, Nx, Ny, Nz, C00, C10, C20, C01, C11, C21, C02, C12, C22);
    Py_RETURN_NONE;
}

PyObject* libvdwxc_init_serial(PyObject* self, PyObject* args)
{
    PyObject* vdwxc_obj;
    if(!PyArg_ParseTuple(args, "O", &vdwxc_obj)) {
        return NULL;
    }
    vdwxc_data* vdw = unpack_vdwxc_pointer(vdwxc_obj);
    vdwxc_init_serial(*vdw);
    Py_RETURN_NONE;
}

PyObject* libvdwxc_calculate(PyObject* self, PyObject* args)
{
    PyObject *vdwxc_obj;
    PyArrayObject *stress_obj, *rho_obj, *sigma_obj, *dedn_obj, *dedsigma_obj;
    if(!PyArg_ParseTuple(args, "OOOOOO",
                         &vdwxc_obj, &stress_obj, &rho_obj, &sigma_obj,
                         &dedn_obj, &dedsigma_obj)) {
        return NULL;
    }
    double *stress_vv;
    if ((PyObject *)stress_obj == Py_None) {
        stress_vv = NULL;
    } else {
        stress_vv = (double*)PyArray_DATA(stress_obj);
    }
    vdwxc_data* vdw = unpack_vdwxc_pointer(vdwxc_obj);
    int nspins = PyArray_DIM(rho_obj, 0);
    double energy;
    if (nspins == 1) {
        double* rho_g = (double*)PyArray_DATA(rho_obj);
        double* sigma_g = (double*)PyArray_DATA(sigma_obj);
        double* dedn_g = (double*)PyArray_DATA(dedn_obj);
        double* dedsigma_g = (double*)PyArray_DATA(dedsigma_obj);
        if (stress_vv == NULL) {
            energy = vdwxc_calculate(*vdw, rho_g, sigma_g, dedn_g, dedsigma_g);
        } else {
#ifdef VDWXC_HAS_STRESS
            energy = vdwxc_stress(*vdw, stress_vv, rho_g, sigma_g,
                                  dedn_g, dedsigma_g);
#else
            PyErr_SetString(PyExc_ValueError,
                            "This libvdwxc version does not implement stress.");
            return NULL;
#endif
        }
    } else if (nspins == 2) {
        // We actually only need two sigmas/dedsigmas.
        // The third one came along because that's what usually happens,
        // but we could save it entirely.
        assert(PyArray_DIM(sigma_obj, 0) == 3);
        assert(PyArray_DIM(dedn_obj, 0) == 2);
        assert(PyArray_DIM(dedsigma_obj, 0) == 3);
        double *rhoa_g = (double*)PyArray_GETPTR1(rho_obj, 0);
        double *rhob_g = (double*)PyArray_GETPTR1(rho_obj, 1);
        double *sigmaa_g = (double*)PyArray_GETPTR1(sigma_obj, 0);
        double *sigmab_g = (double*)PyArray_GETPTR1(sigma_obj, 2);
        double *dedna_g = (double*)PyArray_GETPTR1(dedn_obj, 0);
        double *dednb_g = (double*)PyArray_GETPTR1(dedn_obj, 1);
        double *dedsigmaa_g = (double*)PyArray_GETPTR1(dedsigma_obj, 0);
        double *dedsigmab_g = (double*)PyArray_GETPTR1(dedsigma_obj, 2);
#ifdef VDWXC_HAS_SPIN
        if (stress_vv == NULL) {
            energy = vdwxc_calculate_spin(*vdw, rhoa_g, rhob_g, sigmaa_g,
                                          sigmab_g, dedna_g, dednb_g,
                                          dedsigmaa_g, dedsigmab_g);
        } else {
#ifdef VDWXC_HAS_STRESS
            energy = vdwxc_stress_spin(*vdw, stress_vv, rhoa_g, rhob_g,
                                       sigmaa_g, sigmab_g, dedna_g, dednb_g,
                                       dedsigmaa_g, dedsigmab_g);
#else
            PyErr_SetString(PyExc_ValueError,
                            "This libvdwxc version does not implement stress.");
            return NULL;
#endif
        }
#else
        PyErr_SetString(PyExc_ValueError,
                        "This libvdwxc version does not implement spin.");
        return NULL;
#endif
    } else {
        PyErr_SetString(PyExc_ValueError, "Expected 1 or 2 spins");
        return NULL;
    }
    return Py_BuildValue("d", energy);
}

PyObject* libvdwxc_tostring(PyObject* self, PyObject* args)
{
    PyObject *vdwxc_obj;
    if(!PyArg_ParseTuple(args, "O", &vdwxc_obj)) {
        return NULL;
    }
    vdwxc_data* vdw = unpack_vdwxc_pointer(vdwxc_obj);
    int maxlen = 80 * 200; // up to a few hundred lines
    char str[maxlen];
    vdwxc_tostring(*vdw, maxlen, str);
    return Py_BuildValue("s", str);
}

PyObject* libvdwxc_free(PyObject* self, PyObject* args)
{
    PyObject* vdwxc_obj;
    if(!PyArg_ParseTuple(args, "O", &vdwxc_obj)) {
        return NULL;
    }
    vdwxc_data* vdw = unpack_vdwxc_pointer(vdwxc_obj);
    vdwxc_finalize(vdw);
    Py_RETURN_NONE;
}

#ifdef PARALLEL
MPI_Comm unpack_gpaw_comm(PyObject* gpaw_mpi_obj)
{
    MPIObject* gpaw_comm = (MPIObject *)gpaw_mpi_obj;
    return gpaw_comm->comm;
}
#endif

PyObject* error_parallel_support(void)
{
    // Not a true import error, but pretty close.
#ifndef PARALLEL
    PyErr_SetString(PyExc_ImportError,
                    "GPAW not compiled in parallel");
#endif
#ifndef VDWXC_HAS_MPI
    PyErr_SetString(PyExc_ImportError,
                    "libvdwxc not compiled in parallel.  Recompile libvdwxc with --with-mpi");
#endif
    return NULL;
}

PyObject* libvdwxc_init_mpi(PyObject* self, PyObject* args)
{
    PyObject* vdwxc_obj;
    PyObject* gpaw_comm_obj;
    if(!PyArg_ParseTuple(args, "OO", &vdwxc_obj, &gpaw_comm_obj)) {
        return NULL;
    }

    if(!vdwxc_has_mpi()) {
        PyErr_SetString(PyExc_ImportError, "libvdwxc not compiled with MPI.");
        return NULL;
    }

#if defined(PARALLEL) && defined(VDWXC_HAS_MPI)
    vdwxc_data* vdw = unpack_vdwxc_pointer(vdwxc_obj);
    MPI_Comm comm = unpack_gpaw_comm(gpaw_comm_obj);
    vdwxc_init_mpi(*vdw, comm);
    Py_RETURN_NONE;
#else
    return error_parallel_support();
#endif
}

PyObject* libvdwxc_init_pfft(PyObject* self, PyObject* args)
{
    PyObject* vdwxc_obj;
    PyObject* gpaw_comm_obj;
    int nproc1, nproc2;
    if(!PyArg_ParseTuple(args, "OOii", &vdwxc_obj, &gpaw_comm_obj, &nproc1, &nproc2)) {
        return NULL;
    }

    if(!vdwxc_has_pfft()) {
        PyErr_SetString(PyExc_ImportError, "libvdwxc not compiled with PFFT.");
        return NULL;
    }

#if defined(PARALLEL)
    vdwxc_data* vdw = unpack_vdwxc_pointer(vdwxc_obj);
    MPI_Comm comm = unpack_gpaw_comm(gpaw_comm_obj);
    vdwxc_init_pfft(*vdw, comm, nproc1, nproc2);
    Py_RETURN_NONE;
#else
    return error_parallel_support();
#endif
}

#endif // gpaw_with_libvdwxc
