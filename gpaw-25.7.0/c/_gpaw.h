/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2009  CAMd
 *  Copyright (C) 2007-2010  CSC - IT Center for Science Ltd.
 *  Please see the accompanying LICENSE file for further information. */

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#include <numpy/arrayobject.h>

#ifdef PARALLEL
#include <mpi.h>
#endif
#ifndef GPAW_WITHOUT_LIBXC
#include <xc.h> // If this file is not found, install libxc https://gpaw.readthedocs.io/install.html#libxc-installation
#endif

#ifdef GPAW_HPM
PyObject* ibm_hpm_start(PyObject *self, PyObject *args);
PyObject* ibm_hpm_stop(PyObject *self, PyObject *args);
PyObject* ibm_mpi_start(PyObject *self);
PyObject* ibm_mpi_stop(PyObject *self);
#endif

#ifdef CRAYPAT
#include <pat_api.h>
PyObject* craypat_region_begin(PyObject *self, PyObject *args);
PyObject* craypat_region_end(PyObject *self, PyObject *args);
#endif

PyObject* evaluate_mpa_poly(PyObject *self, PyObject *args);
PyObject* pawexxvv(PyObject* self, PyObject* args);
PyObject* symmetrize(PyObject *self, PyObject *args);
PyObject* symmetrize_ft(PyObject *self, PyObject *args);
PyObject* symmetrize_wavefunction(PyObject *self, PyObject *args);
PyObject* symmetrize_return_index(PyObject *self, PyObject *args);
PyObject* symmetrize_with_index(PyObject *self, PyObject *args);
PyObject* map_k_points(PyObject *self, PyObject *args);
PyObject* GG_shuffle(PyObject *self, PyObject *args);
PyObject* tetrahedron_weight(PyObject *self, PyObject *args);
#ifndef GPAW_WITHOUT_BLAS
PyObject* mmm(PyObject *self, PyObject *args);
PyObject* rk(PyObject *self, PyObject *args);
PyObject* r2k(PyObject *self, PyObject *args);
#endif
PyObject* NewOperatorObject(PyObject *self, PyObject *args);
PyObject* NewWOperatorObject(PyObject *self, PyObject *args);
PyObject* NewSplineObject(PyObject *self, PyObject *args);
PyObject* NewTransformerObject(PyObject *self, PyObject *args);
PyObject* pc_potential(PyObject *self, PyObject *args);
PyObject* add_to_density(PyObject *self, PyObject *args);
PyObject* utilities_gaussian_wave(PyObject *self, PyObject *args);
PyObject* pack(PyObject *self, PyObject *args);
PyObject* unpack(PyObject *self, PyObject *args);
PyObject* unpack_complex(PyObject *self, PyObject *args);
PyObject* hartree(PyObject *self, PyObject *args);
PyObject* integrate_outwards(PyObject *self, PyObject *args);
PyObject* integrate_inwards(PyObject *self, PyObject *args);
PyObject* localize(PyObject *self, PyObject *args);
PyObject* NewXCFunctionalObject(PyObject *self, PyObject *args);
#ifndef GPAW_WITHOUT_LIBXC
PyObject* NewlxcXCFunctionalObject(PyObject *self, PyObject *args);
PyObject* lxcXCFuncNum(PyObject *self, PyObject *args);
#endif
PyObject* exterior_electron_density_region(PyObject *self, PyObject *args);
PyObject* plane_wave_grid(PyObject *self, PyObject *args);
PyObject* tci_overlap(PyObject *self, PyObject *args);
PyObject *pwlfc_expand(PyObject *self, PyObject *args);
PyObject *pwlfc_expand_old(PyObject *self, PyObject *args);
PyObject *pw_insert(PyObject *self, PyObject *args);
PyObject *pw_precond(PyObject *self, PyObject *args);
PyObject *fd_precond(PyObject *self, PyObject *args);
PyObject* vdw(PyObject *self, PyObject *args);
PyObject* vdw2(PyObject *self, PyObject *args);
PyObject* spherical_harmonics(PyObject *self, PyObject *args);
PyObject* spline_to_grid(PyObject *self, PyObject *args);
PyObject* NewLFCObject(PyObject *self, PyObject *args);
#ifdef PARALLEL
PyObject* globally_broadcast_bytes(PyObject *self, PyObject *args);
#endif
#if defined(GPAW_WITH_SL) && defined(PARALLEL)
PyObject* new_blacs_context(PyObject *self, PyObject *args);
PyObject* get_blacs_gridinfo(PyObject* self, PyObject *args);
PyObject* get_blacs_local_shape(PyObject* self, PyObject *args);
PyObject* blacs_destroy(PyObject *self, PyObject *args);
PyObject* scalapack_set(PyObject *self, PyObject *args);
PyObject* scalapack_redist(PyObject *self, PyObject *args);
PyObject* scalapack_diagonalize_dc(PyObject *self, PyObject *args);
PyObject* scalapack_diagonalize_ex(PyObject *self, PyObject *args);
#ifdef GPAW_WITH_INTEL_MKL
PyObject* mklscalapack_diagonalize_geev(PyObject *self, PyObject *args);
#endif
#ifdef GPAW_MR3
PyObject* scalapack_diagonalize_mr3(PyObject *self, PyObject *args);
#endif
PyObject* scalapack_general_diagonalize_dc(PyObject *self, PyObject *args);
PyObject* scalapack_general_diagonalize_ex(PyObject *self, PyObject *args);
#ifdef GPAW_MR3
PyObject* scalapack_general_diagonalize_mr3(PyObject *self, PyObject *args);
#endif
PyObject* scalapack_inverse_cholesky(PyObject *self, PyObject *args);
PyObject* scalapack_inverse(PyObject *self, PyObject *args);
PyObject* scalapack_solve(PyObject *self, PyObject *args);
PyObject* pblas_tran(PyObject *self, PyObject *args);
PyObject* pblas_gemm(PyObject *self, PyObject *args);
PyObject* pblas_hemm_symm(PyObject *self, PyObject *args);
PyObject* pblas_gemv(PyObject *self, PyObject *args);
PyObject* pblas_r2k(PyObject *self, PyObject *args);
PyObject* pblas_rk(PyObject *self, PyObject *args);
#if defined(GPAW_WITH_ELPA)
#include <elpa/elpa.h>
PyObject* pyelpa_init(PyObject *self, PyObject *args);
PyObject* pyelpa_uninit(PyObject *self, PyObject *args);
PyObject* pyelpa_version(PyObject *self, PyObject *args);
PyObject* pyelpa_allocate(PyObject *self, PyObject *args);
PyObject* pyelpa_set(PyObject *self, PyObject *args);
PyObject* pyelpa_set_comm(PyObject *self, PyObject *args);
PyObject* pyelpa_setup(PyObject *self, PyObject *args);
PyObject* pyelpa_diagonalize(PyObject *self, PyObject *args);
PyObject* pyelpa_general_diagonalize(PyObject *self, PyObject *args);
PyObject* pyelpa_constants(PyObject *self, PyObject *args);
PyObject* pyelpa_deallocate(PyObject *self, PyObject *args);
#endif // GPAW_WITH_ELPA
#endif // GPAW_WITH_SL and PARALLEL

#ifdef GPAW_WITH_FFTW
PyObject * FFTWPlan(PyObject *self, PyObject *args);
PyObject * FFTWExecute(PyObject *self, PyObject *args);
PyObject * FFTWDestroy(PyObject *self, PyObject *args);
#endif

// Threading
PyObject* get_num_threads(PyObject *self, PyObject *args);

#ifdef GPAW_PAPI
PyObject* papi_mem_info(PyObject *self, PyObject *args);
#endif

#ifdef GPAW_WITH_LIBVDWXC
PyObject* libvdwxc_create(PyObject *self, PyObject *args);
PyObject* libvdwxc_has(PyObject* self, PyObject *args);
PyObject* libvdwxc_init_serial(PyObject *self, PyObject *args);
PyObject* libvdwxc_calculate(PyObject *self, PyObject *args);
PyObject* libvdwxc_tostring(PyObject *self, PyObject *args);
PyObject* libvdwxc_free(PyObject* self, PyObject* args);
PyObject* libvdwxc_init_mpi(PyObject* self, PyObject* args);
PyObject* libvdwxc_init_pfft(PyObject* self, PyObject* args);
#endif // GPAW_WITH_LIBVDWXC

#ifdef GPAW_GITHASH
// For converting contents of a macro to a string, see
// https://en.wikipedia.org/wiki/C_preprocessor#Token_stringification
#define STR(s) #s
#define XSTR(s) STR(s)
PyObject* githash(PyObject* self, PyObject* args)
{
    return Py_BuildValue("s", XSTR(GPAW_GITHASH));
}
#undef XSTR
#undef STR
#endif // GPAW_GITHASH

// Holonomic constraints
PyObject* adjust_positions(PyObject *self, PyObject *args);
PyObject* adjust_momenta(PyObject *self, PyObject *args);
// TIP3P forces
PyObject* calculate_forces_H2O(PyObject *self, PyObject *args);


#ifdef GPAW_GPU
PyObject* gpaw_gpu_init(PyObject *self, PyObject *args);
PyObject* gpaw_gpu_delete(PyObject *self, PyObject *args);
PyObject* csign_gpu(PyObject *self, PyObject *args);
PyObject* scal_gpu(PyObject *self, PyObject *args);
PyObject* multi_scal_gpu(PyObject *self, PyObject *args);
PyObject* mmm_gpu(PyObject *self, PyObject *args);
PyObject* gemm_gpu(PyObject *self, PyObject *args);
PyObject* gemv_gpu(PyObject *self, PyObject *args);
PyObject* rk_gpu(PyObject *self, PyObject *args);
PyObject* axpy_gpu(PyObject *self, PyObject *args);
PyObject* multi_axpy_gpu(PyObject *self, PyObject *args);
PyObject* r2k_gpu(PyObject *self, PyObject *args);
PyObject* dotc_gpu(PyObject *self, PyObject *args);
PyObject* dotu_gpu(PyObject *self, PyObject *args);
PyObject* multi_dotu_gpu(PyObject *self, PyObject *args);
PyObject* multi_dotc_gpu(PyObject *self, PyObject *args);
PyObject* add_linear_field_gpu(PyObject *self, PyObject *args);
PyObject* elementwise_multiply_add_gpu(PyObject *self, PyObject *args);
PyObject* multi_elementwise_multiply_add_gpu(PyObject *self, PyObject *args);
PyObject* ax2py_gpu(PyObject *self, PyObject *args);
PyObject* multi_ax2py_gpu(PyObject *self, PyObject *args);
PyObject* axpbyz_gpu(PyObject *self, PyObject *args);
PyObject* axpbz_gpu(PyObject *self, PyObject *args);
PyObject* fill_gpu(PyObject *self, PyObject *args);
PyObject* pwlfc_expand_gpu(PyObject *self, PyObject *args);
PyObject* pw_insert_gpu(PyObject *self, PyObject *args);
PyObject* pw_norm_kinetic_gpu(PyObject *self, PyObject *args);
PyObject* pw_norm_gpu(PyObject *self, PyObject *args);

PyObject* pw_amend_insert_realwf_gpu(PyObject *self, PyObject *args);
PyObject* add_to_density_gpu(PyObject* self, PyObject* args);
PyObject* dH_aii_times_P_ani_gpu(PyObject* self, PyObject* args);
PyObject* evaluate_lda_gpu(PyObject* self, PyObject* args);
PyObject* evaluate_pbe_gpu(PyObject* self, PyObject* args);
PyObject* calculate_residual_gpu(PyObject* self, PyObject* args);

#endif // GPAW_GPU

#ifdef GPAW_WITH_MAGMA
#include "magma_gpaw.h"
PyObject* eigh_magma_dsyevd(PyObject* self, PyObject* args);
PyObject* eigh_magma_zheevd(PyObject* self, PyObject* args);
#ifdef GPAW_GPU
PyObject* eigh_magma_dsyevd_gpu(PyObject* self, PyObject* args);
PyObject* eigh_magma_zheevd_gpu(PyObject* self, PyObject* args);
#endif
#endif // GPAW_WITH_MAGMA

static PyMethodDef functions[] = {
    {"pawexxvv", pawexxvv, METH_VARARGS, 0},
    {"evaluate_mpa_poly", evaluate_mpa_poly, METH_VARARGS, 0},
    {"symmetrize", symmetrize, METH_VARARGS, 0},
    {"symmetrize_ft", symmetrize_ft, METH_VARARGS, 0},
    {"symmetrize_wavefunction", symmetrize_wavefunction, METH_VARARGS, 0},
    {"symmetrize_return_index", symmetrize_return_index, METH_VARARGS, 0},
    {"symmetrize_with_index", symmetrize_with_index, METH_VARARGS, 0},
    {"map_k_points", map_k_points, METH_VARARGS, 0},
    {"GG_shuffle", GG_shuffle, METH_VARARGS, 0},
    {"tetrahedron_weight", tetrahedron_weight, METH_VARARGS, 0},
#ifndef GPAW_WITHOUT_BLAS
    {"mmm", mmm, METH_VARARGS, 0},
    {"rk",  rk,  METH_VARARGS, 0},
    {"r2k", r2k, METH_VARARGS, 0},
#endif
    {"Operator", NewOperatorObject, METH_VARARGS, 0},
    {"WOperator", NewWOperatorObject, METH_VARARGS, 0},
    {"Spline", NewSplineObject, METH_VARARGS, 0},
    {"Transformer", NewTransformerObject, METH_VARARGS, 0},
    {"add_to_density", add_to_density, METH_VARARGS, 0},
    {"utilities_gaussian_wave", utilities_gaussian_wave, METH_VARARGS, 0},
    {"eed_region", exterior_electron_density_region, METH_VARARGS, 0},
    {"plane_wave_grid", plane_wave_grid, METH_VARARGS, 0},
    {"pwlfc_expand", pwlfc_expand, METH_VARARGS, 0},
    {"pwlfc_expand_old", pwlfc_expand_old, METH_VARARGS, 0},
    {"pw_insert", pw_insert, METH_VARARGS, 0},
    {"pw_precond", pw_precond, METH_VARARGS, 0},
    {"fd_precond", fd_precond, METH_VARARGS, 0},
    {"pack", pack, METH_VARARGS, 0},
    {"unpack", unpack, METH_VARARGS, 0},
    {"unpack_complex", unpack_complex,           METH_VARARGS, 0},
    {"hartree", hartree, METH_VARARGS, 0},
    {"integrate_outwards", integrate_outwards, METH_VARARGS, 0},
    {"integrate_inwards", integrate_inwards, METH_VARARGS, 0},
    {"localize", localize, METH_VARARGS, 0},
    {"XCFunctional", NewXCFunctionalObject, METH_VARARGS, 0},
#ifndef GPAW_WITHOUT_LIBXC
    {"lxcXCFunctional", NewlxcXCFunctionalObject, METH_VARARGS, 0},
    {"lxcXCFuncNum", lxcXCFuncNum, METH_VARARGS, 0},
#endif
    {"tci_overlap", tci_overlap, METH_VARARGS, 0},
    {"vdw", vdw, METH_VARARGS, 0},
    {"vdw2", vdw2, METH_VARARGS, 0},
    {"spherical_harmonics", spherical_harmonics, METH_VARARGS, 0},
    {"pc_potential", pc_potential, METH_VARARGS, 0},
    {"spline_to_grid", spline_to_grid, METH_VARARGS, 0},
    {"LFC", NewLFCObject, METH_VARARGS, 0},
#ifdef PARALLEL
    {"globally_broadcast_bytes", globally_broadcast_bytes, METH_VARARGS, 0},
#endif
    {"get_num_threads", get_num_threads, METH_VARARGS, 0},
#if defined(GPAW_WITH_SL) && defined(PARALLEL)
    {"new_blacs_context", new_blacs_context, METH_VARARGS, NULL},
    {"get_blacs_gridinfo", get_blacs_gridinfo, METH_VARARGS, NULL},
    {"get_blacs_local_shape", get_blacs_local_shape, METH_VARARGS, NULL},
    {"blacs_destroy", blacs_destroy, METH_VARARGS, 0},
    {"scalapack_set", scalapack_set, METH_VARARGS, 0},
#ifdef GPAW_WITH_INTEL_MKL
    {"mklscalapack_diagonalize_geev", mklscalapack_diagonalize_geev, METH_VARARGS, 0},
#endif
    {"scalapack_redist", scalapack_redist, METH_VARARGS, 0},
    {"scalapack_diagonalize_dc", scalapack_diagonalize_dc, METH_VARARGS, 0},
    {"scalapack_diagonalize_ex", scalapack_diagonalize_ex, METH_VARARGS, 0},
#ifdef GPAW_MR3
    {"scalapack_diagonalize_mr3", scalapack_diagonalize_mr3, METH_VARARGS, 0},
#endif // GPAW_MR3
    {"scalapack_general_diagonalize_dc",
     scalapack_general_diagonalize_dc, METH_VARARGS, 0},
    {"scalapack_general_diagonalize_ex",
     scalapack_general_diagonalize_ex, METH_VARARGS, 0},
#ifdef GPAW_MR3
    {"scalapack_general_diagonalize_mr3",
     scalapack_general_diagonalize_mr3, METH_VARARGS, 0},
#endif // GPAW_MR3
    {"scalapack_inverse_cholesky", scalapack_inverse_cholesky,
     METH_VARARGS, 0},
    {"scalapack_inverse", scalapack_inverse, METH_VARARGS, 0},
    {"scalapack_solve", scalapack_solve, METH_VARARGS, 0},
    {"pblas_tran", pblas_tran, METH_VARARGS, 0},
    {"pblas_gemm", pblas_gemm, METH_VARARGS, 0},
    {"pblas_hemm_symm", pblas_hemm_symm, METH_VARARGS, 0},
    {"pblas_gemv", pblas_gemv, METH_VARARGS, 0},
    {"pblas_r2k", pblas_r2k, METH_VARARGS, 0},
    {"pblas_rk", pblas_rk, METH_VARARGS, 0},
#if defined(GPAW_WITH_ELPA)
    {"pyelpa_init", pyelpa_init, METH_VARARGS, 0},
    {"pyelpa_uninit", pyelpa_uninit, METH_VARARGS, 0},
    {"pyelpa_version", pyelpa_version, METH_VARARGS, 0},
    {"pyelpa_allocate", pyelpa_allocate, METH_VARARGS, 0},
    {"pyelpa_set", pyelpa_set, METH_VARARGS, 0},
    {"pyelpa_setup", pyelpa_setup, METH_VARARGS, 0},
    {"pyelpa_set_comm", pyelpa_set_comm, METH_VARARGS, 0},
    {"pyelpa_diagonalize", pyelpa_diagonalize, METH_VARARGS, 0},
    {"pyelpa_general_diagonalize", pyelpa_general_diagonalize, METH_VARARGS, 0},
    {"pyelpa_constants", pyelpa_constants, METH_VARARGS, 0},
    {"pyelpa_deallocate", pyelpa_deallocate, METH_VARARGS, 0},
#endif // GPAW_WITH_ELPA
#endif // GPAW_WITH_SL && PARALLEL
#ifdef GPAW_WITH_FFTW
    {"FFTWPlan", FFTWPlan, METH_VARARGS, 0},
    {"FFTWExecute", FFTWExecute, METH_VARARGS, 0},
    {"FFTWDestroy", FFTWDestroy, METH_VARARGS, 0},
#endif
#ifdef GPAW_HPM
    {"hpm_start", ibm_hpm_start, METH_VARARGS, 0},
    {"hpm_stop", ibm_hpm_stop, METH_VARARGS, 0},
    {"mpi_start", (PyCFunction) ibm_mpi_start, METH_NOARGS, 0},
    {"mpi_stop", (PyCFunction) ibm_mpi_stop, METH_NOARGS, 0},
#endif // GPAW_HPM
#ifdef CRAYPAT
    {"craypat_region_begin", craypat_region_begin, METH_VARARGS, 0},
    {"craypat_region_end", craypat_region_end, METH_VARARGS, 0},
#endif // CRAYPAT
#ifdef GPAW_PAPI
    {"papi_mem_info", papi_mem_info, METH_VARARGS, 0},
#endif // GPAW_PAPI
#ifdef GPAW_WITH_LIBVDWXC
    {"libvdwxc_create", libvdwxc_create, METH_VARARGS, 0},
    {"libvdwxc_has", libvdwxc_has, METH_VARARGS, 0},
    {"libvdwxc_init_serial", libvdwxc_init_serial, METH_VARARGS, 0},
    {"libvdwxc_calculate", libvdwxc_calculate, METH_VARARGS, 0},
    {"libvdwxc_tostring", libvdwxc_tostring, METH_VARARGS, 0},
    {"libvdwxc_free", libvdwxc_free, METH_VARARGS, 0},
    {"libvdwxc_init_mpi", libvdwxc_init_mpi, METH_VARARGS, 0},
    {"libvdwxc_init_pfft", libvdwxc_init_pfft, METH_VARARGS, 0},
#endif // GPAW_WITH_LIBVDWXC
    {"adjust_positions", adjust_positions, METH_VARARGS, 0},
    {"adjust_momenta", adjust_momenta, METH_VARARGS, 0},
    {"calculate_forces_H2O", calculate_forces_H2O, METH_VARARGS, 0},
#ifdef GPAW_GITHASH
    {"githash", githash, METH_VARARGS, 0},
#endif // GPAW_GITHASH
#ifdef GPAW_GPU
    {"gpaw_gpu_init", gpaw_gpu_init, METH_VARARGS, 0},
    {"gpaw_gpu_delete", gpaw_gpu_delete, METH_VARARGS, 0},
    {"csign_gpu", csign_gpu, METH_VARARGS, 0},
    {"scal_gpu", scal_gpu, METH_VARARGS, 0},
    {"multi_scal_gpu", multi_scal_gpu, METH_VARARGS, 0},
    {"mmm_gpu", mmm_gpu, METH_VARARGS, 0},
    {"gemm_gpu", gemm_gpu, METH_VARARGS, 0},
    {"gemv_gpu", gemv_gpu, METH_VARARGS, 0},
    {"axpy_gpu", axpy_gpu, METH_VARARGS, 0},
    {"multi_axpy_gpu", multi_axpy_gpu, METH_VARARGS, 0},
    {"rk_gpu",  rk_gpu,  METH_VARARGS, 0},
    {"r2k_gpu", r2k_gpu, METH_VARARGS, 0},
    {"dotc_gpu", dotc_gpu, METH_VARARGS, 0},
    {"dotu_gpu", dotu_gpu, METH_VARARGS, 0},
    {"multi_dotu_gpu", multi_dotu_gpu, METH_VARARGS, 0},
    {"multi_dotc_gpu", multi_dotc_gpu, METH_VARARGS, 0},
    {"add_linear_field_gpu", add_linear_field_gpu, METH_VARARGS, 0},
    {"elementwise_multiply_add_gpu", elementwise_multiply_add_gpu,
        METH_VARARGS, 0},
    {"multi_elementwise_multiply_add_gpu", multi_elementwise_multiply_add_gpu,
        METH_VARARGS, 0},
    {"ax2py_gpu", ax2py_gpu, METH_VARARGS, 0},
    {"multi_ax2py_gpu", multi_ax2py_gpu, METH_VARARGS, 0},
    {"axpbyz_gpu", axpbyz_gpu, METH_VARARGS, 0},
    {"axpbz_gpu", axpbz_gpu, METH_VARARGS, 0},
    {"fill_gpu", fill_gpu, METH_VARARGS, 0},
    {"pwlfc_expand_gpu", pwlfc_expand_gpu, METH_VARARGS, 0},
    {"pw_insert_gpu", pw_insert_gpu, METH_VARARGS, 0},
    {"pw_norm_kinetic_gpu", pw_norm_kinetic_gpu, METH_VARARGS, 0},
    {"pw_norm_gpu", pw_norm_gpu, METH_VARARGS, 0},
    {"pw_amend_insert_realwf_gpu", pw_amend_insert_realwf_gpu, METH_VARARGS, 0},
    {"add_to_density_gpu", add_to_density_gpu, METH_VARARGS, 0},
    {"dH_aii_times_P_ani_gpu", dH_aii_times_P_ani_gpu, METH_VARARGS, 0},
    {"evaluate_lda_gpu", evaluate_lda_gpu, METH_VARARGS, 0},
    {"evaluate_pbe_gpu", evaluate_pbe_gpu, METH_VARARGS, 0},
    {"calculate_residuals_gpu", calculate_residual_gpu, METH_VARARGS, 0},

#endif // GPAW_GPU

#ifdef GPAW_WITH_MAGMA
{"eigh_magma_dsyevd", eigh_magma_dsyevd, METH_VARARGS, 0},
{"eigh_magma_zheevd", eigh_magma_zheevd, METH_VARARGS, 0},
#ifdef GPAW_GPU
{"eigh_magma_dsyevd_gpu", eigh_magma_dsyevd_gpu, METH_VARARGS, 0},
{"eigh_magma_zheevd_gpu", eigh_magma_zheevd_gpu, METH_VARARGS, 0},
#endif
#endif // GPAW_WITH_MAGMA


    {0, 0, 0, 0}
};

#ifdef PARALLEL
extern PyTypeObject MPIType;
extern PyTypeObject GPAW_MPI_Request_type;
#endif

extern PyTypeObject LFCType;
extern PyTypeObject OperatorType;
extern PyTypeObject WOperatorType;
extern PyTypeObject SplineType;
extern PyTypeObject TransformerType;
extern PyTypeObject XCFunctionalType;
#ifndef GPAW_WITHOUT_LIBXC
extern PyTypeObject lxcXCFunctionalType;
#endif


static void gpaw_module_cleanup(void *m)
{
#ifdef GPAW_WITH_MAGMA
    // Assuming GPAW calls magma_init() during module startup
    magma_finalize();
#endif
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_gpaw",
    "C-extension for GPAW",
    -1,
    functions,
    NULL,
    NULL,
    NULL,
    gpaw_module_cleanup
};

static PyObject* moduleinit(void)
{
#ifdef PARALLEL
    if (PyType_Ready(&MPIType) < 0)
        return NULL;
    if (PyType_Ready(&GPAW_MPI_Request_type) < 0)
        return NULL;
#endif

    if (PyType_Ready(&LFCType) < 0)
        return NULL;
    if (PyType_Ready(&OperatorType) < 0)
        return NULL;
    if (PyType_Ready(&WOperatorType) < 0)
        return NULL;
    if (PyType_Ready(&SplineType) < 0)
        return NULL;
    if (PyType_Ready(&TransformerType) < 0)
        return NULL;
    if (PyType_Ready(&XCFunctionalType) < 0)
        return NULL;
#ifndef GPAW_WITHOUT_LIBXC
    if (PyType_Ready(&lxcXCFunctionalType) < 0)
        return NULL;
#endif

    PyObject* m = PyModule_Create(&moduledef);

    if (m == NULL)
        return NULL;

#ifdef Py_GIL_DISABLED
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

#ifdef PARALLEL
    Py_INCREF(&MPIType);
    Py_INCREF(&GPAW_MPI_Request_type);
    PyModule_AddObject(m, "Communicator", (PyObject *)&MPIType);
#endif

#ifndef GPAW_WITHOUT_LIBXC
# if XC_MAJOR_VERSION >= 3
    PyObject_SetAttrString(m,
                           "libxc_version",
                           PyUnicode_FromString(xc_version_string()));
# endif
#endif
#ifdef GPAW_GPU
    PyObject_SetAttrString(m, "GPU_ENABLED", Py_True);
#else
    PyObject_SetAttrString(m, "GPU_ENABLED", Py_False);
#endif
#ifdef GPAW_GPU_AWARE_MPI
    PyObject_SetAttrString(m, "gpu_aware_mpi", Py_True);
#else
    PyObject_SetAttrString(m, "gpu_aware_mpi", Py_False);
#endif
#ifdef _OPENMP
    PyObject_SetAttrString(m, "have_openmp", Py_True);
#else
    PyObject_SetAttrString(m, "have_openmp", Py_False);
#endif

#ifdef GPAW_WITH_MAGMA
    PyObject_SetAttrString(m, "have_magma", Py_True);

    // MAGMA needs to be globally initialized, but keeps track of accumulated
    // magma_init() calls. So it's safe to call it inside GPAW, even if other
    // libs are also doing it.

    // FIXME: Where should GPAW call magma_init()?
    // Should not be in GPU-specific init because magma can work without GPU too.
    // However it needs to come AFTER cudaSetValidDevices and cudaSetDeviceFlags.
    // Calling it here could become a problem if Python-side GPU init does more than setDevice(...)
    magma_init();

#else
    PyObject_SetAttrString(m, "have_magma", Py_False);
#endif
    // Version number of C-code.  Keep in sync with gpaw/_broadcast_imports.py
    PyObject_SetAttrString(m, "version", PyLong_FromLong(10));

    Py_INCREF(&LFCType);
    Py_INCREF(&OperatorType);
    Py_INCREF(&WOperatorType);
    Py_INCREF(&SplineType);
    Py_INCREF(&TransformerType);
    Py_INCREF(&XCFunctionalType);
#ifndef GPAW_WITHOUT_LIBXC
    Py_INCREF(&lxcXCFunctionalType);
#endif
    return m;
}
