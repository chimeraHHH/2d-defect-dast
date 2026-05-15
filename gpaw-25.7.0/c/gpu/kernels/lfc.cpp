#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>

#include </usr/include/complex.h>
#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include "../../lfc.h"
#include "../gpu.h"
#include "../gpu-complex.h"

#ifndef GPU_USE_COMPLEX

#define INLINE inline
static INLINE void* gpaw_malloc(int n)
{
    void *p = malloc(n);
    assert(p != NULL);
    return p;
}
#define GPAW_MALLOC(T, n) (T*)(gpaw_malloc((n) * sizeof(T)))

#define BLOCK_Y 16

#endif

#include "lfc-reduce.cpp"


__global__ void Zgpu(add_kernel)(Tgpu *a_G, const Tgpu *c_M, int *G_B1,
                                 int *G_B2, LFVolume_gpu **volume_i,
                                 int *A_gm_i, int *ni, int nimax, int na_G,
                                 int nM, gpuDoubleComplex *phase_i,
                                 int max_k, int q, int nB_gpu)
{
    int G = threadIdx.x;
    int B = blockIdx.x * blockDim.y + threadIdx.y;
    if (B >= nB_gpu)
        return;
    int x = blockIdx.y;

    int nii, Gb, Ga, nG;
    LFVolume_gpu* v;
    double* A_gm;
    const Tgpu* c_Mt;
    int nm;
    int len;

    nii = ni[B];
    Ga = G_B1[B];
    Gb = G_B2[B];
    nG = Gb - Ga;
    a_G += Ga + na_G * x;
    c_M += nM * x;
    Tgpu av = MAKED(0);
    if (G < nG) {
        for (int i=0; i < nii; i++) {
            Tgpu avv;
            v = volume_i[B + i * nB_gpu];
            A_gm = v->A_gm + A_gm_i[B + i * nB_gpu] + G;
            nm = v->nm;
            len = v->len_A_gm;
            c_Mt = c_M + v->M;

            avv = MULTD(c_Mt[0], A_gm[0]);
            for (int m=1; m < nm; m += 2) {
                A_gm += len;
                IADD(avv, MULTD(c_Mt[m], A_gm[0]));
                A_gm += len;
                IADD(avv, MULTD(c_Mt[m+1], A_gm[0]));
            }
#ifdef GPU_USE_COMPLEX
            avv = MULTT(avv,
                        gpuConj(phase_i[max_k * nimax * B + q * nimax + i]));
#endif
            IADD(av, avv);
        }
        IADD(a_G[G] , av);
    }
}

#ifndef GPU_USE_COMPLEX
#define GPU_USE_COMPLEX
#include "lfc.cpp"

extern "C"
void lfc_dealloc_gpu(LFCObject *self)
{
    if (self->use_gpu) {
        for (int W=0; W < self->nW; W++) {
            LFVolume_gpu* volume_gpu = &self->volume_W_gpu_host[W];
            gpuFree(volume_gpu->A_gm);
            gpuFree(volume_gpu->GB1);
            gpuFree(volume_gpu->nGBcum);
            gpuFree(volume_gpu->phase_k);
        }
        free(self->volume_W_gpu_host);
        gpuFree(self->volume_W_gpu);
        gpuFree(self->G_B1_gpu);
        gpuFree(self->G_B2_gpu);
        gpuFree(self->volume_i_gpu);
        gpuFree(self->A_gm_i_gpu);
        gpuFree(self->volume_WMi_gpu);
        gpuFree(self->WMi_gpu);
        gpuFree(self->ni_gpu);
        gpuCheckLastError();
    }
}

extern "C"
void *transp(void *matrix, int rows, int cols, size_t item_size)
{
#define ALIGNMENT 16    /* power of 2 >= minimum array boundary alignment;
                       maybe unnecessary but machine dependent */

    char *cursor;
    char carry[ALIGNMENT];
    size_t block_size, remaining_size;
    int nadir, lag, orbit, ents;

    if (rows == 1 || cols == 1)
        return matrix;

    ents = rows * cols;
    cursor = (char *) matrix;
    remaining_size = item_size;
    while ((block_size = ALIGNMENT < remaining_size ? ALIGNMENT
                                                    : remaining_size)) {
        nadir = 1;
        /* first and last entries are always fixed points so aren't
           visited */
        while (nadir + 1 < ents) {
            memcpy(carry, &cursor[(lag = nadir) * item_size], block_size);
            /* follow a complete cycle */
            while ((orbit = lag / rows + cols * (lag % rows)) > nadir) {
                memcpy(&cursor[lag * item_size],
                       &cursor[orbit * item_size], block_size);
                lag = orbit;
            }
            memcpy(&cursor[lag * item_size], carry, block_size);
            orbit = nadir++;
            /* find the next unvisited index by an exhaustive search */
            while (orbit < nadir && nadir + 1 < ents) {
                orbit = nadir;
                while ((orbit = orbit / rows + cols * (orbit % rows))
                       > nadir);
                if (orbit < nadir)
                    nadir++;
            }
        }
        cursor += block_size;
        remaining_size -= block_size;
    }
    return matrix;
}

extern "C"
PyObject * NewLFCObject_gpu(LFCObject *self, PyObject *args)
{
    PyObject* A_Wgm_obj;
    const PyArrayObject* M_W_obj;
    const PyArrayObject* G_B_obj;
    const PyArrayObject* W_B_obj;
    double dv;
    PyArrayObject* phase_kW_obj;
    int use_gpu = 1;

    if (!PyArg_ParseTuple(args, "OOOOdO|iO",
                          &A_Wgm_obj, &M_W_obj, &G_B_obj, &W_B_obj, &dv,
                          &phase_kW_obj, &use_gpu))
        return NULL;

    if (!use_gpu)
        return (PyObject*) self;

    int nimax = self->nimax;
    int max_k = 0;
    int *GB2s[self->nW];

    LFVolume_gpu* volume_W_gpu;
    volume_W_gpu = GPAW_MALLOC(LFVolume_gpu, self->nW);

    if (self->bloch_boundary_conditions) {
        max_k = PyArray_DIMS(phase_kW_obj)[0];
    }
    self->max_k = max_k;
    self->max_len_A_gm = 0;
    self->max_nG = 0;
    for (int W=0; W < self->nW; W++) {
        LFVolume_gpu* v_gpu = &volume_W_gpu[W];
        LFVolume* v = &self->volume_W[W];

        PyArrayObject* A_gm_obj =
            (PyArrayObject*) PyList_GetItem(A_Wgm_obj, W);

        double *work_A_gm = GPAW_MALLOC(double, self->ngm_W[W]);
        gpuMalloc(&(v_gpu->A_gm), sizeof(double) * self->ngm_W[W]);

        memcpy(work_A_gm, v->A_gm, sizeof(double) * self->ngm_W[W]);
        transp(work_A_gm, PyArray_DIMS(A_gm_obj)[0],
               PyArray_DIMS(A_gm_obj)[1], sizeof(double));
        gpuMemcpy(v_gpu->A_gm, work_A_gm, sizeof(double) * self->ngm_W[W],
                  gpuMemcpyHostToDevice);
        free(work_A_gm);

        v_gpu->nm = v->nm;
        v_gpu->M = v->M;
        v_gpu->W = v->W;
        v_gpu->len_A_gm = 0;
        v_gpu->GB1 = GPAW_MALLOC(int, self->ngm_W[W]);
        GB2s[W] = GPAW_MALLOC(int, self->ngm_W[W]);
        v_gpu->nGBcum = GPAW_MALLOC(int, self->ngm_W[W] + 1);
        v_gpu->nB = 0;
        v_gpu->phase_k = NULL;
    }

    gpuMalloc(&(self->volume_W_gpu), sizeof(LFVolume_gpu) * self->nW);

    int* i_W = self->i_W;
    LFVolume_gpu** volume_i = GPAW_MALLOC(LFVolume_gpu*, nimax);
    int Ga = 0;
    int ni = 0;
    LFVolume_gpu **volume_i_gpu = GPAW_MALLOC(LFVolume_gpu*,
                                              self->nB*nimax);
    int *A_gm_i_gpu = GPAW_MALLOC(int, self->nB*nimax);
    int *ni_gpu = GPAW_MALLOC(int, self->nB);
    int *G_B1_gpu = GPAW_MALLOC(int, self->nB);
    int *G_B2_gpu = GPAW_MALLOC(int, self->nB);

    gpuDoubleComplex *phase_i_gpu = NULL;
    gpuDoubleComplex *phase_i = NULL;

    if (self->bloch_boundary_conditions) {
        phase_i_gpu = GPAW_MALLOC(gpuDoubleComplex,
                                  max_k * self->nB * nimax);
        phase_i = GPAW_MALLOC(gpuDoubleComplex, max_k * nimax);
    }

    int nB_gpu=0;

    for (int B=0; B < self->nB; B++) {
        int Gb = self->G_B[B];
        int nG = Gb - Ga;

        if ((nG > 0) && (ni > 0)) {
            for (int i=0; i < ni; i++) {
                LFVolume_gpu* v = volume_i[i];
                volume_i_gpu[nB_gpu * nimax + i] = self->volume_W_gpu
                                                 + (v - volume_W_gpu);
                A_gm_i_gpu[nB_gpu * nimax + i] = v->len_A_gm;
                if (self->bloch_boundary_conditions) {
                    for (int kk=0; kk < max_k; kk++){
                        phase_i_gpu[i + nB_gpu * nimax * max_k
                                    + kk * nimax]
                            = phase_i[i + kk * nimax];
                    }
                }
                v->len_A_gm += nG;
                int *GB2 = GB2s[v - volume_W_gpu];
                if ((v->nB > 0) && (GB2[v->nB - 1] == Ga)) {
                    GB2[v->nB - 1] = Gb;
                    v->nGBcum[v->nB] += nG;
                } else {
                    v->GB1[v->nB] = Ga;
                    GB2[v->nB] = Gb;

                    if (v->nB == 0)
                        v->nGBcum[v->nB] = 0;
                    v->nGBcum[v->nB + 1] = nG + v->nGBcum[v->nB];

                    v->nB++;
                }
            }
            self->max_nG = MAX(self->max_nG, nG);
            G_B1_gpu[nB_gpu] = Ga;
            G_B2_gpu[nB_gpu] = Gb;
            ni_gpu[nB_gpu] = ni;
            nB_gpu++;
        }
        int Wnew = self->W_B[B];
        if (Wnew >= 0) {
            /* Entering new sphere: */
            volume_i[ni] = &volume_W_gpu[Wnew];
            if (self->bloch_boundary_conditions) {
                for (int i=0; i < max_k; i++) {
                    phase_i[ni + i * nimax].x = creal(
                            self->phase_kW[Wnew + i * self->nW]);
                    phase_i[ni + i * nimax].y = cimag(
                            self->phase_kW[Wnew + i * self->nW]);
                }
            }
            i_W[Wnew] = ni;
            ni++;
        } else {
            /* Leaving sphere: */
            int Wold = -1 - Wnew;
            int iold = i_W[Wold];
            volume_W_gpu[Wold].len_A_gm = volume_i[iold]->len_A_gm;
            ni--;
            volume_i[iold] = volume_i[ni];
            if (self->bloch_boundary_conditions) {
                for (int i=0; i < max_k; i++) {
                    phase_i[iold + i * nimax] = phase_i[ni + i * nimax];
                }
            }
            int Wlast = volume_i[iold]->W;
            i_W[Wlast] = iold;
        }
        Ga = Gb;
    }
    for (int W=0; W < self->nW; W++) {
        LFVolume_gpu* v = &volume_W_gpu[W];
        self->max_len_A_gm = MAX(self->max_len_A_gm, v->len_A_gm);

        int *GB_gpu;
        gpuMalloc(&(GB_gpu), sizeof(int) * v->nB);
        gpuMemcpy(GB_gpu, v->GB1, sizeof(int) * v->nB, gpuMemcpyHostToDevice);
        free(v->GB1);
        v->GB1 = GB_gpu;
        free(GB2s[W]);

        gpuMalloc(&(GB_gpu), sizeof(int) * (v->nB + 1));
        gpuMemcpy(GB_gpu, v->nGBcum, sizeof(int) * (v->nB + 1),
                  gpuMemcpyHostToDevice);
        free(v->nGBcum);
        v->nGBcum = GB_gpu;

        if (self->bloch_boundary_conditions) {
            gpuDoubleComplex phase_k[max_k];
            for (int q=0; q < max_k; q++) {
                phase_k[q].x = creal(self->phase_kW[self->nW*q+W]);
                phase_k[q].y = cimag(self->phase_kW[self->nW*q+W]);
            }
            gpuMalloc(&(v->phase_k), sizeof(gpuDoubleComplex) * max_k);
            gpuMemcpy(v->phase_k, phase_k, sizeof(gpuDoubleComplex) * max_k,
                      gpuMemcpyHostToDevice);
        }
    }

    int WMimax = 0;
    int *WMi_gpu = GPAW_MALLOC(int, self->nW);
    int *volume_WMi_gpu = GPAW_MALLOC(int, self->nW * self->nW);

    self->Mcount = 0;
    for (int W=0; W < self->nW; W++) {
        WMi_gpu[W] = 0;
    }
    for (int W=0; W < self->nW; W++) {
        LFVolume_gpu* v = &volume_W_gpu[W];
        int M = v->M;

        for (int W2=0; W2 <= W; W2++) {
            if (WMi_gpu[W2] > 0) {
                LFVolume_gpu* v2
                    = &volume_W_gpu[volume_WMi_gpu[W2 * self->nW]];
                if (v2->M == M) {
                    volume_WMi_gpu[W2 * self->nW + WMi_gpu[W2]] = W;
                    WMi_gpu[W2]++;
                    WMimax = MAX(WMi_gpu[W2], WMimax);
                    break;
                }
            } else {
                volume_WMi_gpu[W2*self->nW] = W;
                WMi_gpu[W2]++;
                self->Mcount++;
                WMimax = MAX(WMi_gpu[W2], WMimax);
                break;
            }
        }
    }
    int *volume_WMi_gpu2 = GPAW_MALLOC(int, WMimax * self->nW);
    for (int W=0; W < self->Mcount; W++) {
        for (int W2=0; W2 < WMi_gpu[W]; W2++) {
            volume_WMi_gpu2[W * WMimax + W2]
                = volume_WMi_gpu[W * self->nW + W2];
        }
    }
    self->WMimax = WMimax;

    gpuMalloc(&(self->WMi_gpu), sizeof(int) * self->Mcount);
    gpuMemcpy(self->WMi_gpu, WMi_gpu, sizeof(int) * self->Mcount,
              gpuMemcpyHostToDevice);

    gpuMalloc(&(self->volume_WMi_gpu), sizeof(int) * self->Mcount * WMimax);
    gpuMemcpy(self->volume_WMi_gpu, volume_WMi_gpu2,
              sizeof(int) * self->Mcount * WMimax, gpuMemcpyHostToDevice);

    self->nB_gpu = nB_gpu;

    gpuMalloc(&(self->G_B1_gpu), sizeof(int) * nB_gpu);
    gpuMemcpy(self->G_B1_gpu, G_B1_gpu, sizeof(int) * nB_gpu,
              gpuMemcpyHostToDevice);

    gpuMalloc(&(self->G_B2_gpu), sizeof(int) * nB_gpu);
    gpuMemcpy(self->G_B2_gpu, G_B2_gpu, sizeof(int) * nB_gpu,
              gpuMemcpyHostToDevice);

    gpuMalloc(&(self->ni_gpu), sizeof(int) * nB_gpu);
    gpuMemcpy(self->ni_gpu, ni_gpu, sizeof(int) * nB_gpu,
              gpuMemcpyHostToDevice);

    transp(volume_i_gpu, nB_gpu, nimax, sizeof(LFVolume_gpu*));
    gpuMalloc(&(self->volume_i_gpu), sizeof(LFVolume_gpu*) * nB_gpu * nimax);
    gpuMemcpy(self->volume_i_gpu, volume_i_gpu,
              sizeof(LFVolume_gpu*) * nB_gpu * nimax, gpuMemcpyHostToDevice);

    transp(A_gm_i_gpu, nB_gpu, nimax, sizeof(int));
    gpuMalloc(&(self->A_gm_i_gpu), sizeof(int) * nB_gpu * nimax);
    gpuMemcpy(self->A_gm_i_gpu, A_gm_i_gpu, sizeof(int) * nB_gpu * nimax,
              gpuMemcpyHostToDevice);

    if (self->bloch_boundary_conditions) {
        gpuMalloc(&(self->phase_i_gpu),
                  sizeof(gpuDoubleComplex) * max_k * nB_gpu * nimax);
        gpuMemcpy(self->phase_i_gpu, phase_i_gpu,
                  sizeof(gpuDoubleComplex) * max_k * nB_gpu * nimax,
                  gpuMemcpyHostToDevice);
    }
    self->volume_W_gpu_host = volume_W_gpu;

    gpuMemcpy(self->volume_W_gpu, volume_W_gpu,
              sizeof(LFVolume_gpu) * self->nW, gpuMemcpyHostToDevice);
    free(volume_i);
    free(volume_i_gpu);
    free(A_gm_i_gpu);
    free(volume_WMi_gpu);
    free(volume_WMi_gpu2);
    free(WMi_gpu);
    free(ni_gpu);
    free(G_B1_gpu);
    free(G_B2_gpu);
    if (self->bloch_boundary_conditions) {
        free(phase_i_gpu);
    }
    if (PyErr_Occurred())
        return NULL;
    else
        return (PyObject*) self;
}


extern "C"
void parse_shape_xG(PyObject* shape, int* nx, int* nG)
{
    int nd = PyTuple_Size(shape);

    *nx = 1;
    for (int i = 0; i < nd-3; i++) {
        *nx *= (int) PyLong_AsLong(PyTuple_GetItem(shape, i));
    }
    *nG = 1;
    for (int i = nd-3; i < nd; i++)
        *nG *= (int) PyLong_AsLong(PyTuple_GetItem(shape, i));
}


extern "C"
PyObject* integrate_gpu(LFCObject *lfc, PyObject *args)
{
    void *a_xG_gpu;
    void *c_xM_gpu;
    PyObject *shape, *c_shape;
    int q;

    assert(lfc->use_gpu);

    if (!PyArg_ParseTuple(args, "nOnOi", &a_xG_gpu, &shape, &c_xM_gpu,
                          &c_shape, &q))
        return NULL;

    int nx, nG;
    parse_shape_xG(shape, &nx, &nG);

    int c_nd = PyTuple_Size(c_shape);
    int nM = (int) PyLong_AsLong(PyTuple_GetItem(c_shape, c_nd - 1));

    if (nM > 0) {
        if (!lfc->bloch_boundary_conditions) {
            const double* a_G = (const double*) a_xG_gpu;
            double* c_M = (double*) c_xM_gpu;

            lfc_reducemap(lfc, a_G, nG, c_M, nM, nx, q);
            gpuCheckLastError();
        } else {
            const gpuDoubleComplex* a_G = (const gpuDoubleComplex*) a_xG_gpu;
            gpuDoubleComplex* c_M = (gpuDoubleComplex*) c_xM_gpu;

            lfc_reducemapz(lfc, a_G, nG, c_M, nM, nx, q);
            gpuCheckLastError();
        }
    }
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}

extern "C"
PyObject* add_gpu(LFCObject *lfc, PyObject *args)
{
    void *a_xG_gpu;
    void *c_xM_gpu;
    PyObject *shape, *c_shape;
    int q;

    assert(lfc->use_gpu);

    if (!PyArg_ParseTuple(args, "nOnOi", &c_xM_gpu, &c_shape, &a_xG_gpu,
                &shape, &q))
        return NULL;

    int nx, nG;
    parse_shape_xG(shape, &nx, &nG);

    int c_nd = PyTuple_Size(c_shape);
    int nM = (int) PyLong_AsLong(PyTuple_GetItem(c_shape, c_nd - 1));

    if (nM > 0) {
        if (!lfc->bloch_boundary_conditions) {
            double* a_G = (double*) a_xG_gpu;
            const double* c_M = (const double*) c_xM_gpu;
            int blockx = lfc->max_nG;
            int gridx = (lfc->nB_gpu + BLOCK_Y - 1) / BLOCK_Y;
            dim3 dimBlock(blockx, BLOCK_Y);
            dim3 dimGrid(gridx, nx);

            gpuLaunchKernel(
                    add_kernel, dimGrid, dimBlock, 0, 0,
                    a_G, c_M, lfc->G_B1_gpu, lfc->G_B2_gpu,
                    lfc->volume_i_gpu, lfc->A_gm_i_gpu, lfc->ni_gpu,
                    lfc->nimax, nG, nM, lfc->phase_i_gpu, lfc->max_k, q,
                    lfc->nB_gpu);
            gpuCheckLastError();
        } else {
            gpuDoubleComplex* a_G = (gpuDoubleComplex*) a_xG_gpu;
            const gpuDoubleComplex* c_M = (const gpuDoubleComplex*) c_xM_gpu;

            int blockx = lfc->max_nG;
            int gridx = (lfc->nB_gpu + BLOCK_Y - 1) / BLOCK_Y;
            dim3 dimBlock(blockx, BLOCK_Y);
            dim3 dimGrid(gridx, nx);

            gpuLaunchKernel(
                    add_kernelz, dimGrid, dimBlock, 0, 0,
                    a_G, c_M, lfc->G_B1_gpu, lfc->G_B2_gpu,
                    lfc->volume_i_gpu, lfc->A_gm_i_gpu, lfc->ni_gpu,
                    lfc->nimax, nG, nM, lfc->phase_i_gpu, lfc->max_k, q,
                    lfc->nB_gpu);
            gpuCheckLastError();
        }
    }
    if (PyErr_Occurred())
        return NULL;
    else
        Py_RETURN_NONE;
}

#endif
