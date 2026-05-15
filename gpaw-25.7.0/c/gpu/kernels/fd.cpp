#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <sys/types.h>
#include <sys/time.h>

#include "../gpu.h"
#include "../gpu-complex.h"

#undef MYJ_X
#undef NDOUBLE
#undef BLOCK_X
#undef BLOCK_Y

#define BLOCK_X   GPU_DEFAULT_BLOCK_X
#define BLOCK_Y   GPU_DEFAULT_BLOCK_Y

#ifndef GPU_USE_COMPLEX
#define MYJ_X     (MYJ)
#define NDOUBLE   1
#else
#define MYJ_X     (MYJ * NDOUBLE)
#define NDOUBLE   2
#endif

#ifdef MYJ
#define ACACHE_X  (BLOCK_X + 2 * MYJ_X)
#define ACACHE_Y  (BLOCK_Y + 2 * MYJ)

__global__ void FD_kernel(
        const int ncoefs,
        const double *c_coefs, const long *c_offsets,
        const double *c_coefs0, const double *c_coefs1,
        const double *c_coefs2,
        const double *a, double *b,
        const int3 c_n, const int3 a_size, const int3 b_size,
        const int xdiv, const int blocks)
{
    int i2tid = threadIdx.x;
    int i1tid = threadIdx.y;
    int i1, i2;
    int xlen;
    double acache0[MYJ] = {0.0};
    double acache0t[MYJ + 1] = {0.0};
    double *acache12p;

    __shared__ double s_coefs0[MYJ * 2 + 1];
    __shared__ double s_coefs1[MYJ * 2];
    __shared__ double s_coefs2[MYJ * 2];
    __shared__ double acache12[ACACHE_X * ACACHE_Y];
    {
        int xx = gridDim.x / xdiv;
        int yy = gridDim.y / blocks;

        int xind = blockIdx.x / xx;
        i2 = (blockIdx.x - xind * xx) * BLOCK_X + i2tid;

        int blocksi = blockIdx.y / yy;
        i1 = (blockIdx.y - blocksi * yy) * BLOCK_Y + i1tid;

        xlen = (c_n.x + xdiv-1) / xdiv;
        int xstart = xind * xlen;
        if ((c_n.x - xstart) < xlen)
            xlen = c_n.x - xstart;

        a += a_size.x * blocksi + xstart * a_size.y + i1 * a_size.z + i2;
        b += b_size.x * blocksi + xstart * b_size.y + i1 * b_size.z + i2;
    }

    acache12p = acache12 + ACACHE_X * (i1tid + MYJ) + i2tid + MYJ_X;

    if (i2tid <= MYJ * 2)
        s_coefs0[i2tid] = c_coefs0[i2tid];
    if (i2tid < MYJ * 2) {
        s_coefs1[i2tid] = c_coefs1[i2tid];
        s_coefs2[i2tid] = c_coefs2[i2tid];
    }
    __syncthreads();

    for (int c=0; c < MYJ; c++) {
        if ((i1 < c_n.y) && (i2 < c_n.z))
            acache0[c] = a[(c - MYJ) * (a_size.y)];
    }

    for (int i0=0; i0 < xlen; i0++) {
        if (i1 < c_n.y + MYJ) {
            acache12p[-MYJ_X] = a[-MYJ_X];
            if ((i2tid < MYJ_X * 2)
                    && (i2 < c_n.z + MYJ_X - BLOCK_X + MYJ_X)) {
                acache12p[BLOCK_X - MYJ_X] = a[BLOCK_X - MYJ_X];
            }
        }
        if (i1tid < MYJ) {
            acache12p[-ACACHE_X * MYJ] = a[-a_size.z * MYJ];
            if (i1 < c_n.y + MYJ - BLOCK_Y) {
                acache12p[ACACHE_X * BLOCK_Y] = a[a_size.z * BLOCK_Y];
            }
        }
        __syncthreads();

        acache0t[0] = 0.0;

        for (int c=0; c < MYJ; c++)
            acache0t[0] += acache12p[ACACHE_X * (c - MYJ)] * s_coefs1[c];
        for (int c=0; c < MYJ; c++)
            acache0t[0] += acache12p[NDOUBLE * c - MYJ_X] * s_coefs2[c];
        for (int c=0; c < MYJ; c++)
            acache0t[0] += acache12p[NDOUBLE * (c+1)] * s_coefs2[c + MYJ];
        for (int c=0; c < MYJ; c++)
            acache0t[0] += acache12p[ACACHE_X * (c+1)] * s_coefs1[c + MYJ];
        for (int c=0; c < MYJ; c++)
            acache0t[0] += acache0[c] * s_coefs0[c];

        acache0t[0] += acache12p[0] * s_coefs0[MYJ];
        for (int c=0; c < MYJ; c++)
            acache0t[c+1] += acache12p[0] * s_coefs0[c + 1 + MYJ];
        for (int c=0; c < ncoefs; c++)
            acache0t[0] += a[NDOUBLE * c_offsets[c]] * c_coefs[c];

        if (i0 >= MYJ) {
            if ((i1 < c_n.y) && (i2 < c_n.z)) {
                b[0] = acache0t[MYJ];
            }
            b += b_size.y;
        }

        for (int c=0; c < MYJ-1; c++) {
            acache0[c] = acache0[c+1];
        }
        acache0[MYJ-1] = acache12p[0];

        for (int c=MYJ; c > 0; c--) {
            acache0t[c] = acache0t[c-1];
        }
        a += a_size.y;
        __syncthreads();
    }

#pragma unroll
    for (int i0=0; i0 < MYJ; i0++) {
        if ((i1 < c_n.y) && (i2 < c_n.z))
            acache0[0] = a[0];

        if (i0 < 1)
            acache0t[1 - i0] += acache0[0] * s_coefs0[1 + MYJ];
#if MYJ >= 2
        if (i0 < 2)
            acache0t[2 - i0] += acache0[0] * s_coefs0[2 + MYJ];
#endif
#if MYJ >= 3
        if (i0 < 3)
            acache0t[3 - i0] += acache0[0] * s_coefs0[3 + MYJ];
#endif
#if MYJ >= 4
        if (i0 < 4)
            acache0t[4 - i0] += acache0[0] * s_coefs0[4 + MYJ];
#endif
#if MYJ >= 5
        if (i0 < 5)
            acache0t[5 - i0] += acache0[0] * s_coefs0[5 + MYJ];
#endif
        if (i0 + xlen >= MYJ) {
            if ((i1 < c_n.y) && (i2 < c_n.z)) {
                b[0] = acache0t[MYJ - i0];
            }
            b += b_size.y;
        }
        a += a_size.y;
    }
}


__global__ void FD_kernel_onlyb(
        const int ncoefs,
        const double *c_coefs, const long *c_offsets,
        const double *c_coefs0, const double *c_coefs1,
        const double *c_coefs2,
        const double *a, double *b,
        const int3 c_n, const int3 c_jb,
        const int boundary, const int xdiv, const int blocks)
{
    int xx = MAX((c_n.z + BLOCK_X - 1) / BLOCK_X, 1);
    int yy = MAX((c_n.y + BLOCK_Y - 1) / BLOCK_Y, 1);
    int ysiz = c_n.y;
    if ((boundary & GPAW_BOUNDARY_Y0) != 0)
        ysiz -= BLOCK_Y;
    if ((boundary & GPAW_BOUNDARY_Y1) != 0)
        ysiz -= BLOCK_Y;
    int yy2 = MAX((ysiz + BLOCK_Y - 1) / BLOCK_Y, 0);

    int i2bl, i1bl;
    int xlen = c_n.x;
    int xind = 0;
    int xstart = 0;
    int i2pitch = 0;
    int i1pitch = 0;
    int ymax = c_n.y;
    int zmax = c_n.z;
    int xmax = c_n.x;
    int blockix;

    blockix = blockIdx.x;

    if ((boundary & GPAW_BOUNDARY_X0) != 0) {
        if ((blockix >= 0) && (blockix < xx * yy)) {
            i1bl = blockix / xx;
            i2bl = blockix - i1bl * xx;
            xlen = c_jb.x / 2;
            xstart = 0;
        }
        blockix -= xx * yy;
    }
    if ((boundary & GPAW_BOUNDARY_X1) != 0) {
        if ((blockix >= 0) && (blockix < xx * yy)) {
            i1bl = blockix / xx;
            i2bl = blockix - i1bl * xx;
            xlen = c_jb.x / 2;
            xstart += c_n.x - c_jb.x / 2;
        }
        blockix -= xx * yy;
    }
    if (blockix >= 0) {
        if ((boundary & GPAW_BOUNDARY_Y0) != 0) {
            if ((blockix >= 0) && (blockix < xdiv * xx)) {
                xind = blockix / xx;
                i2bl = blockix - xind * xx;
                i1bl = 0;
                ymax = MIN(BLOCK_Y, ymax);
            }
            blockix -= xdiv * xx;
        }
        if ((boundary & GPAW_BOUNDARY_Y1) != 0) {
            if ((blockix >= 0) && (blockix < xdiv * xx)) {
                xind = blockix / xx;
                i2bl = blockix - xind * xx;
                i1bl = 0;
                i1pitch = MAX(c_n.y - BLOCK_Y, 0);
            }
            blockix -= xdiv * xx;
        }
        if ((boundary & GPAW_BOUNDARY_Z0) != 0) {
            if ((blockix >= 0) && (blockix < xdiv * yy2)) {
                xind = blockix / yy2;
                i2bl = 0;
                zmax = MIN(BLOCK_X, zmax);
                i1bl = blockix - xind * yy2;
                if ((boundary & GPAW_BOUNDARY_Y0) != 0)
                    i1pitch = BLOCK_Y;
                if ((boundary & GPAW_BOUNDARY_Y1) != 0)
                    ymax = MAX(c_n.y - BLOCK_Y, 0);
            }
            blockix -= xdiv * yy2;
        }
        if ((boundary & GPAW_BOUNDARY_Z1) != 0) {
            if ((blockix >= 0) && (blockix < xdiv * yy2)) {
                xind = blockix / yy2;
                i2bl = 0;
                i2pitch = MAX(c_n.z - BLOCK_X, 0);
                i1bl = blockix - xind * yy2;
                if ((boundary & GPAW_BOUNDARY_Y0) != 0)
                    i1pitch = BLOCK_Y;
                if ((boundary & GPAW_BOUNDARY_Y1) != 0)
                    ymax = MAX(c_n.y - BLOCK_Y, 0);
            }
            blockix -= xdiv * yy2;
        }
        if ((boundary & GPAW_BOUNDARY_X0) != 0) {
            xstart += c_jb.x / 2;
            xlen -= c_jb.x / 2;
        }
        if ((boundary & GPAW_BOUNDARY_X1) != 0) {
            xlen -= c_jb.x / 2;
            xmax -= c_jb.x / 2;
        }
        xlen = (xlen + xdiv - 1) / xdiv;
        xstart += xind * xlen;
    }

    int i2tid = threadIdx.x;
    int i2 = i2pitch + i2bl * BLOCK_X + i2tid;

    int blocksi = blockIdx.y;

    int i1tid = threadIdx.y;
    int i1 = i1pitch + i1bl * BLOCK_Y + i1tid;

    __shared__ double s_coefs0[MYJ * 2 + 1];
    __shared__ double s_coefs1[MYJ * 2];
    __shared__ double s_coefs2[MYJ * 2];
    __shared__ double acache12[ACACHE_X * ACACHE_Y];

    double acache0[MYJ];
    double acache0t[MYJ + 1];
    double *acache12p;
    int sizez = c_jb.z + c_n.z;
    int sizeyz = (c_jb.y + c_n.y) * sizez;

    if ((xmax-xstart) < xlen)
        xlen = xmax - xstart;

    a += ((c_jb.x + c_n.x) * sizeyz) * blocksi;
    b += (c_n.x * c_n.y * c_n.z) * blocksi;

    acache12p = acache12 + ACACHE_X * (i1tid + MYJ * 2 / 2)
              + i2tid + MYJ_X * 2 / 2;

    if (i2tid <= MYJ * 2)
        s_coefs0[i2tid] = c_coefs0[i2tid];
    if (i2tid < MYJ * 2) {
        s_coefs1[i2tid] = c_coefs1[i2tid];
        s_coefs2[i2tid] = c_coefs2[i2tid];
    }
    __syncthreads();

    a += xstart * sizeyz + i1 * sizez + i2;
    b += xstart * c_n.y * c_n.z + i1 * c_n.z + i2;
    for (int c=0; c < MYJ; c++) {
        if ((i1 < ymax) && (i2 < zmax))
            acache0[c] = a[(c - MYJ) * sizeyz];
    }

    for (int i0=0; i0 < xlen; i0++) {
        if (i1 < ymax + MYJ) {
            acache12p[-MYJ_X] = a[-MYJ_X];
            if ((i2tid < MYJ_X * 2) && (i2 < zmax + MYJ_X - BLOCK_X + MYJ_X))
                acache12p[BLOCK_X - MYJ_X] = a[BLOCK_X - MYJ_X];
        }
        if  (i1tid < MYJ) {
            acache12p[-ACACHE_X * MYJ] = a[-sizez * MYJ];
            if  (i1 < ymax + MYJ - BLOCK_Y)
                acache12p[ACACHE_X * BLOCK_Y] = a[sizez * BLOCK_Y];
        }
        __syncthreads();

        acache0t[0] = 0.0;

        for (int c=0; c < MYJ; c++)
            acache0t[0] += acache12p[ACACHE_X * (c - MYJ)] * s_coefs1[c];
        for (int c=0; c < MYJ; c++)
            acache0t[0] += acache12p[NDOUBLE * c - MYJ_X] * s_coefs2[c];
        for (int c=0; c < MYJ; c++)
            acache0t[0] += acache12p[NDOUBLE * (c+1)] * s_coefs2[c+MYJ];
        for (int c=0; c < MYJ; c++)
            acache0t[0] += acache12p[ACACHE_X * (c+1)] * s_coefs1[c+MYJ];
        for (int c=0; c < MYJ; c++)
            acache0t[0] += acache0[c] * s_coefs0[c];

        acache0t[0] += acache12p[0] * s_coefs0[MYJ];
        for (int c=0; c < MYJ; c++)
            acache0t[c+1] += acache12p[0] * s_coefs0[c + 1 + MYJ];
        for (int c=0; c < ncoefs; c++)
            acache0t[0] += a[NDOUBLE * c_offsets[c]] * c_coefs[c];

        if (i0 >= MYJ) {
            if ((i1 < ymax) && (i2 < zmax)) {
                b[0] = acache0t[MYJ];
            }
            b += c_n.y * c_n.z;
        }

        for (int c=0; c < MYJ - 1; c++) {
            acache0[c] = acache0[c+1];
        }
        acache0[MYJ - 1] = acache12p[0];

        for (int c=MYJ; c > 0;c--) {
            acache0t[c] = acache0t[c-1];
        }
        a += sizeyz;
        __syncthreads();
    }
#pragma unroll
    for (int i0=0; i0 < MYJ; i0++) {
        if ((i1 < c_n.y) && (i2 < c_n.z))
            acache0[0] = a[0];

        if (i0 < 1)
            acache0t[1 - i0] += acache0[0] * s_coefs0[1 + MYJ];
#if MYJ >= 2
        if (i0 < 2)
            acache0t[2 - i0] += acache0[0] * s_coefs0[2 + MYJ];
#endif
#if MYJ >= 3
        if (i0 < 3)
            acache0t[3 - i0] += acache0[0] * s_coefs0[3 + MYJ];
#endif
#if MYJ >= 4
        if (i0 < 4)
            acache0t[4 - i0] += acache0[0] * s_coefs0[4 + MYJ];
#endif
#if MYJ >= 5
        if (i0 < 5)
            acache0t[5 - i0] += acache0[0] * s_coefs0[5 + MYJ];
#endif
        if (i0 + xlen >= MYJ) {
            if ((i1 < ymax) && (i2 < zmax)) {
                b[0] = acache0t[MYJ - i0];
            }
            b += c_n.y * c_n.z;
        }
        a += sizeyz;
    }
}


#else

#define MYJ  (2/2)
#  define FD_kernel Zgpu(fd_kernel2)
#  define FD_kernel_onlyb Zgpu(fd_kernel2_onlyb)
#  include "fd.cpp"
#  undef FD_kernel
#  undef FD_kernel_onlyb
#  undef MYJ
#define MYJ  (4/2)
#  define FD_kernel Zgpu(fd_kernel4)
#  define FD_kernel_onlyb Zgpu(fd_kernel4_onlyb)
#  include "fd.cpp"
#  undef FD_kernel
#  undef FD_kernel_onlyb
#  undef MYJ
#define MYJ  (6/2)
#  define FD_kernel Zgpu(fd_kernel6)
#  define FD_kernel_onlyb Zgpu(fd_kernel6_onlyb)
#  include "fd.cpp"
#  undef FD_kernel
#  undef FD_kernel_onlyb
#  undef MYJ
#define MYJ  (8/2)
#  define FD_kernel Zgpu(fd_kernel8)
#  define FD_kernel_onlyb Zgpu(fd_kernel8_onlyb)
#  include "fd.cpp"
#  undef FD_kernel
#  undef FD_kernel_onlyb
#  undef MYJ
#define MYJ  (10/2)
#  define FD_kernel Zgpu(fd_kernel10)
#  define FD_kernel_onlyb Zgpu(fd_kernel10_onlyb)
#  include "fd.cpp"
#  undef FD_kernel
#  undef FD_kernel_onlyb
#  undef MYJ


extern "C"
bmgsstencil_gpu bmgs_stencil_to_gpu(const bmgsstencil* s);

extern "C"
int bmgs_fd_boundary_test(
        const bmgsstencil_gpu* s, int boundary, int ndouble);

extern "C"
void Zgpu(bmgs_fd_gpu)(
        const bmgsstencil_gpu* s_gpu, const Tgpu* adev, Tgpu* bdev,
        int boundary, int blocks, gpuStream_t stream)
{
    int3 bjb;
    int3 jb;
    int3 hc_bj;
    int3 hc_n;
    int3 hc_j;
    long *offsets_gpu;

    dim3 dimBlock(BLOCK_X, BLOCK_Y);

    if ((boundary & GPAW_BOUNDARY_SKIP) != 0) {
        if (!bmgs_fd_boundary_test(s_gpu, boundary, NDOUBLE))
            return;
    } else if ((boundary & GPAW_BOUNDARY_ONLY) != 0) {
        if (!bmgs_fd_boundary_test(s_gpu, boundary, NDOUBLE)) {
            boundary &= ~GPAW_BOUNDARY_ONLY;
            boundary |= GPAW_BOUNDARY_NORMAL;
        }
    }
    hc_n.x=s_gpu->n[0];
    hc_n.y=s_gpu->n[1];
    hc_n.z=s_gpu->n[2];
    hc_j.x=s_gpu->j[0];
    hc_j.y=s_gpu->j[1];
    hc_j.z=s_gpu->j[2];

    bjb.x=0;
    bjb.y=0;
    bjb.z=0;
    hc_bj.x=0;
    hc_bj.y=0;
    hc_bj.z=0;

    hc_n.z *= NDOUBLE;
    hc_j.x *= NDOUBLE;
    hc_j.y *= NDOUBLE;
    hc_j.z *= NDOUBLE;

    offsets_gpu = s_gpu->offsets_gpu;

    jb.z = hc_j.z;
    jb.y = hc_j.y / (hc_j.z + hc_n.z);
    jb.x = hc_j.x / ((hc_j.z + hc_n.z) * hc_n.y + hc_j.y);
    if ((boundary & GPAW_BOUNDARY_SKIP) != 0) {
        int3 jb1;
        int3 bjb1, bjb2;
        bjb1.x = 0;
        bjb1.y = 0;
        bjb1.z = 0;
        bjb2.x = 0;
        bjb2.y = 0;
        bjb2.z = 0;
        jb1.z = jb.z / 2;
        jb1.x = jb.x / 2;
        jb1.y = jb.y / 2;
        if ((boundary & GPAW_BOUNDARY_X0) != 0) {
            bjb1.x += jb.x / 2;
        }
        if ((boundary & GPAW_BOUNDARY_X1) != 0) {
            bjb2.x += jb.x / 2;
        }
        if ((boundary & GPAW_BOUNDARY_Y0) != 0) {
            bjb1.y += dimBlock.y;
        }
        if ((boundary & GPAW_BOUNDARY_Y1) != 0) {
            bjb2.y += dimBlock.y;
        }
        if ((boundary & GPAW_BOUNDARY_Z0) != 0) {
            bjb1.z += dimBlock.x;
        }
        if ((boundary & GPAW_BOUNDARY_Z1) != 0) {
            bjb2.z += dimBlock.x;
        }
        bjb.x = bjb1.x + bjb2.x;
        bjb.y = bjb1.y + bjb2.y;
        bjb.z = bjb1.z + bjb2.z;

        hc_n.x -= bjb.x;
        hc_n.y -= bjb.y;
        hc_n.z -= bjb.z;

        jb.x += bjb.x;
        jb.y += bjb.y;
        jb.z += bjb.z;
        jb1.x += bjb1.x;
        jb1.y += bjb1.y;
        jb1.z += bjb1.z;

        hc_bj.z = bjb.z;
        hc_bj.y = bjb.y * (hc_bj.z + hc_n.z);
        hc_bj.x = bjb.x * ((hc_bj.z + hc_n.z) * hc_n.y + hc_bj.y);

        hc_j.z = jb.z;
        hc_j.y = jb.y * (hc_j.z + hc_n.z);
        hc_j.x = jb.x * ((hc_j.z + hc_n.z) * hc_n.y + hc_j.y);

        bdev += bjb1.z + bjb1.y * (hc_bj.z + hc_n.z)
              + bjb1.x * ((hc_bj.z + hc_n.z) * hc_n.y + hc_bj.y);
        adev = (Tgpu*) ((double*) adev + jb1.z
             + jb1.y * (hc_j.z + hc_n.z)
             + jb1.x * ((hc_j.z + hc_n.z) * hc_n.y + hc_j.y));
    } else {
        adev = (Tgpu*) ((double*) adev + (hc_j.x + hc_j.y + hc_j.z) / 2);
    }

    if ((hc_n.x <= 0) || (hc_n.y <= 0) || (hc_n.z <= 0))
        return;

    dim3 dimGrid(1,1,1);
    int xdiv = MIN(hc_n.x, MAX((4 + blocks - 1) / blocks, 1));

    if (((boundary & GPAW_BOUNDARY_NORMAL) != 0) ||
            ((boundary & GPAW_BOUNDARY_SKIP) != 0)) {
        dimGrid.x = MAX((hc_n.z + dimBlock.x - 1) / dimBlock.x, 1);
        dimGrid.y = MAX((hc_n.y + dimBlock.y - 1) / dimBlock.y, 1);
        dimGrid.y *= blocks;
        dimGrid.x *= xdiv;
    } else if ((boundary & GPAW_BOUNDARY_ONLY) != 0) {
        int xx = MAX((hc_n.z + dimBlock.x - 1) / dimBlock.x, 1);
        int yy = MAX((hc_n.y + dimBlock.y - 1) / dimBlock.y, 1);
        int ysiz = hc_n.y;
        if ((boundary & GPAW_BOUNDARY_Y0) != 0)
            ysiz -= dimBlock.y;
        if ((boundary & GPAW_BOUNDARY_Y1) != 0)
            ysiz -= dimBlock.y;
        int yy2 = MAX((ysiz + dimBlock.y - 1) / dimBlock.y, 0);
        dimGrid.x = 0;
        if ((boundary & GPAW_BOUNDARY_X0) != 0)
            dimGrid.x += xx * yy;
        if ((boundary & GPAW_BOUNDARY_X1) != 0)
            dimGrid.x += xx * yy;
        if ((boundary & GPAW_BOUNDARY_Y0) != 0)
            dimGrid.x += xdiv * xx;
        if ((boundary & GPAW_BOUNDARY_Y1) != 0)
            dimGrid.x += xdiv * xx;
        if ((boundary & GPAW_BOUNDARY_Z0) != 0)
            dimGrid.x += xdiv * yy2;
        if ((boundary & GPAW_BOUNDARY_Z1) != 0)
            dimGrid.x += xdiv * yy2;
        dimGrid.y = blocks;
    }

    int3 sizea;
    sizea.z = hc_j.z + hc_n.z;
    sizea.y = sizea.z * hc_n.y + hc_j.y;
    sizea.x = sizea.y * hc_n.x + hc_j.x;

    int3 sizeb;
    sizeb.z = hc_bj.z + hc_n.z;
    sizeb.y = sizeb.z * hc_n.y + hc_bj.y;
    sizeb.x = sizeb.y * hc_n.x + hc_bj.x;

    if (((boundary & GPAW_BOUNDARY_NORMAL) != 0) ||
            ((boundary & GPAW_BOUNDARY_SKIP) != 0)) {
        void (*fd_kernel)(const int ncoefs, const double *c_coefs,
                          const long *c_offsets, const double *c_coefs0,
                          const double *c_coefs1, const double *c_coefs2,
                          const double* a, double* b, const int3 c_n,
                          const int3 a_size, const int3 b_size,
                          const int xdiv, const int blocks);
        switch (s_gpu->ncoefs0) {
            case 3:
                fd_kernel = Zgpu(fd_kernel2);
                break;
            case 5:
                fd_kernel = Zgpu(fd_kernel4);
                break;
            case 7:
                fd_kernel = Zgpu(fd_kernel6);
                break;
            case 9:
                fd_kernel = Zgpu(fd_kernel8);
                break;
            case 11:
                fd_kernel = Zgpu(fd_kernel10);
                break;
            default:
                assert(0);
        }
        gpuLaunchKernel(
                (*fd_kernel), dimGrid, dimBlock, 0, stream,
                s_gpu->ncoefs, s_gpu->coefs_gpu, offsets_gpu,
                s_gpu->coefs0_gpu, s_gpu->coefs1_gpu, s_gpu->coefs2_gpu,
                (double*) adev, (double*) bdev, hc_n, sizea, sizeb, xdiv,
                blocks);
    } else if ((boundary & GPAW_BOUNDARY_ONLY) != 0) {
        void (*fd_kernel)(const int ncoefs, const double *c_coefs,
                          const long *c_offsets, const double *c_coefs0,
                          const double *c_coefs1, const double *c_coefs2,
                          const double *a, double *b, const int3 c_n,
                          const int3 c_jb, const int boundary,
                          const int xdiv, const int blocks);
        switch (s_gpu->ncoefs0) {
            case 3:
                fd_kernel = Zgpu(fd_kernel2_onlyb);
                break;
            case 5:
                fd_kernel = Zgpu(fd_kernel4_onlyb);
                break;
            case 7:
                fd_kernel = Zgpu(fd_kernel6_onlyb);
                break;
            case 9:
                fd_kernel = Zgpu(fd_kernel8_onlyb);
                break;
            case 11:
                fd_kernel = Zgpu(fd_kernel10_onlyb);
                break;
            default:
                assert(0);
        }
        gpuLaunchKernel(
                (*fd_kernel), dimGrid, dimBlock, 0, stream,
                s_gpu->ncoefs, s_gpu->coefs_gpu, offsets_gpu,
                s_gpu->coefs0_gpu, s_gpu->coefs1_gpu, s_gpu->coefs2_gpu,
                (double*) adev, (double*) bdev, hc_n, jb, boundary, xdiv,
                blocks);
    }
    gpuCheckLastError();
}

#ifndef GPU_USE_COMPLEX
#define GPU_USE_COMPLEX
#include "fd.cpp"

extern "C"
int bmgs_fd_boundary_test(const bmgsstencil_gpu* s, int boundary,
                          int ndouble)
{
    int3 jb;
    int3 bjb;
    long3 hc_n;
    long3 hc_j;

    dim3 dimBlock(BLOCK_X, BLOCK_Y);

    hc_n.x = s->n[0];
    hc_n.y = s->n[1];
    hc_n.z = s->n[2];
    hc_j.x = s->j[0];
    hc_j.y = s->j[1];
    hc_j.z = s->j[2];

    jb.z = hc_j.z;
    jb.y = hc_j.y / (hc_j.z + hc_n.z);
    jb.x = hc_j.x / ((hc_j.z + hc_n.z) * hc_n.y + hc_j.y);

    int3 bjb1, bjb2;
    bjb1.x=0;
    bjb1.y=0;
    bjb1.z=0;
    bjb2.x=0;
    bjb2.y=0;
    bjb2.z=0;

    if ((boundary & GPAW_BOUNDARY_X0) != 0) {
        bjb1.x += jb.x / 2;
    }
    if ((boundary & GPAW_BOUNDARY_X1) != 0) {
        bjb2.x += jb.x / 2;
    }
    if ((boundary & GPAW_BOUNDARY_Y0) != 0) {
        bjb1.y += dimBlock.y;
    }
    if ((boundary & GPAW_BOUNDARY_Y1) != 0) {
        bjb2.y += dimBlock.y;
    }
    if ((boundary & GPAW_BOUNDARY_Z0) != 0) {
        bjb1.z += dimBlock.x;
    }
    if ((boundary & GPAW_BOUNDARY_Z1) != 0) {
        bjb2.z += dimBlock.x;
    }
    bjb.x = bjb1.x + bjb2.x;
    bjb.y = bjb1.y + bjb2.y;
    bjb.z = bjb1.z + bjb2.z;

    hc_n.x -= bjb.x;
    hc_n.y -= bjb.y;
    hc_n.z -= bjb.z;

    if (hc_n.x < 4 || hc_n.y < 1 || hc_n.z < 1)
        return 0;

    if ((hc_n.y / (dimBlock.y)) * (hc_n.z / (dimBlock.x)) < 20)
        return 0;

    return 1;
}

extern "C"
bmgsstencil_gpu bmgs_stencil_to_gpu(const bmgsstencil* s)
{
    bmgsstencil_gpu s_gpu;
    long offsets[s->ncoefs];
    double coefs[s->ncoefs];
    int ncoefs=0, ncoefs0=0, ncoefs1=0, ncoefs2=0;

    int n2 = (s->n[2] + s->j[2]);
    int n1 = s->j[1] + s->n[1] * n2;
    int jb[3];

    jb[2] = s->j[2];
    jb[1] = s->j[1] / n2;
    jb[0] = s->j[0] / n1;

    s_gpu.n[0] = s->n[0];
    s_gpu.n[1] = s->n[1];
    s_gpu.n[2] = s->n[2];
    s_gpu.j[0] = s->j[0];
    s_gpu.j[1] = s->j[1];
    s_gpu.j[2] = s->j[2];

    ncoefs0 = jb[0] + 1;
    ncoefs1 = jb[1];
    ncoefs2 = jb[2];

    double coefs0[ncoefs0], coefs1[ncoefs1], coefs2[ncoefs2];
    memset(coefs0, 0, sizeof(double) * ncoefs0);
    memset(coefs1, 0, sizeof(double) * ncoefs1);
    memset(coefs2, 0, sizeof(double) * ncoefs2);

    for (int i=0; i < s->ncoefs; i++) {
        int offpoint = s->offsets[i] + (s->j[0] + s->j[1] + s->j[2]) / 2;
        int i0 = offpoint / n1;
        int i1 = (offpoint - i0 * n1) / n2;
        int i2 = (offpoint - i0 * n1 - i1 * n2);
        i0 -= jb[0] / 2;
        i1 -= jb[1] / 2;
        i2 -= jb[2] / 2;
        if (i1 == 0 && i2 == 0 && abs(i0) <= jb[0] / 2) {
            int offset = ncoefs0 / 2 + i0;
            coefs0[offset] = s->coefs[i];
        } else if (i0 == 0 && i1 == 0 && abs(i2) <= jb[2] / 2) {
            int offset = i2 > 0 ? ncoefs2 / 2 + i2 - 1 : ncoefs2 / 2 + i2;
            coefs2[offset] = s->coefs[i];
        } else if (i0 == 0 && i2 == 0 && abs(i1) <= jb[1] / 2) {
            int offset = i1 > 0 ? ncoefs1 / 2 + i1 - 1 : ncoefs1 / 2 + i1;
            coefs1[offset] = s->coefs[i];
        } else {
            offsets[ncoefs] = s->offsets[i];
            coefs[ncoefs] = s->coefs[i];
            ncoefs++;
        }
    }
    s_gpu.ncoefs = ncoefs;
    s_gpu.ncoefs0 = ncoefs0;
    s_gpu.ncoefs1 = ncoefs1;
    s_gpu.ncoefs2 = ncoefs2;

    s_gpu.coef_relax = s->coefs[0];

    if (ncoefs > 0) {
        gpuMalloc(&(s_gpu.coefs_gpu), sizeof(double) * ncoefs);
        gpuMemcpy(s_gpu.coefs_gpu, coefs, sizeof(double) * ncoefs,
                  gpuMemcpyHostToDevice);

        gpuMalloc(&(s_gpu.offsets_gpu), sizeof(long) * ncoefs);
        gpuMemcpy(s_gpu.offsets_gpu, offsets, sizeof(long) * ncoefs,
                  gpuMemcpyHostToDevice);
    }
    gpuMalloc(&(s_gpu.coefs0_gpu), sizeof(double) * ncoefs0);
    gpuMemcpy(s_gpu.coefs0_gpu, coefs0, sizeof(double) * ncoefs0,
              gpuMemcpyHostToDevice);

    gpuMalloc(&(s_gpu.coefs1_gpu), sizeof(double) * ncoefs1);
    gpuMemcpy(s_gpu.coefs1_gpu, coefs1, sizeof(double) * ncoefs1,
              gpuMemcpyHostToDevice);

    gpuMalloc(&(s_gpu.coefs2_gpu), sizeof(double) * ncoefs2);
    gpuMemcpy(s_gpu.coefs2_gpu, coefs2, sizeof(double) * ncoefs2,
              gpuMemcpyHostToDevice);
    return s_gpu;
}

#endif
#endif
