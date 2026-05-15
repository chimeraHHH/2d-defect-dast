#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>

#include "../gpu.h"
#include "../gpu-complex.h"

#ifndef GPU_USE_COMPLEX
#  define BLOCK     (16)
#  define BLOCK_X   (32)
#  define BLOCK_Y   (8)
#  define ACACHE_X  (2 * BLOCK_X + 1)
#  define ACACHE_Y  (2 * BLOCK_Y + 1)
#endif


__global__ void Zgpu(restrict_kernel)(const Tgpu* a, const int3 n,
                                      Tgpu* b, const int3 b_n,
                                      int xdiv, int blocks)
{
    int i2, i1;
    int i2_x2, i1_x2;
    int xlen;
    Tgpu *acache12p;
    Tgpu *acache12p_2x;
    Tgpu b_old;
    __shared__ Tgpu Zgpu(acache12)[ACACHE_X * ACACHE_Y];
    {
        int xx = gridDim.x / xdiv;
        int xind = blockIdx.x / xx;
        int base = (blockIdx.x - xind * xx) * BLOCK_X;
        i2 = base + threadIdx.x;
        i2_x2 = 2 * base + threadIdx.x;

        int yy = gridDim.y / blocks;
        int blocksi = blockIdx.y / yy;
        base = (blockIdx.y - blocksi * yy) * BLOCK_Y;
        i1 = base + threadIdx.y;
        i1_x2 = 2 * base + threadIdx.y;

        xlen = (b_n.x + xdiv - 1) / xdiv;
        int xstart = xind * xlen;
        if ((b_n.x - xstart) < xlen)
            xlen = b_n.x - xstart;

        a += n.x * n.y * n.z * blocksi + 2 * xstart * n.y * n.z
           + i1_x2 * n.z + i2_x2;
        b += b_n.x * b_n.y * b_n.z * blocksi + xstart * b_n.y * b_n.z
           + i1 * b_n.z + i2;
    }
    acache12p = Zgpu(acache12) + ACACHE_X * threadIdx.y + threadIdx.x;
    acache12p_2x = Zgpu(acache12) + ACACHE_X * (2 * threadIdx.y)
                 + 2 * threadIdx.x;

    acache12p[0] = a[0];
    acache12p[BLOCK_X] = a[BLOCK_X];
    if  (threadIdx.x < 1) {
        acache12p[2 * BLOCK_X] = a[2 * BLOCK_X];
        acache12p[BLOCK_Y * ACACHE_X + 2 * BLOCK_X]
            = a[BLOCK_Y * n.z + 2 * BLOCK_X];
    }

    acache12p[BLOCK_Y * ACACHE_X + 0] = a[BLOCK_Y * n.z];
    acache12p[BLOCK_Y * ACACHE_X + BLOCK_X] = a[BLOCK_Y * n.z + BLOCK_X];
    if (threadIdx.y < 1) {
        acache12p[2 * BLOCK_Y * ACACHE_X] = a[2 * BLOCK_Y * n.z];
        acache12p[2 * BLOCK_Y * ACACHE_X + BLOCK_X]
            = a[2 * BLOCK_Y * n.z + BLOCK_X];
        if (threadIdx.x < 1)
            acache12p[2 * BLOCK_Y * ACACHE_X + 2 * BLOCK_X]
                = a[2 * BLOCK_Y * n.z + 2 * BLOCK_X];
    }
    __syncthreads();

    b_old = ADD3(MULTD(acache12p_2x[ACACHE_X * 1 + 1], 0.0625),
                 MULTD(ADD4(acache12p_2x[ACACHE_X * 1 + 0],
                            acache12p_2x[ACACHE_X * 1 + 2],
                            acache12p_2x[ACACHE_X * 0 + 1],
                            acache12p_2x[ACACHE_X * 2 + 1]),
                       0.03125),
                 MULTD(ADD4(acache12p_2x[ACACHE_X * 0 + 0],
                            acache12p_2x[ACACHE_X * 0 + 2],
                            acache12p_2x[ACACHE_X * 2 + 0],
                            acache12p_2x[ACACHE_X * 2 + 2]),
                       0.015625));
    __syncthreads();

    for (int i0=0; i0 < xlen; i0++) {
        a += n.y * n.z;
        acache12p[0] = a[0];
        acache12p[BLOCK_X] = a[BLOCK_X];
        if (threadIdx.x < 1) {
            acache12p[2 * BLOCK_X] = a[2 * BLOCK_X];
            acache12p[BLOCK_Y * ACACHE_X + 2 * BLOCK_X]
                = a[BLOCK_Y * n.z + 2 * BLOCK_X];
        }
        acache12p[BLOCK_Y * ACACHE_X + 0] = a[BLOCK_Y * n.z];
        acache12p[BLOCK_Y * ACACHE_X + BLOCK_X] = a[BLOCK_Y * n.z + BLOCK_X];
        if (threadIdx.y < 1) {
            acache12p[2 * BLOCK_Y * ACACHE_X] = a[2 * BLOCK_Y * n.z];
            acache12p[2 * BLOCK_Y * ACACHE_X + BLOCK_X]
                = a[2 * BLOCK_Y * n.z + BLOCK_X];
            if (threadIdx.x < 1)
                acache12p[2 * BLOCK_Y * ACACHE_X + 2 * BLOCK_X]
                    = a[2 * BLOCK_Y * n.z + 2 * BLOCK_X];
        }
        __syncthreads();

        IADD(b_old, ADD3(MULTD(acache12p_2x[ACACHE_X * 1 + 1], 0.125),
                         MULTD(ADD4(acache12p_2x[ACACHE_X * 1 + 0],
                                    acache12p_2x[ACACHE_X * 1 + 2],
                                    acache12p_2x[ACACHE_X * 0 + 1],
                                    acache12p_2x[ACACHE_X * 2 + 1]),
                               0.0625),
                         MULTD(ADD4(acache12p_2x[ACACHE_X * 0 + 0],
                                    acache12p_2x[ACACHE_X * 0 + 2],
                                    acache12p_2x[ACACHE_X * 2 + 0],
                                    acache12p_2x[ACACHE_X * 2 + 2]),
                               0.03125)));
        __syncthreads();

        a += n.y * n.z;
        if (i0 == b_n.x - 1) {
            if (i1_x2 < n.y) {
                if (i2_x2 < n.z) {
                    acache12p[0] = a[0];
                    if (i2_x2 + BLOCK_X < n.z) {
                        acache12p[BLOCK_X] = a[BLOCK_X];
                        if (threadIdx.x < 1) {
                            if (i2_x2 + 2 * BLOCK_X < n.z)
                                acache12p[2 * BLOCK_X] = a[2 * BLOCK_X];
                        }
                    }
                }
            }
            if (i1_x2 + BLOCK_Y < n.y) {
                if (i2_x2 < n.z) {
                    acache12p[BLOCK_Y * ACACHE_X + 0] = a[BLOCK_Y * n.z];
                    if (i2_x2 + BLOCK_X < n.z) {
                        acache12p[BLOCK_Y * ACACHE_X + BLOCK_X]
                            = a[BLOCK_Y * n.z + BLOCK_X];
                        if (threadIdx.x < 1) {
                            if (i2_x2 + 2 * BLOCK_X < n.z)
                                acache12p[BLOCK_Y * ACACHE_X + 2 * BLOCK_X]
                                    =a[BLOCK_Y * n.z + 2 * BLOCK_X];
                        }
                    }
                }
            }
            if (threadIdx.y < 1) {
                if (i1_x2 + 2 * BLOCK_Y < n.y) {
                    if (i2_x2 < n.z) {
                        acache12p[2 * BLOCK_Y * ACACHE_X]
                            = a[2 * BLOCK_Y * n.z];
                        if (i2_x2 + BLOCK_X < n.z) {
                            acache12p[2 * BLOCK_Y * ACACHE_X + BLOCK_X]
                                = a[2 * BLOCK_Y * n.z + BLOCK_X];
                            if (threadIdx.x < 1)
                                if (i2_x2 + 2 * BLOCK_X < n.z)
                                    acache12p[2 * BLOCK_Y * ACACHE_X
                                              + 2 * BLOCK_X]
                                        = a[2 * BLOCK_Y * n.z + 2 * BLOCK_X];
                        }
                    }
                }
            }
        } else {
            acache12p[0] = a[0];
            acache12p[BLOCK_X] = a[BLOCK_X];
            if  (threadIdx.x < 1) {
                acache12p[2 * BLOCK_X] = a[2 * BLOCK_X];
                acache12p[BLOCK_Y * ACACHE_X + 2 * BLOCK_X]
                    = a[BLOCK_Y * n.z + 2 * BLOCK_X];
            }
            acache12p[BLOCK_Y * ACACHE_X + 0] = a[BLOCK_Y * n.z];
            acache12p[BLOCK_Y * ACACHE_X + BLOCK_X]
                = a[BLOCK_Y * n.z + BLOCK_X];
            if (threadIdx.y < 1) {
                acache12p[2 * BLOCK_Y * ACACHE_X] = a[2 * BLOCK_Y * n.z];
                acache12p[2 * BLOCK_Y * ACACHE_X + BLOCK_X]
                    = a[2 * BLOCK_Y * n.z + BLOCK_X];
                if (threadIdx.x < 1)
                    acache12p[2 * BLOCK_Y * ACACHE_X + 2 * BLOCK_X]
                        =a[2 * BLOCK_Y * n.z + 2 * BLOCK_X];
            }
        }
        __syncthreads();

        Tgpu b_new=ADD3(MULTD(acache12p_2x[ACACHE_X * 1 + 1], 0.0625),
                        MULTD(ADD4(acache12p_2x[ACACHE_X * 1 + 0],
                                   acache12p_2x[ACACHE_X * 1 + 2],
                                   acache12p_2x[ACACHE_X * 0 + 1],
                                   acache12p_2x[ACACHE_X * 2 + 1]),
                              0.03125),
                        MULTD(ADD4(acache12p_2x[ACACHE_X * 0 + 0],
                                   acache12p_2x[ACACHE_X * 0 + 2],
                                   acache12p_2x[ACACHE_X * 2 + 0],
                                   acache12p_2x[ACACHE_X * 2 + 2]),
                              0.015625));
        if (i1 < b_n.y && i2 < b_n.z)
            b[0] = ADD(b_old, b_new);
        b_old = b_new;
        __syncthreads();
        b += b_n.y * b_n.z;
    }
}

extern "C"
void Zgpu(bmgs_restrict_gpu)(int k, const Tgpu* a, const int size[3],
                             Tgpu* b, const int sizeb[3], int blocks)
{
    if (k != 2)
        assert(0);

    dim3 dimBlock(BLOCK_X, BLOCK_Y);

    int xdiv = MIN(MAX(sizeb[0] / 2, 1),
                   MAX((4 + blocks - 1) / blocks, 1));

    int gridy = blocks * ((sizeb[1] + dimBlock.y - 1) / dimBlock.y);
    int gridx = xdiv * ((sizeb[2] + dimBlock.x - 1) / dimBlock.x);
    dim3 dimGrid(gridx, gridy);

    int3 n = {size[0], size[1], size[2]};
    int3 b_n = {sizeb[0], sizeb[1], sizeb[2]};

    gpuLaunchKernel(
            Zgpu(restrict_kernel), dimGrid, dimBlock, 0, 0,
            a, n ,b, b_n, xdiv, blocks);
    gpuCheckLastError();
}

#define K 2
#define RST1D Zgpu(restrict1D2)
#define RST1D_kernel Zgpu(restrict1D2_kernel)
#include "restrict-stencil.cpp"
#undef RST1D
#undef RST1D_kernel
#undef K

#define K 4
#define RST1D Zgpu(restrict1D4)
#define RST1D_kernel Zgpu(restrict1D4_kernel)
#include "restrict-stencil.cpp"
#undef RST1D
#undef RST1D_kernel
#undef K

#define K 6
#define RST1D Zgpu(restrict1D6)
#define RST1D_kernel Zgpu(restrict1D6_kernel)
#include "restrict-stencil.cpp"
#undef RST1D
#undef RST1D_kernel
#undef K

#define K 8
#define RST1D Zgpu(restrict1D8)
#define RST1D_kernel Zgpu(restrict1D8_kernel)
#include "restrict-stencil.cpp"
#undef RST1D
#undef RST1D_kernel
#undef K

extern "C"
void Zgpu(bmgs_restrict_stencil_gpu)(int k, Tgpu* a, const int na[3],
                                     Tgpu* b, const int nb[3],
                                     Tgpu* w, int blocks)
{
    void (*func)(const Tgpu*, int, int, Tgpu*, int, int, int);
    int ang = na[0] * na[1] * na[2];
    int bng = nb[0] * nb[1] * nb[2];

    if (k == 2)
        func = Zgpu(restrict1D2);
    else if (k == 4)
        func = Zgpu(restrict1D4);
    else if (k == 6)
        func = Zgpu(restrict1D6);
    else
        func = Zgpu(restrict1D8);

    int e = k * 2 - 3;
    func(a, (na[2] - e) / 2, na[0] * na[1], w, ang, ang, blocks);
    func(w, (na[1] - e) / 2, na[0] * (na[2] - e) / 2, a, ang, ang, blocks);
    func(a, (na[0] - e) / 2, (na[1] - e) * (na[2] - e) / 4, b, ang, bng,
         blocks);
}

#ifndef GPU_USE_COMPLEX
#define GPU_USE_COMPLEX
#include "restrict.cpp"
#endif
