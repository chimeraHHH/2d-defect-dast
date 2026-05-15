#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>

#include "../gpu.h"
#include "../gpu-complex.h"

#ifndef GPU_USE_COMPLEX
#  define BLOCK_SIZEX 32
#  define BLOCK_SIZEY 16
#  define BLOCK_MAX 32
#  define GRID_MAX 65535
#  define BLOCK_TOTALMAX 512
#  define XDIV 4
#  define Tfunc launch_func

typedef void (*launch_func)(const double *, const int *,
                            double *, const int *, const int *, int,
                            gpuStream_t);
typedef void (*launch_funcz)(const gpuDoubleComplex *, const int *,
                             gpuDoubleComplex *, const int *, const int *, int,
                             gpuStream_t);
#else
#  undef Tfunc
#  define Tfunc launch_funcz
#endif


/*
 * GPU kernel to copy a smaller array into a given position in a
 * larger one.
 */
__global__ void Zgpu(bmgs_paste_kernel)(
        const double* a, const int3 c_sizea, double* b, const int3 c_sizeb,
        int blocks, int xdiv)
{
    int xx = gridDim.x / xdiv;
    int yy = gridDim.y / blocks;

    int blocksi = blockIdx.y / yy;
    int i1 = (blockIdx.y - blocksi * yy) * blockDim.y + threadIdx.y;

    int xind = blockIdx.x / xx;
    int i2 = (blockIdx.x - xind * xx) * blockDim.x + threadIdx.x;

    b += i2 + (i1 + (xind + blocksi * c_sizeb.x) * c_sizeb.y) * c_sizeb.z;
    a += i2 + (i1 + (xind + blocksi * c_sizea.x) * c_sizea.y) * c_sizea.z;

    while (xind < c_sizea.x) {
        if ((i2 < c_sizea.z) && (i1 < c_sizea.y)) {
            b[0] = a[0];
        }
        b += xdiv * c_sizeb.y * c_sizeb.z;
        a += xdiv * c_sizea.y * c_sizea.z;
        xind += xdiv;
    }
}

/*
 * GPU kernel to copy a smaller array into a given position in a
 * larger one and set all other elements to 0.
 */
__global__ void Zgpu(bmgs_paste_zero_kernel)(
        const Tgpu* a, const int3 c_sizea, Tgpu* b, const int3 c_sizeb,
        const int3 c_startb, const int3 c_blocks_bc, int blocks)
{
    int xx = gridDim.x / XDIV;
    int yy = gridDim.y / blocks;

    int blocksi = blockIdx.y / yy;
    int i1bl = blockIdx.y - blocksi * yy;
    int i1tid = threadIdx.y;
    int i1 = i1bl * BLOCK_SIZEY + i1tid;

    int xind = blockIdx.x / xx;
    int i2bl = blockIdx.x - xind * xx;
    int i2tid = threadIdx.x;
    int i2 = i2bl * BLOCK_SIZEX + i2tid;

    int xlen = (c_sizea.x + XDIV - 1) / XDIV;
    int xstart = xind * xlen;
    int xend = MIN(xstart + xlen, c_sizea.x);

    b += c_sizeb.x * c_sizeb.y * c_sizeb.z * blocksi;
    a += c_sizea.x * c_sizea.y * c_sizea.z * blocksi;

    // zero x = 0 .. startb.x
    if (xind==0) {
        Tgpu *bb = b + i2 + i1 * c_sizeb.z;
#pragma unroll 3
        for (int i0=0; i0 < c_startb.x; i0++) {
            if ((i2 < c_sizeb.z) && (i1 < c_sizeb.y)) {
                bb[0] = MAKED(0);
            }
            bb += c_sizeb.y * c_sizeb.z;
        }
    }
    // zero x = startb.x+sizea.x .. <end>
    if (xind == XDIV - 1) {
        Tgpu *bb = b + (c_startb.x + c_sizea.x) * c_sizeb.y * c_sizeb.z
                  + i2 + i1 * c_sizeb.z;
#pragma unroll 3
        for (int i0 = c_startb.x + c_sizea.x; i0 < c_sizeb.x; i0++) {
            if ((i2 < c_sizeb.z) && (i1 < c_sizeb.y)) {
                bb[0] = MAKED(0);
            }
            bb += c_sizeb.y * c_sizeb.z;
        }
    }

    int i1blbc = gridDim.y / blocks - i1bl - 1;
    int i2blbc = gridDim.x / XDIV - i2bl - 1;

    if (i1blbc<c_blocks_bc.y || i2blbc<c_blocks_bc.z) {
        int i1bc = i1blbc * BLOCK_SIZEY + i1tid;
        int i2bc = i2blbc * BLOCK_SIZEX + i2tid;

        b += (c_startb.x + xstart) * c_sizeb.y * c_sizeb.z;
        for (int i0=xstart; i0 < xend; i0++) {
            // zero y = 0 .. startb.y
            if ((i1bc < c_startb.y) && (i2 < c_sizeb.z)) {
                b[i2 + i1bc * c_sizeb.z] = MAKED(0);
            }
            // zero y = startb.y+sizea.y .. <end>
            if ((i1bc + c_sizea.y + c_startb.y < c_sizeb.y)
                    && (i2 < c_sizeb.z)) {
                b[i2 + i1bc * c_sizeb.z
                  + (c_sizea.y + c_startb.y) * c_sizeb.z] = MAKED(0);
            }
            // zero z = 0 .. startb.z
            if ((i2bc < c_startb.z) && (i1 < c_sizeb.y)) {
                b[i2bc + i1 * c_sizeb.z] = MAKED(0);
            }
            // zero z = startb.z+sizea.z .. <end>
            if ((i2bc + c_sizea.z + c_startb.z < c_sizeb.z)
                    && (i1 < c_sizeb.y)) {
                b[i2bc + i1 * c_sizeb.z + c_sizea.z + c_startb.z] = MAKED(0);
            }
            b += c_sizeb.y * c_sizeb.z;
        }
    } else {
        b += c_startb.z + (c_startb.y + c_startb.x * c_sizeb.y) * c_sizeb.z;

        b += i2 + i1 * c_sizeb.z + xstart * c_sizeb.y * c_sizeb.z;
        a += i2 + i1 * c_sizea.z + xstart * c_sizea.y * c_sizea.z;
        for (int i0=xstart; i0 < xend; i0++) {
            if ((i2 < c_sizea.z) && (i1 < c_sizea.y)) {
                b[0] = a[0];
            }
            b += c_sizeb.y * c_sizeb.z;
            a += c_sizea.y * c_sizea.z;
        }
    }
}

/*
 * Copy a smaller array into a given position in a larger one.
 *
 * For example:
 *                  . . . .
 *   a = 1 2 -> b = . 1 2 .
 *       3 4        . 3 4 .
 *                  . . . .
 */
extern "C"
void Zgpu(bmgs_paste_gpu)(const Tgpu* a, const int sizea[3],
                          Tgpu* b, const int sizeb[3],
                          const int startb[3], int blocks,
                          gpuStream_t stream)
{
    if (!(sizea[0] && sizea[1] && sizea[2]))
        return;

    int3 hc_sizea, hc_sizeb;
    hc_sizea.x = sizea[0];
    hc_sizea.y = sizea[1];
    hc_sizea.z = sizea[2] * sizeof(Tgpu) / sizeof(double);
    hc_sizeb.x = sizeb[0];
    hc_sizeb.y = sizeb[1];
    hc_sizeb.z = sizeb[2] * sizeof(Tgpu) / sizeof(double);

    BLOCK_GRID(hc_sizea);

    b += startb[2] + (startb[1] + startb[0] * sizeb[1]) * sizeb[2];
    gpuLaunchKernel(
            Zgpu(bmgs_paste_kernel), dimGrid, dimBlock, 0, stream,
            (double*) a, hc_sizea, (double*) b, hc_sizeb, blocks, xdiv);
    gpuCheckLastError();
}

/*
 * Copy a smaller array into a given position in a larger one and
 * set all other elements to 0.
 */
extern "C"
void Zgpu(bmgs_paste_zero_gpu)(const Tgpu* a, const int sizea[3],
                                    Tgpu* b, const int sizeb[3],
                                    const int startb[3], int blocks,
                                    gpuStream_t stream)
{
    if (!(sizea[0] && sizea[1] && sizea[2]))
        return;

    int3 bc_blocks;
    int3 hc_sizea, hc_sizeb, hc_startb;
    hc_sizea.x = sizea[0];
    hc_sizea.y = sizea[1];
    hc_sizea.z = sizea[2];
    hc_sizeb.x = sizeb[0];
    hc_sizeb.y = sizeb[1];
    hc_sizeb.z = sizeb[2];
    hc_startb.x = startb[0];
    hc_startb.y = startb[1];
    hc_startb.z = startb[2];

    bc_blocks.y = hc_sizeb.y - hc_sizea.y > 0
                ? MAX((hc_sizeb.y - hc_sizea.y + BLOCK_SIZEY - 1)
                        / BLOCK_SIZEY, 1)
                : 0;
    bc_blocks.z = hc_sizeb.z - hc_sizea.z > 0
                ? MAX((hc_sizeb.z - hc_sizea.z + BLOCK_SIZEX - 1)
                        / BLOCK_SIZEX, 1)
                : 0;

    int gridy = blocks * ((sizeb[1] + BLOCK_SIZEY - 1) / BLOCK_SIZEY
                          + bc_blocks.y);
    int gridx = XDIV * ((sizeb[2] + BLOCK_SIZEX - 1) / BLOCK_SIZEX
                        + bc_blocks.z);

    dim3 dimBlock(BLOCK_SIZEX, BLOCK_SIZEY);
    dim3 dimGrid(gridx, gridy);

    gpuLaunchKernel(
            Zgpu(bmgs_paste_zero_kernel), dimGrid, dimBlock, 0, stream,
            (Tgpu*) a, hc_sizea, (Tgpu*) b, hc_sizeb, hc_startb,
            bc_blocks, blocks);
    gpuCheckLastError();
}

#ifndef GPU_USE_COMPLEX
#define GPU_USE_COMPLEX
#include "paste.cpp"
#endif
