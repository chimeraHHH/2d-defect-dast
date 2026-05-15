#include <string.h>

#include "../gpu.h"
#include "../gpu-complex.h"

#ifndef GPU_USE_COMPLEX
#define BLOCK_MAX 32
#define GRID_MAX 65535
#define BLOCK_TOTALMAX 256
#endif


/*
 * GPU kernel to copy a slice of an array.
 */
__global__ void Zgpu(bmgs_cut_kernel)(
        const Tgpu* a, const int3 c_sizea, Tgpu* b, const int3 c_sizeb,
#ifdef GPU_USE_COMPLEX
        gpuDoubleComplex phase,
#endif
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

    while (xind < c_sizeb.x) {
        if ((i2 < c_sizeb.z) && (i1 < c_sizeb.y)) {
#ifndef GPU_USE_COMPLEX
            b[0] = a[0];
#else
            b[0] = MULTT(phase, a[0]);
#endif
        }
        b += xdiv * c_sizeb.y * c_sizeb.z;
        a += xdiv * c_sizea.y * c_sizea.z;
        xind += xdiv;
    }
}

/*
 * Copy a slice of an array on the GPU. If the array contains complex
 * numbers, then multiply each element with the given phase.
 *
 * For example:
 *       . . . .               (OR for complex numbers)
 *   a = . 1 2 . -> b = 1 2     -> b = phase*1 phase*2
 *       . 3 4 .        3 4            phase*3 phase*4
 *       . . . .
 *
 * arguments:
 *   a      -- input array
 *   sizea  -- dimensions of the array a
 *   starta -- offset to the start of the slice
 *   b      -- output array
 *   sizeb  -- dimensions of the array b
 *   phase  -- phase (only for complex)
 *   blocks -- number of blocks
 *   stream -- GPU stream to use
 */
extern "C"
void Zgpu(bmgs_cut_gpu)(
        const Tgpu* a, const int sizea[3], const int starta[3],
        Tgpu* b, const int sizeb[3],
#ifdef GPU_USE_COMPLEX
        gpuDoubleComplex phase,
#endif
        int blocks, gpuStream_t stream)
{
    if (!(sizea[0] && sizea[1] && sizea[2]))
        return;
    int3 hc_sizea, hc_sizeb;
    hc_sizea.x = sizea[0];
    hc_sizea.y = sizea[1];
    hc_sizea.z = sizea[2];
    hc_sizeb.x = sizeb[0];
    hc_sizeb.y = sizeb[1];
    hc_sizeb.z = sizeb[2];

    BLOCK_GRID(hc_sizeb);

    a += starta[2] + (starta[1] + starta[0] * hc_sizea.y) * hc_sizea.z;

    gpuLaunchKernel(Zgpu(bmgs_cut_kernel), dimGrid, dimBlock, 0, stream,
                    (Tgpu*) a, hc_sizea, (Tgpu*) b, hc_sizeb,
#ifdef GPU_USE_COMPLEX
                    phase,
#endif
                    blocks, xdiv);
    gpuCheckLastError();
}

#ifndef GPU_USE_COMPLEX
#define GPU_USE_COMPLEX
#include "cut.cpp"
#endif
