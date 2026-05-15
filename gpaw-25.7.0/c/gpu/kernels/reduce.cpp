#ifndef REDUCE
#include "../gpu.h"

#define REDUCE_MAX_THREADS  (256)
#define REDUCE_MAX_BLOCKS   (64)
#define REDUCE_MAX_NVEC     (128*1024)
#define REDUCE_BUFFER_SIZE  ((REDUCE_MAX_NVEC + 2 * GPU_BLOCKS_MAX \
                              * REDUCE_MAX_BLOCKS) * 16)

static void *reduce_buffer = NULL;

extern "C"
void reduce_init_buffers_gpu()
{
    reduce_buffer = NULL;
}

extern "C"
void reduce_dealloc_gpu()
{
    gpuFree(reduce_buffer);
    gpuCheckLastError();
    reduce_init_buffers_gpu();
}

static void reduceNumBlocksAndThreads(int n, int *blocks, int *threads)
{
    *threads = (n < REDUCE_MAX_THREADS * 2) ? nextPow2((n + 1) / 2)
                                            : REDUCE_MAX_THREADS;
    *blocks = MIN((n + (*threads * 2 - 1)) / (*threads * 2),
                  REDUCE_MAX_BLOCKS);
}

#endif
#define REDUCE

#define INFUNC(a,b) MAPFUNC(a,b)

#define INNAME(f) MAPNAME(f ## _map512)
#define REDUCE_THREADS   512
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## _map256)
#define REDUCE_THREADS   256
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## _map128)
#define REDUCE_THREADS   128
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## _map64)
#define REDUCE_THREADS   64
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## _map32)
#define REDUCE_THREADS   32
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## _map16)
#define REDUCE_THREADS   16
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## _map8)
#define REDUCE_THREADS   8
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## _map4)
#define REDUCE_THREADS   4
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## _map2)
#define REDUCE_THREADS   2
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## _map1)
#define REDUCE_THREADS   1
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#undef  INFUNC
#define INFUNC(a,b) (a)

#define INNAME(f) MAPNAME(f ## 512)
#define REDUCE_THREADS   512
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## 256)
#define REDUCE_THREADS   256
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## 128)
#define REDUCE_THREADS   128
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## 64)
#define REDUCE_THREADS   64
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## 32)
#define REDUCE_THREADS   32
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## 16)
#define REDUCE_THREADS   16
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## 8)
#define REDUCE_THREADS   8
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## 4)
#define REDUCE_THREADS   4
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## 2)
#define REDUCE_THREADS   2
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) MAPNAME(f ## 1)
#define REDUCE_THREADS   1
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#undef  INFUNC

void MAPNAME(reducemap)(const Tgpu *d_idata1, const Tgpu *d_idata2,
                        Tgpu *d_odata, int size, int nvec)
{
    int blocks, threads;

    if (reduce_buffer == NULL) {
        gpuMalloc(&reduce_buffer, REDUCE_BUFFER_SIZE);
    }
    reduceNumBlocksAndThreads(size, &blocks, &threads);

    int min_wsize = blocks;
    int work_buffer_size = ((REDUCE_BUFFER_SIZE) / sizeof(Tgpu) - nvec) / 2;

    assert(min_wsize < work_buffer_size);

    int mynvec = MAX(MIN(work_buffer_size / min_wsize, nvec), 1);

    Tgpu *result_gpu = (Tgpu*) d_odata;
    Tgpu *work_buffer1 = (Tgpu*) reduce_buffer;
    Tgpu *work_buffer2 = work_buffer1 + work_buffer_size;

    int smemSize = (threads <= 32) ? 2 * threads * sizeof(Tgpu)
                                   : threads * sizeof(Tgpu);

    for (int i=0; i < nvec; i += mynvec) {
        int cunvec = MIN(mynvec, nvec - i);
        dim3 dimBlock(threads, 1, 1);
        dim3 dimGrid(blocks, cunvec, 1);
        int block_out = blocks;
        int s = size;
        switch (threads) {
            case 512:
                gpuLaunchKernel(
                        MAPNAME(reduce_kernel_map512),
                        dimGrid, dimBlock, smemSize, 0,
                        d_idata1 + i * size, d_idata2 + i * size,
                        (Tgpu*) work_buffer1, result_gpu + i, s, size,
                        block_out, cunvec);
                break;
            case 256:
                gpuLaunchKernel(
                        MAPNAME(reduce_kernel_map256),
                        dimGrid, dimBlock, smemSize, 0,
                        d_idata1 + i * size, d_idata2 + i * size,
                        (Tgpu*) work_buffer1, result_gpu + i, s, size,
                        block_out, cunvec);
                break;
            case 128:
                gpuLaunchKernel(
                        MAPNAME(reduce_kernel_map128),
                        dimGrid, dimBlock, smemSize, 0,
                        d_idata1 + i * size, d_idata2 + i * size,
                        (Tgpu*) work_buffer1, result_gpu + i, s, size,
                        block_out, cunvec);
                break;
            case 64:
                gpuLaunchKernel(
                        MAPNAME(reduce_kernel_map64),
                        dimGrid, dimBlock, smemSize, 0,
                        d_idata1 + i * size, d_idata2 + i * size,
                        (Tgpu*) work_buffer1, result_gpu + i, s, size,
                        block_out, cunvec);
                break;
            case 32:
                gpuLaunchKernel(
                        MAPNAME(reduce_kernel_map32),
                        dimGrid, dimBlock, smemSize, 0,
                        d_idata1 + i * size, d_idata2 + i * size,
                        (Tgpu*) work_buffer1, result_gpu + i, s, size,
                        block_out, cunvec);
                break;
            case 16:
                gpuLaunchKernel(
                        MAPNAME(reduce_kernel_map16),
                        dimGrid, dimBlock, smemSize, 0,
                        d_idata1 + i * size, d_idata2 + i * size,
                        (Tgpu*) work_buffer1, result_gpu + i, s, size,
                        block_out, cunvec);
                break;
            case  8:
                gpuLaunchKernel(
                        MAPNAME(reduce_kernel_map8),
                        dimGrid, dimBlock, smemSize, 0,
                        d_idata1 + i * size, d_idata2 + i * size,
                        (Tgpu*) work_buffer1, result_gpu + i, s, size,
                        block_out, cunvec);
                break;
            case  4:
                gpuLaunchKernel(
                        MAPNAME(reduce_kernel_map4),
                        dimGrid, dimBlock, smemSize, 0,
                        d_idata1 + i * size, d_idata2 + i * size,
                        (Tgpu*) work_buffer1, result_gpu + i, s, size,
                        block_out, cunvec);
                break;
            case  2:
                gpuLaunchKernel(
                        MAPNAME(reduce_kernel_map2),
                        dimGrid, dimBlock, smemSize, 0,
                        d_idata1 + i * size, d_idata2 + i * size,
                        (Tgpu*) work_buffer1, result_gpu + i, s, size,
                        block_out, cunvec);
                break;
            case  1:
                gpuLaunchKernel(
                        MAPNAME(reduce_kernel_map1),
                        dimGrid, dimBlock, smemSize, 0,
                        d_idata1 + i * size, d_idata2 + i * size,
                        (Tgpu*) work_buffer1, result_gpu + i, s, size,
                        block_out, cunvec);
                break;
            default:
                assert(0);
        }
        gpuCheckLastError();

        s = blocks;
        int count = 0;
        while (s > 1)  {
            int blocks2, threads2;
            int block_in = block_out;
            reduceNumBlocksAndThreads(s, &blocks2, &threads2);
            block_out = blocks2;
            dim3 dimBlock(threads2, 1, 1);
            dim3 dimGrid(blocks2, cunvec, 1);
            int smemSize = (threads2 <= 32) ? 2 * threads2 * sizeof(Tgpu)
                                            : threads2 * sizeof(Tgpu);

            Tgpu *work1 = (count % 2) ? work_buffer2 : work_buffer1;
            Tgpu *work2 = (count % 2) ? work_buffer1 : work_buffer2;
            count++;

            switch (threads2) {
                case 512:
                    gpuLaunchKernel(
                            MAPNAME(reduce_kernel512),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i, s, block_in, block_out, cunvec);
                    break;
                case 256:
                    gpuLaunchKernel(
                            MAPNAME(reduce_kernel256),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i, s, block_in, block_out, cunvec);
                    break;
                case 128:
                    gpuLaunchKernel(
                            MAPNAME(reduce_kernel128),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i, s, block_in, block_out, cunvec);
                    break;
                case 64:
                    gpuLaunchKernel(
                            MAPNAME(reduce_kernel64),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i, s, block_in, block_out, cunvec);
                    break;
                case 32:
                    gpuLaunchKernel(
                            MAPNAME(reduce_kernel32),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i, s, block_in, block_out, cunvec);
                    break;
                case 16:
                    gpuLaunchKernel(
                            MAPNAME(reduce_kernel16),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i, s, block_in, block_out, cunvec);
                    break;
                case  8:
                    gpuLaunchKernel(
                            MAPNAME(reduce_kernel8),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i, s, block_in, block_out, cunvec);
                    break;
                case  4:
                    gpuLaunchKernel(
                            MAPNAME(reduce_kernel4),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i, s, block_in, block_out, cunvec);
                    break;
                case  2:
                    gpuLaunchKernel(
                            MAPNAME(reduce_kernel2),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i, s, block_in, block_out, cunvec);
                    break;
                case  1:
                    gpuLaunchKernel(
                            MAPNAME(reduce_kernel1),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i, s, block_in, block_out, cunvec);
                    break;
                default:
                    assert(0);
            }
            gpuCheckLastError();

            s = (s + (threads2 * 2 - 1)) / (threads2 * 2);
        }
    }
}
