#ifndef REDUCE_LFC

#define REDUCE_LFC_MAX_THREADS  (64)
#define REDUCE_LFC_MAX_THREADS2 (64)
#define REDUCE_LFC_MAX_BLOCKS   (32)
#define REDUCE_LFC_MAX_BLOCKS2  (32)
#define REDUCE_LFC_MAX_YBLOCKS  (65535)
#define REDUCE_LFC_BUFFER_SIZE  ((2 * GPU_BLOCKS_MAX \
                                 * MAX(REDUCE_LFC_MAX_BLOCKS, \
                                       REDUCE_LFC_MAX_BLOCKS2)) * 16)

static void *lfc_reduce_buffer = NULL;
static int lfc_reduce_buffer_size = 0;

extern "C"
void lfc_reduce_init_buffers_gpu()
{
    lfc_reduce_buffer = NULL;
    lfc_reduce_buffer_size = 0;
}

extern "C"
void lfc_reduce_dealloc_gpu()
{
    gpuFree(lfc_reduce_buffer);
    gpuCheckLastError();
    lfc_reduce_init_buffers_gpu();
}

static void lfc_reduceNumBlocksAndThreads(int n, int *blocks, int *threads)
{
    *threads = (n < REDUCE_LFC_MAX_THREADS) ? nextPow2(n)
                                            : REDUCE_LFC_MAX_THREADS;
    *blocks = MIN((n + (*threads - 1)) / (*threads), REDUCE_LFC_MAX_BLOCKS);
}

static void lfc_reduceNumBlocksAndThreads2(int n,int *blocks, int *threads)
{
    *threads = (n < REDUCE_LFC_MAX_THREADS2 * 2) ? nextPow2((n + 1) / 2)
                                                 : REDUCE_LFC_MAX_THREADS2;
    *blocks = MIN((n + (*threads * 2 - 1)) / (*threads * 2),
                  REDUCE_LFC_MAX_BLOCKS2);
}

#endif
#define REDUCE_LFC

#define INNAME(f) Zgpu(f ## _map512)
#define REDUCE_LFC_THREADS  512
#include "lfc-reduce-kernel.cpp"
#undef  REDUCE_LFC_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## _map256)
#define REDUCE_LFC_THREADS  256
#include "lfc-reduce-kernel.cpp"
#undef  REDUCE_LFC_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## _map128)
#define REDUCE_LFC_THREADS  128
#include "lfc-reduce-kernel.cpp"
#undef  REDUCE_LFC_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## _map64)
#define REDUCE_LFC_THREADS  64
#include "lfc-reduce-kernel.cpp"
#undef  REDUCE_LFC_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## _map32)
#define REDUCE_LFC_THREADS  32
#include "lfc-reduce-kernel.cpp"
#undef  REDUCE_LFC_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## _map16)
#define REDUCE_LFC_THREADS  16
#include "lfc-reduce-kernel.cpp"
#undef  REDUCE_LFC_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## _map8)
#define REDUCE_LFC_THREADS  8
#include "lfc-reduce-kernel.cpp"
#undef  REDUCE_LFC_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## _map4)
#define REDUCE_LFC_THREADS  4
#include "lfc-reduce-kernel.cpp"
#undef  REDUCE_LFC_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## _map2)
#define REDUCE_LFC_THREADS  2
#include "lfc-reduce-kernel.cpp"
#undef  REDUCE_LFC_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## _map1)
#define REDUCE_LFC_THREADS  1
#include "lfc-reduce-kernel.cpp"
#undef  REDUCE_LFC_THREADS
#undef  INNAME

#undef  INFUNC
#undef  REDUCE_THREADS

#define INFUNC(a,b) (a)
#define INNAME(f) Zgpu(f ## 512)
#define REDUCE_THREADS  512
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## 256)
#define REDUCE_THREADS  256
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## 128)
#define REDUCE_THREADS  128
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## 64)
#define REDUCE_THREADS  64
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## 32)
#define REDUCE_THREADS  32
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## 16)
#define REDUCE_THREADS  16
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## 8)
#define REDUCE_THREADS  8
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## 4)
#define REDUCE_THREADS  4
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## 2)
#define REDUCE_THREADS  2
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME

#define INNAME(f) Zgpu(f ## 1)
#define REDUCE_THREADS  1
#include "reduce-kernel.cpp"
#undef  REDUCE_THREADS
#undef  INNAME
#undef  INFUNC


void Zgpu(lfc_reducemap)(LFCObject *lfc, const Tgpu *a_G, int nG,
        Tgpu *c_xM, int nM, int nvec, int q)
{
    int blocks, threads;

    if (lfc_reduce_buffer_size < nM * REDUCE_LFC_BUFFER_SIZE) {
        lfc_reduce_dealloc_gpu();
        gpuMalloc(&lfc_reduce_buffer, nM * REDUCE_LFC_BUFFER_SIZE);
        lfc_reduce_buffer_size = nM * REDUCE_LFC_BUFFER_SIZE;
    }
    lfc_reduceNumBlocksAndThreads(lfc->max_len_A_gm, &blocks, &threads);

    int min_wsize = blocks * nM;
    int work_buffer_size = (lfc_reduce_buffer_size / sizeof(Tgpu)) / 2;

    assert(min_wsize < work_buffer_size);

    int mynvec = MAX(MIN(work_buffer_size / min_wsize, nvec), 1);

    mynvec = MIN(mynvec, (REDUCE_LFC_MAX_YBLOCKS) / nM);

    Tgpu *work_buffer1 = (Tgpu*) lfc_reduce_buffer;
    Tgpu *work_buffer2 = work_buffer1 + work_buffer_size;
    Tgpu *result_gpu = c_xM;

    int smemSize = (threads <= 32) ? 2 * threads * sizeof(Tgpu)
                                   : threads * sizeof(Tgpu);

    for (int i=0; i < nvec; i += mynvec) {
        int cunvec = MIN(mynvec, nvec - i);
        int innvec = 1;

        dim3 dimBlock(threads, 1, 1);
        dim3 dimGrid(blocks, lfc->Mcount, 1);
        int block_out = blocks;

        innvec = cunvec;

        switch (threads) {
            case 512:
                gpuLaunchKernel(
                        Zgpu(integrate_mul_kernel_map512),
                        dimGrid, dimBlock, smemSize, 0,
                        a_G + i * nG, nG, lfc->volume_W_gpu,
                        lfc->volume_WMi_gpu, lfc->WMi_gpu, lfc->WMimax, q,
                        (Tgpu*) work_buffer1, block_out, result_gpu + i * nM,
                        lfc->Mcount, nM, innvec);
                break;
            case 256:
                gpuLaunchKernel(
                        Zgpu(integrate_mul_kernel_map256),
                        dimGrid, dimBlock, smemSize, 0,
                        a_G + i * nG, nG, lfc->volume_W_gpu,
                        lfc->volume_WMi_gpu, lfc->WMi_gpu, lfc->WMimax, q,
                        (Tgpu*) work_buffer1, block_out, result_gpu + i * nM,
                        lfc->Mcount, nM, innvec);
                break;
            case 128:
                gpuLaunchKernel(
                        Zgpu(integrate_mul_kernel_map128),
                        dimGrid, dimBlock, smemSize, 0,
                        a_G + i * nG, nG, lfc->volume_W_gpu,
                        lfc->volume_WMi_gpu, lfc->WMi_gpu, lfc->WMimax, q,
                        (Tgpu*) work_buffer1, block_out, result_gpu + i * nM,
                        lfc->Mcount, nM, innvec);
                break;
            case 64:
                gpuLaunchKernel(
                        Zgpu(integrate_mul_kernel_map64),
                        dimGrid, dimBlock, smemSize, 0,
                        a_G + i * nG, nG, lfc->volume_W_gpu,
                        lfc->volume_WMi_gpu, lfc->WMi_gpu, lfc->WMimax, q,
                        (Tgpu*) work_buffer1, block_out, result_gpu + i * nM,
                        lfc->Mcount, nM, innvec);
                break;
            case 32:
                gpuLaunchKernel(
                        Zgpu(integrate_mul_kernel_map32),
                        dimGrid, dimBlock, smemSize, 0,
                        a_G + i * nG, nG, lfc->volume_W_gpu,
                        lfc->volume_WMi_gpu, lfc->WMi_gpu, lfc->WMimax, q,
                        (Tgpu*) work_buffer1, block_out, result_gpu + i * nM,
                        lfc->Mcount, nM, innvec);
                break;
            case 16:
                gpuLaunchKernel(
                        Zgpu(integrate_mul_kernel_map16),
                        dimGrid, dimBlock, smemSize, 0,
                        a_G + i * nG, nG, lfc->volume_W_gpu,
                        lfc->volume_WMi_gpu, lfc->WMi_gpu, lfc->WMimax, q,
                        (Tgpu*) work_buffer1, block_out, result_gpu + i * nM,
                        lfc->Mcount, nM, innvec);
                break;
            case  8:
                gpuLaunchKernel(
                        Zgpu(integrate_mul_kernel_map8),
                        dimGrid, dimBlock, smemSize, 0,
                        a_G + i * nG, nG, lfc->volume_W_gpu,
                        lfc->volume_WMi_gpu, lfc->WMi_gpu, lfc->WMimax, q,
                        (Tgpu*) work_buffer1, block_out, result_gpu + i * nM,
                        lfc->Mcount, nM, innvec);
                break;
            case  4:
                gpuLaunchKernel(
                        Zgpu(integrate_mul_kernel_map4),
                        dimGrid, dimBlock, smemSize, 0,
                        a_G + i * nG, nG, lfc->volume_W_gpu,
                        lfc->volume_WMi_gpu, lfc->WMi_gpu, lfc->WMimax, q,
                        (Tgpu*) work_buffer1, block_out, result_gpu + i * nM,
                        lfc->Mcount, nM, innvec);
                break;
            case  2:
                gpuLaunchKernel(
                        Zgpu(integrate_mul_kernel_map2),
                        dimGrid, dimBlock, smemSize, 0,
                        a_G + i * nG, nG,lfc->volume_W_gpu,
                        lfc->volume_WMi_gpu, lfc->WMi_gpu, lfc->WMimax, q,
                        (Tgpu*) work_buffer1, block_out, result_gpu + i * nM,
                        lfc->Mcount, nM, innvec);
                break;
            case  1:
                gpuLaunchKernel(
                        Zgpu(integrate_mul_kernel_map1),
                        dimGrid, dimBlock, smemSize, 0,
                        a_G + i * nG, nG, lfc->volume_W_gpu,
                        lfc->volume_WMi_gpu, lfc->WMi_gpu, lfc->WMimax, q,
                        (Tgpu*) work_buffer1, block_out, result_gpu + i * nM,
                        lfc->Mcount, nM, innvec);
                break;
            default:
                assert(0);
        }
        assert(!gpuCheckLastError());

        int s = blocks;
        int count = 0;
        while (s > 1) {
            int blocks2, threads2;
            int block_in = block_out;
            lfc_reduceNumBlocksAndThreads2(s, &blocks2, &threads2);
            block_out = blocks2;
            dim3 dimBlock(threads2, 1, 1);
            dim3 dimGrid(blocks2, cunvec * nM, 1);
            int smemSize = (threads2 <= 32) ? 2 * threads2 * sizeof(Tgpu)
                                            : threads2 * sizeof(Tgpu);

            Tgpu *work1 = (count % 2) ? work_buffer2 : work_buffer1;
            Tgpu *work2 = (count % 2) ? work_buffer1 : work_buffer2;
            count++;

            switch (threads2) {
                case 512:
                    gpuLaunchKernel(
                            Zgpu(reduce_kernel512),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i * nM, s, block_in, block_out,
                            cunvec * nM);
                    break;
                case 256:
                    gpuLaunchKernel(
                            Zgpu(reduce_kernel256),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i * nM, s, block_in, block_out,
                            cunvec * nM);
                    break;
                case 128:
                    gpuLaunchKernel(
                            Zgpu(reduce_kernel128),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i * nM, s, block_in, block_out,
                            cunvec * nM);
                    break;
                case 64:
                    gpuLaunchKernel(
                            Zgpu(reduce_kernel64),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i * nM, s, block_in, block_out,
                            cunvec * nM);
                    break;
                case 32:
                    gpuLaunchKernel(
                            Zgpu(reduce_kernel32),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i * nM, s, block_in, block_out,
                            cunvec * nM);
                    break;
                case 16:
                    gpuLaunchKernel(
                            Zgpu(reduce_kernel16),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i * nM, s, block_in, block_out,
                            cunvec * nM);
                    break;
                case  8:
                    gpuLaunchKernel(
                            Zgpu(reduce_kernel8),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i * nM, s, block_in, block_out,
                            cunvec * nM);
                    break;
                case  4:
                    gpuLaunchKernel(
                            Zgpu(reduce_kernel4),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i * nM, s, block_in, block_out,
                            cunvec * nM);
                    break;
                case  2:
                    gpuLaunchKernel(
                            Zgpu(reduce_kernel2),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i * nM, s, block_in, block_out,
                            cunvec * nM);
                    break;
                case  1:
                    gpuLaunchKernel(
                            Zgpu(reduce_kernel1),
                            dimGrid, dimBlock, smemSize, 0,
                            (Tgpu*) work1, NULL, (Tgpu*) work2,
                            result_gpu + i * nM, s, block_in, block_out,
                            cunvec * nM);
                    break;
                default:
                    assert(0);
            }
            assert(!gpuCheckLastError());
            s = (s + (threads2 * 2 - 1)) / (threads2 * 2);
        }
    }
}
