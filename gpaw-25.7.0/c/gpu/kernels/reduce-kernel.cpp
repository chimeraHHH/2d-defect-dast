__device__ unsigned int INNAME(retirementCount) = {0};

__global__ void INNAME(reduce_kernel)(
        const Tgpu *g_idata1, const Tgpu *g_idata2, Tgpu *g_odata,
        Tgpu *results, unsigned int n, unsigned int block_in,
        int block_out, int nvec)
{
    extern __shared__ Tgpu Zgpu(sdata)[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = REDUCE_THREADS * 2 * gridDim.x;

    unsigned int i_vec = blockIdx.y;
    unsigned int i = blockIdx.x * (REDUCE_THREADS * 2) + threadIdx.x;
    Tgpu mySum = MAKED(0);

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n) {
        IADD(mySum, INFUNC(g_idata1[i + block_in * i_vec],
                           g_idata2[i + block_in * i_vec]));
        // ensure we don't read out of bounds
        if (i + REDUCE_THREADS < n) {
            IADD(mySum,
                 INFUNC(g_idata1[i + block_in * i_vec + REDUCE_THREADS],
                        g_idata2[i + block_in * i_vec + REDUCE_THREADS]));
        }
        i += gridSize;
    }
    Zgpu(sdata)[tid] = mySum;
    __syncthreads();

    if (REDUCE_THREADS >= 512) {
        if (tid < 256) {
            Zgpu(sdata)[tid] = mySum = ADD(mySum, Zgpu(sdata)[tid + 256]);
        }
        __syncthreads();
    }
    if (REDUCE_THREADS >= 256) {
        if (tid < 128) {
            Zgpu(sdata)[tid] = mySum = ADD(mySum, Zgpu(sdata)[tid + 128]);
        }
        __syncthreads();
    }
    if (REDUCE_THREADS >= 128) {
        if (tid <  64) {
            Zgpu(sdata)[tid] = mySum = ADD(mySum, Zgpu(sdata)[tid + 64]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile Tgpu *smem = Zgpu(sdata);
#ifdef GPU_USE_COMPLEX
        if (REDUCE_THREADS >= 64) {
            smem[tid].x = mySum.x = mySum.x + smem[tid + 32].x;
            smem[tid].y = mySum.y = mySum.y + smem[tid + 32].y;
        }
        if (REDUCE_THREADS >= 32) {
            smem[tid].x = mySum.x = mySum.x + smem[tid + 16].x;
            smem[tid].y = mySum.y = mySum.y + smem[tid + 16].y;
        }
        if (REDUCE_THREADS >= 16) {
            smem[tid].x = mySum.x = mySum.x + smem[tid + 8].x;
            smem[tid].y = mySum.y = mySum.y + smem[tid + 8].y;
        }
        if (REDUCE_THREADS >= 8) {
            smem[tid].x = mySum.x = mySum.x + smem[tid + 4].x;
            smem[tid].y = mySum.y = mySum.y + smem[tid + 4].y;
        }
        if (REDUCE_THREADS >= 4) {
            smem[tid].x = mySum.x = mySum.x + smem[tid + 2].x;
            smem[tid].y = mySum.y = mySum.y + smem[tid + 2].y;
        }
        if (REDUCE_THREADS >= 2) {
            smem[tid].x = mySum.x = mySum.x + smem[tid + 1].x;
            smem[tid].y = mySum.y = mySum.y + smem[tid + 1].y;
        }
#else
        if (REDUCE_THREADS >= 64)
            smem[tid] = mySum = ADD(mySum, smem[tid + 32]);
        if (REDUCE_THREADS >= 32)
            smem[tid] = mySum = ADD(mySum, smem[tid + 16]);
        if (REDUCE_THREADS >= 16)
            smem[tid] = mySum = ADD(mySum, smem[tid + 8]);
        if (REDUCE_THREADS >= 8)
            smem[tid] = mySum = ADD(mySum, smem[tid + 4]);
        if (REDUCE_THREADS >= 4)
            smem[tid] = mySum = ADD(mySum, smem[tid + 2]);
        if (REDUCE_THREADS >= 2)
            smem[tid] = mySum = ADD(mySum, smem[tid + 1]);
#endif
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x + block_out * i_vec] = Zgpu(sdata)[0];

    if (gridDim.x == 1) {
        __shared__ bool amLast;
        __threadfence();
        if (tid == 0) {
            unsigned int ticket = atomicInc(&INNAME(retirementCount),
                                            gridDim.y);
            amLast = (ticket == gridDim.y - 1);
        }
        __syncthreads();
        if (amLast) {
            for (int i=tid; i < nvec; i += blockDim.x)
                results[i] = g_odata[i * block_out];
            INNAME(retirementCount) = 0;
        }
    }
}
