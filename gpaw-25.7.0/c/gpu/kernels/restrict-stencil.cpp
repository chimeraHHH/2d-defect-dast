#define ACACHE_K (2 * (K - 1))

__global__ void RST1D_kernel(
        const Tgpu* a, int n, int m, Tgpu* b, int ang, int bng, int blocks)
{
    __shared__ Tgpu ac[ACACHE_Y * ACACHE_X];
    Tgpu *acp;

    int jtid = threadIdx.x;
    int j = blockIdx.x * BLOCK;

    int itid = threadIdx.y;
    int yy = gridDim.y / blocks;
    int blocksi = blockIdx.y / yy;
    int ibl = blockIdx.y - yy * blocksi;
    int i = ibl * BLOCK;

    int sizex = n * 2 + K * 2 - 3;
    int aind = (j + itid) * sizex + i * 2 + jtid + K - 1;

    a += blocksi * ang + aind;
    b += blocksi * bng + (j + jtid) + (i + itid) * m;

    acp = ac + ACACHE_X * itid + jtid + ACACHE_K / 2;
    if (aind < ang)
        acp[0] = a[0];
    if ((aind + BLOCK) < ang)
        acp[BLOCK] = a[BLOCK];
    if  (jtid < ACACHE_K / 2) {
        if (aind - ACACHE_K / 2 < ang)
            acp[-ACACHE_K / 2] = a[-ACACHE_K / 2];
        if (aind + 2 * BLOCK < ang)
            acp[2 * BLOCK] = a[2 * BLOCK];
    }
    acp = ac + ACACHE_X * (jtid) + 2 * itid + ACACHE_K / 2;
    __syncthreads();

    if (((i + itid) < n) && ((j + jtid) < m)) {
        if      (K == 2)
            b[0] = MULDT(0.5,
                         ADD(acp[0],
                             MULDT(0.5, ADD(acp[1], acp[-1]))));
        else if (K == 4)
            b[0] = MULDT(0.5,
                         ADD(acp[0],
                             ADD(MULDT( 0.5625, ADD(acp[1], acp[-1])),
                                 MULDT(-0.0625, ADD(acp[3], acp[-3])))));
        else if (K == 6)
            b[0] = MULDT(0.5,
                         ADD(ADD(acp[0],
                                 MULDT( 0.58593750, ADD(acp[1], acp[-1]))),
                             ADD(MULDT(-0.09765625, ADD(acp[3], acp[-3])),
                                 MULDT( 0.01171875, ADD(acp[5], acp[-5])))));
        else
            b[0] = MULDT(
                    0.5,
                    ADD(acp[0],
                        ADD(ADD(MULDT( 0.59814453125, ADD(acp[1], acp[-1])),
                                MULDT(-0.11962890625, ADD(acp[3], acp[-3]))),
                            ADD(MULDT( 0.02392578125, ADD(acp[5], acp[-5])),
                                MULDT(-0.00244140625, ADD(acp[7], acp[-7]))))));
    }
}

void RST1D(const Tgpu* a, int n, int m, Tgpu* b, int ang, int bng, int blocks)
{
    int gridx = (m + BLOCK - 1) / BLOCK;
    int gridy = (n + BLOCK - 1) / BLOCK;

    dim3 dimBlock(BLOCK, BLOCK);
    dim3 dimGrid(gridx, gridy * blocks);

    gpuLaunchKernel(
            RST1D_kernel, dimGrid, dimBlock, 0, 0,
            a, n, m, b, ang, bng, blocks);
    gpuCheckLastError();
}

#undef ACACHE_K
