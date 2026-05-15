__device__ unsigned int INNAME(lfc_retirementCount) = {0};

__global__ void INNAME(integrate_mul_kernel)(
        const Tgpu *a_G, int nG,
        const LFVolume_gpu *volume_W,
        const int *volume_WMi_gpu,
        const int *WMi_gpu,
        int WMimax,
        int q,
        Tgpu *out, int block_out,
        Tgpu *results, int Mcount, int nM, int nvec)
{
    int yy = gridDim.y / Mcount;

    int bloy = blockIdx.y / yy;
    int block = blockIdx.y - bloy * yy;

    unsigned int tid = threadIdx.x;
    unsigned int gridSize = REDUCE_LFC_THREADS * gridDim.x;
    unsigned int i_b = blockIdx.x * (REDUCE_LFC_THREADS) + tid;

    extern __shared__ Tgpu Zgpu(sdata)[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    a_G += nG * block;
    for (int vv=0; vv < WMi_gpu[bloy]; vv++) {
        const LFVolume_gpu *v = &volume_W[volume_WMi_gpu[bloy * WMimax
                                                         + vv]];
        int *nGBcum = v->nGBcum;
#ifdef GPU_USE_COMPLEX
        Tgpu phase = v->phase_k[q];
#endif
        int len_A_gm = v->len_A_gm;
        Tgpu *out_t = out + v->M * block_out + block * nM * block_out;
        int a_ind, ai=0, acum=0;

        if (i_b < len_A_gm) {
            int bi = v->nB;
            int ci;
            int bcum = nGBcum[bi];
            int ccum;
            while (bi - ai > 1) {
                ci = ai + 1 + (bi - ai - 2) * (i_b - acum) / (bcum - acum);
                ccum = nGBcum[ci];
                if (ccum <= i_b) {
                    ai = ci;
                    acum = ccum;
                } else {
                    bi = ci;
                    bcum = ccum;
                }
            }
            a_ind = v->GB1[ai] + i_b - acum;
        }
        for (int i=0; i < nvec; i++) {
            Tgpu a_Gv;
            double *A_gm2 = v->A_gm;
            Tgpu *out_t2 = out_t;
            if (i_b < len_A_gm) {
#ifdef GPU_USE_COMPLEX
                a_Gv = MULTT(a_G[i * nG + a_ind], phase);
#else
                a_Gv = a_G[i * nG + a_ind];
#endif
            }
            for (int m=0; m < v->nm; m++) {
                Tgpu mySum = MAKED(0);
                if (i_b < len_A_gm) {
                    mySum = MULTD(a_Gv, A_gm2[i_b]);
                }
                if (len_A_gm > gridSize) {
                    unsigned int i_bb = i_b + gridSize;
                    int aai = ai;
                    int aacum = acum;
                    while (i_bb < len_A_gm) {
                        int bi = v->nB;
                        int ci;
                        int bcum = nGBcum[bi];
                        int ccum;
                        while (bi - aai > 1) {
                            ci = aai + 1 + (bi - aai - 2) * (i_bb - aacum)
                                           / (bcum - aacum);
                            ccum = nGBcum[ci];
                            if (ccum <= i_bb) {
                                aai = ci;
                                aacum = ccum;
                            } else {
                                bi = ci;
                                bcum = ccum;
                            }
                        }
#ifdef GPU_USE_COMPLEX
                        IADD(mySum, MULTD(MULTT(a_G[i * nG + v->GB1[aai]
                                                    + i_bb - aacum], phase),
                                          A_gm2[i_bb]));
#else
                        IADD(mySum, MULTD(a_G[i * nG + v->GB1[aai] + i_bb
                                              - aacum],
                                          A_gm2[i_bb]));
#endif
                        i_bb += gridSize;
                    }
                }
                Zgpu(sdata)[tid] = mySum;
                __syncthreads();

                if (REDUCE_LFC_THREADS >= 512) {
                    if (tid < 256) {
                        Zgpu(sdata)[tid] = mySum
                                         = ADD(mySum,
                                               Zgpu(sdata)[tid + 256]);
                    }
                    __syncthreads();
                }
                if (REDUCE_LFC_THREADS >= 256) {
                    if (tid < 128) {
                        Zgpu(sdata)[tid] = mySum
                                         = ADD(mySum,
                                               Zgpu(sdata)[tid + 128]);
                    }
                    __syncthreads();
                }
                if (REDUCE_LFC_THREADS >= 128) {
                    if (tid <  64) {
                        Zgpu(sdata)[tid] = mySum
                                         = ADD(mySum,
                                               Zgpu(sdata)[tid + 64]);
                    }
                    __syncthreads();
                }

                if (tid < 32) {
                    volatile Tgpu *smem = Zgpu(sdata);
#ifdef GPU_USE_COMPLEX
                    if (REDUCE_LFC_THREADS >= 64) {
                        smem[tid].x = mySum.x = mySum.x + smem[tid + 32].x;
                        smem[tid].y = mySum.y = mySum.y + smem[tid + 32].y;
                    }
                    if (REDUCE_LFC_THREADS >= 32) {
                        smem[tid].x = mySum.x = mySum.x + smem[tid + 16].x;
                        smem[tid].y = mySum.y = mySum.y + smem[tid + 16].y;
                    }
                    if (REDUCE_LFC_THREADS >= 16) {
                        smem[tid].x = mySum.x = mySum.x + smem[tid + 8].x;
                        smem[tid].y = mySum.y = mySum.y + smem[tid + 8].y;
                    }
                    if (REDUCE_LFC_THREADS >= 8) {
                        smem[tid].x = mySum.x = mySum.x + smem[tid + 4].x;
                        smem[tid].y = mySum.y = mySum.y + smem[tid + 4].y;
                    }
                    if (REDUCE_LFC_THREADS >= 4) {
                        smem[tid].x = mySum.x = mySum.x + smem[tid + 2].x;
                        smem[tid].y = mySum.y = mySum.y + smem[tid + 2].y;
                    }
                    if (REDUCE_LFC_THREADS >= 2) {
                        smem[tid].x = mySum.x = mySum.x + smem[tid + 1].x;
                        smem[tid].y = mySum.y = mySum.y + smem[tid + 1].y;
                    }
#else
                    if (REDUCE_LFC_THREADS >= 64)
                        smem[tid] = mySum = ADD(mySum, smem[tid + 32]);
                    if (REDUCE_LFC_THREADS >= 32)
                        smem[tid] = mySum = ADD(mySum, smem[tid + 16]);
                    if (REDUCE_LFC_THREADS >= 16)
                        smem[tid] = mySum = ADD(mySum, smem[tid + 8]);
                    if (REDUCE_LFC_THREADS >= 8)
                        smem[tid] = mySum = ADD(mySum, smem[tid + 4]);
                    if (REDUCE_LFC_THREADS >= 4)
                        smem[tid] = mySum = ADD(mySum, smem[tid + 2]);
                    if (REDUCE_LFC_THREADS >= 2)
                        smem[tid] = mySum = ADD(mySum, smem[tid + 1]);
#endif
                }

                // write result for this block to global mem
                if (tid==0) {
                    if (vv==0)
                        out_t2[blockIdx.x] = Zgpu(sdata)[0];
                    else
                        IADD(out_t2[blockIdx.x], Zgpu(sdata)[0]);
                }
                A_gm2 += len_A_gm;
                out_t2 += block_out;
                __syncthreads();
            }
            out_t += nM * block_out;
        }
    }

    if (gridDim.x==1) {
        __shared__ bool amLast;
        __threadfence();
        if (tid == 0) {
            unsigned int ticket = atomicInc(&INNAME(lfc_retirementCount),
                                            gridDim.y);
            amLast = (ticket == gridDim.y - 1);
        }
        __syncthreads();
        if ((amLast)) {
            for (int i=tid; i < nM * yy * nvec; i += blockDim.x) {
                results[i] = out[i * block_out];
            }
            INNAME(lfc_retirementCount) = 0;
        }
    }
}
