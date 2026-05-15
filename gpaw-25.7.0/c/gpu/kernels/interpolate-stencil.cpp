__global__ void IP1D_kernel(
        const Tgpu* a, int n, int m, Tgpu* b, int skip0, int skip1)
{
    a += K / 2 - 1;

    int j = blockIdx.x * BLOCK_X + threadIdx.x;
    int i = blockIdx.y * BLOCK_Y + threadIdx.y;

    if (j >= m || i >= n) {
        return;
    }

    a += j * (K - 1 - skip1 + n) + i;
    b += j + (2 * m * i);

    if (skip0) {
        b -= m;
    }

    if (i > 0 || !skip0) {
        b[0] = a[0];
    }

    if (i == n - 1 && skip1) {
        b -= m;
    } else {
        if (K == 2) {
            b[m] = MULDT(0.5, ADD(a[0], a[1]));
        } else if (K == 4) {
            b[m] = ADD(MULDT( 0.5625, ADD(a[ 0], a[1])),
                       MULDT(-0.0625, ADD(a[-1], a[2])));
        } else if (K == 6) {
            b[m] = ADD(ADD(MULDT( 0.58593750, ADD(a[ 0], a[1])),
                           MULDT(-0.09765625, ADD(a[-1], a[2]))),
                       MULDT(0.01171875, ADD(a[-2], a[3])));
        } else {
            b[m] = ADD(ADD(MULDT( 0.59814453125, ADD(a[ 0], a[1])),
                           MULDT(-0.11962890625, ADD(a[-1], a[2]))),
                       ADD(MULDT( 0.02392578125, ADD(a[-2], a[3])),
                           MULDT(-0.00244140625, ADD(a[-3], a[4]))));
        }
    }
}

void IP1D(const Tgpu* a, int n, int m, Tgpu* b, int skip[2])
{
    int gridx = (m + BLOCK_X - 1) / BLOCK_X;
    int gridy = (n + BLOCK_Y - 1) / BLOCK_Y;

    dim3 dimBlock(BLOCK_X, BLOCK_Y);
    dim3 dimGrid(gridx, gridy);

    gpuLaunchKernel(
            IP1D_kernel, dimGrid, dimBlock, 0, 0,
            a, n, m, b, skip[0], skip[1]);

    gpuCheckLastError();
}
