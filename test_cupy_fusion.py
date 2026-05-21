import os
os.environ['CUPY_NVCC_GENERATE_PTX_ARGS'] = '-I/usr/local/cuda/include'
import cupy as cp

@cp.fuse()
def my_fuse(x, y):
    return x * y + x

x = cp.arange(10, dtype=cp.float32)
y = cp.arange(10, dtype=cp.float32)
print(my_fuse(x, y))
