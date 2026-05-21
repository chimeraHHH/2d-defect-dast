import os
os.environ['CUPY_NVCC_GENERATE_PTX_ARGS'] = '-I/usr/local/cuda/include'
import cupy as cp
x = cp.arange(10)
y = x * 2
print(y)
