import os
import cupy as cp
# 告诉 cupy 编译器包含 cuda 头文件路径
if not hasattr(cp.cuda.compiler, '_extra_options'):
    cp.cuda.compiler._extra_options = []
# CuPy NVRTC options
os.environ['NVCC'] = 'nvcc'
os.environ['CUPY_NVCC_GENERATE_PTX_ARGS'] = '-I/usr/local/cuda/include'

@cp.fuse()
def my_fuse(x, y):
    return x * y + x
x = cp.arange(10, dtype=cp.float32)
y = cp.arange(10, dtype=cp.float32)
print(my_fuse(x, y))
