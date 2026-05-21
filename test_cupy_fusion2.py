import cupy as cp
cp.cuda.compiler.nvrtc_options = ('-I/usr/local/cuda/include',)
@cp.fuse()
def my_fuse(x, y):
    return x * y + x
x = cp.arange(10, dtype=cp.float32)
y = cp.arange(10, dtype=cp.float32)
print(my_fuse(x, y))
