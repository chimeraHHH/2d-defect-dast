import os

compiler = 'gcc'
mpi = True        # 开启 MPI
fftw = True
scalapack = False  # 单节点通常不需要 scalapack

# 链接的底层数学和物理库，注意添加 mpi 相关的链接库
libraries = ['xc', 'fftw3', 'openblas', 'mpi']

# 指向 Conda 虚拟环境的库路径
conda_prefix = os.environ.get('CONDA_PREFIX', '/root/miniconda3/envs/gpaw-env')
library_dirs = [os.path.join(conda_prefix, 'lib')]
include_dirs = [os.path.join(conda_prefix, 'include')]
runtime_library_dirs = [os.path.join(conda_prefix, 'lib')]

# 核心 GPU 配置
gpu = True
gpu_target = 'cuda'
gpu_compiler = 'nvcc'

# RTX 5090 属于 Blackwell 架构，这里使用 sm_89 (Ada架构) 以保证最高兼容性，同时能充分调用 Tensor Core
gpu_compile_args = ['-O3', '-g', '-arch=sm_89']

libraries += ['cudart', 'cublas']
