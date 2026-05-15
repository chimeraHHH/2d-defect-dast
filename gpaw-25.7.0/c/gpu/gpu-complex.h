#ifdef GPAW_CUDA
#include <cuComplex.h>
#endif
#ifdef GPAW_HIP
#include <hip/hip_complex.h>
#endif
#include "gpu-runtime.h"

#undef Tgpu
#undef Zgpu
#undef MULTD
#undef MULDT
#undef ADD
#undef ADD3
#undef ADD4
#undef IADD
#undef MAKED
#undef MULTT
#undef CONJ
#undef REAL
#undef IMAG
#undef NEG

#ifndef GPU_USE_COMPLEX
#  define Tgpu           double
#  define Zgpu(f)        f
#  define MULTT(a,b)     ((a) * (b))
#  define MULTD(a,b)     ((a) * (b))
#  define MULDT(a,b)     ((a) * (b))
#  define ADD(a,b)       ((a) + (b))
#  define ADD3(a,b,c)    ((a) + (b) + (c))
#  define ADD4(a,b,c,d)  ((a) + (b) + (c) + (d))
#  define IADD(a,b)      ((a) += (b))
#  define MAKED(a)       (a)
#  define CONJ(a)        (a)
#  define REAL(a)        (a)
#  define IMAG(a)        (0)
#  define NEG(a)         (-(a))
#else
#  define Tgpu           gpuDoubleComplex
#  define Zgpu(f)        f ## z
#  define MULTT(a,b)     gpuCmul((a), (b))
#  define MULTD(a,b)     gpuCmulD((a), (b))
#  define MULDT(b,a)     MULTD((a), (b))
#  define ADD(a,b)       gpuCadd((a), (b))
#  define ADD3(a,b,c)    gpuCadd3((a), (b), (c))
#  define ADD4(a,b,c,d)  gpuCadd4((a), (b), (c), (d))
#  define IADD(a,b)      {(a).x += gpuCreal(b); (a).y += gpuCimag(b);}
#  define MAKED(a)       make_gpuDoubleComplex(a, 0)
#  define CONJ(a)        gpuConj(a)
#  define REAL(a)        gpuCreal(a)
#  define IMAG(a)        gpuCimag(a)
#  define NEG(a)         gpuCneg(a)
#endif

#ifndef GPU_COMPLEX_H
#define GPU_COMPLEX_H

__host__ __device__ static __inline__ gpuDoubleComplex gpuCmulD(
        gpuDoubleComplex x, double y)
{
    return make_gpuDoubleComplex(gpuCreal(x) * y, gpuCimag(x) * y);
}

#ifdef __cplusplus
__host__ __device__ static __inline__ gpuDoubleComplex operator*(
        gpuDoubleComplex x, double y)
{
    return gpuCmulD(x, y);
}

__host__ __device__ static __inline__ gpuFloatComplex operator*(
        gpuFloatComplex x, float y)
{
    return make_gpuFloatComplex(x.x * y, x.y * y);
}

__host__ __device__ static __inline__ gpuDoubleComplex operator*(
        gpuDoubleComplex x, gpuDoubleComplex y)
{
    return gpuCmul(x, y);
}

__host__ __device__ static __inline__ gpuFloatComplex operator*(
        gpuFloatComplex x, gpuFloatComplex y)
{
    return gpuCmulf(x, y);
}
__host__ __device__ static __inline__ gpuDoubleComplex operator-(
        gpuDoubleComplex x, gpuDoubleComplex y)
{
    return gpuCsub(x, y);
}
__host__ __device__ static __inline__ gpuFloatComplex operator-(
        gpuFloatComplex x, gpuFloatComplex y)
{
    return gpuCsubf(x, y);
}
__host__ __device__ static __inline__ gpuDoubleComplex operator+(
        gpuDoubleComplex x, gpuDoubleComplex y)
{
    return gpuCadd(x, y);
}
__host__ __device__ static __inline__ gpuFloatComplex operator+(
        gpuFloatComplex x, gpuFloatComplex y)
{
    return gpuCaddf(x, y);
}
#endif

__host__ __device__ static __inline__ gpuDoubleComplex gpuCneg(
        gpuDoubleComplex x)
{
    return make_gpuDoubleComplex(-gpuCreal(x), -gpuCimag(x));
}

__host__ __device__ static __inline__ gpuDoubleComplex gpuCadd3(
        gpuDoubleComplex x, gpuDoubleComplex y, gpuDoubleComplex z)
{
    return make_gpuDoubleComplex(gpuCreal(x) + gpuCreal(y) + gpuCreal(z),
                                 gpuCimag(x) + gpuCimag(y) + gpuCimag(z));
}

__host__ __device__ static __inline__ gpuDoubleComplex gpuCadd4(
        gpuDoubleComplex x, gpuDoubleComplex y, gpuDoubleComplex z,
        gpuDoubleComplex w)
{
    return make_gpuDoubleComplex(
            gpuCreal(x) + gpuCreal(y) + gpuCreal(z) + gpuCreal(w),
            gpuCimag(x) + gpuCimag(y) + gpuCimag(z) + gpuCimag(w));
}

#endif
