#ifndef GPU_ALIGN_H
#define GPU_ALIGN_H

#if defined(__CUDACC__)
#define ALIGN(x)  __align__(x)
#else
#if defined(__GNUC__)
#define ALIGN(x)  __attribute__ ((aligned (x)))
#else
#define ALIGN(x)
#endif
#endif

#endif
