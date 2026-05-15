#ifndef GPU_BMGS_H
#define GPU_BMGS_H

#include "gpu.h"
#include "gpu-complex.h"

int bmgs_fd_boundary_test(const bmgsstencil_gpu* s, int boundary,
                          int ndouble);

bmgsstencil_gpu bmgs_stencil_to_gpu(bmgsstencil *s);

void bmgs_fd_gpu(const bmgsstencil_gpu* s, const double* adev,
                 double* bdev, int boundary, int blocks,
                 gpuStream_t stream);

void bmgs_relax_gpu(const int relax_method, const bmgsstencil_gpu* s,
                    double* adev, double* bdev, const double* src,
                    const double w, int boundary, gpuStream_t stream);

void bmgs_cut_gpu(const double* a, const int n[3], const int c[3],
                  double* b, const int m[3],int blocks, gpuStream_t stream);

void bmgs_paste_gpu(const double* a, const int n[3],
                    double* b, const int m[3], const int c[3],
                    int blocks, gpuStream_t stream);

void bmgs_paste_zero_gpu(const double* a, const int n[3],
                         double* b, const int m[3], const int c[3],
                         int blocks, gpuStream_t stream);

void bmgs_translate_gpu(double* a, const int sizea[3], const int size[3],
                        const int start1[3], const int start2[3],
                        int blocks, gpuStream_t stream);

void bmgs_restrict_gpu(int k, double* a, const int n[3], double* b,
                       const int nb[3], int blocks);

void bmgs_restrict_stencil_gpu(int k, double* a, const int na[3],
                               double* b, const int nb[3],
                               double* w, int blocks);

void bmgs_interpolate_gpu(int k, int skip[3][2],
                          const double* a, const int n[3],
                          double* b, const int sizeb[3],
                          int blocks);

void bmgs_interpolate_stencil_gpu(int k, int skip[3][2],
                                  const double* a, const int sizea[3],
                                  double* b, const int sizeb[3],
                                  double* w, int blocks);

// complex routines:
void bmgs_fd_gpuz(const bmgsstencil_gpu* s, const gpuDoubleComplex* adev,
                  gpuDoubleComplex* bdev, int boundary, int blocks,
                  gpuStream_t stream);

void bmgs_cut_gpuz(const gpuDoubleComplex* a, const int n[3],
                   const int c[3], gpuDoubleComplex* b, const int m[3],
                   gpuDoubleComplex, int blocks, gpuStream_t stream);

void bmgs_paste_gpuz(const gpuDoubleComplex* a, const int n[3],
                     gpuDoubleComplex* b, const int m[3], const int c[3],
                     int blocks, gpuStream_t stream);

void bmgs_paste_zero_gpuz(const gpuDoubleComplex* a, const int n[3],
                          gpuDoubleComplex* b, const int m[3],
                          const int c[3], int blocks,
                          gpuStream_t stream);

void bmgs_translate_gpuz(gpuDoubleComplex* a, const int sizea[3],
                         const int size[3], const int start1[3],
                         const int start2[3], gpuDoubleComplex,
                         int blocks, gpuStream_t stream);

void bmgs_restrict_gpuz(int k, gpuDoubleComplex* a, const int n[3],
                        gpuDoubleComplex* b, const int nb[3],
                        int blocks);

void bmgs_restrict_stencil_gpuz(int k, gpuDoubleComplex* a, const int na[3],
                                gpuDoubleComplex* b, const int nb[3],
                                gpuDoubleComplex* w, int blocks);

void bmgs_interpolate_gpuz(int k, int skip[3][2],
                           const gpuDoubleComplex* a, const int n[3],
                           gpuDoubleComplex* b, const int sizeb[3],
                           int blocks);

void bmgs_interpolate_stencil_gpuz(int k, int skip[3][2],
                                  const gpuDoubleComplex* a, const int sizea[3],
                                  gpuDoubleComplex* b, const int sizeb[3],
                                  gpuDoubleComplex* w, int blocks);

void reducemap_dotuz(const gpuDoubleComplex* a_gpu,
                     const gpuDoubleComplex* b_gpu, gpuDoubleComplex* result,
                     int n, int nvec);

#endif
