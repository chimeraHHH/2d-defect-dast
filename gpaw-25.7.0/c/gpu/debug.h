#ifndef GPU_DEBUG_H
#define GPU_DEBUG_H

#define GPAW_MALLOC(T, n) (T*)(malloc((n) * sizeof(T)))

extern "C" void bmgs_paste_cpu(const double *a_cpu, const int sizea[3],
                               double *b_cpu, const int sizeb[3],
                               const int startb[3]);
extern "C" void bmgs_pastez_cpu(const double *a_cpu, const int sizea[3],
                                double *b_cpu, const int sizeb[3],
                                const int startb[3]);
extern "C" void bmgs_cut_cpu(const double *a_cpu, const int sizea[3],
                             const int starta[3],
                             double *b_cpu, const int sizeb[3]);
extern "C" void bmgs_cutz_cpu(const double *a_cpu, const int sizea[3],
                              const int starta[3],
                              double *b_cpu, const int sizeb[3]);
extern "C" void bmgs_cutmz_cpu(const void *a_cpu, const int sizea[3],
                               const int starta[3],
                               void *b_cpu, const int sizeb[3],
                               void *phase);
#endif
