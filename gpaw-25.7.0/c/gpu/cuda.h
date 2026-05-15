#ifndef GPU_CUDA_H
#define GPU_CUDA_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define gpuMemcpyKind             cudaMemcpyKind
#define gpuMemcpyDeviceToHost     cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice     cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice   cudaMemcpyDeviceToDevice
#define gpuSuccess                cudaSuccess
#define gpuEventDefault           cudaEventDefault
#define gpuEventBlockingSync      cudaEventBlockingSync
#define gpuEventDisableTiming     cudaEventDisableTiming

#define gpuStream_t               cudaStream_t
#define gpuEvent_t                cudaEvent_t
#define gpuError_t                cudaError_t
#define gpuDeviceProp             cudaDeviceProp

#define gpuDoubleComplex          cuDoubleComplex
#define gpuFloatComplex           cuFloatComplex
#define gpublasDoubleComplex      cuDoubleComplex
#define make_gpuDoubleComplex     make_cuDoubleComplex
#define make_gpuFloatComplex      make_cuFloatComplex
#define gpuCreal                  cuCreal
#define gpuCimag                  cuCimag
#define gpuCadd                   cuCadd
#define gpuCaddf                  cuCaddf
#define gpuCsub                   cuCsub
#define gpuCsubf                  cuCsubf
#define gpuCmul                   cuCmul
#define gpuCmulf                  cuCmulf
#define gpuConj                   cuConj

#define gpuCheckLastError()       gpuSafeCall(cudaGetLastError())
#define gpuGetErrorString(err)    cudaGetErrorString(err)

#define gpuSetDevice(id)          gpuSafeCall(cudaSetDevice(id))
#define gpuGetDevice(dev)         gpuSafeCall(cudaGetDevice(dev))
#define gpuGetDeviceProperties(prop, dev) \
        gpuSafeCall(cudaGetDeviceProperties(prop, dev))
#define gpuDeviceSynchronize()    gpuSafeCall(cudaDeviceSynchronize())

#define gpuFree(p)                if ((p) != NULL) gpuSafeCall(cudaFree(p))
#define gpuFreeHost(p)            if ((p) != NULL) gpuSafeCall(cudaFreeHost(p))
#define gpuMalloc(pp, size)       gpuSafeCall(cudaMalloc((void**) (pp), size))
#define gpuHostAlloc(pp, size) \
        gpuSafeCall(cudaHostAlloc((void**) (pp), size, cudaHostAllocPortable))
#define gpuMemcpy(dst, src, count, kind) \
        gpuSafeCall(cudaMemcpy(dst, src, count, kind))
#define gpuMemcpyAsync(dst, src, count, kind, stream) \
        gpuSafeCall(cudaMemcpyAsync(dst, src, count, kind, stream))

#define gpuStreamCreate(stream)   gpuSafeCall(cudaStreamCreate(stream))
#define gpuStreamDestroy(stream)  gpuSafeCall(cudaStreamDestroy(stream))
#define gpuStreamWaitEvent(stream, event, flags) \
        gpuSafeCall(cudaStreamWaitEvent(stream, event, flags))
#define gpuStreamSynchronize(stream) \
        gpuSafeCall(cudaStreamSynchronize(stream))

#define gpuEventCreate(event)     gpuSafeCall(cudaEventCreate(event))
#define gpuEventCreateWithFlags(event, flags) \
        gpuSafeCall(cudaEventCreateWithFlags(event, flags))
#define gpuEventDestroy(event)    gpuSafeCall(cudaEventDestroy(event))
#define gpuEventQuery(event)      cudaEventQuery(event)
#define gpuEventRecord(event, stream) \
        gpuSafeCall(cudaEventRecord(event, stream))
#define gpuEventSynchronize(event) \
        gpuSafeCall(cudaEventSynchronize(event))
#define gpuEventElapsedTime(ms, start, end) \
        gpuSafeCall(cudaEventElapsedTime(ms, start, end))

#define gpuLaunchKernel(kernel, dimGrid, dimBlock, shared, stream, ...) \
        kernel<<<dimGrid, dimBlock, shared, stream>>>(__VA_ARGS__)

#define gpublasStatus_t           cublasStatus_t
#define gpublasHandle_t           cublasHandle_t
#define gpublasOperation_t        cublasOperation_t

#define gpublasCreate             cublasCreate
#define gpublasSetStream          cublasSetStream
#define gpublasGetMatrixAsync     cublasGetMatrixAsync
#define gpublasSetMatrixAsync     cublasSetMatrixAsync
#define gpublasDsyrk              cublasDsyrk
#define gpublasDsyr2k             cublasDsyr2k
#define gpublasDscal              cublasDscal
#define gpublasZscal              cublasZscal
#define gpublasDgemm              cublasDgemm
#define gpublasZgemm              cublasZgemm
#define gpublasDgemv              cublasDgemv
#define gpublasZgemv              cublasZgemv
#define gpublasDaxpy              cublasDaxpy
#define gpublasZaxpy              cublasZaxpy
#define gpublasZherk              cublasZherk
#define gpublasZher2k             cublasZher2k
#define gpublasDdot               cublasDdot
#define gpublasZdotc              cublasZdotc
#define gpublasZdotu              cublasZdotu

#define GPUBLAS_OP_N                     CUBLAS_OP_N
#define GPUBLAS_OP_T                     CUBLAS_OP_T
#define GPUBLAS_OP_C                     CUBLAS_OP_C
#define GPUBLAS_FILL_MODE_UPPER          CUBLAS_FILL_MODE_UPPER
#define GPUBLAS_STATUS_SUCCESS           CUBLAS_STATUS_SUCCESS
#define GPUBLAS_STATUS_NOT_INITIALIZED   CUBLAS_STATUS_NOT_INITIALIZED
#define GPUBLAS_STATUS_ALLOC_FAILED      CUBLAS_STATUS_ALLOC_FAILED
#define GPUBLAS_STATUS_INVALID_VALUE     CUBLAS_STATUS_INVALID_VALUE
#define GPUBLAS_STATUS_ARCH_MISMATCH     CUBLAS_STATUS_ARCH_MISMATCH
#define GPUBLAS_STATUS_MAPPING_ERROR     CUBLAS_STATUS_MAPPING_ERROR
#define GPUBLAS_STATUS_EXECUTION_FAILED  CUBLAS_STATUS_EXECUTION_FAILED
#define GPUBLAS_STATUS_INTERNAL_ERROR    CUBLAS_STATUS_INTERNAL_ERROR

#endif
