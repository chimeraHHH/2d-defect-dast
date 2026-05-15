#include "../extensions.h"
#include "../bc.h"
#include "bmgs.h"
#include "gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

static int bc_init_count = 0;

static gpuStream_t bc_recv_stream;
static int bc_streams = 0;
static gpuEvent_t bc_sendcpy_event[3][2];
static gpuEvent_t bc_recv_event[3][2];

static int bc_recv_done[3][2];

static double *bc_rbuff[3][2];
static double *bc_sbuff[3][2];
static double *bc_rbuffs=NULL;
static double *bc_sbuffs=NULL;
static double *bc_rbuff_gpu[3][2];
static double *bc_sbuff_gpu[3][2];
static double *bc_rbuffs_gpu=NULL;
static double *bc_sbuffs_gpu=NULL;
static int bc_rbuffs_size=0;
static int bc_sbuffs_size=0;
static int bc_rbuffs_max=0;
static int bc_sbuffs_max=0;

#ifdef NDEBUG
#  define check_mpi(s) (s)
#else
#  define check_mpi(s) (assert((s) == MPI_SUCCESS))
#endif

void bc_init_gpu(boundary_conditions* bc)
{
    int nsends=0;
    int nrecvs=0;

    for (int i=0; i<3; i++) {
        for (int d=0; d<2; d++) {
            nsends += NEXTPITCHDIV(bc->nsend[i][d]);
            nrecvs += NEXTPITCHDIV(bc->nrecv[i][d]);
        }
    }
    bc_sbuffs_max = MAX(nsends, bc_sbuffs_max);
    bc_rbuffs_max = MAX(nrecvs, bc_rbuffs_max);

    bc_init_count++;
}

void bc_init_buffers_gpu()
{
#ifndef GPAW_GPU_AWARE_MPI
    bc_rbuffs = NULL;
    bc_sbuffs = NULL;
    bc_streams = 0;
#endif
    bc_rbuffs_gpu = NULL;
    bc_sbuffs_gpu = NULL;
    bc_rbuffs_size = 0;
    bc_sbuffs_size = 0;
    bc_init_count = 0;
}

static inline void _reallocate_buffer_host(double **buffer, const int size) {
    gpuFreeHost(*buffer);
    gpuCheckLastError();
    gpuHostAlloc(buffer, sizeof(double) * size);
}

static inline void _reallocate_buffer_device(double **buffer, const int size) {
    gpuFree(*buffer);
    gpuCheckLastError();
    gpuMalloc(buffer, sizeof(double) * size);
}

static inline void _create_stream_events() {
    if (!bc_streams) {
        gpuStreamCreate(&bc_recv_stream);
        bc_streams = 1;
        for (int d=0; d<3; d++) {
            for (int i=0; i<2; i++) {
                gpuEventCreateWithFlags(&bc_sendcpy_event[d][i],
                        gpuEventDefault|gpuEventDisableTiming);
                gpuEventCreateWithFlags(&bc_recv_event[d][i],
                        gpuEventDefault|gpuEventDisableTiming);
            }
        }
    }
}

static void _allocate_buffers(const boundary_conditions* bc, int blocks)
{
    int nsends=0;
    int nrecvs=0;

    for (int i=0; i<3; i++) {
        for (int d=0; d<2; d++) {
            nsends += NEXTPITCHDIV(bc->nsend[i][d] * blocks);
            nrecvs += NEXTPITCHDIV(bc->nrecv[i][d] * blocks);
        }
    }

    bc_sbuffs_max=MAX(nsends, bc_sbuffs_max);
    if (bc_sbuffs_max > bc_sbuffs_size) {
#ifndef GPAW_GPU_AWARE_MPI
        _reallocate_buffer_host(&bc_sbuffs, bc_sbuffs_max);
#endif
        _reallocate_buffer_device(&bc_sbuffs_gpu, bc_sbuffs_max);
        bc_sbuffs_size = bc_sbuffs_max;
    }

    bc_rbuffs_max=MAX(nrecvs, bc_rbuffs_max);
    if (bc_rbuffs_max > bc_rbuffs_size) {
#ifndef GPAW_GPU_AWARE_MPI
        _reallocate_buffer_host(&bc_rbuffs, bc_rbuffs_max);
#endif
        _reallocate_buffer_device(&bc_rbuffs_gpu, bc_rbuffs_max);
        bc_rbuffs_size = bc_rbuffs_max;
    }
#ifndef GPAW_GPU_AWARE_MPI
    _create_stream_events();
#endif
}

void bc_dealloc_gpu(int force)
{
    if (force)
        bc_init_count = 1;

    if (bc_init_count == 1) {
#ifndef GPAW_GPU_AWARE_MPI
        gpuFreeHost(bc_sbuffs);
        gpuFreeHost(bc_rbuffs);
        if (bc_streams) {
            gpuStreamDestroy(bc_recv_stream);
            for (int d=0; d<3; d++) {
                for (int i=0; i<2; i++) {
                    gpuEventDestroy(bc_sendcpy_event[d][i]);
                    gpuEventDestroy(bc_recv_event[d][i]);
                }
            }
        }
#endif
        gpuFree(bc_sbuffs_gpu);
        gpuFree(bc_rbuffs_gpu);
        bc_init_buffers_gpu();
        return;
    }
    if (bc_init_count > 0)
        bc_init_count--;
}

static inline void _check_msg_size(boundary_conditions* bc, int nin)
{
    int maxrecv, maxsend;

    for (int i=0; i<3; i++) {
        maxrecv = MAX(bc->nrecv[i][0], bc->nrecv[i][1]) * nin * sizeof(double);
        maxsend = MAX(bc->nsend[i][0], bc->nsend[i][1]) * nin * sizeof(double);

        bc->gpu_rjoin[i] = 0;
        if (bc->recvproc[i][0] >= 0 && bc->recvproc[i][1] >= 0) {
            if (maxrecv < GPU_RJOIN_SIZE)
                bc->gpu_rjoin[i] = 1;
            else if ((maxrecv < GPU_RJOIN_SAME_SIZE) &&
                    (bc->recvproc[i][0] == bc->recvproc[i][1]))
                bc->gpu_rjoin[i] = 1;
        }
        bc->gpu_sjoin[i] = 0;
        if (bc->sendproc[i][0] >= 0 && bc->sendproc[i][1] >= 0) {
            if (maxsend < GPU_SJOIN_SIZE)
                bc->gpu_sjoin[i] = 1;
            else if ((maxsend < GPU_SJOIN_SAME_SIZE) &&
                    (bc->sendproc[i][0] == bc->sendproc[i][1]))
                bc->gpu_sjoin[i] = 1;
        }

        if (MAX(maxsend, maxrecv) < GPU_ASYNC_SIZE)
            bc->gpu_async[i] = 0;
        else
            bc->gpu_async[i] = 1;
    }
}

static inline void _prepare_buffers_host(boundary_conditions* bc, int nin)
{
    int recvp=0;
    int sendp=0;
    for (int i=0; i<3; i++) {
        bc_sbuff[i][0] = bc_sbuffs + sendp;
        if (!bc->gpu_async[i] || bc->gpu_sjoin[i]) {
            bc_sbuff[i][1] = bc_sbuffs + sendp + bc->nsend[i][0] * nin;
            sendp += NEXTPITCHDIV((bc->nsend[i][0] + bc->nsend[i][1]) * nin);
        } else {
            sendp += NEXTPITCHDIV(bc->nsend[i][0] * nin);
            bc_sbuff[i][1] = bc_sbuffs + sendp;
            sendp += NEXTPITCHDIV(bc->nsend[i][1] * nin);
        }

        bc_rbuff[i][0] = bc_rbuffs + recvp;
        if (!bc->gpu_async[i] || bc->gpu_rjoin[i]) {
            bc_rbuff[i][1] = bc_rbuffs + recvp + bc->nrecv[i][0] * nin;
            recvp += NEXTPITCHDIV((bc->nrecv[i][0] + bc->nrecv[i][1]) * nin);
        } else {
            recvp += NEXTPITCHDIV(bc->nrecv[i][0] * nin);
            bc_rbuff[i][1] = bc_rbuffs + recvp;
            recvp += NEXTPITCHDIV(bc->nrecv[i][1] * nin);
        }
    }
}

static inline void _prepare_buffers_gpu(boundary_conditions* bc, int nin)
{
    int recvp=0;
    int sendp=0;
    for (int i=0; i<3; i++) {
        bc_sbuff_gpu[i][0] = bc_sbuffs_gpu + sendp;
        if (!bc->gpu_async[i] || bc->gpu_sjoin[i]) {
            bc_sbuff_gpu[i][1] = bc_sbuffs_gpu + sendp + bc->nsend[i][0] * nin;
            sendp += NEXTPITCHDIV((bc->nsend[i][0] + bc->nsend[i][1]) * nin);
        } else {
            sendp += NEXTPITCHDIV(bc->nsend[i][0] * nin);
            bc_sbuff_gpu[i][1] = bc_sbuffs_gpu + sendp;
            sendp += NEXTPITCHDIV(bc->nsend[i][1] * nin);
        }

        bc_rbuff_gpu[i][0] = bc_rbuffs_gpu + recvp;
        if (!bc->gpu_async[i] || bc->gpu_rjoin[i]) {
            bc_rbuff_gpu[i][1] = bc_rbuffs_gpu + recvp + bc->nrecv[i][0] * nin;
            recvp += NEXTPITCHDIV((bc->nrecv[i][0] + bc->nrecv[i][1]) * nin);
        } else {
            recvp += NEXTPITCHDIV(bc->nrecv[i][0] * nin);
            bc_rbuff_gpu[i][1] = bc_rbuffs_gpu + recvp;
            recvp += NEXTPITCHDIV(bc->nrecv[i][1] * nin);
        }
    }
}

void bc_unpack_paste_gpu(boundary_conditions* bc,
        const double* aa1, double* aa2,
        MPI_Request recvreq[3][2],
        gpuStream_t kernel_stream, int nin)
{
    bool real = (bc->ndouble == 1);

    _allocate_buffers(bc, nin);
    // Copy data:
    // Zero all of a2 array.  We should only zero the bounaries
    // that are not periodic, but it's simpler to zero everything!

    // Copy data from a1 to central part of a2 and zero boundaries:
    if (real)
        bmgs_paste_zero_gpu(aa1, bc->size1, aa2,
                bc->size2, bc->sendstart[0][0], nin, kernel_stream);
    else
        bmgs_paste_zero_gpuz((const gpuDoubleComplex*)(aa1),
                bc->size1, (gpuDoubleComplex*) aa2,
                bc->size2, bc->sendstart[0][0], nin,
                kernel_stream);

#ifndef GPAW_GPU_AWARE_MPI
    _check_msg_size(bc, nin);
    _prepare_buffers_host(bc, nin);
#endif
    _prepare_buffers_gpu(bc, nin);

    for (int i=0; i<3; i++) {
        for (int d=0; d<2; d++) {
            int p = bc->recvproc[i][d];
            if (p >= 0) {
#ifndef GPAW_GPU_AWARE_MPI
                MPI_Irecv(bc_rbuff[i][d], bc->nrecv[i][d] * nin,
                        MPI_DOUBLE, p, d + 1000 * i, bc->comm, &recvreq[i][d]);
#else
                MPI_Irecv(bc_rbuff_gpu[i][d], bc->nrecv[i][d] * nin,
                        MPI_DOUBLE, p, d + 1000 * i, bc->comm, &recvreq[i][d]);
#endif
                bc_recv_done[i][d] = 0;
            } else {
                bc_recv_done[i][d] = 1;
            }
        }
    }
}

void bc_unpack_gpu_sync(const boundary_conditions* bc,
        double* aa2, int i,
        MPI_Request recvreq[3][2],
        MPI_Request sendreq[2],
        const double_complex phases[2],
        gpuStream_t kernel_stream, int nin)
{
    bool real = (bc->ndouble == 1);
    _allocate_buffers(bc, nin);

#ifdef PARALLEL
    for (int d=0; d<2; d++) {
        if (bc->sendproc[i][d] >= 0) {
            const int* start = bc->sendstart[i][d];
            const int* size = bc->sendsize[i][d];
            if (real)
                bmgs_cut_gpu(aa2, bc->size2, start,
                             bc_sbuff_gpu[i][d],
                             size, nin, kernel_stream);
            else {
                gpuDoubleComplex phase = {creal(phases[d]), cimag(phases[d])};
                bmgs_cut_gpuz((gpuDoubleComplex*)(aa2), bc->size2, start,
                              (gpuDoubleComplex*)(bc_sbuff_gpu[i][d]),
                              size, phase, nin, kernel_stream);
            }
        }
    }

#ifndef GPAW_GPU_AWARE_MPI
    if (bc->sendproc[i][0] >= 0 || bc->sendproc[i][1] >= 0)
        gpuMemcpy(bc_sbuff[i][0], bc_sbuff_gpu[i][0],
                  sizeof(double) * (bc->nsend[i][0] + bc->nsend[i][1]) * nin,
                  gpuMemcpyDeviceToHost);
#endif

    // Start sending:
    for (int d=0; d<2; d++) {
        sendreq[d] = 0;
        int p = bc->sendproc[i][d];
        if (p >= 0) {
#ifndef GPAW_GPU_AWARE_MPI
            check_mpi(MPI_Isend(bc_sbuff[i][d], bc->nsend[i][d] * nin,
                                MPI_DOUBLE, p, 1 - d + 1000 * i, bc->comm,
                                &sendreq[d]));
#else
            gpuStreamSynchronize(kernel_stream);
            check_mpi(MPI_Isend(bc_sbuff_gpu[i][d], bc->nsend[i][d] * nin,
                                MPI_DOUBLE, p, 1 - d + 1000 * i, bc->comm,
                                &sendreq[d]));
#endif
        }
    }
#endif

    // Copy data for periodic boundary conditions:
    for (int d=0; d<2; d++) {
        if (bc->sendproc[i][d] == COPY_DATA) {
            if (real) {
                bmgs_translate_gpu(aa2, bc->size2, bc->sendsize[i][d],
                        bc->sendstart[i][d], bc->recvstart[i][1 - d],
                        nin, kernel_stream);
            } else {
                gpuDoubleComplex phase = {creal(phases[d]), cimag(phases[d])};
                bmgs_translate_gpuz((gpuDoubleComplex*)(aa2),
                        bc->size2, bc->sendsize[i][d],
                        bc->sendstart[i][d], bc->recvstart[i][1 - d],
                        phase, nin, kernel_stream);
            }
        }
    }

#ifdef PARALLEL
    for (int d=0; d<2; d++) {
        if (!bc_recv_done[i][d]) {
            check_mpi(MPI_Wait(&recvreq[i][d], MPI_STATUS_IGNORE));
        }
    }
    if (!bc_recv_done[i][0] || !bc_recv_done[i][1]) {
#ifndef GPAW_GPU_AWARE_MPI
        gpuMemcpy(bc_rbuff_gpu[i][0], bc_rbuff[i][0],
                  sizeof(double) * (bc->nrecv[i][0] + bc->nrecv[i][1]) * nin,
                  gpuMemcpyHostToDevice);
#endif
        bc_recv_done[i][0] = 1;
        bc_recv_done[i][1] = 1;
    }
    for (int d=0; d<2; d++) {
        if (bc->recvproc[i][d] >= 0)  {
            if (real)
                bmgs_paste_gpu(bc_rbuff_gpu[i][d], bc->recvsize[i][d],
                        aa2, bc->size2, bc->recvstart[i][d], nin,
                        kernel_stream);

            else
                bmgs_paste_gpuz(
                        (const gpuDoubleComplex*)(bc_rbuff_gpu[i][d]),
                        bc->recvsize[i][d],
                        (gpuDoubleComplex*)(aa2),
                        bc->size2, bc->recvstart[i][d], nin,
                        kernel_stream);
        }
    }
    // This does not work on the ibm with gcc!  We do a blocking send instead.
    for (int d=0; d<2; d++)
        if (bc->sendproc[i][d] >= 0)
            check_mpi(MPI_Wait(&sendreq[d], MPI_STATUS_IGNORE));
#endif
}

static inline void _bc_unpack_gpu_async(const boundary_conditions* bc,
        double* aa2, int i,
        MPI_Request recvreq[3][2],
        MPI_Request sendreq[2],
        const double_complex phases[2],
        gpuStream_t kernel_stream,
        int nin)
{
    bool real = (bc->ndouble == 1);
    _allocate_buffers(bc, nin);

#ifdef PARALLEL
    // Prepare send-buffers
    int send_done[2] = {0,0};

    if (bc->sendproc[i][0] >= 0 || bc->sendproc[i][1] >= 0) {
        for (int d=0; d<2; d++) {
            sendreq[d] = 0;
            if (bc->sendproc[i][d] >= 0) {
                const int* start = bc->sendstart[i][d];
                const int* size = bc->sendsize[i][d];
                if (real)
                    bmgs_cut_gpu(aa2, bc->size2, start,
                                 bc_sbuff_gpu[i][d],
                                 size, nin, kernel_stream);
                else {
                    gpuDoubleComplex phase = {creal(phases[d]),
                                             cimag(phases[d])};
                    bmgs_cut_gpuz((gpuDoubleComplex*)(aa2), bc->size2, start,
                                  (gpuDoubleComplex*)(bc_sbuff_gpu[i][d]),
                                  size, phase, nin, kernel_stream);
                }
                if (!bc->gpu_sjoin[i]) {
                    gpuMemcpyAsync(bc_sbuff[i][d],
                                   bc_sbuff_gpu[i][d],
                                   sizeof(double) * bc->nsend[i][d] * nin,
                                   gpuMemcpyDeviceToHost,
                                   kernel_stream);
                    gpuEventRecord(bc_sendcpy_event[i][d], kernel_stream);
                }
            }
        }
        if (bc->gpu_sjoin[i]) {
            gpuMemcpyAsync(bc_sbuff[i][0],
                           bc_sbuff_gpu[i][0],
                           sizeof(double)
                               * (bc->nsend[i][0] + bc->nsend[i][1]) * nin,
                           gpuMemcpyDeviceToHost,
                           kernel_stream);
            gpuEventRecord(bc_sendcpy_event[i][0], kernel_stream);
        }
    }
    for (int d=0; d<2; d++)
        if (!(bc->sendproc[i][d] >= 0))
            send_done[d] = 1;

    int dd=0;
    if (send_done[dd])
        dd=1;

    int loopc=MIN(2, 3-i);

    int ddd[loopc];
    for (int ii=0; ii < loopc; ii++) {
        ddd[ii] = 1;
        if (bc_recv_done[ii+i][ddd[ii]])
            ddd[ii] = 1 - ddd[ii];
    }

    do {
        if (!send_done[dd] &&
                gpuEventQuery(bc_sendcpy_event[i][dd]) == gpuSuccess) {
            MPI_Isend(bc_sbuff[i][dd],
                    bc->nsend[i][dd] * nin, MPI_DOUBLE,
                    bc->sendproc[i][dd],
                    1 - dd + 1000 * i, bc->comm,
                    &sendreq[dd]);
            send_done[dd] = 1;
            dd = 1;
            if (bc->gpu_sjoin[i]) {
                MPI_Isend(bc_sbuff[i][dd],
                        bc->nsend[i][dd] * nin, MPI_DOUBLE,
                        bc->sendproc[i][dd],
                        1 - dd + 1000 * i, bc->comm,
                        &sendreq[dd]);
                send_done[dd] = 1;
            }
            loopc = 1;
        }
        for (int i2=0; i2 < loopc; i2++) {
            int i3 = i2+i;
            if (i==0 && i2==1 &&
                    bc_recv_done[i3][0] && bc_recv_done[i3][1]) {
                i3 = 2;
            }
            if (!bc->gpu_async[i3])
                continue;

            if (i2==0 && bc->gpu_rjoin[i3] &&
                    !bc_recv_done[i3][0] && !bc_recv_done[i3][1]) {
                int status;
                MPI_Testall(2, recvreq[i3], &status, MPI_STATUSES_IGNORE);
                if (status) {
                    gpuMemcpyAsync(bc_rbuff_gpu[i3][0], bc_rbuff[i3][0],
                                   sizeof(double)
                                       * (bc->nrecv[i3][0] + bc->nrecv[i3][1])
                                       * nin,
                                   gpuMemcpyHostToDevice,
                                   bc_recv_stream);
                    for (int d=0; d<2; d++) {
                        if (!bc_recv_done[i3][d]) {
                            if (real)
                                bmgs_paste_gpu(bc_rbuff_gpu[i3][d],
                                        bc->recvsize[i3][d], aa2, bc->size2,
                                        bc->recvstart[i3][d], nin,
                                        bc_recv_stream);
                            else
                                bmgs_paste_gpuz(
                                        (const gpuDoubleComplex*)(bc_rbuff_gpu[i3][d]),
                                        bc->recvsize[i3][d],
                                        (gpuDoubleComplex*)(aa2),
                                        bc->size2, bc->recvstart[i3][d], nin,
                                        bc_recv_stream);
                            gpuEventRecord(bc_recv_event[i3][d],
                                    bc_recv_stream);
                            bc_recv_done[i3][d] = 1;
                        }
                    }
                }
            } else if (!bc_recv_done[i3][ddd[i2]]) {
                int status;
                MPI_Test(&recvreq[i3][ddd[i2]], &status, MPI_STATUS_IGNORE);
                if (status) {
                    gpuMemcpyAsync(bc_rbuff_gpu[i3][ddd[i2]],
                                   bc_rbuff[i3][ddd[i2]],
                                   sizeof(double)
                                       * (bc->nrecv[i3][ddd[i2]]) * nin,
                                   gpuMemcpyHostToDevice,
                                   bc_recv_stream);
                    if (real)
                        bmgs_paste_gpu(bc_rbuff_gpu[i3][ddd[i2]],
                                bc->recvsize[i3][ddd[i2]],
                                aa2, bc->size2, bc->recvstart[i3][ddd[i2]], nin,
                                bc_recv_stream);
                    else
                        bmgs_paste_gpuz(
                                (const gpuDoubleComplex*)(bc_rbuff_gpu[i3][ddd[i2]]),
                                bc->recvsize[i3][ddd[i2]],
                                (gpuDoubleComplex*)(aa2),
                                bc->size2, bc->recvstart[i3][ddd[i2]], nin,
                                bc_recv_stream);
                    gpuEventRecord(bc_recv_event[i3][ddd[i2]], bc_recv_stream);
                    bc_recv_done[i3][ddd[i2]] = 1;
                }
            }
            if (!bc_recv_done[i3][1-ddd[i2]])
                ddd[i2] = 1 - ddd[i2];
        }
    } while (!bc_recv_done[i][0] || !bc_recv_done[i][1]
            || !send_done[0] || !send_done[1]);
#endif

    // Copy data for periodic boundary conditions:
    for (int d=0; d<2; d++) {
        if (bc->sendproc[i][d] == COPY_DATA)  {
            if (real) {
                bmgs_translate_gpu(aa2, bc->size2, bc->sendsize[i][d],
                        bc->sendstart[i][d], bc->recvstart[i][1 - d],
                        nin, kernel_stream);
            } else {
                gpuDoubleComplex phase = {creal(phases[d]), cimag(phases[d])};
                bmgs_translate_gpuz((gpuDoubleComplex*)(aa2),
                        bc->size2, bc->sendsize[i][d],
                        bc->sendstart[i][d], bc->recvstart[i][1 - d],
                        phase, nin, kernel_stream);
            }
        }
    }
#ifdef PARALLEL
    // This does not work on the ibm with gcc!  We do a blocking send instead.
    for (int d=0; d<2; d++) {
        if (bc->sendproc[i][d] >= 0)
            check_mpi(MPI_Wait(&sendreq[d], MPI_STATUS_IGNORE));
    }
    for (int d=0; d<2; d++) {
        if (bc->recvproc[i][d] >= 0)
            gpuStreamWaitEvent(kernel_stream, bc_recv_event[i][d], 0);
    }
#endif
}

void bc_unpack_gpu_async(const boundary_conditions* bc,
        double* aa2, int i,
        MPI_Request recvreq[3][2],
        MPI_Request sendreq[2],
        const double_complex phases[2],
        gpuStream_t kernel_stream,
        int nin)
{
#ifndef GPAW_GPU_AWARE_MPI
    _bc_unpack_gpu_async(bc, aa2, i, recvreq, sendreq, phases,
                         kernel_stream, nin);
#else
    bc_unpack_gpu_sync(bc, aa2, i, recvreq, sendreq, phases,
                       kernel_stream, nin);
#endif
}

void bc_unpack_gpu(const boundary_conditions* bc,
        double* aa2, int i,
        MPI_Request recvreq[3][2],
        MPI_Request sendreq[2],
        const double_complex phases[2],
        gpuStream_t kernel_stream,
        int nin)
{
#ifndef GPAW_GPU_AWARE_MPI
    if (!bc->gpu_async[i]) {
        bc_unpack_gpu_sync(bc, aa2, i, recvreq, sendreq,
                phases, kernel_stream, nin);
    }  else {
        bc_unpack_gpu_async(bc, aa2, i, recvreq, sendreq,
                phases, kernel_stream, nin);
    }
#else
    bc_unpack_gpu_sync(bc, aa2, i, recvreq, sendreq,
            phases, kernel_stream, nin);
#endif
}
