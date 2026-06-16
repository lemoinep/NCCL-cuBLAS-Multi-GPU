#include "gpu_utils.hpp"
#include "nccl_utils.hpp"
#include "linalg_utils.hpp"
#include "owner_utils.hpp"
#include "task_scheduler.hpp"
#include "validation.hpp"

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <nccl.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>



#define CUDACHECK(cmd) do { \
  cudaError_t e = cmd; \
  if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

#define MPICHECK(cmd) do { \
  int e = cmd; \
  if (e != MPI_SUCCESS) { \
    std::fprintf(stderr, "MPI error %s:%d\n", __FILE__, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)



static void fill_spd(std::vector<double>& A, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) A[i * n + j] = (i == j) ? double(n) + 50.0 : 0.01;
    A[i * n + i] += n;
  }
}

static void check_async_nccl(ncclComm_t comm, MPI_Comm mpi_comm) {
  ncclResult_t asyncErr = ncclSuccess;
  NCCLCHECK(ncclCommGetAsyncError(comm, &asyncErr));
  if (asyncErr != ncclSuccess) {
    std::fprintf(stderr, "Async NCCL error: %s\n", ncclGetErrorString(asyncErr));
    ncclCommAbort(comm);
    MPI_Abort(mpi_comm, 1);
  }
}

static void potrf_block(double* dA, int lda, int n, cudaStream_t stream, cusolverDnHandle_t solver) {
  CUSOLVERCHECK(cusolverDnSetStream(solver, stream));
  int work_size = 0;
  CUSOLVERCHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, n, dA, lda, &work_size));
  double* dWork = nullptr;
  int* devInfo = nullptr;
  CUDACHECK(cudaMalloc(&dWork, size_t(work_size) * sizeof(double)));
  CUDACHECK(cudaMalloc(&devInfo, sizeof(int)));
  CUSOLVERCHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, n, dA, lda, dWork, work_size, devInfo));
  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaFree(dWork));
  CUDACHECK(cudaFree(devInfo));
}



int main(int argc, char** argv) {
  MPICHECK(MPI_Init(&argc, &argv));
  MPI_Comm mpi_comm = MPI_COMM_WORLD;

  int rank = 0, nranks = 0;
  MPICHECK(MPI_Comm_rank(mpi_comm, &rank));
  MPICHECK(MPI_Comm_size(mpi_comm, &nranks));

  int devCount = 0;
  CUDACHECK(cudaGetDeviceCount(&devCount));
  CUDACHECK(cudaSetDevice(rank % std::max(1, devCount)));

  int n = 4096, nb = 256;
  if (argc > 1) n = std::atoi(argv[1]);
  if (argc > 2) nb = std::atoi(argv[2]);
  if (n <= 0) n = 4096;
  if (nb <= 0) nb = 256;

  if (rank == 0) {
    auto gpus = detect_gpus();
    std::printf("Detected %zu GPU(s)\n", gpus.size());
  }

  TaskScheduler sched(n, nb, nranks, rank);

  std::vector<double> hA;
  if (rank == 0) {
    hA.resize(size_t(n) * n);
    fill_spd(hA, n);
  }

  ncclUniqueId id;
  if (rank == 0) NCCLCHECK(ncclGetUniqueId(&id));
  MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm));

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));

  cudaStream_t stream_comm, stream_comp;
  CUDACHECK(cudaStreamCreate(&stream_comm));
  CUDACHECK(cudaStreamCreate(&stream_comp));

  cublasHandle_t cublas;
  cusolverDnHandle_t solver;
  CUBLASCHECK(cublasCreate(&cublas));
  CUSOLVERCHECK(cusolverDnCreate(&solver));
  CUBLASCHECK(cublasSetStream(cublas, stream_comp));
  CUSOLVERCHECK(cusolverDnSetStream(solver, stream_comp));

  double* dA = nullptr;
  CUDACHECK(cudaMalloc(&dA, size_t(n) * n * sizeof(double)));
  if (rank == 0) CUDACHECK(cudaMemcpy(dA, hA.data(), size_t(n) * n * sizeof(double), cudaMemcpyHostToDevice));

  std::vector<cudaEvent_t> panel_ready(sched.tasks.size());
  for (auto& e : panel_ready) CUDACHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));

  std::vector<bool> potrf_done(sched.tasks.size(), false), broadcast_done(sched.tasks.size(), false), update_done(sched.tasks.size(), false);
  std::vector<double> block_time(sched.tasks.size(), 0.0);

  size_t done = 0;
  const double alpha = 1.0, beta = 1.0, minus_one = -1.0;

  while (done < sched.tasks.size()) {
    check_async_nccl(comm, mpi_comm);

    for (int tid = 0; tid < (int)sched.tasks.size(); ++tid) {
      auto& t = sched.tasks[tid];
      if (t.owner == rank && !potrf_done[tid]) {
        auto t0 = std::chrono::high_resolution_clock::now();
        double* dPanel = dA + size_t(t.k) * n + t.k;
        potrf_block(dPanel, n, t.nb, stream_comp, solver);
        auto t1 = std::chrono::high_resolution_clock::now();
        block_time[tid] += std::chrono::duration<double>(t1 - t0).count();
        potrf_done[tid] = true;
        CUDACHECK(cudaEventRecord(panel_ready[tid], stream_comp));
        break;
      }
    }

    for (int tid = 0; tid < (int)sched.tasks.size(); ++tid) {
      auto& t = sched.tasks[tid];
      if (!potrf_done[tid] || broadcast_done[tid]) continue;
      CUDACHECK(cudaStreamWaitEvent(stream_comm, panel_ready[tid], 0));
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclBroadcast((const void*)(dA + size_t(t.k) * n + t.k),
                              (void*)(dA + size_t(t.k) * n + t.k),
                              size_t(t.nb) * t.nb,
                              ncclDouble,
                              t.owner,
                              comm,
                              stream_comm));
      NCCLCHECK(ncclGroupEnd());
      broadcast_done[tid] = true;
      break;
    }

    for (int tid = 0; tid < (int)sched.tasks.size(); ++tid) {
      auto& t = sched.tasks[tid];
      if (!(potrf_done[tid] && broadcast_done[tid]) || update_done[tid]) continue;
      int next_k = t.k + t.nb;
      if (next_k < n) {
        auto t0 = std::chrono::high_resolution_clock::now();
        int m = n - next_k;
        double* dPanel = dA + size_t(t.k) * n + t.k;
        double* dTrail = dA + size_t(next_k) * n + t.k;

        CUBLASCHECK(cublasDtrsm(cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                CUBLAS_DIAG_NON_UNIT, m, t.nb, &alpha, dPanel, n, dTrail, m));

        CUBLASCHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, m, m, t.nb,
                                &minus_one, dTrail, m, dTrail, m, &beta,
                                dA + size_t(next_k) * n + next_k, n));
        auto t1 = std::chrono::high_resolution_clock::now();
        block_time[tid] += std::chrono::duration<double>(t1 - t0).count();
      }
      update_done[tid] = true;
      ++done;
      break;
    }

    CUDACHECK(cudaStreamSynchronize(stream_comm));
    CUDACHECK(cudaStreamSynchronize(stream_comp));
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  if (rank == 0) {
    std::vector<double> hL(size_t(n) * n);
    CUDACHECK(cudaMemcpy(hL.data(), dA, size_t(n) * n * sizeof(double), cudaMemcpyDeviceToHost));
    double resid = frob_norm_diff_lower(hA, hL, n);
    std::printf("Cholesky final done. L[0][0] = %f\n", hL[0]);
    std::printf("Residual Frobenius norm ||A-LL^T|| = %.6e\n", resid);
    for (size_t i = 0; i < block_time.size(); ++i) {
      std::printf("Block %zu time %.6f s\n", i, block_time[i]);
    }
  }

  for (auto& e : panel_ready) cudaEventDestroy(e);
  CUBLASCHECK(cublasDestroy(cublas));
  CUSOLVERCHECK(cusolverDnDestroy(solver));
  CUDACHECK(cudaFree(dA));
  CUDACHECK(cudaStreamDestroy(stream_comm));
  CUDACHECK(cudaStreamDestroy(stream_comp));
  NCCLCHECK(ncclCommDestroy(comm));
  MPI_Finalize();
  return 0;
}



