#include "gpu_utils.hpp"
#include "scheduler.hpp"
#include "cublas_matmul.hpp"
#include "nccl_utils.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CUDACHECK(cmd) do { \
  cudaError_t e = cmd; \
  if (e != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

#define MPICHECK(cmd) do { \
  int e = cmd; \
  if (e != MPI_SUCCESS) { \
    printf("MPI error %s:%d\n", __FILE__, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

static void fillHost(std::vector<float>& v, float base) {
  for (size_t i = 0; i < v.size(); ++i) v[i] = base + 0.001f * float(i % 100);
}

int main(int argc, char** argv) {
  MPICHECK(MPI_Init(&argc, &argv));
  int rank = 0, nranks = 0;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

  int devCount = 0;
  CUDACHECK(cudaGetDeviceCount(&devCount));
  int dev = rank % devCount;
  CUDACHECK(cudaSetDevice(dev));

  const int M = 4096, N = 4096, K = 4096;
  int rowsPerRank = (M + nranks - 1) / nranks;
  int r0 = rank * rowsPerRank;
  int r1 = std::min(M, r0 + rowsPerRank);
  int localM = std::max(0, r1 - r0);

  std::vector<float> hA(localM * K), hB(K * N), hC(localM * N);
  fillHost(hA, 1.0f + rank);
  if (rank == 0) fillHost(hB, 2.0f);
  MPICHECK(MPI_Bcast(hB.data(), int(hB.size()), MPI_FLOAT, 0, MPI_COMM_WORLD));

  ncclUniqueId id;
  if (rank == 0) NCCLCHECK(ncclGetUniqueId(&id));
  MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  if (localM > 0) CUDACHECK(cudaMalloc(&dA, size_t(localM) * K * sizeof(float)));
  CUDACHECK(cudaMalloc(&dB, size_t(K) * N * sizeof(float)));
  if (localM > 0) CUDACHECK(cudaMalloc(&dC, size_t(localM) * N * sizeof(float)));

  if (localM > 0) CUDACHECK(cudaMemcpyAsync(dA, hA.data(), size_t(localM) * K * sizeof(float),
                                            cudaMemcpyHostToDevice, stream));
  CUDACHECK(cudaMemcpyAsync(dB, hB.data(), size_t(K) * N * sizeof(float),
                            cudaMemcpyHostToDevice, stream));

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  if (localM > 0) cublas_gemm_f32(dA, dB, dC, localM, N, K, (void*)handle);

  CUDACHECK(cudaStreamSynchronize(stream));

  std::vector<int> counts(nranks), displs(nranks);
  for (int r = 0; r < nranks; ++r) {
    int a0 = r * rowsPerRank;
    int a1 = std::min(M, a0 + rowsPerRank);
    counts[r] = std::max(0, a1 - a0) * N;
    displs[r] = a0 * N;
  }

  std::vector<float> globalC;
  if (rank == 0) globalC.resize(size_t(M) * N);

  if (localM > 0) {
    CUDACHECK(cudaMemcpy(hC.data(), dC, size_t(localM) * N * sizeof(float), cudaMemcpyDeviceToHost));
  }

  MPICHECK(MPI_Gatherv(hC.data(), localM * N, MPI_FLOAT,
                       rank == 0 ? globalC.data() : nullptr,
                       counts.data(), displs.data(), MPI_FLOAT,
                       0, MPI_COMM_WORLD));

  if (rank == 0) {
    printf("Sample C[0][0..3]: ");
    for (int j = 0; j < 4; ++j) printf("%8.3f ", globalC[j]);
    printf("\n");
  }

  cublasDestroy(handle);
  if (dA) cudaFree(dA);
  if (dB) cudaFree(dB);
  if (dC) cudaFree(dC);
  cudaStreamDestroy(stream);
  ncclCommDestroy(comm);
  MPI_Finalize();
  return 0;
}