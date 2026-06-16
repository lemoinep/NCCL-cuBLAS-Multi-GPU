#include "gpu_utils.hpp"
#include "scheduler.hpp"
#include "cublas_matmul.hpp"
#include "nccl_utils.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CUDACHECK(cmd) do { \
  cudaError_t e = cmd; \
  if (e != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

static void fillHost(std::vector<float>& v, float base) {
  for (size_t i = 0; i < v.size(); ++i) v[i] = base + 0.001f * float(i % 100);
}


int main() {
  auto gpus = detect_gpus();
  int nGPU = (int)gpus.size();
  if (nGPU == 0) return 0;

  printf("Detected %d GPU(s)\n", nGPU);
  for (const auto& g : gpus) {
    printf("GPU %d: %s, %zu MB, CC %d.%d\n", g.id, g.name.c_str(), g.totalMemMB, g.major, g.minor);
  }
  print_topology();

  const int M = 4096, N = 4096, K = 4096;
  std::vector<float> hA(M * K), hB(K * N), hC(M * N, 0.0f);
  fillHost(hA, 1.0f);
  fillHost(hB, 2.0f);

  auto chunks = make_row_chunks(M, nGPU);
  std::vector<cudaStream_t> streams(nGPU);
  std::vector<cublasHandle_t> handles(nGPU);
  std::vector<float*> dA(nGPU, nullptr), dB(nGPU, nullptr), dC(nGPU, nullptr);

  for (int g = 0; g < nGPU; ++g) {
    CUDACHECK(cudaSetDevice(g));
    CUDACHECK(cudaStreamCreate(&streams[g]));
    cublasCreate(&handles[g]);
    cublasSetStream(handles[g], streams[g]);
    CUDACHECK(cudaMalloc(&dB[g], size_t(K) * N * sizeof(float)));
    CUDACHECK(cudaMemcpyAsync(dB[g], hB.data(), size_t(K) * N * sizeof(float),
                              cudaMemcpyHostToDevice, streams[g]));
  }

  for (const auto& c : chunks) {
    int g = c.gpu;
    int localM = c.row1 - c.row0;
    CUDACHECK(cudaSetDevice(g));
    CUDACHECK(cudaMalloc(&dA[g], size_t(localM) * K * sizeof(float)));
    CUDACHECK(cudaMalloc(&dC[g], size_t(localM) * N * sizeof(float)));
    CUDACHECK(cudaMemcpyAsync(dA[g], hA.data() + size_t(c.row0) * K,
                              size_t(localM) * K * sizeof(float),
                              cudaMemcpyHostToDevice, streams[g]));
    cublas_gemm_f32(dA[g], dB[g], dC[g], localM, N, K, (void*)handles[g]);
  }

  for (int g = 0; g < nGPU; ++g) CUDACHECK(cudaStreamSynchronize(streams[g]));

  for (const auto& c : chunks) {
    int g = c.gpu;
    int localM = c.row1 - c.row0;
    CUDACHECK(cudaMemcpy(hC.data() + size_t(c.row0) * N, dC[g],
                         size_t(localM) * N * sizeof(float),
                         cudaMemcpyDeviceToHost));
  }

  printf("Sample C[0][0..3]: ");
  for (int j = 0; j < 4; ++j) printf("%8.3f ", hC[j]);
  printf("\n");

  for (int g = 0; g < nGPU; ++g) {
    CUDACHECK(cudaSetDevice(g));
    if (dA[g]) cudaFree(dA[g]);
    if (dB[g]) cudaFree(dB[g]);
    if (dC[g]) cudaFree(dC[g]);
    cudaStreamDestroy(streams[g]);
    cublasDestroy(handles[g]);
  }
  return 0;
}