#include "gpu_utils.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

static void cuda_check(cudaError_t e, const char* file, int line) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
    std::exit(EXIT_FAILURE);
  }
}
#define CUDACHECK(x) cuda_check((x), __FILE__, __LINE__)

std::vector<GpuInfo> detect_gpus() {
  int n = 0;
  CUDACHECK(cudaGetDeviceCount(&n));
  std::vector<GpuInfo> out;
  for (int i = 0; i < n; ++i) {
    cudaDeviceProp p{};
    CUDACHECK(cudaGetDeviceProperties(&p, i));
    out.push_back({i, p.name, p.major, p.minor, p.totalGlobalMem / (1024 * 1024)});
  }
  return out;
}

