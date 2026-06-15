#include "cublas_matmul.hpp"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

static void cublas_check(cublasStatus_t s, const char* file, int line) {
  if (s != CUBLAS_STATUS_SUCCESS) {
    printf("cuBLAS error %s:%d\n", file, line);
    std::exit(EXIT_FAILURE);
  }
}
#define CUBLASCHECK(x) cublas_check((x), __FILE__, __LINE__)

void cublas_gemm_f32(const float* dA, const float* dB, float* dC,
                     int M, int N, int K, void* handle) {
  cublasHandle_t h = reinterpret_cast<cublasHandle_t>(handle);
  const float alpha = 1.0f, beta = 0.0f;
  CUBLASCHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                          N, M, K,
                          &alpha,
                          dB, N,
                          dA, K,
                          &beta,
                          dC, N));
}