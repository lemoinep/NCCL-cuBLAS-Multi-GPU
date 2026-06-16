#include "linalg_utils.hpp"
#include <cstdio>
#include <cstdlib>

void cublas_check(cublasStatus_t s, const char* file, int line) {
  if (s != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", file, line, int(s));
    std::exit(EXIT_FAILURE);
  }
}

void cusolver_check(cusolverStatus_t s, const char* file, int line) {
  if (s != CUSOLVER_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuSOLVER error %s:%d: %d\n", file, line, int(s));
    std::exit(EXIT_FAILURE);
  }
}

