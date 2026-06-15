#include "nccl_utils.hpp"
#include <cstdio>
#include <cstdlib>

void nccl_check(ncclResult_t r, const char* file, int line) {
  if (r != ncclSuccess) {
    printf("NCCL error %s:%d: %s\n", file, line, ncclGetErrorString(r));
    std::exit(EXIT_FAILURE);
  }
}