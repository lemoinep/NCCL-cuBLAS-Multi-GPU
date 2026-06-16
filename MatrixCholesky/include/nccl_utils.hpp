#pragma once
#include <nccl.h>

void nccl_check(ncclResult_t r, const char* file, int line);
#define NCCLCHECK(x) nccl_check((x), __FILE__, __LINE__)
