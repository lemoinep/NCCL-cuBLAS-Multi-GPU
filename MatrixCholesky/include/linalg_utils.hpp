#pragma once
#include <cublas_v2.h>
#include <cusolverDn.h>

void cublas_check(cublasStatus_t s, const char* file, int line);
void cusolver_check(cusolverStatus_t s, const char* file, int line);

#define CUBLASCHECK(x) cublas_check((x), __FILE__, __LINE__)
#define CUSOLVERCHECK(x) cusolver_check((x), __FILE__, __LINE__)
