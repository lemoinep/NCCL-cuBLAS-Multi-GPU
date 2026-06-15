#pragma once
void cublas_gemm_f32(const float* dA, const float* dB, float* dC,
                     int M, int N, int K, void* cublasHandle);
