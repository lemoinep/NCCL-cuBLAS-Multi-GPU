#include "cublas_ops.hpp"
#include <iostream>

bool initialize_cublas() {
    return true;
}

void shutdown_cublas() {}

ComputeResult run_cublas_stage(int gpu_id, int matrix_size) {
    ComputeResult result;
    result.compute_ms = 0.0;
    result.gflops = 0.0;
    std::cout << "Running cuBLAS stage on GPU " << gpu_id
              << " with matrix size " << matrix_size << "\n";
    return result;
}