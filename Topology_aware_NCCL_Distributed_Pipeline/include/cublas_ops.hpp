#pragma once
#include <vector>

bool initialize_cublas();
void shutdown_cublas();

struct ComputeResult {
    double compute_ms;
    double gflops;
};

ComputeResult run_cublas_stage(int gpu_id, int matrix_size);