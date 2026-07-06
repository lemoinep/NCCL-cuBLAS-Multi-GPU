#pragma once

#include "topology.hpp"
#include "cublas_ops.hpp"

struct DemoConfig {
    std::string mode;
    int num_gpus;
    int matrix_size;
};

struct PipelineResult {
    double total_ms;
    double compute_ms;
    double communication_ms;
    double bandwidth_gbps;
};

bool initialize_nccl(int num_gpus);
void shutdown_nccl();
PipelineResult run_pipeline(const DemoConfig& config, const TopologyInfo& topo, const RankMapping& mapping);
void print_result(const PipelineResult& result);