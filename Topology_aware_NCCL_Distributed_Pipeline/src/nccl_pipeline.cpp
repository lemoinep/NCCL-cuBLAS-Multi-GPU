#include "nccl_pipeline.hpp"

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>


#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess) {                                                 \
            std::cerr << "[CUDA] " << cudaGetErrorString(_e)                     \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            MPI_Abort(MPI_COMM_WORLD, -1);                                       \
        }                                                                        \
    } while (0)

#define NCCL_CHECK(call)                                                         \
    do {                                                                         \
        ncclResult_t _e = (call);                                                \
        if (_e != ncclSuccess) {                                                 \
            std::cerr << "[NCCL] " << ncclGetErrorString(_e)                     \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            MPI_Abort(MPI_COMM_WORLD, -1);                                       \
        }                                                                        \
    } while (0)

static bool g_nccl_initialized = false;

bool initialize_nccl(int num_gpus) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        std::cerr << "MPI must be initialized before initialize_nccl()." << std::endl;
        return false;
    }

    int rank = -1, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "[init] MPI world size = " << size
                  << ", expected GPUs = " << num_gpus << std::endl;
    }

    g_nccl_initialized = true;
    return true;
}

void shutdown_nccl() {
    g_nccl_initialized = false;
}

PipelineResult run_pipeline(const DemoConfig& config,
                            const TopologyInfo& topo,
                            const RankMapping& mapping) {
    PipelineResult result{};
    if (!g_nccl_initialized) {
        std::cerr << "NCCL not initialized. Call initialize_nccl() first." << std::endl;
        return result;
    }

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        std::cerr << "MPI not initialized. Call MPI_Init first." << std::endl;
        return result;
    }

    int rank = -1, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    if (nRanks <= 0) {
        std::cerr << "Invalid MPI world size." << std::endl;
        return result;
    }

    int gpu_id = 0;
    if (static_cast<int>(mapping.rank_to_gpu.size()) > rank) {
        gpu_id = mapping.rank_to_gpu[rank];
    } else {
        gpu_id = rank % topo.num_gpus;
    }

    CUDA_CHECK(cudaSetDevice(gpu_id));

    if (rank == 0) {
        std::cout << "[run] NCCL + MPI multi-process pipeline\n";
        std::cout << "      MPI ranks: " << nRanks
                  << ", topology GPUs: " << topo.num_gpus
                  << ", mode: " << config.mode << std::endl;
    }

    const int num_elems = config.matrix_size * config.matrix_size;
    const size_t bytes = static_cast<size_t>(num_elems) * sizeof(float);

    ncclUniqueId id;
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&id));
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);  

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, nRanks, id, rank));

    float* sendbuff = nullptr;
    float* recvbuff = nullptr;
    cudaStream_t stream;

    auto t0 = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaMalloc(&sendbuff, bytes));
    CUDA_CHECK(cudaMalloc(&recvbuff, bytes));
    CUDA_CHECK(cudaMemset(sendbuff, 0, bytes));
    CUDA_CHECK(cudaMemset(recvbuff, 0, bytes));
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto t_compute_start = std::chrono::steady_clock::now();
    auto comp = run_cublas_stage(gpu_id, config.matrix_size);

    result.compute_ms = comp.compute_ms;
    auto t_compute_end = std::chrono::steady_clock::now();

    cudaEvent_t start_evt, end_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&end_evt));

    CUDA_CHECK(cudaEventRecord(start_evt, stream));

    NCCL_CHECK(ncclAllReduce(
        (const void*)sendbuff,
        (void*)recvbuff,
        num_elems,
        ncclFloat,
        ncclSum,
        comm,
        stream
    ));

    CUDA_CHECK(cudaEventRecord(end_evt, stream));
    CUDA_CHECK(cudaEventSynchronize(end_evt));

    float comm_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&comm_ms, start_evt, end_evt));
    result.communication_ms = static_cast<double>(comm_ms);

    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(end_evt));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    auto t1 = std::chrono::steady_clock::now();
    result.compute_ms = std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count();
    result.total_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (result.communication_ms > 0.0) {
        double total_bytes = static_cast<double>(bytes) * static_cast<double>(nRanks);
        double comm_s = result.communication_ms / 1000.0;
        double gb = total_bytes / (1024.0 * 1024.0 * 1024.0);
        result.bandwidth_gbps = gb / comm_s;
    } else {
        result.bandwidth_gbps = 0.0;
    }


    struct {
        double total_ms;
        double compute_ms;
        double communication_ms;
        double bandwidth_gbps;
    } local_vals{result.total_ms, result.compute_ms, result.communication_ms, result.bandwidth_gbps},
      global_vals{};

    MPI_Reduce(&local_vals, &global_vals, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);  

    if (rank == 0) {
        std::cout << "[metrics] (max over ranks)\n";
        std::cout << "  total_ms:         " << global_vals.total_ms << "\n";
        std::cout << "  compute_ms:       " << global_vals.compute_ms << "\n";
        std::cout << "  communication_ms: " << global_vals.communication_ms << "\n";
        std::cout << "  bandwidth_gbps:   " << global_vals.bandwidth_gbps << "\n";
    }

    if (sendbuff) CUDA_CHECK(cudaFree(sendbuff));
    if (recvbuff) CUDA_CHECK(cudaFree(recvbuff));
    if (stream)   CUDA_CHECK(cudaStreamDestroy(stream));
    if (comm)     ncclCommDestroy(comm);

    return result;
}

void print_result(const PipelineResult& result) {
    int rank = -1;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    if (rank > 0) return;

    std::cout << "Pipeline result (local rank or aggregated)\n";
    std::cout << "  total_ms:         " << result.total_ms << "\n";
    std::cout << "  compute_ms:       " << result.compute_ms << "\n";
    std::cout << "  communication_ms: " << result.communication_ms << "\n";
    std::cout << "  bandwidth_gbps:   " << result.bandwidth_gbps << "\n";
}