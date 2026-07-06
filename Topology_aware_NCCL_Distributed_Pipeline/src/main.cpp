#include <iostream>
#include <string>
#include <map>
#include <vector>

#include <mpi.h>

#include "topology.hpp"
#include "cublas_ops.hpp"
#include "nccl_pipeline.hpp"

static void print_usage(int rank, const char* prog) {
    if (rank != 0) return;
    std::cout << "Usage: " << prog
        << " [--mode baseline|topology-aware]"
        << " [--matrix-size N]"
        << " [--gpus TOTAL_GPUS(optional)]"
        << " [--dump-json [path]]"
        << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = -1, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    DemoConfig config;
    config.mode = "baseline";
    config.matrix_size = 4096;
    config.num_gpus = world_size;

    bool dump_json = false;
    std::string json_path = "topology_mapping.json";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            config.mode = argv[++i];
        }
        else if (arg == "--matrix-size" && i + 1 < argc) {
            config.matrix_size = std::stoi(argv[++i]);
        }
        else if (arg == "--gpus" && i + 1 < argc) {
            config.num_gpus = std::stoi(argv[++i]);
        }
        else if (arg == "--dump-json") {
            dump_json = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                json_path = argv[++i];
            }
        }
        else if (arg == "--help" || arg == "-h") {
            print_usage(rank, argv[0]);
            MPI_Finalize();
            return 0;
        }
    }

    char my_name[MPI_MAX_PROCESSOR_NAME];
    int name_len = 0;
    MPI_Get_processor_name(my_name, &name_len);
    std::string my_host(my_name, name_len);

    std::vector<int> name_lengths(world_size);
    int my_len = name_len;
    MPI_Allgather(&my_len, 1, MPI_INT, name_lengths.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_chars = 0;
    for (int len : name_lengths) total_chars += len;

    std::vector<char> all_names(total_chars);
    std::vector<int> displs(world_size);
    int offset = 0;
    for (int i = 0; i < world_size; ++i) {
        displs[i] = offset;
        offset += name_lengths[i];
    }

    MPI_Allgatherv(my_name, name_len, MPI_CHAR,
        all_names.data(), name_lengths.data(), displs.data(), MPI_CHAR,
        MPI_COMM_WORLD);

    std::map<std::string, int> host_to_node;
    int next_node_id = 0;
    for (int r = 0; r < world_size; ++r) {
        std::string host(all_names.data() + displs[r], name_lengths[r]);
        if (host_to_node.find(host) == host_to_node.end()) {
            host_to_node[host] = next_node_id++;
        }
    }

    int my_node_id = host_to_node[my_host];

    if (rank == 0) {
        std::cout << "[main] MPI world_size = " << world_size
            << ", mode = " << config.mode
            << ", matrix_size = " << config.matrix_size
            << ", num_gpus(cfg) = " << config.num_gpus
            << ", nodes = " << host_to_node.size()
            << std::endl;
    }

    TopologyInfo topo = detect_topology();

    for (auto& g : topo.gpus) {
        g.node_id = my_node_id;
    }

    if (rank == 0) {
        print_topology(topo);
    }

    RankMapping mapping;

    if (config.mode == "baseline") {
        mapping.rank_to_gpu.resize(world_size);
        for (int r = 0; r < world_size; ++r) {
            mapping.rank_to_gpu[r] = (topo.num_gpus > 0) ? (r % topo.num_gpus) : 0;
        }
    }
    else {
        mapping = build_rank_mapping(topo, world_size, config.mode);
    }

    if (rank == 0) {
        print_mapping(mapping);
        if (dump_json) {
            dump_topology_and_mapping_json(topo, mapping, json_path);
        }
    }

    if (!initialize_cublas()) {
        if (rank == 0) {
            std::cerr << "Failed to initialize cuBLAS" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (!initialize_nccl(config.num_gpus)) {
        if (rank == 0) {
            std::cerr << "Failed to initialize NCCL" << std::endl;
        }
        shutdown_cublas();
        MPI_Finalize();
        return 1;
    }

    PipelineResult local_result = run_pipeline(config, topo, mapping);
    print_result(local_result);

    shutdown_nccl();
    shutdown_cublas();
    MPI_Finalize();
    return 0;
}

