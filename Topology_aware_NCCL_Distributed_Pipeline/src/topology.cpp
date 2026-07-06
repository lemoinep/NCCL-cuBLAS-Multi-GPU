#include "topology.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <sstream>


static std::string cuda_error_string(cudaError_t err) {
    return std::string(cudaGetErrorString(err));
}

TopologyInfo detect_topology() {
    TopologyInfo topo;
    int device_count = 0;

    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        std::cerr << "CUDA device discovery failed: " << cuda_error_string(err) << "\n";
        topo.num_gpus = 0;
        return topo;
    }

    topo.num_gpus = device_count;
    topo.gpus.reserve(device_count);

    for (int dev = 0; dev < device_count; ++dev) {
        GpuInfo info;
        info.id = dev;
        info.numa_node = -1;
        info.node_id = -1;

        cudaDeviceProp prop{};
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            info.name = "unknown";
        } else {
            info.name = prop.name;
        }

        for (int peer = 0; peer < device_count; ++peer) {
            if (peer == dev) continue;
            int can_access = 0;
            cudaError_t peer_err = cudaDeviceCanAccessPeer(&can_access, dev, peer);
            if (peer_err == cudaSuccess && can_access) {
                int perf_rank = 0;
                cudaError_t attr_err = cudaDeviceGetP2PAttribute(
                    &perf_rank,
                    cudaDevP2PAttrPerformanceRank,
                    dev,
                    peer
                );
                if (attr_err == cudaSuccess) {
                    LinkInfo link;
                    link.peer_id = peer;
                    link.perf_rank = perf_rank;  
                    info.links.push_back(link);
                }
            }
        }

        topo.gpus.push_back(info);
    }

    return topo;
}

void print_topology(const TopologyInfo& topo) {
    std::cout << "Detected GPUs: " << topo.num_gpus << "\n";
    for (const auto& gpu : topo.gpus) {
        std::cout << "  GPU " << gpu.id << " : " << gpu.name
                  << ", NUMA " << gpu.numa_node
                  << ", node_id " << gpu.node_id
                  << ", links: ";
        if (gpu.links.empty()) {
            std::cout << "none";
        } else {
            for (size_t i = 0; i < gpu.links.size(); ++i) {
                std::cout << "(" << gpu.links[i].peer_id
                          << ", rank=" << gpu.links[i].perf_rank << ")";
                if (i + 1 < gpu.links.size()) std::cout << " ";
            }
        }
        std::cout << "\n";
    }
}

static std::vector<std::vector<int>> build_gpu_islands(const TopologyInfo& topo, int perf_threshold) {
    int n = topo.num_gpus;
    std::vector<std::vector<int>> islands;
    if (n <= 0) return islands;

    std::vector<bool> visited(n, false);

    for (int g = 0; g < n; ++g) {
        if (visited[g]) continue;
        std::vector<int> stack{g};
        std::vector<int> island;
        visited[g] = true;

        while (!stack.empty()) {
            int cur = stack.back();
            stack.pop_back();
            island.push_back(cur);

            const auto& gi = topo.gpus[cur];
            for (const auto& link : gi.links) {
                if (link.perf_rank <= perf_threshold && !visited[link.peer_id]) {
                    visited[link.peer_id] = true;
                    stack.push_back(link.peer_id);
                }
            }
        }

        std::sort(island.begin(), island.end());
        islands.push_back(island);
    }

    std::sort(islands.begin(), islands.end(),
              [](const std::vector<int>& a, const std::vector<int>& b) {
                  if (a.size() != b.size()) return a.size() > b.size();
                  return a.front() < b.front();
              });

    return islands;
}


RankMapping build_rank_mapping(const TopologyInfo& topo, int num_ranks, const std::string& mode) {
    RankMapping mapping;
    mapping.rank_to_gpu.resize(num_ranks, 0);

    if (topo.num_gpus == 0 || num_ranks <= 0) {
        return mapping;
    }

    if (mode == "baseline") {
        for (int r = 0; r < num_ranks; ++r) {
            mapping.rank_to_gpu[r] = r % topo.num_gpus;
        }
        return mapping;
    }

    int perf_threshold = 0;  
    auto islands = build_gpu_islands(topo, perf_threshold);

    if (islands.empty()) {
        for (int r = 0; r < num_ranks; ++r) {
            mapping.rank_to_gpu[r] = r % topo.num_gpus;
        }
        return mapping;
    }

    std::vector<std::vector<int>> islands_sorted;
    islands_sorted.reserve(islands.size());

    for (const auto& island : islands) {
        std::vector<int> sorted_island = island;

        std::sort(sorted_island.begin(), sorted_island.end(),
            [&topo](int a, int b) {
                const auto& ga = topo.gpus[a];
                const auto& gb = topo.gpus[b];
                if (ga.node_id != gb.node_id) return ga.node_id < gb.node_id;
                if (ga.numa_node != gb.numa_node) return ga.numa_node < gb.numa_node;
                return ga.id < gb.id;
            });

        islands_sorted.push_back(sorted_island);
    }

    std::sort(islands_sorted.begin(), islands_sorted.end(),
        [&topo](const std::vector<int>& a, const std::vector<int>& b) {
            if (a.size() != b.size()) return a.size() > b.size();

            const auto& ga = topo.gpus[a.front()];
            const auto& gb = topo.gpus[b.front()];

            if (ga.node_id != gb.node_id) return ga.node_id < gb.node_id;
            if (ga.numa_node != gb.numa_node) return ga.numa_node < gb.numa_node;
            return ga.id < gb.id;
        });

    std::vector<int> topo_order;
    topo_order.reserve(topo.num_gpus);

    for (const auto& island : islands_sorted) {
        topo_order.insert(topo_order.end(), island.begin(), island.end());
    }

    std::set<int> in_order(topo_order.begin(), topo_order.end());
    std::vector<int> isolated;
    for (int g = 0; g < topo.num_gpus; ++g) {
        if (!in_order.count(g)) {
            isolated.push_back(g);
        }
    }
    std::sort(isolated.begin(), isolated.end(),
        [&topo](int a, int b) {
            const auto& ga = topo.gpus[a];
            const auto& gb = topo.gpus[b];
            if (ga.node_id != gb.node_id) return ga.node_id < gb.node_id;
            if (ga.numa_node != gb.numa_node) return ga.numa_node < gb.numa_node;
            return ga.id < gb.id;
        });
    topo_order.insert(topo_order.end(), isolated.begin(), isolated.end());

    for (int r = 0; r < num_ranks; ++r) {
        int idx = r % static_cast<int>(topo_order.size());
        mapping.rank_to_gpu[r] = topo_order[idx];
    }

    return mapping;
}


void dump_topology_and_mapping_json(const TopologyInfo& topo,
    const RankMapping& mapping,
    const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "Failed to open JSON output file: " << path << "\n";
        return;
    }

    ofs << "{\n";
    ofs << "  \"num_gpus\": " << topo.num_gpus << ",\n";

    ofs << "  \"gpus\": [\n";
    for (size_t i = 0; i < topo.gpus.size(); ++i) {
        const auto& g = topo.gpus[i];
        ofs << "    {\n";
        ofs << "      \"id\": " << g.id << ",\n";
        ofs << "      \"name\": \"" << g.name << "\",\n";
        ofs << "      \"numa_node\": " << g.numa_node << ",\n";
        ofs << "      \"node_id\": " << g.node_id << ",\n";
        ofs << "      \"links\": [";
        for (size_t j = 0; j < g.links.size(); ++j) {
            const auto& link = g.links[j];
            ofs << "{ \"peer_id\": " << link.peer_id
                << ", \"perf_rank\": " << link.perf_rank << "}";
            if (j + 1 < g.links.size()) ofs << ", ";
        }
        ofs << "]\n";
        ofs << "    }";
        if (i + 1 < topo.gpus.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ],\n";

    ofs << "  \"rank_mapping\": [\n";
    for (size_t r = 0; r < mapping.rank_to_gpu.size(); ++r) {
        ofs << "    { \"rank\": " << r
            << ", \"gpu_id\": " << mapping.rank_to_gpu[r] << " }";
        if (r + 1 < mapping.rank_to_gpu.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ]\n";

    ofs << "}\n";
    ofs.close();

    std::cout << "Topology and mapping JSON written to " << path << "\n";
}