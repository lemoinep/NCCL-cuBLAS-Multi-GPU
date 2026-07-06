#pragma once
#include <string>
#include <vector>

struct LinkInfo {
    int peer_id;    
    int perf_rank;  
};

struct GpuInfo {
    int id;
    std::string name;
    int numa_node;   
    int node_id;     
    std::vector<LinkInfo> links; 
};

struct TopologyInfo {
    int num_gpus;
    std::vector<GpuInfo> gpus;
};

struct RankMapping {
    std::vector<int> rank_to_gpu;
};

TopologyInfo detect_topology();
void print_topology(const TopologyInfo& topo);

RankMapping build_rank_mapping(const TopologyInfo& topo,
                               int num_ranks,
                               const std::string& mode);

void print_mapping(const RankMapping& mapping);