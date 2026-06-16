#pragma once
#include <vector>
#include <string>

struct GpuInfo {
  int id;
  std::string name;
  int major;
  int minor;
  size_t totalMemMB;
};

std::vector<GpuInfo> detect_gpus();
