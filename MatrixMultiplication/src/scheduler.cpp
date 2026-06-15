#include "scheduler.hpp"
#include <algorithm>

std::vector<Chunk> make_row_chunks(int M, int ngpu) {
  std::vector<Chunk> chunks;
  if (ngpu <= 0) return chunks;
  int rowsPerGPU = (M + ngpu - 1) / ngpu;
  for (int g = 0; g < ngpu; ++g) {
    int r0 = g * rowsPerGPU;
    int r1 = std::min(M, r0 + rowsPerGPU);
    if (r0 < r1) chunks.push_back({g, r0, r1});
  }
  return chunks;
}