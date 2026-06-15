#pragma once
#include <vector>

struct Chunk {
  int gpu;
  int row0;
  int row1;
};

std::vector<Chunk> make_row_chunks(int M, int ngpu);