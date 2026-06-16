#pragma once
#include <vector>

struct BlockTask {
  int k;
  int nb;
  int owner;
  bool potrf_done;
  bool broadcast_done;
  bool update_done;
};

struct TaskScheduler {
  int n, nb, nranks, rank;
  std::vector<BlockTask> tasks;
  explicit TaskScheduler(int n_, int nb_, int nranks_, int rank_);


};

