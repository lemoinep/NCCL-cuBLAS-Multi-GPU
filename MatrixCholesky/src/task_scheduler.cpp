#include "task_scheduler.hpp"
#include "owner_utils.hpp"
#include <algorithm>

TaskScheduler::TaskScheduler(int n_, int nb_, int nranks_, int rank_)
  : n(n_), nb(nb_), nranks(nranks_), rank(rank_) {
  for (int k = 0; k < n; k += nb) {
    int bk = std::min(nb, n - k);
    tasks.push_back({k, bk, panel_owner(k, nb, nranks), false, false, false});
  }
}

