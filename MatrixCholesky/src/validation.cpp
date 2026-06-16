#include "validation.hpp"
#include <cmath>
#include <algorithm>

double frob_norm_diff_lower(const std::vector<double>& A, const std::vector<double>& L, int n) {
  auto idx = [n](int i, int j){ return size_t(i) * n + j; };
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double ll = 0.0;
      for (int k = 0; k <= std::min(i,j); ++k) ll += L[idx(i,k)] * L[idx(j,k)];
      double d = A[idx(i,j)] - ll;
      s += d * d;
    }
  }
  return std::sqrt(s);
}

