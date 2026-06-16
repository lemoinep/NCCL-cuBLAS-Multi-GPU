#include "owner_utils.hpp"

int block_index_of(int k, int nb) { return k / nb; }
int block_owner(int block_index, int nranks) { return block_index % nranks; }
int panel_owner(int k, int nb, int nranks) { return block_owner(block_index_of(k, nb), nranks); }


