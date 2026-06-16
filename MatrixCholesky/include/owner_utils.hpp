#pragma once
int block_index_of(int k, int nb);
int block_owner(int block_index, int nranks);
int panel_owner(int k, int nb, int nranks);

