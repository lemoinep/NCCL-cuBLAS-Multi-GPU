set -euo pipefail

EXE=${EXE:-./nccl-topopipe}
MATRIX_SIZE=${MATRIX_SIZE:-4096}

NP_TOTAL=${NP_TOTAL:-8}
HOSTS=${HOSTS:-"node0:4,node1:4"}

echo "[multi_node_mpi] Running baseline mode on multi-node"
mpirun -np "${NP_TOTAL}" \
       -H "${HOSTS}" \
       --map-by ppr:4:node --bind-to core \
       "${EXE}" \
       --mode baseline \
       --matrix-size "${MATRIX_SIZE}"

echo "[multi_node_mpi] Running topology-aware mode on multi-node"
mpirun -np "${NP_TOTAL}" \
       -H "${HOSTS}" \
       --map-by ppr:4:node --bind-to core \
       "${EXE}" \
       --mode topology-aware \
       --matrix-size "${MATRIX_SIZE}" \
       --dump-json multi_node_topology_mapping.json