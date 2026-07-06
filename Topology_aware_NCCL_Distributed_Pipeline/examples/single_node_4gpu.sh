set -euo pipefail

EXE=${EXE:-./nccl-topopipe}
MATRIX_SIZE=${MATRIX_SIZE:-4096}
NP=${NP:-4}  

echo "[single_node_4gpu] Running baseline mode on 1 node, 4 GPUs"
mpirun -np "${NP}" \
       --map-by ppr:${NP}:node --bind-to core \
       "${EXE}" \
       --mode baseline \
       --matrix-size "${MATRIX_SIZE}"

echo "[single_node_4gpu] Running topology-aware mode on 1 node, 4 GPUs"
mpirun -np "${NP}" \
       --map-by ppr:${NP}:node --bind-to core \
       "${EXE}" \
       --mode topology-aware \
       --matrix-size "${MATRIX_SIZE}" \
       --dump-json single_node_topology_mapping.json

