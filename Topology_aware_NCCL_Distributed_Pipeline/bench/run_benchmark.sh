set -euo pipefail

EXE=${EXE:-./nccl-topopipe}
MATRIX_SIZE=${MATRIX_SIZE:-4096}
RESULTS_DIR=${RESULTS_DIR:-results}

mkdir -p "${RESULTS_DIR}"
mkdir -p examples

echo "[bench] Using executable: ${EXE}"
echo "[bench] Matrix size: ${MATRIX_SIZE}"
echo "[bench] Results directory: ${RESULTS_DIR}"

echo "[bench] Running single_node_4gpu baseline/topology-aware"
EXE="${EXE}" MATRIX_SIZE="${MATRIX_SIZE}" NP=4 \
    bash examples/single_node_4gpu.sh \
    > "${RESULTS_DIR}/single_node_4gpu.log" 2>&1

echo "[bench] Running multi_node_mpi baseline/topology-aware"
EXE="${EXE}" MATRIX_SIZE="${MATRIX_SIZE}" NP_TOTAL=8 HOSTS="${HOSTS:-node0:4,node1:4}" \
    bash examples/multi_node_mpi.sh \
    > "${RESULTS_DIR}/multi_node_mpi.log" 2>&1

echo "[bench] Extracting metrics from logs"

python3 python/parse_results.py \
    "${RESULTS_DIR}/bench_metrics.csv" \
    "${RESULTS_DIR}/single_node_4gpu.log" \
    "${RESULTS_DIR}/multi_node_mpi.log"

echo "[bench] Benchmark metrics written to ${RESULTS_DIR}/bench_metrics.csv"