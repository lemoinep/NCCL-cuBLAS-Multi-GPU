set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-results}
CSV_NAME=${CSV_NAME:-bench_metrics.csv}
OUT_PREFIX=${OUT_PREFIX:-bench}

echo "[collect_metrics] Results directory: ${RESULTS_DIR}"
echo "[collect_metrics] Output CSV: ${CSV_NAME}"
echo "[collect_metrics] Output prefix for plots: ${OUT_PREFIX}"

SINGLE_LOG="${RESULTS_DIR}/single_node_4gpu.log"
MULTI_LOG="${RESULTS_DIR}/multi_node_mpi.log"

if [[ ! -f "${SINGLE_LOG}" ]]; then
    echo "Missing ${SINGLE_LOG}. Run bench/run_benchmark.sh first."
    exit 1
fi

if [[ ! -f "${MULTI_LOG}" ]]; then
    echo "Missing ${MULTI_LOG}. Run bench/run_benchmark.sh first."
    exit 1
fi

echo "[collect_metrics] Generating CSV from logs"
python3 python/parse_results.py \
    "${RESULTS_DIR}/${CSV_NAME}" \
    "${SINGLE_LOG}" \
    "${MULTI_LOG}"

echo "[collect_metrics] Generating plots from CSV"
python3 python/plot_bench.py \
    "${RESULTS_DIR}/${CSV_NAME}" \
    "${OUT_PREFIX}"

echo "[collect_metrics] Done. CSV: ${RESULTS_DIR}/${CSV_NAME}, plots: ${OUT_PREFIX}_comm_ms.png, ${OUT_PREFIX}_bandwidth.png"