import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["total_ms", "compute_ms", "communication_ms", "bandwidth_gbps"]:
                if row.get(key):
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = None
            rows.append(row)
    return rows


def classify_scenario(row):
    fname = Path(row["file"]).name
    if "single_node_4gpu" in fname:
        return "single_node_4gpu"
    elif "multi_node_mpi" in fname:
        return "multi_node_mpi"
    else:
        return "unknown"


def plot_metrics(rows, out_prefix="bench"):
    scenarios = {}
    for row in rows:
        scenario = classify_scenario(row)
        scenarios.setdefault(scenario, []).append(row)

    labels = []
    comm_ms_vals = []
    bw_vals = []

    for scenario, srows in scenarios.items():
        if not srows:
            continue

        comm_ms = [r["communication_ms"] for r in srows if r["communication_ms"] is not None]
        bw = [r["bandwidth_gbps"] for r in srows if r["bandwidth_gbps"] is not None]
        if not comm_ms or not bw:
            continue
        labels.append(scenario)
        comm_ms_vals.append(sum(comm_ms) / len(comm_ms))
        bw_vals.append(sum(bw) / len(bw))

    if not labels:
        print("No metrics to plot.")
        return

    plt.figure(figsize=(6, 4))
    plt.bar(labels, comm_ms_vals, color=["steelblue", "darkorange"])
    plt.ylabel("Communication latency (ms)")
    plt.title("NCCL pipeline communication latency per scenario")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_comm_ms.png")
    print(f"Saved {out_prefix}_comm_ms.png")

    plt.figure(figsize=(6, 4))
    plt.bar(labels, bw_vals, color=["seagreen", "firebrick"])
    plt.ylabel("Bandwidth (GB/s)")
    plt.title("NCCL pipeline bandwidth per scenario")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_bandwidth.png")
    print(f"Saved {out_prefix}_bandwidth.png")


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_bench.py bench_metrics.csv [out_prefix]")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else "bench"

    rows = load_metrics(csv_path)
    plot_metrics(rows, out_prefix)


if __name__ == "__main__":
    main()