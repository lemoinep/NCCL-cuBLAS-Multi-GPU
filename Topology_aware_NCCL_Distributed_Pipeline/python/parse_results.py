import re
import sys
import csv


def parse_file(path):
    data = {"total_ms": None, "compute_ms": None, "communication_ms": None, "bandwidth_gbps": None}
    with open(path, "r") as f:
        text = f.read()

    patterns = {
        "total_ms": r"total_ms:\s*([0-9.]+)",
        "compute_ms": r"compute_ms:\s*([0-9.]+)",
        "communication_ms": r"communication_ms:\s*([0-9.]+)",
        "bandwidth_gbps": r"bandwidth_gbps:\s*([0-9.]+)",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            data[key] = float(m.group(1))
    return data


def main():
    if len(sys.argv) < 3:
        print("Usage: parse_results.py output.csv file1 file2 ...")
        sys.exit(1)

    out_csv = sys.argv[1]
    rows = []
    for path in sys.argv[2:]:
        row = parse_file(path)
        row["file"] = path
        rows.append(row)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "total_ms", "compute_ms", "communication_ms", "bandwidth_gbps"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()