import argparse
import os
import subprocess
import sys


def build_cmd(args):
    cmd = [args.executable]
    cmd += ["--mode", args.mode]
    cmd += ["--gpus", str(args.gpus)]
    cmd += ["--matrix-size", str(args.matrix_size)]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Launch NCCL/cuBLAS topology-aware demo")
    parser.add_argument("--executable", default="./topology_aware_pipeline")
    parser.add_argument("--mode", choices=["baseline", "topology-aware"], default="baseline")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--np", type=int, default=4)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cmd = build_cmd(args)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    elif args.mpi:
        mpi_cmd = ["mpirun", "-np", str(args.np)] + cmd
        subprocess.run(mpi_cmd, check=False)
    else:
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    sys.exit(main())