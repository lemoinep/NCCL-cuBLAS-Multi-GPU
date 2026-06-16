#!/bin/bash
#SBATCH --job-name=nccl-cholesky
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,NET,COLL
export NCCL_DEBUG_FILE=logs/nccl_%h_%p.log
export NCCL_ASYNC_ERROR_HANDLING=1

srun ./nccl_cusolver_cholesky 4096 256