#!/bin/bash
# ============================================================================
# SLURM batch script for MINT+ model training on HKUST HPC (HPC4 / SuperPOD)
#
# Usage:
#   sbatch scripts/train.slurm                          # default config
#   CONFIG=configs/recipes/frozen_35M.yaml sbatch ...    # override config
#
# Optional environment variable overrides:
#   CONFIG=configs/recipes/frozen_150M.yaml    (default)
#   CONFIG=configs/recipes/frozen_35M.yaml
#   CONFIG=configs/recipes/frozen_650M_flex.yaml
#   CONFIG=configs/recipes/no_frozen_35M.yaml
#   CONFIG=configs/recipes/frozen_lora_8M.yaml
#
#   ACCOUNT=my_project        (SLURM account — check with sacctmgr show user $USER withassoc)
#   PARTITION=gpu-a30         (GPU partition — HPC4: gpu-a30, gpu-l20; SuperPOD: normal)
#   GPUS=1                    (GPUs per node)
#   CPUS=8                    (CPUs per task — match to num_workers + headroom)
#   TIME=48:00:00             (walltime)
#   CONDA_ENV=mint_plus       (conda env name)
#   DATA_DIR=./data/diamond   (where STRING data lives)
#   CKPT_DIR=./ckpts          (where checkpoints go)
#
# Environment setup for HPC4 (HKUST):
#   source /opt/shared/.spack-edge/dist/bin/setup-env.sh -y
#   module load cuda/12.4.1
#   module load anaconda3/2025
#
# Reference: https://hkust-hpc-docs.readthedocs.io/latest/index.html
# ============================================================================

#SBATCH --job-name=mint-train
#SBATCH --output=slurm/train-%j.out
#SBATCH --open-mode=truncate
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=wtliaf
#SBATCH --partition=gpu-rtx5880
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G

set -euo pipefail

# ============================================================
# 1. Overridable defaults
# ============================================================
CONFIG="${CONFIG:-configs/recipes/frozen_150M.yaml}"
CONDA_ENV="${CONDA_ENV:-mint_plus}"
DATA_DIR="${DATA_DIR:-./data/diamond}"
CKPT_DIR="${CKPT_DIR:-./ckpts}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-8}"
# If your conda env ships its own CUDA runtime (nvidia-* wheels), set this to
# skip the system CUDA module. The NVIDIA kernel driver is enough at runtime.
# Only set USE_SYSTEM_CUDA=true if you need nvcc or CUDA headers from Spack.
USE_SYSTEM_CUDA="${USE_SYSTEM_CUDA:-false}"

# ============================================================
# 2. Activate HPC4 Spack edge instance + load modules
# ============================================================
# On HPC4, the edge Spack instance is at /opt/shared/.spack-edge.
# The `-y` flag skips the TOS prompt for non-interactive batch scripts.
source /opt/shared/.spack-edge/dist/bin/setup-env.sh -y

# System CUDA module — only loaded when explicitly needed.
# Your conda env already has CUDA 13 baked in:
#   nvidia-cuda-runtime-cu13=13.0.96
#   nvidia-cuda-nvrtc-cu13=13.0.88
#   nvidia-cuda-nvshmem-cu13=3.4.5
# These provide libcudart, libnvrtc, and the driver API at runtime.
# The only thing from the OS we need is the NVIDIA kernel driver (nvidia.ko),
# which is always present on GPU nodes regardless of loaded modules.
#
# Set USE_SYSTEM_CUDA=true only if you need nvcc from Spack (e.g., compiling
# a custom CUDA extension that doesn't ship as a pip wheel).
if [ "${USE_SYSTEM_CUDA}" = "true" ]; then
    module load cuda/12.4.1
    echo "System CUDA module loaded."
else
    echo "Using conda-env CUDA (nvidia-* wheels). System CUDA module skipped."
fi

# Load Anaconda3 so we can activate our pre-created conda environment
module load anaconda3/2025

# ============================================================
# 3. Activate conda environment
# ============================================================
# First-time setup (run on a login node or GPU interactive session):
#   source /opt/shared/.spack-edge/dist/bin/setup-env.sh -y
#   module load cuda/12.4.1
#   module load anaconda3/2025
#   conda create -y -n mint_plus python=3.11
#   conda activate mint_plus
#   pip install -r requirements.txt  # includes PyTorch 2.12.1, Triton 3.7.1, etc.

# For HPC4, conda environments live under ~/.conda/envs/ by default.
# If you prefer a shared location (faster on parallel filesystem):
#   conda create -y -p /scratch/$USER/conda/mint_plus python=3.11
# Then set CONDA_ENV=/scratch/$USER/conda/mint_plus below.
if [ -d "${CONDA_ENV}" ]; then
    # CONDA_ENV is a path (e.g., /scratch/$USER/conda/mint_plus)
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
else
    # CONDA_ENV is a name
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}" 2>/dev/null || {
        echo "ERROR: Conda env '${CONDA_ENV}' not found."
        echo "Create it first:"
        echo "  conda create -y -n ${CONDA_ENV} python=3.11"
        echo "  conda activate ${CONDA_ENV}"
        echo "  pip install -r requirements.txt"
        exit 1
    }
fi

# ============================================================
# 4. Prepare project + data directories
# ============================================================
# On HPC, always use /scratch (Tier-1, NVMe/SSD-backed, fast) for data
# and checkpoints. /home is NFS-backed and slow for I/O-heavy ML workloads.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Ensure output directories exist
mkdir -p "${CKPT_DIR}" slurm

# ============================================================
# 5. Print job diagnostics
# ============================================================
echo "================================================"
echo "MINT+ Training Job"
echo "================================================"
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Node:           $(hostname)"
echo "GPUs allocated: ${GPUS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "CUDA version:   $(nvcc --version 2>/dev/null | tail -1 || echo 'N/A')"
echo "Python:         $(which python) ($(python --version 2>&1))"
echo "PyTorch:        $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "Config:         ${CONFIG}"
echo "Data dir:       ${DATA_DIR}"
echo "Ckpt dir:       ${CKPT_DIR}"
echo "Conda env:      ${CONDA_ENV}"
echo "Workers:        ${CPUS}"
echo "Start time:     $(date -Iseconds)"
echo "================================================"

# ============================================================
# 6. Runtime environment tuning
# ============================================================
# train.py hardcodes CUDA_VISIBLE_DEVICES=0 for local dev.
# On HPC, SLURM sets CUDA_VISIBLE_DEVICES automatically based on
# --gpus-per-node, so we must unset it to avoid conflicts.
unset CUDA_VISIBLE_DEVICES

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# NCCL tuning for HPC4 InfiniBand fabric
export NCCL_DEBUG="WARN"               # "INFO" for first run to verify IB detecgtion
export NCCL_IB_DISABLE="0"             # enable InfiniBand (HPC4 nodes have IB)
export NCCL_NET_GDR_LEVEL="PHB"        # GPU Direct RDMA where available
export NCCL_P2P_DISABLE="0"
export CUDA_DEVICE_MAX_CONNECTIONS="1" # reduce NCCL connection overhead

# Lightning/NCCL robustness against transient network errors
export TORCH_NCCL_BLOCKING_WAIT="1"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"

# Triton kernel cache — persists compiled kernels between job restarts
export TRITON_CACHE_DIR="${CKPT_DIR}/triton_cache"
mkdir -p "${TRITON_CACHE_DIR}"

# Pre-allocate Triton's shared memory pool (reduces jitter)
export TRITON_MAX_SHARED_MEMORY="200000"  # bytes, adjust for your GPU

# ============================================================
# 7. Launch training
# ============================================================
# The MINTTrainer loads config via from_config() → the load_config() function
# supports YAML inheritance via the `extends:` key. No CLI args needed.
#
# If you need to override specific config values at submit time (e.g., batch
# size or learning rate), the from_config() method accepts an `overrides` dict.
# Uncomment the Python wrapper below and customise as needed.
python -c "
import os
import sys

# Use the DATA_DIR and CKPT_DIR env vars to override config at runtime
from mint_plus.training.trainer import MINTTrainer

overrides = {
    'data.data_dir': '${DATA_DIR}',
    'output.run_name': os.environ.get('RUN_NAME', None),
}

# Strip None overrides
overrides = {k: v for k, v in overrides.items() if v is not None}

trainer = MINTTrainer.from_config('${CONFIG}', overrides=overrides if overrides else None)
trainer.fit()
"

EXIT_CODE=$?

echo "================================================"
echo "Finish time: $(date -Iseconds)"
echo "Exit code:   ${EXIT_CODE}"
echo "================================================"

exit $EXIT_CODE
