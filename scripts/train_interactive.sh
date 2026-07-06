#!/bin/bash
# Interactive training script for MINT+
# Run this INSIDE an interactive SLURM session (srun --pty bash)
#
# Usage on the compute node:
#   bash scripts/train_interactive.sh
#
# Or override config:
#   CONFIG=configs/recipes/frozen_35M.yaml bash scripts/train_interactive.sh
#
# To get an interactive session first:
#   srun --partition=gpu-a30 --account=danglab --gpus-per-node=4 \
#        --cpus-per-task=16 --time=01:00:00 --mem=128G --pty bash

set -euo pipefail

# Overridable defaults
CONFIG="${CONFIG:-configs/recipes/no_frozen_8M.yaml}"
CONDA_ENV="${CONDA_ENV:-mint}"

# 1. Load HPC4 environment
source /opt/shared/.spack-edge/dist/bin/setup-env.sh -y
module load anaconda3/2025

# qt-main_activate.sh references QT_XCB_GL_INTEGRATION without a default;
# pre-export it to avoid unbound-variable errors under set -u.
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-}"

# 2. Activate conda
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}" 2>/dev/null || {
    echo "ERROR: Conda env '${CONDA_ENV}' not found."
    exit 1
}

# 3. Go to project root (resolve from script location, not cwd)
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "${SCRIPT_DIR}/.."
echo "Project root: ${PWD}"

# 4. Set PYTHONPATH and environment
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
unset CUDA_VISIBLE_DEVICES

# 5. Diagnostics
echo "================================================"
echo "MINT+ Interactive Training"
echo "================================================"
echo "Host:    $(hostname)"
/usr/bin/nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo "Python:  $(which python) ($(python --version 2>&1))"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"
echo "Config:  ${CONFIG}"
echo "GPUs:    $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'N/A')"
echo "================================================"
echo ""

# 6. Launch training
python -c "
import os, sys
from mint_plus.training.trainer import MINTTrainer

trainer = MINTTrainer.from_config('${CONFIG}')
trainer.fit()
"
