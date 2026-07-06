# SLURM Beginner's Guide — HKUST HPC4

**Last updated: 2026-07-03**

This guide covers the basics of using SLURM (Simple Linux Utility for Resource
Management) on HKUST's HPC4 cluster for training ML models. If you're new to
HPC scheduling, start here.

---

## Table of Contents

1. [What is SLURM?](#1-what-is-slurm)
2. [Checking available resources](#2-checking-available-resources)
3. [Submitting a batch job](#3-submitting-a-batch-job)
4. [Monitoring jobs](#4-monitoring-jobs)
5. [Cancelling jobs](#5-cancelling-jobs)
6. [Interactive sessions](#6-interactive-sessions)
7. [GPU partitions reference](#7-gpu-partitions-reference)
8. [Environment setup (one-time)](#8-environment-setup-one-time)
9. [Common pitfalls](#9-common-pitfalls)
10. [Quick reference card](#10-quick-reference-card)

---

## 1. What is SLURM?

SLURM is the job scheduler on HPC4. You don't run GPU workloads directly on
the login node — you write a **batch script**, submit it with `sbatch`, and
SLURM finds an available compute node with the resources you requested, runs
your script there, and saves the output.

The key idea: **you describe what you need (GPUs, CPUs, memory, time), and
SLURM fits you into an available node.**

---

## 2. Checking available resources

### Which partitions exist and how busy are they?

```bash
sinfo
```

This shows every partition, how many nodes are idle/allocated/mixed/draining,
and which specific nodes. Example output:

```
PARTITION      AVAIL  TIMELIMIT  NODES  STATE NODELIST
gpu-a30          up   infinite      8   idle gpu[02-07,12-13]
gpu-l20          up   infinite      2   idle gpu[16-17]
gpu-rtx5880      up   infinite      0   idle (none)
```

- **idle** = free right now, your job launches immediately
- **mix** = some GPUs free, some used
- **alloc** = fully occupied
- **drain\*** = node is faulty, being repaired

### What GPUs does each partition have?

```bash
sinfo -o "%P %G"
```

The `%G` column shows the generic resource schedule (GPU type). On HPC4:
- `gpu-a30` → NVIDIA A30 (24 GB VRAM)
- `gpu-l20` → NVIDIA L20 (48 GB VRAM)
- `gpu-rtx4090d` → RTX 4090D (24 GB VRAM)
- `gpu-rtx5880` → RTX 5880 Ada (48 GB VRAM)
- `temgpu` → varies (check with `sinfo -N -p temgpu`)

### What's your allocation/account?

```bash
sacctmgr show user $USER withassoc
```

Shows which SLURM **accounts** you can charge jobs to and which **partitions**
each account can access. Each account may have access to different partitions,
so you need to match the account to the partition.

For user `wtliaf`, the available accounts are:

| Account | Accessible partitions |
|---|---|
| `danglab` | gpu-a30, gpu-l20, gpu-rtx5880, gpu-rtx4090d, amd, intel, temgpu |
| `danglabmet` | gpu-a30, gpu-l20, gpu-rtx5880, gpu-rtx4090d, amd, intel |
| `danglabves` | gpu-a30, gpu-l20, gpu-rtx5880, gpu-rtx4090d, amd, intel |
| `danglabmp` | gpu-a30, gpu-l20, gpu-rtx5880, gpu-rtx4090d, amd, intel |
| `migrate` | gpu-a30, gpu-l20, gpu-rtx5880, gpu-rtx4090d, amd, intel |

Run `sacctmgr show user $USER withassoc` yourself to see your own accounts
and their partition access.

### What's currently queued or running?

```bash
squeue -u $USER          # your jobs only
squeue -p gpu-a30        # all jobs on a specific partition
squeue                    # everyone's jobs (noisy)
```

---

## 3. Submitting a batch job

### The default way

This project already has a SLURM script at `scripts/train.slurm`. From the
project root:

```bash
sbatch scripts/train.slurm
```

This submits the default config (`frozen_150M.yaml`) requesting 1 GPU, 8 CPUs,
48 hours, on `gpu-rtx5880` under account `danglabmet`.

### How the script works

The script reads two kinds of variables:

**1. Environment variables (set before `sbatch`) — these actually work:**

| Env var | What it controls | Default |
|---|---|---|
| `CONFIG` | Recipe YAML path | `configs/recipes/frozen_150M.yaml` |
| `CONDA_ENV` | Conda env name or path | `mint_plus` |
| `DATA_DIR` | STRING data directory | `./data/diamond` |
| `CKPT_DIR` | Checkpoint output directory | `./ckpts` |
| `RUN_NAME` | Run name for WandB/logging | (none) |
| `USE_SYSTEM_CUDA` | Load system CUDA module? | `false` |

These are read by the script at runtime via `"${VAR:-default}"`.

**2. SBATCH directives (hardcoded with `#SBATCH`) — these only change via CLI flags:**

The script has these hardcoded:

- `#SBATCH --partition=gpu-rtx5880`
- `#SBATCH --account=danglabmet`
- `#SBATCH --gpus-per-node=1`
- `#SBATCH --cpus-per-task=8`
- `#SBATCH --time=48:00:00`

The env vars `PARTITION`, `ACCOUNT`, `GPUS`, `CPUS`, `TIME` are **NOT read by SBATCH** — they only appear in the diagnostic header. To change the actual allocation, use `sbatch` command-line flags.

### Overriding SBATCH directives

Pass `--flag=value` after `sbatch` and before the script path:

```bash
# Override partition + account + GPUs + CPUs
CONFIG=configs/recipes/no_frozen_8M.yaml \
    sbatch --partition=gpu-a30 --account=danglab \
    --gpus-per-node=4 --cpus-per-task=16 \
    scripts/train.slurm
```

Any `--flag=value` you pass on the command line **overrides** the corresponding
`#SBATCH` directive in the script.

### The right pattern — env vars + SBATCH flags

Use env vars for things the script reads (`CONFIG`, `CONDA_ENV`, `DATA_DIR`,
`CKPT_DIR`). Use `--flag=value` for things SBATCH controls (partition, account,
GPUs, CPUs, time, memory).

```bash
# Frozen 150M on gpu-rtx5880 (default) — just override config
CONFIG=configs/recipes/frozen_35M.yaml sbatch scripts/train.slurm

# No-frozen 8M on gpu-a30 with 4 GPUs
CONFIG=configs/recipes/no_frozen_8M.yaml \
    sbatch --partition=gpu-a30 --account=danglab \
    --gpus-per-node=4 --cpus-per-task=16 \
    scripts/train.slurm

# Short debug run on temgpu
CONFIG=configs/recipes/frozen_8M.yaml \
    sbatch --partition=temgpu --account=danglab \
    --time=02:00:00 --gpus-per-node=1 \
    scripts/train.slurm
```

### Wrong patterns that look like they should work but don't

```bash
# WRONG — env var after sbatch, SLURM tries to open it as a file
sbatch --gpus-per-node=4 CPUS=16 scripts/train.slurm

# WRONG — PARTITION env var is ignored, script still uses gpu-rtx5880
PARTITION=gpu-a30 CPUS=16 sbatch scripts/train.slurm

# WRONG — ACCOUNT env var is ignored, script still uses danglabmet
ACCOUNT=danglab sbatch --partition=gpu-a30 scripts/train.slurm
```

If you see `sbatch: error: Unable to open file <name>`, you put an env var
after `sbatch`. Move it before.

If you see `Invalid account or account/partition combination specified`,
either the account doesn't have access to that partition, or the `--account`
flag wasn't passed at all and the script's hardcoded account is incompatible
with the partition you selected.

### After submission

You get a job ID printed to the terminal:

```
Submitted batch job 1234567
```

Output goes to `slurm/train-1234567.out`.

---

## 4. Monitoring jobs

### Is it running yet?

```bash
squeue -j 1234567
```

States you'll see:
- **PD (PENDING)** — waiting for resources
- **R (RUNNING)** — actively executing
- **CG (COMPLETING)** — almost done
- **CD (COMPLETED)** — finished (won't show in squeue; use sacct)

### Watch output in real time

```bash
tail -f slurm/train-1234567.out
```

You'll see the diagnostic header, then PyTorch Lightning progress bars and
loss/metrics as training proceeds.

### Check finished job details

```bash
sacct -j 1234567 --format=JobID,State,Elapsed,ExitCode,MaxRSS,ReqMem
```

This shows whether it completed successfully or failed, how long it ran, peak
memory, and exit code.

### GPU utilisation

The script prints `nvidia-smi` output at startup, but those metrics are a
snapshot. For per-step GPU stats, check the Lightning output — it logs
GPU utilisation and memory if configured.

---

## 5. Cancelling jobs

```bash
scancel 1234567          # cancel by job ID
scancel -u $USER         # cancel ALL your jobs (use carefully!)
scancel -n mint-train    # cancel by job name
```

No confirmation — it just goes away.

---

## 6. Interactive sessions

Sometimes you need to debug or test interactively instead of writing a batch
script. Request a GPU node with a shell:

```bash
srun --partition=gpu-a30 --account=danglab \
     --gpus-per-node=1 --cpus-per-task=8 \
     --mem=64G --time=01:00:00 --pty bash
```

Then set up your environment manually and run Python:

```bash
source /opt/shared/.spack-edge/dist/bin/setup-env.sh -y
module load anaconda3/2025
conda activate mint_plus
cd /path/to/mint_plus
python -c "from mint_plus.training.trainer import MINTTrainer; ..."
```

This is useful for:
- Debugging import errors or config parsing
- Testing a quick forward pass
- Profiling with `nvidia-smi` / `nsys`

**Always release the node when done.** `exit` closes the session and SLURM
reclaims the resources automatically.

Interactive vs batch rule of thumb:

| Situation | Best approach |
|---|---|
| One-time quick test (< 1h) | `srun --pty bash` |
| Repeated / overnight training | `sbatch scripts/train.slurm` |
| Debugging a crash | `srun --pty bash` then run Python |
| Hyperparameter sweep | Multiple `sbatch` submissions |

---

## 7. GPU partitions reference

| Partition | GPU | VRAM | Use for |
|---|---|---|---|
| `gpu-a30` | A30 | 24 GB | Small-to-mid experiments, 35M models |
| `gpu-l20` | L20 | 48 GB | Mid-sized models, LoRA, 150M |
| `gpu-rtx4090d` | RTX 4090D | 24 GB | 8M models, quick tests |
| `gpu-rtx5880` | RTX 5880 Ada | 48 GB | Default, 150M+ models |
| `temgpu` | varies | varies | Debug / throwaway jobs |

Always check current availability with `sinfo`. If your partition is full,
pick one with idle nodes and use the matching account:

```bash
# Pass both partition AND account on the sbatch line
sbatch --partition=gpu-a30 --account=danglab scripts/train.slurm
```

---

## 8. Environment setup (one-time)

These steps run once on a GPU login node to create your conda environment:

```bash
# 1. Load HPC4 software stack
source /opt/shared/.spack-edge/dist/bin/setup-env.sh -y

# 2. Load CUDA (needed during pip install for compiling CUDA extensions)
module load cuda/12.4.1

# 3. Load Anaconda
module load anaconda3/2025

# 4. Create environment
conda create -y -n mint_plus python=3.11

# 5. Activate and install dependencies
conda activate mint_plus
pip install -r requirements.txt
```

**Optional: use /scratch for faster I/O**

Your home directory (`/home/$USER`) is NFS-backed and slow for ML data.
For better performance, put your conda env on the scratch filesystem:

```bash
conda create -y -p /scratch/$USER/conda/mint_plus python=3.11
conda activate /scratch/$USER/conda/mint_plus
pip install -r requirements.txt
```

Then submit jobs with:

```bash
CONDA_ENV=/scratch/$USER/conda/mint_plus sbatch scripts/train.slurm
```

---

## 9. Common pitfalls

### Job stuck in PD (PENDING) forever

Check why:

```bash
squeue -j 1234567 -o "%i %P %t %R"
```

The `%R` column shows the reason:
- **Resources** — not enough idle nodes (try a less busy partition)
- **Priority** — other users' jobs are ahead (wait)
- **PartitionNodeLimit** — too many GPUs requested for available nodes
- **AssociationJobLimit** — hit your max running jobs as a user

Fix: switch to a partition with idle nodes (see section 7), or request fewer
GPUs.

### "Invalid account or account/partition combination"

Two causes:

1. You didn't pass `--account=...` and the script's hardcoded account
   (`danglabmet`) doesn't have access to the partition you chose.
   **Fix:** always pass `--account=danglab` when using a non-default partition.

2. The account you chose genuinely doesn't have access to that partition.
   **Fix:** run `sacctmgr show user $USER withassoc` and pick an account that
   lists the partition.

### Environment variables ignored (PARTITION, CPUS, GPUS, etc.)

These env vars only get printed in the job header — they do NOT change the
SBATCH allocation. Use `--partition=`, `--cpus-per-task=`, `--gpus-per-node=`
on the `sbatch` command line instead.

### "sbatch: error: Unable to open file <name>"

You put a variable after `sbatch`:

```bash
# WRONG
sbatch --gpus-per-node=4 CPUS=16 scripts/train.slurm

# RIGHT — variable before sbatch
CPUS=16 sbatch --gpus-per-node=4 scripts/train.slurm

# ALSO RIGHT — pass as --flag
sbatch --gpus-per-node=4 --cpus-per-task=16 scripts/train.slurm
```

### Conda env not found

The script prints the exact commands to create the env. Run them on a login
node first.

### NCCL / InfiniBand errors

On your first run, edit the script to set `NCCL_DEBUG="INFO"` (line ~117) to
see IB detection details. Once it works, switch back to `"WARN"`.

### CUDA version mismatch

The conda env ships its own CUDA 13 runtime — the script skips loading the
system `cuda/12.4.1` module by default. If training fails with library
version errors, try setting `USE_SYSTEM_CUDA=true` at submit time. This loads
the Spack CUDA module, which provides `nvcc` and CUDA headers.

### OOM (Out of Memory)

If the job fails with an OOM error:

1. Try a partition with more VRAM (e.g., `gpu-l20` has 48 GB vs 24 GB)
2. Reduce batch size in the config
3. Use a frozen-backbone config (uses much less memory)
4. Check `sacct` for peak memory: `sacct -j 1234567 --format=MaxRSS`

### Output not appearing

SLURM buffers stdout. You can flush explicitly in Python (`sys.stdout.flush()`)
or use `--open-mode=append` vs `truncate` in the `#SBATCH --open-mode` line.
The script already uses `truncate` so output starts from scratch each run.

---

## 10. Quick reference card

```bash
# CHECK YOUR ACCOUNTS & PARTITION ACCESS
sacctmgr show user $USER withassoc

# CHECK AVAILABLE PARTITIONS
sinfo
sinfo -o "%P %G %D %t %N"

# --- SUBMIT (default config, default partition) ---
sbatch scripts/train.slurm

# --- SUBMIT (override config only) ---
CONFIG=configs/recipes/frozen_35M.yaml sbatch scripts/train.slurm

# --- SUBMIT (different partition + account) ---
CONFIG=configs/recipes/no_frozen_8M.yaml \
    sbatch --partition=gpu-a30 --account=danglab \
    --gpus-per-node=1 --cpus-per-task=8 \
    scripts/train.slurm

# --- SUBMIT (multi-GPU) ---
CONFIG=configs/recipes/no_frozen_8M.yaml \
    sbatch --partition=gpu-a30 --account=danglab \
    --gpus-per-node=4 --cpus-per-task=16 \
    scripts/train.slurm

# CHECK QUEUE
squeue -u $USER

# CHECK WHY PENDING
squeue -j 1234567 -o "%i %P %t %R"

# WATCH OUTPUT
tail -f slurm/train-*.out

# CHECK COMPLETED JOB
sacct -j 1234567 --format=JobID,State,Elapsed,ExitCode,MaxRSS

# CANCEL
scancel 1234567

# INTERACTIVE SESSION (1 hour, 1 GPU on gpu-a30)
srun --partition=gpu-a30 --account=danglab \
     --gpus-per-node=1 --cpus-per-task=8 \
     --mem=64G --time=01:00:00 --pty bash
```
