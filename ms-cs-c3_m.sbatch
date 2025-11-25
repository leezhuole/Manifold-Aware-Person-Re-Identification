#!/bin/bash
#SBATCH --job-name=bau-train                # Job name
#SBATCH --gres=gpu:1,VRAM:12G               # Request 2 GPUs with at least 12GB VRAM each                         
#SBATCH --nodes=1                           # Run all processes on a single node
#SBATCH --cpus-per-task=5                  # Number of CPU cores per task
#SBATCH --mem=32G                           # Total memory for the entire job
#SBATCH --time=24:00:00                     # Time limit hrs:min:sec
#SBATCH --constraint="GPU_MODEL:rtx_5000"   # Use only rtx5000 for compatibility
#SBATCH --output=logs/%x-%j.out             # Standard output and error log  (e.g., logs/bau-train-12345.out)
#SBATCH --error=logs/%x-%j.err              # Separate error log (e.g., logs/bau-train-12345.err)
# Optional if you need a specific GPU gen:
# SBATCH --constraint=<feature>              # PDF: Choosing nodes → By features/type (p.4–5)

set -euo pipefail
mkdir -p logs

# Activate your environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU   # or module load <python/conda>

# Slurm already sets CUDA_VISIBLE_DEVICES for the allocated GPU
srun --unbuffered python examples/train_bau.py \
  -a resnet50 \
  -ds msmt17 cuhk03 cuhksysu \
  -dt market1501 \
  -b 32 \
  --lam 1.5 --k 10 --iters 500 \
  --logs-dir "/home/stud/leez/reid/src/BAU/examples/logs/ms-cs-c3_m" \
  --manifold-aware