#!/bin/bash
#SBATCH --job-name=bau-train                  # Job name
#SBATCH --gres=gpu:2,VRAM:48G                 # Request 2 GPUs with at least 48GB VRAM each                         
#SBATCH --nodes=1                             # Run all processes on a single node
#SBATCH --cpus-per-task=10                    # Number of CPU cores per task
#SBATCH --mem=64G                             # Total memory for the entire job
#SBATCH --time=24:00:00                       # Time limit hrs:min:sec
#SBATCH --constraint="GPU_MODEL:a40"          # Use only rtx5000 for compatibility
#SBATCH --output=logs/%x-%j.out               # Standard output and error log  (e.g., logs/bau-train-12345.out)
#SBATCH --error=logs/%x-%j.err                # Separate error log (e.g., logs/bau-train-12345.err)
# Optional if you need a specific GPU gen:
# SBATCH --constraint=<feature>               # PDF: Choosing nodes → By features/type (p.4–5)

set -euo pipefail
mkdir -p logs

# Activate the environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU   


# Slurm already sets CUDA_VISIBLE_DEVICES for the allocated GPU
srun --unbuffered python examples/train_bau.py \
  -a resnet50 \
  -ds market1501 msmt17 cuhksysu \
  -dt cuhk03 \
  -b  256 \
  --lam 1.5 --k 10 --iters 500 \
  --logs-dir "/home/stud/leez/reid/src/BAU/examples/logs/m-ms-cs_c3" \
  --manifold-aware \
  --curvature 1.0
  