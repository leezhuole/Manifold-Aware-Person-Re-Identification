#!/bin/bash
#SBATCH --job-name=bau-ablation             # Job name
#SBATCH --gres=gpu:2,VRAM:24G               # Request 2 GPUs with at least 24GB VRAM each                         
#SBATCH --nodes=1                           # Run all processes on a single node
#SBATCH --cpus-per-task=10                  # Number of CPU cores per task
#SBATCH --mem=32G                           # Total memory for the entire job
#SBATCH --time=24:00:00                     # Time limit hrs:min:sec
#SBATCH --exclude=node7,node9                       # Exclude problematic nodes
# #SBATCH --constraint="GPU_MODEL:rtx_5000"   # Use only rtx5000 for compatibility
#SBATCH --output=logs/%x-%j.out             # Standard output and error log  (e.g., logs/bau-train-12345.out)
#SBATCH --error=logs/%x-%j.err              # Separate error log (e.g., logs/bau-train-12345.err)

set -euo pipefail
# mkdir -p logs/ablation

# Activate your environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU   # or module load <python/conda>

LOGS_DIR="$(pwd)/logs/ablation/job_${SLURM_JOB_ID}"
WANDB_NAME="ablationHyp_MSCSC3_to_M_Baseline"


# Slurm already sets CUDA_VISIBLE_DEVICES for the allocated GPU
srun --unbuffered python examples/train_bau.py \
  -a resnet50 \
  -ds msmt17 cuhksysu cuhk03 -dt market1501 \
  -b 256 \
  --lam 1.5 --k 10 --iters 500 \
  --epochs 60 \
  --eval-epochs 5 10 15 20 25 30 35 40 45 50 55 59 \
  --warmup-step 10 \
  --manifold-aware False \
  --curvature 0.0 \
  --logs-dir "${LOGS_DIR}" \
  --manifold-chunk-size 500 \
  --wandb-name "${WANDB_NAME}" \
  --verbosity 1 \
  --no-align --no-uniform --no-domain --prob 0.0
