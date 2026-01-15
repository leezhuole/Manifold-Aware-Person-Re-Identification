#!/bin/bash
#SBATCH --job-name=bau-sweep-batchsize        # Job name
#SBATCH --array=0-3                           # Run 4 jobs (indices 0, 1, 2, 3)
#SBATCH --gres=gpu:2,VRAM:48G                 # Request 2 GPUs with at least 48GB VRAM each                         
#SBATCH --nodes=1                             # Run all processes on a single node
#SBATCH --cpus-per-task=10                    # Number of CPU cores per task
#SBATCH --mem=64G                             # Total memory for the entire job
#SBATCH --time=24:00:00                       # Time limit hrs:min:sec
#SBATCH --constraint="GPU_MODEL:a40"          # Use only rtx5000 for compatibility
#SBATCH --output=logs/%x-%A_%a.out            # %A=Job ID, %a=Array Index
#SBATCH --error=logs/%x-%A_%a.err             # Separate logs for each run
# Optional if you need a specific GPU gen:
# SBATCH --constraint=<feature>               # PDF: Choosing nodes → By features/type 

set -euo pipefail
mkdir -p logs

# --- PARAMETER CONFIGURATION ---

# 1. Define the lists of values you want to test
# (Ensure the number of items matches the --array range above)
batch_size=(64 64 128 128)
manifold_aware=("True" "False" "True" "False")

# 2. Select the values for the current run using the Array Task ID
CURRENT_BATCH_SIZE=${batch_size[$SLURM_ARRAY_TASK_ID]}
CURRENT_MANIFOLD_AWARE=${manifold_aware[$SLURM_ARRAY_TASK_ID]}

echo "Starting Task ID: $SLURM_ARRAY_TASK_ID"
echo "Parameters: Batch Size=$CURRENT_BATCH_SIZE, Manifold Aware=$CURRENT_MANIFOLD_AWARE"

# -------------------------------


# Activate the environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU   

# Slurm already sets CUDA_VISIBLE_DEVICES for the allocated GPU
srun --unbuffered python examples/train_bau.py \
  -a resnet50 \
  -ds market1501 msmt17 cuhksysu \
  -dt cuhk03 \
  -b  $CURRENT_BATCH_SIZE \
  --lam 1.5 --k 10 --iters 500 \
  --logs-dir "/home/stud/leez/reid/src/BAU/examples/logs/m-ms-cs_c3_job${SLURM_ARRAY_JOB_ID}" \
  --manifold-aware $CURRENT_MANIFOLD_AWARE \
  --curvature 1.0 \
  --wandb-name-add "bsize-${CURRENT_BATCH_SIZE}_manifold-${CURRENT_MANIFOLD_AWARE}"
