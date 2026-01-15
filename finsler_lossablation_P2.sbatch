#!/bin/bash
#SBATCH --job-name=bau-ablation-fins             # Job name
#SBATCH --array=0-14                             # 5 loss configs * 5 runs each (mean/variance per config)
#SBATCH --gres=gpu:1,VRAM:28G                   # Request two 24GB-class GPUs 
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --cpus-per-task=5                       # Number of CPU cores per task
#SBATCH --mem=36G                               # Total memory for the entire job
#SBATCH --time=12:00:00                         # Time limit hrs:min:sec
#SBATCH --exclude=node7,node9,node1,node13    # Exclude problematic nodes
# #SBATCH --constraint="GPU_MODEL:a40"          # Ensure A40-class GPU
#SBATCH --output=logs/ablationFins-%A_%a.out     # Standard output log
#SBATCH --error=logs/ablationFins-%A_%a.err      # Error log

set -euo pipefail
mkdir -p logs/ablationFinsler-P2

# Activate the environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU

# Ensure local repo is on Python path for srun processes
export PYTHONPATH=/home/stud/leez/reid/src/bau_finsler

# Base hyper-parameters (shared across jobs)
ARCH=resnet50                   # Default
BATCH_SIZE=256                  # Default
LAM=1.5                         # Default
KNN=10                          # Default
ITERS=500                       # Default
WARMUP_STEP=10                  # Default
EPOCHS=60                       # Default
ALPHA=0.0                       # Finsler alpha value

# Loss configurations (ordered to match paper rows)
LOSS_CONFIGS=(
  "--no-align --no-uniform --no-domain --no-triplet --prob 0.0"   # L_ce
  "--no-align --no-uniform --no-domain --no-ce --prob 0.0"        # L_triplet
  "--no-ce --no-uniform --no-domain --no-triplet"                 # L_align
)
LOSS_NAMES=(                                                         
  "L_ce"
  "L_triplet"
  "L_align"
)

# Number of repeated runs per loss config for mean/variance estimation
RUNS_PER_CONFIG=5

# Determine indices for this task
LOSS_IDX=$((SLURM_ARRAY_TASK_ID / RUNS_PER_CONFIG))
RUN_IDX=$((SLURM_ARRAY_TASK_ID % RUNS_PER_CONFIG + 1))

SELECTED_LOSS_CONFIG=${LOSS_CONFIGS[$LOSS_IDX]}
SELECTED_LOSS_NAME=${LOSS_NAMES[$LOSS_IDX]}

LOGS_DIR="$(pwd)/logs/ablationFinsler-P2/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SELECTED_LOSS_NAME}_run${RUN_IDX}"
WANDB_NAME="ablation_${SELECTED_LOSS_NAME}_run${RUN_IDX}_alpha${ALPHA}"

echo "Starting ablation task ${SLURM_ARRAY_TASK_ID}"
echo "  Loss config: ${SELECTED_LOSS_NAME} -> ${SELECTED_LOSS_CONFIG}"
echo "  Run index: ${RUN_IDX}/${RUNS_PER_CONFIG}"
echo "  Logs dir: ${LOGS_DIR}"

mkdir -p "${LOGS_DIR}"
read -r -a LOSS_ARGS <<< "${SELECTED_LOSS_CONFIG}"

srun --unbuffered python examples/train_bau.py \
  -a "${ARCH}" \
  -ds market1501 msmt17 cuhksysu \
  -dt cuhk03 \
  -b "${BATCH_SIZE}" \
  --lam "${LAM}" --k "${KNN}" --iters "${ITERS}" \
  --epochs "${EPOCHS}" \
  --eval-epochs 5 10 15 20 25 30 35 40 45 50 55 59 \
  --warmup-step "${WARMUP_STEP}" \
  --logs-dir "${LOGS_DIR}" \
  --wandb-name "${WANDB_NAME}" \
  --verbosity 2 \
  --alpha "${ALPHA}" \
  "${LOSS_ARGS[@]}"

