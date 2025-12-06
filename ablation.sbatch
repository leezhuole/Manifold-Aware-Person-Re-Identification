#!/bin/bash
#SBATCH --job-name=bau-ablation               # Job name
#SBATCH --array=0-14                          # 15 runs: 3 dataset targets × 5 loss configs
#SBATCH --gres=gpu:1,VRAM:24G                 # Request one 24GB-class GPU (A40)
#SBATCH --nodes=1                             # Run all processes on a single node
#SBATCH --cpus-per-task=5                     # Number of CPU cores per task
#SBATCH --mem=32G                             # Total memory for the entire job
#SBATCH --time=24:00:00                       # Time limit hrs:min:sec
# #SBATCH --constraint="GPU_MODEL:a40"          # Ensure A40-class GPU
#SBATCH --output=logs/ablation-%A_%a.out      # Standard output log
#SBATCH --error=logs/ablation-%A_%a.err       # Error log

set -euo pipefail
mkdir -p logs/ablation

# Activate the environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU

# Base hyper-parameters (shared across jobs)
ARCH=resnet50                   # Default
BATCH_SIZE=256                  # Default
CURVATURE=1.0                   # Default
LAM=1.5                         # Default
KNN=10                          # Default
ITERS=500                       # Default
WARMUP_STEP=10                  # Default
EPOCHS=60                       # Default
CHUNK_SIZE=500                  # Chunk to be safe
MANIFOLD_AWARE=False            # Set to false first to reproduce the paper's ablation results

# Loss configurations (ordered to match paper rows)
LOSS_CONFIGS=(
  "--no-align --no-uniform --no-domain --prob 0.0"                    # Baseline
  "--no-align --no-uniform --no-domain --use-aug-ce"                  # L_ce
  "--no-uniform --no-domain"                                          # L_align
  "--no-domain"                                                       # L_align + L_uniform
  ""                                                                  # L_align + L_uniform + L_domain
)
LOSS_NAMES=(
  "Baseline"
  "L_ce"
  "L_align"
  "L_align_uniform"
  "L_align_uniform_domain"
)

# Dataset transfer configurations (match table columns)
DATASET_CONFIGS=(
  "-ds market1501 msmt17 cuhksysu -dt cuhk03"
  "-ds market1501 cuhksysu cuhk03 -dt msmt17"
  "-ds msmt17 cuhksysu cuhk03 -dt market1501"
)
DATASET_NAMES=(
  "MMSCS_to_C3"
  "MCSCS_to_MS"
  "MSCS_to_M"
)

# Determine indices for this task
LOSS_IDX=$((SLURM_ARRAY_TASK_ID % ${#LOSS_CONFIGS[@]}))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / ${#LOSS_CONFIGS[@]}))

SELECTED_LOSS_CONFIG=${LOSS_CONFIGS[$LOSS_IDX]}
SELECTED_LOSS_NAME=${LOSS_NAMES[$LOSS_IDX]}
SELECTED_DATASET_CONFIG=${DATASET_CONFIGS[$DATASET_IDX]}
SELECTED_DATASET_NAME=${DATASET_NAMES[$DATASET_IDX]}

LOGS_DIR="$(pwd)/logs/ablation/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
WANDB_NAME="ablation_${SELECTED_DATASET_NAME}_${SELECTED_LOSS_NAME}"

echo "Starting ablation task ${SLURM_ARRAY_TASK_ID}"
echo "  Dataset config: ${SELECTED_DATASET_NAME} -> ${SELECTED_DATASET_CONFIG}"
echo "  Loss config: ${SELECTED_LOSS_NAME} -> ${SELECTED_LOSS_CONFIG}"
echo "  Logs dir: ${LOGS_DIR}"

mkdir -p "${LOGS_DIR}"

read -r -a LOSS_ARGS <<< "${SELECTED_LOSS_CONFIG}"

srun --unbuffered python examples/train_bau.py \
  -a "${ARCH}" \
  ${SELECTED_DATASET_CONFIG} \
  -b "${BATCH_SIZE}" \
  --lam "${LAM}" --k "${KNN}" --iters "${ITERS}" \
  --epochs "${EPOCHS}" \
  --eval-epochs 5 10 15 20 25 30 35 40 45 50 55 59 \
  --warmup-step "${WARMUP_STEP}" \
  --manifold-aware "${MANIFOLD_AWARE}" \
  --curvature "${CURVATURE}" \
  --logs-dir "${LOGS_DIR}" \
  --manifold-chunk-size ${CHUNK_SIZE} \
  --wandb-name "${WANDB_NAME}" \
  --verbosity 1 \
  "${LOSS_ARGS[@]}"

