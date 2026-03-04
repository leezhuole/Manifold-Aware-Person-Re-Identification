#!/bin/bash
#SBATCH --job-name=bau-ablation-euc             # Job name
#SBATCH --array=0-4                             # 5 runs: 5 loss configs (only for cuhk03 target)
#SBATCH --gres=gpu:1,VRAM:28G                   # Request two 24GB-class GPUs 
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --cpus-per-task=5                       # Number of CPU cores per task
#SBATCH --mem=24G                               # Total memory for the entire job
#SBATCH --time=24:00:00                         # Time limit hrs:min:sec
#SBATCH --exclude=node7,node9,node1             # Exclude problematic nodes
# #SBATCH --constraint="GPU_MODEL:a40"          # Ensure A40-class GPU
#SBATCH --output=logs/ablationEuc-%A_%a.out     # Standard output log
#SBATCH --error=logs/ablationEuc-%A_%a.err      # Error log

set -euo pipefail
ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p ${ROOT}/logs/ablationEuclidean-P3

# Activate the environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU

# Ensure we run from the repo root and import the local bau package
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH}"

# Base hyper-parameters (shared across jobs)
ARCH=resnet50                   # Default
BATCH_SIZE=256                  # Default
CURVATURE=0.0                   # Default
LAM=1.5                         # Default
KNN=10                          # Default
ITERS=1000                      # Default
MANIFOLD_AWARE=False            # Euclidean

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

# Determine indices for this task
LOSS_IDX=$((SLURM_ARRAY_TASK_ID % ${#LOSS_CONFIGS[@]}))
SELECTED_LOSS_CONFIG=${LOSS_CONFIGS[$LOSS_IDX]}
SELECTED_LOSS_NAME=${LOSS_NAMES[$LOSS_IDX]}

LOGS_DIR="${ROOT}/logs/ablationEuclidean-P3/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
WANDB_NAME="ablationEuc-P3_m-ms-cs_c3_${SELECTED_LOSS_NAME}"

echo "Starting ablation task ${SLURM_ARRAY_TASK_ID}"
echo "  Dataset config: market1501 msmt17 cuhksysu -> cuhk03"
echo "  Loss config: ${SELECTED_LOSS_NAME} -> ${SELECTED_LOSS_CONFIG}"
echo "  Logs dir: ${LOGS_DIR}"

mkdir -p "${LOGS_DIR}"

read -r -a LOSS_ARGS <<< "${SELECTED_LOSS_CONFIG}"

srun --unbuffered python examples/train_bau.py \
  -a "${ARCH}" \
  -ds market1501dg msmt17dg cuhksysu -dt cuhk03 \
  -b "${BATCH_SIZE}" \
  --lam "${LAM}" --k "${KNN}" --iters "${ITERS}" \
  --eval-epochs 5 10 15 20 25 30 35 40 45 50 55 59 \
  --manifold-aware "${MANIFOLD_AWARE}" \
  --curvature "${CURVATURE}" \
  --logs-dir "${LOGS_DIR}" \
  --wandb-name "${WANDB_NAME}" \
  --verbosity 1 \
  "${LOSS_ARGS[@]}"

