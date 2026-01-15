#!/bin/bash
#SBATCH --job-name=bau-ablation-hyp             # Job name
#SBATCH --array=0-13                            # 14 runs: 1 dataset target × 14 loss configs
#SBATCH --gres=gpu:2,VRAM:24G                   # Request two 24GB-class GPUs 
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --cpus-per-task=10                      # Number of CPU cores per task
#SBATCH --mem=32G                               # Total memory for the entire job
#SBATCH --time=24:00:00                         # Time limit hrs:min:sec
#SBATCH --exclude=node7,node9,node1             # Exclude problematic nodes
# #SBATCH --constraint="GPU_MODEL:a40"          # Ensure A40-class GPU
#SBATCH --output=logs/ablationHyp-%A_%a.out     # Standard output log
#SBATCH --error=logs/ablationHyp-%A_%a.err      # Error log

set -euo pipefail
ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p ${ROOT}/logs/ablationHyperbolic-P3

# Activate the environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU

# Ensure we run from the repo root and import the local bau package
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH}"

# Base hyper-parameters (shared across jobs)
ARCH=resnet50                   # Default
BATCH_SIZE=256                  # Default
CURVATURE=1.0                   # Default
LAM=1.5                         # Default
KNN=10                          # Default
ITERS=1000                      # Default
CHUNK_SIZE=500                  # Chunk to be safe
MANIFOLD_AWARE=True             # Poincare Ball

# Loss configurations (ordered to match paper rows)
LOSS_CONFIGS=(
  
  # --- ID 1-5: Single-loss settings ---
  "--no-align --no-uniform --no-domain --no-triplet --prob 0.0"   # L_ce
  "--no-align --no-uniform --no-domain --no-ce --prob 0.0"        # L_triplet
  "--no-ce --no-uniform --no-domain --no-triplet"                 # L_align
  "--no-align --no-ce --no-domain --no-triplet"                   # L_uniform
  "--no-align --no-uniform --no-ce --no-triplet"                  # L_domain  
  
  # --- ID 6: Baseline w/o augmentations (CE + Triplet) ---
  "--no-align --no-uniform --no-domain --prob 0.0"    # Baseline (CE + Triplet, no augmentations)

  # --- 3. Dual Losses (excluding Baseline) ---
  # "--no-align --no-uniform --no-domain"               # L_ce + L_triplet
  # "--no-uniform --no-domain --no-triplet"             # L_ce + L_align
  # "--no-align --no-domain --no-triplet"               # L_ce + L_uniform
  # "--no-align --no-uniform --no-triplet"              # L_ce + L_domain
  # "--no-ce --no-uniform --no-domain"                  # L_triplet + L_align
  # "--no-ce --no-align --no-domain"                    # L_triplet + L_uniform
  # "--no-ce --no-align --no-uniform"                   # L_triplet + L_domain
  # "--no-ce --no-triplet --no-domain"                  # L_align + L_uniform
  # "--no-ce --no-triplet --no-uniform"                 # L_align + L_domain
  # "--no-ce --no-triplet --no-align"                   # L_uniform + L_domain

  # --- ID 7-10. Triple Losses ---
  "--no-uniform --no-domain --no-align --use-aug-ce"    # L_ce + L_triplet + L_ce (aug)
  "--no-uniform --no-domain"                            # L_ce + L_triplet + L_align
  "--no-align --no-domain"                              # L_ce + L_triplet + L_uniform
  "--no-align --no-uniform"                             # L_ce + L_triplet + L_domain
  # "--no-triplet --no-domain"                          # L_ce + L_align + L_uniform
  # "--no-triplet --no-uniform"                         # L_ce + L_align + L_domain
  # "--no-triplet --no-align"                           # L_ce + L_uniform + L_domain
  # "--no-ce --no-domain"                               # L_triplet + L_align + L_uniform
  # "--no-ce --no-uniform"                              # L_triplet + L_align + L_domain
  # "--no-ce --no-align"                                # L_triplet + L_uniform + L_domain
  # "--no-ce --no-triplet"                              # L_align + L_uniform + L_domain

  # --- ID 11-13: Quadruple Losses ---
  "--no-domain"                                         # All except Domain
  "--no-uniform"                                        # All except Uniform
  "--no-align"                                          # All except Align
  # "--no-triplet"                                      # All except Triplet
  # "--no-ce"                                           # All except CE

  # --- ID 14: All Losses ---
  ""                                                  # All Losses
)
LOSS_NAMES=(
  "L_ce"
  "L_triplet"
  "L_align"
  "L_uniform"
  "L_domain"
  "Baseline"
  # "L_ce_triplet"
  # "L_ce_align"
  # "L_ce_uniform"
  # "L_ce_domain"
  # "L_triplet_align"
  # "L_triplet_uniform"
  # "L_triplet_domain"
  # "L_align_uniform"
  # "L_align_domain"
  # "L_uniform_domain"
  "L_ce_triplet_ceaug"
  "L_ce_triplet_align"
  "L_ce_triplet_uniform"
  "L_ce_triplet_domain"
  # "L_ce_align_uniform"
  # "L_ce_align_domain"
  # "L_ce_uniform_domain"
  # "L_triplet_align_uniform"
  # "L_triplet_align_domain"
  # "L_triplet_uniform_domain"
  # "L_align_uniform_domain"
  "L_ce_triplet_align_uniform"
  "L_ce_triplet_align_domain"
  "L_ce_triplet_uniform_domain"
  # "L_ce_align_uniform_domain"
  # "L_triplet_align_uniform_domain"
  "L_all"
)

# Basic sanity checks to prevent out-of-range indices or mismatched metadata
CONFIG_COUNT=${#LOSS_CONFIGS[@]}
NAME_COUNT=${#LOSS_NAMES[@]}
if [[ ${CONFIG_COUNT} -ne ${NAME_COUNT} ]]; then
  echo "Mismatch between LOSS_CONFIGS (${CONFIG_COUNT}) and LOSS_NAMES (${NAME_COUNT})." >&2
  exit 1
fi
if (( SLURM_ARRAY_TASK_ID >= CONFIG_COUNT )); then
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is out of range (0..$((CONFIG_COUNT-1)))." >&2
  exit 1
fi

# Determine indices for this task
LOSS_IDX=$((SLURM_ARRAY_TASK_ID % ${#LOSS_CONFIGS[@]}))
SELECTED_LOSS_CONFIG=${LOSS_CONFIGS[$LOSS_IDX]}
SELECTED_LOSS_NAME=${LOSS_NAMES[$LOSS_IDX]}

LOGS_DIR="${ROOT}/logs/ablationHyperbolic-P3/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SELECTED_LOSS_NAME}"
WANDB_NAME="ablationHyp-P3_m-ms-cs_c3_${SELECTED_LOSS_NAME}"

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
  --manifold-chunk-size ${CHUNK_SIZE} \
  --wandb-name "${WANDB_NAME}" \
  --verbosity 1 \
  "${LOSS_ARGS[@]}"

