#!/bin/bash
#SBATCH --job-name=bau-ablation-finsler         # Job name
#SBATCH --array=0-10                             # 11 runs: 0-10
#SBATCH --gres=gpu:1,VRAM:28G                   # Request two 24GB-class GPUs 
#SBATCH --nodes=1                               # Run all processes on a single node
#SBATCH --cpus-per-task=5                       # Number of CPU cores per task
#SBATCH --mem=24G                               # Total memory for the entire job
#SBATCH --time=12:00:00                         # Time limit hrs:min:sec
#SBATCH --exclude=node7,node9,node1,node13    # Exclude problematic nodes
# #SBATCH --constraint="GPU_MODEL:a40"          # Ensure A40-class GPU
#SBATCH --output=logs/ablationFinsler-P2-learnableAlpha-%A_%a.out     # Standard output log
#SBATCH --error=logs/ablationFinsler-P2-learnableAlpha-%A_%a.err      # Error log

set -euo pipefail
ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p ${ROOT}/logs/ablationFinsler-P2-learnableAlpha

# Activate the environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU

# Ensure we run from the repo root and import the local bau package
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH}"

# Base hyper-parameters (shared across jobs)
ARCH=resnet50                   # Default
BATCH_SIZE=256                  # Default
LAM=1.5                         # Default
KNN=10                          # Default
ITERS=500                       # Default
WARMUP_STEP=10                  # Default
EPOCHS=60                       # Default

# Finsler parameters
ALPHA_INIT=0.1
ALPHA_MAX=1.0
ALPHA_TEMP=1.0

# Loss configurations (Hierarchy: Baseline -> Single -> Dual -> Triple -> Full)
LOSS_CONFIGS=(
  "--no-align --no-uniform --no-domain --prob 0.0"
  "--no-align --no-uniform --no-domain --no-ce"
  "--no-align --no-uniform --no-domain"
  "--no-triplet --no-uniform --no-domain"
  "--no-uniform --no-domain"
  "--no-align --no-domain"
  "--no-align --no-uniform"
  "--no-domain"
  "--no-uniform"
  "--no-align"
  ""
)
LOSS_NAMES=(
  "1_Abs_Baseline"
  "2_Pure_Triplet"
  "3_Std_Baseline"
  "4_CE_Align"
  "5_Base_Align"
  "6_Base_Uniform"
  "7_Base_Domain"
  "8_Base_Align_Uni"
  "9_Base_Align_Dom"
  "10_Base_Uni_Dom"
  "11_Full_Method"
)

# Determine indices for this task
LOSS_IDX=$((SLURM_ARRAY_TASK_ID % ${#LOSS_CONFIGS[@]}))
SELECTED_LOSS_CONFIG=${LOSS_CONFIGS[$LOSS_IDX]}
SELECTED_LOSS_NAME=${LOSS_NAMES[$LOSS_IDX]}

LOGS_DIR="${ROOT}/logs/ablationFinsler-P2-learnableAlpha/job_alphainit0.1_${SELECTED_LOSS_NAME}_cuhk03"
WANDB_NAME="ablationFinsler_alphainit0.1_${SELECTED_LOSS_NAME}_cuhk03"

echo "Starting ablation task ${SLURM_ARRAY_TASK_ID}"
echo "  Loss Name: ${SELECTED_LOSS_NAME}"
echo "  Loss Config: ${SELECTED_LOSS_CONFIG}"
echo "  Alpha Init: ${ALPHA_INIT}"

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
  --alpha-init "${ALPHA_INIT}" \
  --alpha-max "${ALPHA_MAX}" \
  --alpha-temp "${ALPHA_TEMP}" \
  "${LOSS_ARGS[@]}"

