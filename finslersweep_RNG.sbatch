#!/bin/bash
#SBATCH --job-name=bau-finsler-RNG            # Job name
#SBATCH --array=0-9                           # Two loss configs * 5 runs each
#SBATCH --gres=gpu:1,VRAM:28G                 # Request one 48GB-class GPU (reduce OOM risk)
#SBATCH --nodes=1                             # Run all processes on a single node
#SBATCH --cpus-per-task=5                     # Number of CPU cores per task
#SBATCH --mem=24G                             # Total memory for the entire job
#SBATCH --time=12:00:00                       # Time limit hrs:min:sec per task
#SBATCH --exclude=node7,node9,node1,node13    # Exclude problematic nodes
# #SBATCH --constraint="GPU_MODEL:a40"          # Ensure specific GPU model if available
#SBATCH --output=logs/finslerRNG-%A_%a.out       # Standard output log per array task
#SBATCH --error=logs/finslerRNG-%A_%a.err        # Error log per array task

set -euo pipefail
ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p ${ROOT}/logs/finsler_duet_rng

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

# Finsler alpha values (sweep narrowed to three)
# ALPHAS=None
# RUNS_PER_ALPHA=5

# ALPHA_INDEX=$((SLURM_ARRAY_TASK_ID / RUNS_PER_ALPHA))
# RUN_INDEX=$((SLURM_ARRAY_TASK_ID % RUNS_PER_ALPHA + 1))
# ALPHA=${ALPHAS[ALPHA_INDEX]}

# Finsler parameters (learnable alpha as in ablation)
ALPHA_INIT=0.1
ALPHA_MAX=1.0
ALPHA_TEMP=1.0

# Loss configs 9 and 11 from ablation script
LOSS_CONFIGS=(
  "--no-uniform"
  ""
)
LOSS_NAMES=(
  "9_Base_Align_Dom"
  "11_Full_Method"
)
RUNS_PER_CONFIG=5

LOSS_INDEX=$((SLURM_ARRAY_TASK_ID / RUNS_PER_CONFIG))
RUN_INDEX=$((SLURM_ARRAY_TASK_ID % RUNS_PER_CONFIG + 1))
SELECTED_LOSS_CONFIG=${LOSS_CONFIGS[$LOSS_INDEX]}
SELECTED_LOSS_NAME=${LOSS_NAMES[$LOSS_INDEX]}

RUN_TAG="${SELECTED_LOSS_NAME}_run${RUN_INDEX}"
LOGS_DIR="${ROOT}/logs/finsler_duet_rng/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${RUN_TAG}"
WANDB_NAME="finsler_duet_rng_${RUN_TAG}"

echo "Starting finsler task ${SLURM_ARRAY_TASK_ID} | Loss ${SELECTED_LOSS_NAME} | Run ${RUN_INDEX}"
echo "Logs dir: ${LOGS_DIR}"

mkdir -p "${LOGS_DIR}"

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
  ${SELECTED_LOSS_CONFIG}
