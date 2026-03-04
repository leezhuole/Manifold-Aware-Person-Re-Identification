#!/bin/bash
#SBATCH --job-name=bau-finsler-sweep          # Job name
#SBATCH --array=0-7                           # Sweep over baseline/learnableAlpha/learnableOmega configs
#SBATCH --gres=gpu:1,VRAM:25G                 # Request one 25GB-class GPU (reduce OOM risk)
#SBATCH --nodes=1                             # Run all processes on a single node
#SBATCH --cpus-per-task=5                     # Number of CPU cores per task
#SBATCH --mem=12G                             # Total memory for the entire job
#SBATCH --time=15:00:00                       # Time limit hrs:min:sec per task
#SBATCH --exclude=node7,node9,node1,node13    # Exclude problematic nodes
# #SBATCH --constraint="GPU_MODEL:a40"          # Ensure specific GPU model if available
#SBATCH --output=logs/AGReIDv2_sweep/finsler-%j.out          # Standard output log
#SBATCH --error=logs/AGReIDv2_sweep/finsler-%j.err           # Error log

set -euo pipefail

# Activate the environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU

# Ensure local repo is on Python path for srun processes
export PYTHONPATH=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification

# Base hyper-parameters (shared across jobs)
ARCH=resnet50_finsler           # Changed to learnableOmega model
BATCH_SIZE=256                  # Default
LAM=1.5                         # Default
KNN=10                          # Default
ITERS=500                       # Default
EPOCHS=20                       # Reduced from 60 to 20 for faster iteration and to fit within time limits
WARMUP_STEP=3                   # Proportionally scale warmup steps to 20 epochs (originally 10 for 60 epochs)
MILESTONES="10 15"              # Proportionally scale milestones to 20 epochs (originally 30 and 50 for 60 epochs)

# Sweep configurations
CONFIG_MODE=(
  learnableOmega
  learnableOmega
  learnableOmega
  learnableOmega
  learnableOmega
  learnableOmega
  learnableOmega
  learnableOmega
)

CONFIG_DRIFT_DIM=(
  16
  32
  64
  128
  256
  512
  1024
  2048
)

MODE=${CONFIG_MODE[SLURM_ARRAY_TASK_ID]}
DRIFT_DIM=${CONFIG_DRIFT_DIM[SLURM_ARRAY_TASK_ID]}
# Sweep specific
RUN_NAME="FHSigGate+ModFinsDist+BiDirectTri"

LOGS_DIR="$(pwd)/logs/AGReIDv2_sweep/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${MODE}_dt_aerial_to_cctv_${RUN_NAME}_drift${DRIFT_DIM}"
WANDB_NAME="AGReIDv2_${MODE}_dt_aerial_to_cctv_${RUN_NAME}_drift${DRIFT_DIM}"
echo "Starting sweep | Mode ${MODE} | Drift Dim ${DRIFT_DIM}"
echo "Logs dir: ${LOGS_DIR}"

mkdir -p "${LOGS_DIR}"

# --- Snapshot code to isolate job from git branch changes ---
SNAP_DIR="${LOGS_DIR}/code_snapshot"
mkdir -p "${SNAP_DIR}"
cp -r bau examples "${SNAP_DIR}/"
echo "Snapshotted code to: ${SNAP_DIR}"
# -----------------------------------------------------------

EXTRA_ARGS=()
if [[ "${ARCH}" == "resnet50_finsler" ]]; then
  EXTRA_ARGS+=(--eval-drift false --memory-bank-mode full --drift-dim "${DRIFT_DIM}" --bidirectional-triplet)
fi
if [[ "${MODE}" == "baseline" ]]; then
  EXTRA_ARGS+=()
fi
# Point PYTHONPATH to the snapshot instead of the repo root
export PYTHONPATH="${SNAP_DIR}"

srun --unbuffered python "${SNAP_DIR}/examples/train_bau.py" \
  -a "${ARCH}" \
  -ds agreidv2_aerial agreidv2_wearable agreidv2_cctv \
  -dt agreidv2_aerial_to_cctv \
  -b "${BATCH_SIZE}" \
  --lam "${LAM}" --k "${KNN}" --iters "${ITERS}" \
  --epochs "${EPOCHS}" \
  --eval-epochs 5 10 15 20 25 30 35 40 45 50 55 59 \
  --warmup-step "${WARMUP_STEP}" \
  --milestones ${MILESTONES} \
  --logs-dir "${LOGS_DIR}" \
  --wandb-name "${WANDB_NAME}" \
  --verbosity 2 \
  "${EXTRA_ARGS[@]}"

