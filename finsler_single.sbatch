#!/bin/bash
#SBATCH --job-name=bau-finsler                # Job name
#SBATCH --gres=gpu:1,VRAM:25G                 # Request one 25GB-class GPU (reduce OOM risk)
#SBATCH --nodes=1                             # Run all processes on a single node
#SBATCH --cpus-per-task=5                     # Number of CPU cores per task
#SBATCH --mem=24G                             # Total memory for the entire job
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
ARCH=resnet50                   # Default
BATCH_SIZE=256                  # Default
LAM=1.5                         # Default
KNN=10                          # Default
ITERS=500                       # Default
EPOCHS=60                       # Increased from 20 to 60 for full training
# WARMUP_STEP=10                   # Proportionally scale warmup steps to 60 epochs (originally 10 for 60 epochs)
# MILESTONES="10 15"              # Proportionally scale milestones to 60 epochs (originally 30 and 50 for 60 epochs)
SOURCE_DATASET="msmt17 cuhk03 cuhksysu"
TARGET_DATASET=market1501

# Run Specific
RUN_NAME="EucBaseline"

# # Finsler alpha learnable configurations
# ALPHA_INIT=0.0                 # Learnable alpha initial value
# ALPHA_MAX=1.0                  # Learnable alpha max value
# ALPHA_TEMP=1.0                 # Learnable alpha temperature

LOGS_DIR="$(pwd)/logs/AGReIDv2_sweep/job_${SLURM_JOB_ID}_dt_${TARGET_DATASET}_${RUN_NAME}"
WANDB_NAME="AGReIDv2_learnableOmega_dt_${TARGET_DATASET}_${RUN_NAME}"
echo "Starting finsler single experimentation"
echo "Logs dir: ${LOGS_DIR}"e

mkdir -p "${LOGS_DIR}"

# --- Snapshot code to isolate job from git branch changes ---
SNAP_DIR="${LOGS_DIR}/code_snapshot"
mkdir -p "${SNAP_DIR}"
cp -r bau examples "${SNAP_DIR}/"
echo "Snapshotted code to: ${SNAP_DIR}"
# -----------------------------------------------------------

# Point PYTHONPATH to the snapshot instead of the repo root
export PYTHONPATH="${SNAP_DIR}"

  # -ds market1501 msmt17 cuhksysu \
  # -dt cuhk03 \

srun --unbuffered python "${SNAP_DIR}/examples/train_bau.py" \
  -a "${ARCH}" \
  -ds ${SOURCE_DATASET} \
  -dt "${TARGET_DATASET}" \
  -b "${BATCH_SIZE}" \
  --lam "${LAM}" --k "${KNN}" --iters "${ITERS}" \
  --epochs "${EPOCHS}" \
  --eval-epochs 5 10 15 20 25 30 35 40 45 50 55 59 \
  --logs-dir "${LOGS_DIR}" \
  --wandb-name "${WANDB_NAME}" \
  --verbosity 2 
  # --warmup-step "${WARMUP_STEP}" \
  # --alpha-init "${ALPHA_INIT}" \
  # --alpha-max "${ALPHA_MAX}" \
  # --alpha-temp "${ALPHA_TEMP}"
  # -ds agreidv2_aerial agreidv2_wearable agreidv2_cctv \
  # -dt agreidv2_aerial_to_cctv \
