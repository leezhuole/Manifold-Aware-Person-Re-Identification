#!/bin/bash
#SBATCH --job-name=bau-finsler                # Job name
#SBATCH --gres=gpu:1,VRAM:36G                 # Request one 36GB-class GPU (reduce OOM risk)
#SBATCH --nodes=1                             # Run all processes on a single node
#SBATCH --cpus-per-task=5                     # Number of CPU cores per task
#SBATCH --mem=36G                             # Total memory for the entire job
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
EPOCHS=60                       # Reduced from 60 to 20 for faster iteration and to fit within time limits
# WARMUP_STEP=3                   # Proportionally scale warmup steps to 20 epochs (originally 10 for 60 epochs)
# MILESTONES="10 15"              # Proportionally scale milestones to 20 epochs (originally 30 and 50 for 60 epochs)
OMEGA_REG_WEIGHT=1.5            # New hyper-parameter for omega regularization weight
DRIFT_METHOD=analytical    # Method for calculating the finsler distance [constant_drift, symmetric_trapezoidal, slerp, analytical]
# SOURCE_DATASET="market1501 cuhksysu cuhk03"
# TARGET_DATASET=msmt17
SOURCE_DATASET="market1501 msmt17 cuhksysu"
TARGET_DATASET=cuhk03

# Run Specific
RUN_NAME="FHSigGate_Drift${DRIFT_METHOD}_BidirectTri__OmegaReg${OMEGA_REG_WEIGHT}_OrthoEnforced"
# RUN_NAME="FHSigGate+ModFinsDist_BidirectTri_OmegaReg${OMEGA_REG_WEIGHT}"

LOGS_DIR="$(pwd)/logs/AGReIDv2_sweep/job_${SLURM_JOB_ID}_dt_${TARGET_DATASET}_${RUN_NAME}"
WANDB_NAME="learnableOmega_dt_${TARGET_DATASET}_${RUN_NAME}"
echo "Starting finsler single experimentation"
echo "Logs dir: ${LOGS_DIR}"

mkdir -p "${LOGS_DIR}"

# --- Snapshot code to isolate job from git branch changes ---
SNAP_DIR="${LOGS_DIR}/code_snapshot"
mkdir -p "${SNAP_DIR}"
cp -r bau examples "${SNAP_DIR}/"
echo "Snapshotted code to: ${SNAP_DIR}"
# -----------------------------------------------------------

# Point PYTHONPATH to the snapshot instead of the repo root
export PYTHONPATH="${SNAP_DIR}"

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
  --verbosity 2 \
  --eval-drift false \
  --memory-bank-mode full \
  --bidirectional-triplet \
  --drift-method "${DRIFT_METHOD}" \
  --use-omega-reg \
  --omega-reg-weight "${OMEGA_REG_WEIGHT}" \
  # --use-drift-align \
  # --drift-dim 32 \


# -- CLI INPUT GRAVEYARD --
  # --warmup-step "${WARMUP_STEP}" \
  # --milestones ${MILESTONES} \
  # -ds market1501 msmt17 cuhksysu \
  # -dt cuhk03 \

  # -ds agreidv2_aerial agreidv2_wearable agreidv2_cctv \
  # -dt agreidv2_aerial_to_cctv \