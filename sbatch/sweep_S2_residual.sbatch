#!/bin/bash
#SBATCH --job-name=S2-residual
#SBATCH --array=0-3
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=28G
#SBATCH --time=08:00:00
#SBATCH --exclude=node7,node9,node1,node13
#SBATCH --output=logs/slurm_logs/S2-residual-%A_%a.out
#SBATCH --error=logs/slurm_logs/S2-residual-%A_%a.err

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Stage 2: Instance Relaxation Sweep                             ║
# ║                                                                  ║
# ║  Sweeps domain_residual_scale with the optimal routing params   ║
# ║  from Stage 1. Tests whether per-instance drift correction      ║
# ║  improves over the pure domain prior.                           ║
# ║                                                                  ║
# ║  4 configs: residual_scale in {0.00, 0.05, 0.10, 0.20}        ║
# ║                                                                  ║
# ║  IMPORTANT: Before submitting, fill in BEST_DTW and BEST_TAU   ║
# ║  from Stage 1 results.                                          ║
# ║                                                                  ║
# ║  See changelogs/sweep_domain_drift.md for full rationale.       ║
# ╚══════════════════════════════════════════════════════════════════╝

set -euo pipefail
ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p "${ROOT}/logs/slurm_logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU
cd "${ROOT}"

# ── Fill in from Stage 1 results ───────────────────────────────────
# Replace these with the optimal values from Stage 1 before submitting.
BEST_DTW=0.10       # <-- UPDATE from Stage 1 best domain_token_loss_weight
BEST_TAU=2.0        # <-- UPDATE from Stage 1 best domain_temperature

# ── Sweep grid ──────────────────────────────────────────────────────
RESIDUAL_SCALES=(0.00 0.05 0.10 0.20)
NAMES=(
  "res0.00_replication"
  "res0.05"
  "res0.10"
  "res0.20"
)

IDX=${SLURM_ARRAY_TASK_ID}
RES_SCALE=${RESIDUAL_SCALES[$IDX]}
RUN_NAME=${NAMES[$IDX]}

# ── Fixed hyperparameters (from current best model) ────────────────
ARCH=resnet50_finsler
BATCH_SIZE=256
LAM=1.5
KNN=10
ITERS=500
EPOCHS=60
OMEGA_REG_WEIGHT=1.5
DRIFT_METHOD=symmetric_trapezoidal
SOURCE_DATASET="market1501 msmt17 cuhksysu"
TARGET_DATASET=cuhk03
DRT_CONDITIONING=domain
SAMPLER="RandomMultipleGallery"
DOMAIN_EMBED_DIM=64

# ── Logging ────────────────────────────────────────────────────────
WANDB_TAGS=(sweep-S2 residual domain-drift "res-${RES_SCALE}" "dtw-${BEST_DTW}" "tau-${BEST_TAU}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
LOGS_DIR="${ROOT}/logs/sweep_S2_residual/job_${SLURM_ARRAY_JOB_ID}_${RUN_NAME}"
WANDB_NAME="S2_${RUN_NAME}_dtw${BEST_DTW}_tau${BEST_TAU}"

echo "=== Stage 2 Instance Relaxation Sweep: ${RUN_NAME} ==="
echo "  domain_token_loss_weight = ${BEST_DTW}  (from Stage 1)"
echo "  domain_temperature       = ${BEST_TAU}  (from Stage 1)"
echo "  domain_residual_scale    = ${RES_SCALE}"
echo "  Logs dir: ${LOGS_DIR}"

mkdir -p "${LOGS_DIR}"

# ── Code snapshot ──────────────────────────────────────────────────
SNAP_DIR="${LOGS_DIR}/code_snapshot"
mkdir -p "${SNAP_DIR}"
cp -r bau examples "${SNAP_DIR}/"
echo "Snapshotted code to: ${SNAP_DIR}"
export PYTHONPATH="${SNAP_DIR}"

# ── Launch training ────────────────────────────────────────────────
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
  --drift-method "${DRIFT_METHOD}" \
  --use-omega-reg \
  --omega-reg-weight "${OMEGA_REG_WEIGHT}" \
  --drift-conditioning "${DRT_CONDITIONING}" \
  --domain-embed-dim "${DOMAIN_EMBED_DIM}" \
  --domain-temperature "${BEST_TAU}" \
  --domain-residual-scale "${RES_SCALE}" \
  --domain-token-loss-weight "${BEST_DTW}" \
  --sampler "${SAMPLER}" \
  --wandb-tags "${WANDB_TAGS[@]}" \
  --bidirectional-triplet
