#!/bin/bash
#SBATCH --job-name=S1-routing-finsler
#SBATCH --array=0-11
#SBATCH --gres=gpu:1,VRAM:28G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=28G
#SBATCH --time=10:00:00
#SBATCH --exclude=node7,node9,node1,node13
#SBATCH --output=logs/slurm_logs/S1-routing-%A_%a.out
#SBATCH --error=logs/slurm_logs/S1-routing-%A_%a.err

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Stage 1: Domain Routing Quality Sweep                           ║
# ║                                                                  ║
# ║  Sweeps domain_token_loss_weight x domain_temperature with       ║
# ║  residual_scale=0 to isolate the routing mechanism.              ║
# ║                                                                  ║
# ║  12 configs (4 loss weights x 3 temperatures):                   ║
# ║    loss_weight  in {0.01, 0.05, 0.10, 0.25}                      ║
# ║    temperature  in {1.0,  2.0,  5.0}                             ║
# ║                                                                  ║
# ║  See changelogs/sweep_domain_drift.md for full rationale.        ║ 
# ╚══════════════════════════════════════════════════════════════════╝

set -euo pipefail
ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p "${ROOT}/logs/slurm_logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU
cd "${ROOT}"

# ── Sweep grid ──────────────────────────────────────────────────────
DOM_TOKEN_WEIGHTS=(0.01 0.01 0.01  0.05 0.05 0.05  0.10 0.10 0.10  0.25 0.25 0.25)
DOM_TEMPERATURES=( 1.0  2.0  5.0   1.0  2.0  5.0   1.0  2.0  5.0   1.0  2.0  5.0)
NAMES=(
  "dtw0.01_tau1.0"
  "dtw0.01_tau2.0"
  "dtw0.01_tau5.0"
  "dtw0.05_tau1.0"
  "dtw0.05_tau2.0"
  "dtw0.05_tau5.0"
  "dtw0.10_tau1.0"
  "dtw0.10_tau2.0"
  "dtw0.10_tau5.0"
  "dtw0.25_tau1.0"
  "dtw0.25_tau2.0"
  "dtw0.25_tau5.0"
)

IDX=${SLURM_ARRAY_TASK_ID}
DTW=${DOM_TOKEN_WEIGHTS[$IDX]}
TAU=${DOM_TEMPERATURES[$IDX]}
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
DOMAIN_RESIDUAL_SCALE=0.0

# ── Logging ────────────────────────────────────────────────────────
WANDB_TAGS=(sweep-S1 routing domain-drift "dtw-${DTW}" "tau-${TAU}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
LOGS_DIR="${ROOT}/logs/sweep_S1_routing/job_${SLURM_ARRAY_JOB_ID}_${RUN_NAME}"
WANDB_NAME="S1_${RUN_NAME}"

echo "=== Stage 1 Routing Sweep: ${RUN_NAME} ==="
echo "  domain_token_loss_weight = ${DTW}"
echo "  domain_temperature       = ${TAU}"
echo "  domain_residual_scale    = ${DOMAIN_RESIDUAL_SCALE}"
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
  --domain-temperature "${TAU}" \
  --domain-residual-scale "${DOMAIN_RESIDUAL_SCALE}" \
  --domain-token-loss-weight "${DTW}" \
  --sampler "${SAMPLER}" \
  --wandb-tags "${WANDB_TAGS[@]}" \
  --bidirectional-triplet
