#!/bin/bash
#SBATCH --job-name=S3-instnorm
#SBATCH --array=0-3
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=28G
#SBATCH --time=08:00:00
#SBATCH --exclude=node7,node9,node1,node13
#SBATCH --output=logs/slurm_logs/S3-instnorm-%A_%a.out
#SBATCH --error=logs/slurm_logs/S3-instnorm-%A_%a.err

# ╔══════════════════════════════════════════════════════════════════╗
# ║  Stage 3: Backbone InstanceNorm Ablation                        ║
# ║                                                                  ║
# ║  Tests whether the inherited shallow-layer InstanceNorm2d       ║
# ║  (from IBN-Net / Jia et al. BMVC 2019) is complementary or     ║
# ║  redundant with the domain-conditioned drift head.              ║
# ║                                                                  ║
# ║  4 configs: backbone_in_stages in {1,2,3  1,2  1  none}        ║
# ║                                                                  ║
# ║  IMPORTANT: Before submitting, fill in BEST_DTW, BEST_TAU,     ║
# ║  and BEST_RES from Stage 1 + Stage 2 results.                  ║
# ║                                                                  ║
# ║  See changelogs/sweep_domain_drift.md for full rationale.       ║
# ╚══════════════════════════════════════════════════════════════════╝

set -euo pipefail
ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p "${ROOT}/logs/slurm_logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU
cd "${ROOT}"

# ── Fill in from Stage 1 + Stage 2 results ───────────────────────
# Replace these with the optimal values before submitting.
BEST_DTW=0.10       # <-- UPDATE from Stage 1 best domain_token_loss_weight
BEST_TAU=2.0        # <-- UPDATE from Stage 1 best domain_temperature
BEST_RES=0.00       # <-- UPDATE from Stage 2 best domain_residual_scale

# ── Sweep grid ──────────────────────────────────────────────────────
IN_STAGES=("1,2,3" "1,2" "1" "none")
NAMES=(
  "in_123_replication"
  "in_12"
  "in_1"
  "in_none"
)

IDX=${SLURM_ARRAY_TASK_ID}
IN_STG=${IN_STAGES[$IDX]}
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
WANDB_TAGS=(sweep-S3 instancenorm domain-drift "in-${IN_STG}" "dtw-${BEST_DTW}" "tau-${BEST_TAU}" "res-${BEST_RES}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
LOGS_DIR="${ROOT}/logs/sweep_S3_instancenorm/job_${SLURM_ARRAY_JOB_ID}_${RUN_NAME}"
WANDB_NAME="S3_${RUN_NAME}_dtw${BEST_DTW}_tau${BEST_TAU}_res${BEST_RES}"

echo "=== Stage 3 InstanceNorm Ablation: ${RUN_NAME} ==="
echo "  backbone_in_stages       = ${IN_STG}"
echo "  domain_token_loss_weight = ${BEST_DTW}  (from Stage 1)"
echo "  domain_temperature       = ${BEST_TAU}  (from Stage 1)"
echo "  domain_residual_scale    = ${BEST_RES}  (from Stage 2)"
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
  --domain-residual-scale "${BEST_RES}" \
  --domain-token-loss-weight "${BEST_DTW}" \
  --backbone-in-stages "${IN_STG}" \
  --sampler "${SAMPLER}" \
  --wandb-tags "${WANDB_TAGS[@]}" \
  --bidirectional-triplet
