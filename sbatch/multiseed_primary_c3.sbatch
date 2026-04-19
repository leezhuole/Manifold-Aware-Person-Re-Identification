#!/bin/bash
#SBATCH --job-name=finsler-multiseed-c3
#SBATCH --array=0-4
#SBATCH --gres=gpu:a40:1           # Pin to A40 (Ampere CC 8.6, nodes 15-18). The primary
#SBATCH --nodes=1                  # best-result run was confirmed on node17 (A40) via sacct.
#SBATCH --cpus-per-task=5          # Without this, cudnn.benchmark selects different convolution
#SBATCH --mem=32G                  # algorithms on e.g. RTX A6000 (node14) or L40S (node20),
#SBATCH --time=15:00:00            # conflating hardware variance with seed variance (~1.4 pp).
#SBATCH --output=logs/slurm_logs/finsler-multiseed-c3-%A_%a.out
#SBATCH --error=logs/slurm_logs/finsler-multiseed-c3-%A_%a.err

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Multi-seed variance estimate — primary recipe, M+MS+CS → C3 (P2)        ║
# ║                                                                          ║
# ║  Identical to finsler_primary_recipe.sbatch across 5 seeds.              ║
# ║  Task 0 (seed=1) targets the primary result (44.9% mAP, 45.8% R1);      ║
# ║  primary confirmed on node17 (A40). All tasks pinned to A40 to           ║
# ║  avoid hardware-induced cuDNN algorithm variance (~1.4 pp observed).     ║
# ║                                                                          ║
# ║  SLURM_ARRAY_TASK_ID → seed:                                             ║
# ║    0 → 1 (primary),  1 → 2,  2 → 3,  3 → 4,  4 → 5                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p "${ROOT}/logs/slurm_logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU
cd "${ROOT}"

# ── Recipe (frozen from primary recipe / sweep job 1521019) ─────────────────
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
DRT_CONDITIONING=instance          # residual drift (no domain routing)
SAMPLER=RandomIdentity
DCC_WEIGHT=0.1

SEEDS=(1 2 3 4 5)
IDX=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEEDS[$IDX]}

RUN_NAME="primary_unified_1c_w${DCC_WEIGHT}_driftInst_seed${SEED}"
LOGS_DIR="${ROOT}/logs/multiseed_primary_c3/job_${SLURM_JOB_ID}_${RUN_NAME}"
WANDB_NAME="multiseed_c3_seed${SEED}"
WANDB_TAGS=(multiseed primary-recipe unified idea1c "dcc-w${DCC_WEIGHT}" driftInst "dt-${TARGET_DATASET}" "seed-${SEED}")

echo "=== Multi-seed primary recipe (task ${IDX}, seed ${SEED}) ==="
echo "  Logs: ${LOGS_DIR}"
mkdir -p "${LOGS_DIR}"

SNAP_DIR="${LOGS_DIR}/code_snapshot"
mkdir -p "${SNAP_DIR}"
cp -r bau examples "${SNAP_DIR}/"
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
  --eval-drift true \
  --memory-bank-mode full \
  --drift-method "${DRIFT_METHOD}" \
  --drift-conditioning "${DRT_CONDITIONING}" \
  --use-omega-reg \
  --omega-reg-weight "${OMEGA_REG_WEIGHT}" \
  --sampler "${SAMPLER}" \
  --use-drift-only-cross-contrastive \
  --drift-only-cross-contrastive-weight "${DCC_WEIGHT}" \
  --seed "${SEED}" \
  --wandb-tags "${WANDB_TAGS[@]}" \
  --bidirectional-triplet

  # Deliberately omitted (same as primary recipe):
  # --identity-triplet-only, --bidirectional-triplet, --use-domain-triplet
