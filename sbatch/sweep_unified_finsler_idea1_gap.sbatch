#!/bin/bash
#SBATCH --job-name=idea1-gap
#SBATCH --array=0-7
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --exclude=node7,node9,node1,node13
#SBATCH --output=logs/slurm_logs/idea1-gap-%A_%a.out
#SBATCH --error=logs/slurm_logs/idea1-gap-%A_%a.err

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Idea-1 gap sweep: domain-triplet-only (ARM=0) OR unified Finsler    ║
# ║  PID triplet (ARM=1) × drift conditioning × L_domain flavor          ║
# ║                                                                      ║
# ║  Indexing: TASK_ID = 0..7                                            ║
# ║    COND       = TASK_ID % 2         0 instance / 1 domain              ║
# ║    L_DOM_KIND = (TASK_ID / 2) % 2   0 Finsler L_dom+omega+full bank  ║
# ║                                     1 Euclidean L_dom (identity bank)║
# ║    ARM        = TASK_ID / 4                                          ║
# ║      0 — Domain triplet only (no PID batch-hard triplet):            ║
# ║         --no-triplet --use-domain-triplet (camera, DOM_TRI_W).       ║
# ║         This is the only TripletLoss-derived term in the sum.        ║
# ║      1 — Unified Finsler: single batch-hard triplet on PIDs using    ║
# ║         full [identity|drift] embedding (no --identity-triplet-only, ║
# ║         no --bidirectional-triplet).                                 ║
# ║                                                                      ║
# ║  Note: without --no-triplet, --use-domain-triplet ADDS loss_tri_dom   ║
# ║  on top of the main triplet; see bau/trainers.py total loss.          ║
# ╚══════════════════════════════════════════════════════════════════════╝

set -euo pipefail
ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p "${ROOT}/logs/slurm_logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU
cd "${ROOT}"

IDX=${SLURM_ARRAY_TASK_ID}
COND=$((IDX % 2))
L_DOM_KIND=$(( (IDX / 2) % 2 ))
ARM=$((IDX / 4))

# ── Fixed recipe (match sweep_loss_ablation_Idea1_EuclideanDom) ──
ARCH=resnet50_finsler
BATCH_SIZE=256
LAM=1.5
KNN=10
ITERS=500
EPOCHS=60
DRIFT_METHOD=symmetric_trapezoidal
OMEGA_REG_WEIGHT=1.5
SOURCE_DATASET="market1501 msmt17 cuhksysu"
TARGET_DATASET=cuhk03
DOM_TRI_W=1.0

if (( COND == 0 )); then
  DRT_CONDITIONING=instance
  COND_TAG="driftInst"
  SAMPLER="RandomIdentity"
else
  DRT_CONDITIONING=domain
  COND_TAG="driftDom"
  SAMPLER="RandomMultipleGallery"
fi

if (( L_DOM_KIND == 0 )); then
  LDOM_TAG="LdomFins"
  LDOM_ARGS=(
    --memory-bank-mode full
    --use-omega-reg
    --omega-reg-weight "${OMEGA_REG_WEIGHT}"
  )
else
  LDOM_TAG="LdomEucl"
  LDOM_ARGS=(
    --use-euclidean-domain-loss
    --memory-bank-mode identity
  )
fi

if (( ARM == 0 )); then
  ARM_TAG="idea1bOnly_domTriW${DOM_TRI_W}"
  ARM_EXTRA_ARGS=(
    --no-triplet
    --use-domain-triplet
    --domain-triplet-mining-label camera
    --domain-triplet-weight "${DOM_TRI_W}"
  )
  WANDB_ARM_TAGS=( "${ARM_TAG}" domain-triplet-only no-pid-triplet "tri-w${DOM_TRI_W}" )
else
  ARM_TAG="unifiedFin_onlyTri"
  ARM_EXTRA_ARGS=( )
  WANDB_ARM_TAGS=( "${ARM_TAG}" unified-finsler-triplet-only )
fi

RUN_NAME="${ARM_TAG}__${COND_TAG}__${LDOM_TAG}"
WANDB_TAGS=(
  sweep-idea1-gap
  "${WANDB_ARM_TAGS[@]}"
  "${COND_TAG}"
  "${LDOM_TAG}"
  "dt-${TARGET_DATASET}"
  "Sampler-${SAMPLER}"
)

LOGS_DIR="${ROOT}/logs/sweep_unified_finsler_idea1_gap/job_${SLURM_ARRAY_JOB_ID}_${RUN_NAME}"
WANDB_NAME="idea1Gap_${RUN_NAME}"

echo "=== Idea-1 gap sweep task=${IDX} ARM=${ARM} COND=${COND} (${DRT_CONDITIONING}) L_DOM_KIND=${L_DOM_KIND} (${LDOM_TAG}) ==="
echo "  RUN_NAME=${RUN_NAME}"
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
  --eval-drift false \
  --drift-method "${DRIFT_METHOD}" \
  --drift-conditioning "${DRT_CONDITIONING}" \
  --sampler "${SAMPLER}" \
  --wandb-tags "${WANDB_TAGS[@]}" \
  "${LDOM_ARGS[@]}" \
  "${ARM_EXTRA_ARGS[@]}"
