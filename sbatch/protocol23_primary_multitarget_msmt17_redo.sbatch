#!/bin/bash
#SBATCH --job-name=finsler-p23-msmt17-redo
#SBATCH --array=0-1
#SBATCH --gres=gpu:a40:1           # Pin to A40 (Ampere CC 8.6, nodes 15-18) — matches
#SBATCH --nodes=1                  # primary run (node17); eliminates cuDNN algo variance.
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G                  # 64G required: MSMT17 gallery has 82K images.
#SBATCH --time=15:00:00
#SBATCH --output=logs/slurm_logs/finsler-p23-msmt17-redo-%A_%a.out
#SBATCH --error=logs/slurm_logs/finsler-p23-msmt17-redo-%A_%a.err

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  Protocol 2 + Protocol 3 — MSMT17-target redo (OOM fix)                    ║
# ║                                                                             ║
# ║  Re-runs only the two multitarget jobs where -dt msmt17 (array tasks 1 & 4   ║
# ║  in protocol23_primary_multitarget.sbatch). Original job 1521463: cgroup    ║
# ║  OOM during MSMT17 evaluation (large gallery); this script bumps RAM.      ║
# ║                                                                             ║
# ║  SLURM_ARRAY_TASK_ID:                                                       ║
# ║    0  P2: -ds market1501 cuhksysu cuhk03          -dt msmt17   iters=200    ║
# ║    1  P3: -ds market1501dg cuhksysu cuhk03dg      -dt msmt17   iters=400    ║
# ║                                                                             ║
# ║  Hyperparameters match protocol23_primary_multitarget.sbatch.               ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p "${ROOT}/logs/slurm_logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU
cd "${ROOT}"

# ── Shared hyperparameters (frozen from primary recipe / sweep job 1521019) ─
ARCH=resnet50_finsler
BATCH_SIZE=256
LAM=1.5
KNN=10
EPOCHS=60
OMEGA_REG_WEIGHT=1.5
DRIFT_METHOD=symmetric_trapezoidal
DRT_CONDITIONING=instance
SAMPLER=RandomIdentity
DCC_WEIGHT=0.1

IDX=${SLURM_ARRAY_TASK_ID:-0}
case "${IDX}" in
  0)
    PROTOCOL=p2
    SOURCE_DATASET="market1501 cuhksysu cuhk03"
    TARGET_DATASET=msmt17
    PROTOCOL_TAG="p2_m-cs-c3_dt-msmt17"
    ITERS=200
    ;;
  1)
    PROTOCOL=p3
    SOURCE_DATASET="market1501dg cuhksysu cuhk03dg"
    TARGET_DATASET=msmt17
    PROTOCOL_TAG="p3_mdg-cs-c3dg_dt-msmt17"
    ITERS=400
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${IDX} (expected 0-1)" >&2
    exit 1
    ;;
esac

RUN_NAME="primary_unified_1c_w${DCC_WEIGHT}_driftInst_${PROTOCOL_TAG}_redo_oomfix"
LOGS_DIR="${ROOT}/logs/protocol23_primary/job_${SLURM_JOB_ID}_${RUN_NAME}"
WANDB_NAME="primary_${PROTOCOL_TAG}_redo"
WANDB_TAGS=(primary-recipe unified idea1c "dcc-w${DCC_WEIGHT}" driftInst "${PROTOCOL}" "${PROTOCOL_TAG}" "dt-${TARGET_DATASET}" protocol23-msmt17-redo oomfix-mem64G)

echo "=== Protocol 2+3 MSMT17-target redo (task ${IDX}) ==="
echo "  Protocol: ${PROTOCOL}"
echo "  Sources:  ${SOURCE_DATASET}"
echo "  Target:   ${TARGET_DATASET}"
echo "  ITERS:    ${ITERS}"
echo "  Logs:     ${LOGS_DIR}"
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
  --seed 1 \
  --wandb-tags "${WANDB_TAGS[@]}" \
  --bidirectional-triplet
