#!/bin/bash
#SBATCH --job-name=finsler-p23-primary
#SBATCH --array=0-5
#SBATCH --gres=gpu:a40:1           # Pin to A40 (Ampere CC 8.6, nodes 15-18) — matches
#SBATCH --nodes=1                  # primary run (node17, A40). Prevents cudnn.benchmark
#SBATCH --cpus-per-task=5          # from selecting different algorithms across GPU gens.
#SBATCH --mem=64G                  # 64G: OOM fix for MSMT17-target tasks (82K gallery).
#SBATCH --time=15:00:00
#SBATCH --output=logs/slurm_logs/finsler-p23-primary-%A_%a.out
#SBATCH --error=logs/slurm_logs/finsler-p23-primary-%A_%a.err

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  Protocol 2 + Protocol 3 multi-target sweep — primary Finsler recipe        ║
# ║                                                                             ║
# ║  Arm 1 from sweep job 1521019: Unified Finsler + L_dcc (w=0.1),             ║
# ║  residual drift, with bidirectional triplet, no L_tri^dom.                  ║
# ║                                                                             ║
# ║  SLURM_ARRAY_TASK_ID:                                                       ║
# ║    0  P2: -ds market1501 msmt17 cuhksysu         -dt cuhk03    iters=500    ║
# ║    1  P2: -ds market1501 cuhksysu cuhk03          -dt msmt17   iters=200    ║
# ║    2  P2: -ds msmt17 cuhk03 cuhksysu              -dt market1501 iters=500  ║
# ║    3  P3: -ds market1501dg msmt17dg cuhksysu      -dt cuhk03  iters=1000    ║
# ║    4  P3: -ds market1501dg cuhksysu cuhk03dg      -dt msmt17  iters=400     ║
# ║    5  P3: -ds msmt17dg cuhk03dg cuhksysu          -dt market1501 iters=1000 ║
# ║                                                                             ║
# ║  Task 0 is a sanity-check against the primary result (44.9% mAP, 45.8% R1). ║
# ║  Tasks 1–5 fill tab:template_multiprotocol_bau.                             ║
# ║  Source/target/iters from old_bau/BAU/train.sh.                             ║
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
DRT_CONDITIONING=instance          # residual drift (no domain routing)
SAMPLER=RandomIdentity
DCC_WEIGHT=0.1

IDX=${SLURM_ARRAY_TASK_ID:-0}
case "${IDX}" in
  0)
    PROTOCOL=p2
    SOURCE_DATASET="market1501 msmt17 cuhksysu"
    TARGET_DATASET=cuhk03
    PROTOCOL_TAG="p2_m-ms-cs_dt-cuhk03"
    ITERS=500
    ;;
  1)
    PROTOCOL=p2
    SOURCE_DATASET="market1501 cuhksysu cuhk03"
    TARGET_DATASET=msmt17
    PROTOCOL_TAG="p2_m-cs-c3_dt-msmt17"
    ITERS=200
    ;;
  2)
    PROTOCOL=p2
    SOURCE_DATASET="msmt17 cuhk03 cuhksysu"
    TARGET_DATASET=market1501
    PROTOCOL_TAG="p2_ms-c3-cs_dt-mkt1501"
    ITERS=500
    ;;
  3)
    PROTOCOL=p3
    SOURCE_DATASET="market1501dg msmt17dg cuhksysu"
    TARGET_DATASET=cuhk03
    PROTOCOL_TAG="p3_mdg-msdg-cs_dt-cuhk03"
    ITERS=1000
    ;;
  4)
    PROTOCOL=p3
    SOURCE_DATASET="market1501dg cuhksysu cuhk03dg"
    TARGET_DATASET=msmt17
    PROTOCOL_TAG="p3_mdg-cs-c3dg_dt-msmt17"
    ITERS=400
    ;;
  5)
    PROTOCOL=p3
    SOURCE_DATASET="msmt17dg cuhk03dg cuhksysu"
    TARGET_DATASET=market1501
    PROTOCOL_TAG="p3_msdg-c3dg-cs_dt-mkt1501"
    ITERS=1000
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${IDX} (expected 0-5)" >&2
    exit 1
    ;;
esac

RUN_NAME="primary_unified_1c_w${DCC_WEIGHT}_driftInst_${PROTOCOL_TAG}"
LOGS_DIR="${ROOT}/logs/protocol23_primary/job_${SLURM_JOB_ID}_${RUN_NAME}"
WANDB_NAME="primary_${PROTOCOL_TAG}"
WANDB_TAGS=(primary-recipe unified idea1c "dcc-w${DCC_WEIGHT}" driftInst "${PROTOCOL}" "${PROTOCOL_TAG}" "dt-${TARGET_DATASET}")

echo "=== Protocol 2+3 primary recipe sweep (task ${IDX}) ==="
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
  # Deliberately omitted vs primary recipe:
  # --identity-triplet-only   → unified Finsler triplet (not identity-sliced)
  # --use-domain-triplet      → interferes with L_dom (see §3.7)
