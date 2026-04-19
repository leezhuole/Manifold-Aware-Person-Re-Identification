#!/bin/bash
#SBATCH --job-name=bau-euc-baseline
#SBATCH --array=0-5
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --exclude=node7,node9,node1,node13
#SBATCH --output=logs/slurm_logs/bau-euc-baseline-%A_%a.out
#SBATCH --error=logs/slurm_logs/bau-euc-baseline-%A_%a.err

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  Configuration-matched BAU Euclidean baseline — Protocol 2 + Protocol 3    ║
# ║                                                                            ║
# ║  ResNet-50 (no Finsler/drift branch). Source/target/iters mirror the       ║
# ║  primary Finsler recipe (finsler_primary_recipe.sbatch / job 1521019)      ║
# ║  so results are table-safe for tab:extended_ranks and                      ║
# ║  tab:template_multiprotocol_bau. ITERS matches the Finsler recipe          ║
# ║  (500/200/500/1000/400/1000); the paper caption saying "ITERS=1000"        ║
# ║  for the Euclidean baseline refers to a stale version and needs            ║
# ║  correction before submission.                                             ║
# ║                                                                            ║
# ║  SLURM_ARRAY_TASK_ID:                                                      ║
# ║    0  P2: -ds market1501 msmt17 cuhksysu         -dt cuhk03    iters=500   ║
# ║    1  P2: -ds market1501 cuhksysu cuhk03          -dt msmt17   iters=200   ║
# ║    2  P2: -ds msmt17 cuhk03 cuhksysu              -dt market1501 iters=500 ║
# ║    3  P3: -ds market1501dg msmt17dg cuhksysu      -dt cuhk03  iters=1000   ║
# ║    4  P3: -ds market1501dg cuhksysu cuhk03dg      -dt msmt17  iters=400    ║
# ║    5  P3: -ds msmt17dg cuhk03dg cuhksysu          -dt market1501 iters=1000║
# ╚════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p "${ROOT}/logs/slurm_logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU
cd "${ROOT}"

# ── Shared hyperparameters (configuration-matched to primary Finsler recipe) ─
ARCH=resnet50
BATCH_SIZE=256
LAM=1.5
KNN=10
EPOCHS=60

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

RUN_NAME="bau_euc_baseline_${PROTOCOL_TAG}"
LOGS_DIR="${ROOT}/logs/euclidean_baseline/job_${SLURM_JOB_ID}_${RUN_NAME}"
WANDB_NAME="euc_baseline_${PROTOCOL_TAG}"
WANDB_TAGS=(euclidean-baseline config-matched resnet50 "${PROTOCOL}" "${PROTOCOL_TAG}" "dt-${TARGET_DATASET}")

echo "=== BAU Euclidean baseline (task ${IDX}) ==="
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
  --wandb-tags "${WANDB_TAGS[@]}"
  # Finsler-specific flags deliberately omitted (arch=resnet50 has no drift branch):
  # --eval-drift, --drift-method, --drift-conditioning
  # --use-omega-reg, --omega-reg-weight
  # --use-drift-only-cross-contrastive, --drift-only-cross-contrastive-weight
