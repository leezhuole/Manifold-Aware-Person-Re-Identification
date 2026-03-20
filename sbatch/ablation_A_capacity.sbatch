#!/bin/bash
#SBATCH --job-name=ablA-capacity
#SBATCH --array=0-2
#SBATCH --gres=gpu:1,VRAM:36G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=36G
#SBATCH --time=15:00:00
#SBATCH --exclude=node7,node9,node1,node13
#SBATCH --output=logs/slurm_logs/ablA-%A_%a.out
#SBATCH --error=logs/slurm_logs/ablA-%A_%a.err

# ╔══════════════════════════════════════════════════════════════╗
# ║  Ablation A: Capacity Control                               ║
# ║  Tests whether Finsler gains come from asymmetric geometry   ║
# ║  or from doubled embedding dimensionality (2048 → 4096).    ║
# ║                                                              ║
# ║  A1: resnet50         (2048-d Euclidean)                     ║
# ║  A2: resnet50_finsler (4096-d Euclidean, --force-euclidean)  ║
# ║  A3: resnet50_finsler (4096-d Finsler)                       ║
# ║                                                              ║
# ║  If A2 ≈ A3 > A1: capacity, not geometry.                   ║
# ║  If A3 > A2 ≈ A1: geometry helps beyond capacity.           ║
# ╚══════════════════════════════════════════════════════════════╝

set -euo pipefail
ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p "${ROOT}/logs/slurm_logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU
cd "${ROOT}"

BATCH_SIZE=256
LAM=1.5
KNN=10
ITERS=500
EPOCHS=60
SOURCE_DATASET="market1501 msmt17 cuhksysu"
TARGET_DATASET=cuhk03
SAMPLER="RandomMultipleGallery"
DRIFT_METHOD=symmetric_trapezoidal

ARCH_LIST=(resnet50       resnet50_finsler resnet50_finsler)
EUCLID=(   ""             "--force-euclidean" "")
NAMES=(    "A1_euc2048"   "A2_euc4096"     "A3_finsler4096")

IDX=${SLURM_ARRAY_TASK_ID}
ARCH=${ARCH_LIST[$IDX]}
FORCE_EUC=${EUCLID[$IDX]}
RUN_NAME=${NAMES[$IDX]}

LOGS_DIR="${ROOT}/logs/ablationA_capacity/job_${SLURM_ARRAY_JOB_ID}_${RUN_NAME}"
WANDB_NAME="ablA_${RUN_NAME}"
WANDB_TAGS=(ablation-A capacity "${RUN_NAME}" "dt-${TARGET_DATASET}")

echo "=== Ablation A: ${RUN_NAME} (arch=${ARCH}) ==="
mkdir -p "${LOGS_DIR}"

SNAP_DIR="${LOGS_DIR}/code_snapshot"
mkdir -p "${SNAP_DIR}"
cp -r bau examples "${SNAP_DIR}/"
export PYTHONPATH="${SNAP_DIR}"

EXTRA_ARGS=()
if [[ "${ARCH}" == "resnet50_finsler" ]]; then
  EXTRA_ARGS+=(--drift-method "${DRIFT_METHOD}" --drift-conditioning instance --sampler "${SAMPLER}" --use-omega-reg --omega-reg-weight 1.5 --eval-drift false --memory-bank-mode full)
fi
if [[ -n "${FORCE_EUC}" ]]; then
  EXTRA_ARGS+=(${FORCE_EUC})
fi

srun --unbuffered python "${SNAP_DIR}/examples/train_bau.py" \
  -a "${ARCH}" \
  -ds ${SOURCE_DATASET} \
  -dt "${TARGET_DATASET}" \
  -b "${BATCH_SIZE}" \
  --lam "${LAM}" --k "${KNN}" --iters "${ITERS}" --epochs "${EPOCHS}" \
  --eval-epochs 5 10 15 20 25 30 35 40 45 50 55 59 \
  --logs-dir "${LOGS_DIR}" \
  --wandb-name "${WANDB_NAME}" \
  --wandb-tags "${WANDB_TAGS[@]}" \
  --verbosity 2 \
  "${EXTRA_ARGS[@]}"
