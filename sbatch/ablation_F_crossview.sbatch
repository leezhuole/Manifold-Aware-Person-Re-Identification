#!/bin/bash
#SBATCH --job-name=ablF-crossview
#SBATCH --array=0-3
#SBATCH --gres=gpu:1,VRAM:36G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=36G
#SBATCH --time=15:00:00
#SBATCH --exclude=node7,node9,node1,node13
#SBATCH --output=logs/slurm_logs/ablF-%A_%a.out
#SBATCH --error=logs/slurm_logs/ablF-%A_%a.err

# ╔══════════════════════════════════════════════════════════════╗
# ║  Ablation F: Directional Cross-View Protocol (AG-ReIDv2)     ║
# ║                                                              ║
# ║  Tests whether asymmetry helps in directional retrieval:     ║
# ║  aerial→CCTV vs CCTV→aerial.                                ║
# ║                                                              ║
# ║  F1: Euclidean,  aerial_to_cctv                              ║
# ║  F2: Finsler,    aerial_to_cctv                              ║
# ║  F3: Euclidean,  cctv_to_aerial                              ║
# ║  F4: Finsler,    cctv_to_aerial                              ║
# ║                                                              ║
# ║  If F2 > F1 but F4 ≈ F3 (or vice versa): directional       ║
# ║  asymmetry evidence. If both or neither: no evidence.        ║
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
SAMPLER="RandomMultipleGallery"
DRIFT_METHOD=symmetric_trapezoidal

SOURCE_DATASET="agreidv2_aerial agreidv2_wearable agreidv2_cctv"

NAMES=(
  "F1_euc_aerial2cctv"
  "F2_finsler_aerial2cctv"
  "F3_euc_cctv2aerial"
  "F4_finsler_cctv2aerial"
)
ARCH_LIST=(resnet50  resnet50_finsler  resnet50  resnet50_finsler)
TARGET_LIST=(agreidv2_aerial_to_cctv  agreidv2_aerial_to_cctv  agreidv2_cctv_to_aerial  agreidv2_cctv_to_aerial)

IDX=${SLURM_ARRAY_TASK_ID}
ARCH=${ARCH_LIST[$IDX]}
TARGET=${TARGET_LIST[$IDX]}
RUN_NAME=${NAMES[$IDX]}

LOGS_DIR="${ROOT}/logs/ablationF_crossview/job_${SLURM_ARRAY_JOB_ID}_${RUN_NAME}"
WANDB_NAME="ablF_${RUN_NAME}"
WANDB_TAGS=(ablation-F crossview agreidv2 "${RUN_NAME}" "dt-${TARGET}")

echo "=== Ablation F: ${RUN_NAME} (arch=${ARCH}, target=${TARGET}) ==="
mkdir -p "${LOGS_DIR}"

SNAP_DIR="${LOGS_DIR}/code_snapshot"
mkdir -p "${SNAP_DIR}"
cp -r bau examples "${SNAP_DIR}/"
export PYTHONPATH="${SNAP_DIR}"

EXTRA_ARGS=()
if [[ "${ARCH}" == "resnet50_finsler" ]]; then
  EXTRA_ARGS+=(
    --drift-method "${DRIFT_METHOD}"
    --drift-conditioning domain
    --sampler "${SAMPLER}"
    --use-omega-reg --omega-reg-weight 1.5
    --eval-drift false
    --memory-bank-mode full
  )
fi

srun --unbuffered python "${SNAP_DIR}/examples/train_bau.py" \
  -a "${ARCH}" \
  -ds ${SOURCE_DATASET} \
  -dt "${TARGET}" \
  -b "${BATCH_SIZE}" \
  --lam "${LAM}" --k "${KNN}" --iters "${ITERS}" --epochs "${EPOCHS}" \
  --eval-epochs 5 10 15 20 25 30 35 40 45 50 55 59 \
  --logs-dir "${LOGS_DIR}" \
  --wandb-name "${WANDB_NAME}" \
  --wandb-tags "${WANDB_TAGS[@]}" \
  --verbosity 2 \
  "${EXTRA_ARGS[@]}"
