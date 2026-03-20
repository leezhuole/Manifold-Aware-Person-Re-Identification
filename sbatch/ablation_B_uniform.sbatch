#!/bin/bash
#SBATCH --job-name=ablB-uniform
#SBATCH --array=0-3
#SBATCH --gres=gpu:1,VRAM:36G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=36G
#SBATCH --time=15:00:00
#SBATCH --exclude=node7,node9,node1,node13
#SBATCH --output=logs/slurm_logs/ablB-%A_%a.out
#SBATCH --error=logs/slurm_logs/ablB-%A_%a.err

# ╔══════════════════════════════════════════════════════════════╗
# ║  Ablation B: Uniform Loss × Geometry Compatibility           ║
# ║                                                              ║
# ║  Tests whether uniform loss suppresses asymmetry (alpha→0).  ║
# ║  After the fix, uniform loss uses Euclidean distance on      ║
# ║  identity-only features. This ablation compares:             ║
# ║                                                              ║
# ║  B1: Finsler, full losses (CE+Tri+Align+Uniform+Domain)     ║
# ║  B2: Finsler, no uniform  (CE+Tri+Align+Domain)             ║
# ║  B3: Finsler, identity-only Euclidean uniform (fixed code)   ║
# ║  B4: Euclidean baseline    (CE+Tri+Align+Uniform+Domain)    ║
# ║                                                              ║
# ║  Track alpha convergence in all runs.                        ║
# ║  NOTE: B3 is the new default after the uniform loss fix.     ║
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

NAMES=(
  "B1_finsler_fullLoss"
  "B2_finsler_noUniform"
  "B3_finsler_eucUniform"
  "B4_euclidean_fullLoss"
)

# B1: Finsler full (uniform on full embedding with Finsler dist - OLD behavior)
# B2: Finsler no uniform
# B3: Finsler with Euclidean uniform on identity-only (NEW default after fix)
# B4: Euclidean baseline
ARCH_LIST=(resnet50_finsler  resnet50_finsler  resnet50_finsler  resnet50)
LOSS_FLAGS=(
  ""
  "--no-uniform"
  ""
  ""
)

IDX=${SLURM_ARRAY_TASK_ID}
ARCH=${ARCH_LIST[$IDX]}
RUN_NAME=${NAMES[$IDX]}
SELECTED_LOSS=${LOSS_FLAGS[$IDX]}

LOGS_DIR="${ROOT}/logs/ablationB_uniform/job_${SLURM_ARRAY_JOB_ID}_${RUN_NAME}"
WANDB_NAME="ablB_${RUN_NAME}"
WANDB_TAGS=(ablation-B uniform-loss "${RUN_NAME}" "dt-${TARGET_DATASET}")

echo "=== Ablation B: ${RUN_NAME} (arch=${ARCH}) ==="
mkdir -p "${LOGS_DIR}"

SNAP_DIR="${LOGS_DIR}/code_snapshot"
mkdir -p "${SNAP_DIR}"
cp -r bau examples "${SNAP_DIR}/"
export PYTHONPATH="${SNAP_DIR}"

EXTRA_ARGS=()
if [[ "${ARCH}" == "resnet50_finsler" ]]; then
  EXTRA_ARGS+=(--drift-method "${DRIFT_METHOD}" --drift-conditioning instance --sampler "${SAMPLER}" --use-omega-reg --omega-reg-weight 1.5 --eval-drift false --memory-bank-mode full --alpha-init 0.5)
fi
if [[ -n "${SELECTED_LOSS}" ]]; then
  read -r -a LOSS_ARGS <<< "${SELECTED_LOSS}"
  EXTRA_ARGS+=("${LOSS_ARGS[@]}")
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
