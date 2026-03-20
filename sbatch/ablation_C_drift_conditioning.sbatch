#!/bin/bash
#SBATCH --job-name=ablC-drift
#SBATCH --array=0-3
#SBATCH --gres=gpu:1,VRAM:36G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=36G
#SBATCH --time=15:00:00
#SBATCH --exclude=node7,node9,node1,node13
#SBATCH --output=logs/slurm_logs/ablC-%A_%a.out
#SBATCH --error=logs/slurm_logs/ablC-%A_%a.err

# ╔══════════════════════════════════════════════════════════════╗
# ║  Ablation C: Drift Conditioning Level                        ║
# ║                                                              ║
# ║  Tests whether domain-level drift is more tractable than     ║
# ║  instance-level, and whether the residual correction helps.  ║
# ║                                                              ║
# ║  C1: Instance-conditioned drift  (FinslerDriftHead)          ║
# ║  C2: Domain-conditioned drift    (DomainConditionedDriftHead)║
# ║  C3: Domain-conditioned, no residual (residual_scale=0)      ║
# ║  C4: Euclidean baseline          (no drift at all)           ║
# ║                                                              ║
# ║  Report: identity-only & full-embedding metrics, drift norm  ║
# ║  evolution, domain classifier accuracy.                      ║
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
  "C1_instance_drift"
  "C2_domain_drift"
  "C3_domain_drift_noResidual"
  "C4_euclidean_noDrift"
)
ARCH_LIST=(resnet50_finsler  resnet50_finsler  resnet50_finsler  resnet50)
CONDITIONING=(instance  domain  domain  "")
RESIDUAL_SCALE=(0.1  0.1  0.0  0.0)

IDX=${SLURM_ARRAY_TASK_ID}
ARCH=${ARCH_LIST[$IDX]}
RUN_NAME=${NAMES[$IDX]}

LOGS_DIR="${ROOT}/logs/ablationC_drift/job_${SLURM_ARRAY_JOB_ID}_${RUN_NAME}"
WANDB_NAME="ablC_${RUN_NAME}"
WANDB_TAGS=(ablation-C drift-conditioning "${RUN_NAME}" "dt-${TARGET_DATASET}")

echo "=== Ablation C: ${RUN_NAME} (arch=${ARCH}) ==="
mkdir -p "${LOGS_DIR}"

SNAP_DIR="${LOGS_DIR}/code_snapshot"
mkdir -p "${SNAP_DIR}"
cp -r bau examples "${SNAP_DIR}/"
export PYTHONPATH="${SNAP_DIR}"

EXTRA_ARGS=()
if [[ "${ARCH}" == "resnet50_finsler" ]]; then
  EXTRA_ARGS+=(
    --drift-method "${DRIFT_METHOD}"
    --drift-conditioning "${CONDITIONING[$IDX]}"
    --domain-residual-scale "${RESIDUAL_SCALE[$IDX]}"
    --sampler "${SAMPLER}"
    --use-omega-reg --omega-reg-weight 1.5
    --eval-drift false
    --memory-bank-mode full
  )
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
