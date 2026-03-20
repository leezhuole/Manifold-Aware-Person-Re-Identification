#!/bin/bash
#SBATCH --job-name=domain-pivot
#SBATCH --gres=gpu:1,VRAM:36G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=36G
#SBATCH --time=15:00:00
#SBATCH --exclude=node7,node9,node1,node13
#SBATCH --output=logs/slurm_logs/domain-pivot-%j.out
#SBATCH --error=logs/slurm_logs/domain-pivot-%j.err

# ╔══════════════════════════════════════════════════════════════╗
# ║  Validate Domain-Conditioned Drift Pivot                     ║
# ║                                                              ║
# ║  First experiment with domain-conditioned drift head using   ║
# ║  the corrected loss routing:                                 ║
# ║    - Uniform loss: Euclidean on identity-only features       ║
# ║    - k-NN weights: computed from identity-only features      ║
# ║    - Alignment:    identity-only (as before)                 ║
# ║    - Triplet:      Finsler on full [identity | drift]        ║
# ║    - Domain:       Finsler on full [identity | drift]        ║
# ╚══════════════════════════════════════════════════════════════╝

set -euo pipefail
ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p "${ROOT}/logs/slurm_logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU
cd "${ROOT}"

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

RUN_NAME="DomainDrift_CorrectedLossRouting"
WANDB_TAGS=(domain-pivot corrected-losses "drift-condition-${DRT_CONDITIONING}" "drift-method-${DRIFT_METHOD}" "dt-${TARGET_DATASET}")

LOGS_DIR="${ROOT}/logs/domain_pivot/job_${SLURM_JOB_ID}_${RUN_NAME}"
WANDB_NAME="domainPivot_${RUN_NAME}"
echo "=== Domain Pivot Validation: ${RUN_NAME} ==="
echo "Logs dir: ${LOGS_DIR}"

mkdir -p "${LOGS_DIR}"

SNAP_DIR="${LOGS_DIR}/code_snapshot"
mkdir -p "${SNAP_DIR}"
cp -r bau examples "${SNAP_DIR}/"
echo "Snapshotted code to: ${SNAP_DIR}"
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
  --memory-bank-mode full \
  --drift-method "${DRIFT_METHOD}" \
  --use-omega-reg \
  --omega-reg-weight "${OMEGA_REG_WEIGHT}" \
  --drift-conditioning "${DRT_CONDITIONING}" \
  --sampler "${SAMPLER}" \
  --wandb-tags "${WANDB_TAGS[@]}"
