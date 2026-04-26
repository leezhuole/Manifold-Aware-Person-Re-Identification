#!/bin/bash
#SBATCH --job-name=finsler-primary
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --exclude=node7,node9,node1,node13
#SBATCH --output=logs/slurm_logs/finsler-primary-%j.out
#SBATCH --error=logs/slurm_logs/finsler-primary-%j.err

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Primary recipe — single deterministic run                               ║
# ║                                                                          ║
# ║  Arm 1 from sweep job 1521019 (EXP=1, COND=0, array task 2):             ║
# ║    Unified Finsler triplet + L_dcc (w=0.1), residual drift               ║
# ║    With bidirectional triplet, no L_tri^dom                              ║
# ║    M+MS+CS → C3 (CUHK03 target)                                          ║
# ║    Result: 44.9% mAP, 45.8% Rank-1                                       ║
# ║                                                                          ║
# ║  Reference logs:                                                         ║
# ║    logs/sweep_auxiliary_loss_configs/                                    ║
# ║        job_1521019_01_unified_1c_w0.1_driftInst/                         ║
# ║    logs/slurm_logs/aux-loss-configs-1521019_2.out                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p "${ROOT}/logs/slurm_logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU
cd "${ROOT}"

# ── Recipe (frozen from sweep job 1521019, array task 2) ─────────────────
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

RUN_NAME="primary_unified_1c_w${DCC_WEIGHT}_driftInst"
LOGS_DIR="${ROOT}/logs/finsler_primary/job_${SLURM_JOB_ID}_${RUN_NAME}"
WANDB_NAME="primary_${RUN_NAME}"
WANDB_TAGS=(primary-recipe unified idea1c "dcc-w${DCC_WEIGHT}" driftInst "dt-${TARGET_DATASET}")

echo "=== Primary recipe run ==="
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
  --wandb-tags "${WANDB_TAGS[@]}" \
  --bidirectional-triplet
  # Deliberately omitted vs sweep template:
  # --identity-triplet-only   → unified Finsler triplet (not identity-sliced)
  # --bidirectional-triplet   → not in primary recipe
  # --use-domain-triplet      → interferes with L_dom (see §3.7)
