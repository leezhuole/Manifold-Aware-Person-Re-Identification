#!/bin/bash
#SBATCH --job-name=1e-no-norm
#SBATCH --array=0-7
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --exclude=node7,node9,node1,node13
#SBATCH --output=logs/slurm_logs/1e-no-norm-%A_%a.out
#SBATCH --error=logs/slurm_logs/1e-no-norm-%A_%a.err

# ╔════════════════════════════════════════════════════════════════════╗
# ║  Re-run of 1e-involving arms with L2 normalization REMOVED from  ║
# ║  drift_cross_camera_uniformity_loss (Wang–Isola now operates on  ║
# ║  raw drift vectors in the Randers ball, not on the hypersphere). ║
# ║                                                                  ║
# ║  Indexing: TASK_ID = 0..7,  EXP = TASK_ID / 2,  COND = TASK_ID%2║
# ║    COND=0 → --drift-conditioning instance                        ║
# ║    COND=1 → --drift-conditioning domain                          ║
# ║                                                                  ║
# ║  EXP (loss arm):                                                 ║
# ║    0  baseline      — 1a only (reference, unchanged)             ║
# ║    1  1a+1e         — cross-camera drift uniformity (no L2 norm) ║
# ║    2  1a+1d+1e      — both 1d and 1e (1e without L2 norm)        ║
# ║    3  1a+1c+1d+1e   — full stack (1e without L2 norm)            ║
# ╚════════════════════════════════════════════════════════════════════╝

set -euo pipefail
ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p "${ROOT}/logs/slurm_logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU
cd "${ROOT}"

IDX=${SLURM_ARRAY_TASK_ID}
EXP=$((IDX / 2))
COND=$((IDX % 2))

# ── Fixed recipe (matches sweep_loss_ablation_Idea1_1d1e) ────
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
DRIFT_SC_W=0.1
DRIFT_CU_W=0.1
DRIFT_CU_T=2.0
DRIFT_X_W=0.1

if (( COND == 0 )); then
  DRT_CONDITIONING=instance
  COND_TAG="driftInst"
  SAMPLER="RandomIdentity"
else
  DRT_CONDITIONING=domain
  COND_TAG="driftDom"
  SAMPLER="RandomMultipleGallery"
fi

case "${EXP}" in
  0)
    RUN_NAME="00_baseline_1a_only_${COND_TAG}"
    WANDB_TAGS=(sweep-1e-no-norm baseline idea1a-only "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    EXTRA_ARGS=( )
    ;;
  1)
    RUN_NAME="01_1e_noNorm_crossCamUniform_w${DRIFT_CU_W}_t${DRIFT_CU_T}_${COND_TAG}"
    WANDB_TAGS=(sweep-1e-no-norm idea1e no-l2-norm cross-cam-uniform "cu-w${DRIFT_CU_W}" "cu-t${DRIFT_CU_T}" "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    EXTRA_ARGS=(
      --use-drift-cross-cam-uniform
      --drift-cross-cam-uniform-weight "${DRIFT_CU_W}"
      --drift-cross-cam-uniform-t "${DRIFT_CU_T}"
    )
    ;;
  2)
    RUN_NAME="02_1d1e_noNorm_combined_scW${DRIFT_SC_W}_cuW${DRIFT_CU_W}_${COND_TAG}"
    WANDB_TAGS=(sweep-1e-no-norm idea1d idea1e no-l2-norm combined "sc-w${DRIFT_SC_W}" "cu-w${DRIFT_CU_W}" "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    EXTRA_ARGS=(
      --use-drift-same-cam-attract
      --drift-same-cam-attract-weight "${DRIFT_SC_W}"
      --use-drift-cross-cam-uniform
      --drift-cross-cam-uniform-weight "${DRIFT_CU_W}"
      --drift-cross-cam-uniform-t "${DRIFT_CU_T}"
    )
    ;;
  3)
    RUN_NAME="03_1c1d1e_noNorm_full_driftW${DRIFT_X_W}_scW${DRIFT_SC_W}_cuW${DRIFT_CU_W}_${COND_TAG}"
    WANDB_TAGS=(sweep-1e-no-norm idea1c idea1d idea1e no-l2-norm full-drift-stack "drift-w${DRIFT_X_W}" "sc-w${DRIFT_SC_W}" "cu-w${DRIFT_CU_W}" "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    EXTRA_ARGS=(
      --use-drift-only-cross-contrastive
      --drift-only-cross-contrastive-weight "${DRIFT_X_W}"
      --use-drift-same-cam-attract
      --drift-same-cam-attract-weight "${DRIFT_SC_W}"
      --use-drift-cross-cam-uniform
      --drift-cross-cam-uniform-weight "${DRIFT_CU_W}"
      --drift-cross-cam-uniform-t "${DRIFT_CU_T}"
    )
    ;;
  *)
    echo "Invalid derived EXP=${EXP} from SLURM_ARRAY_TASK_ID=${IDX}" >&2
    exit 1
    ;;
esac

LOGS_DIR="${ROOT}/logs/sweep_1e_no_norm/job_${SLURM_ARRAY_JOB_ID}_${RUN_NAME}"
WANDB_NAME="1eNoNorm_${RUN_NAME}"

echo "=== 1e no-norm rerun task=${IDX} EXP=${EXP} COND=${COND} (${DRT_CONDITIONING}) ==="
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
  --memory-bank-mode full \
  --drift-method "${DRIFT_METHOD}" \
  --drift-conditioning "${DRT_CONDITIONING}" \
  --use-omega-reg \
  --omega-reg-weight "${OMEGA_REG_WEIGHT}" \
  --sampler "${SAMPLER}" \
  --identity-triplet-only \
  --wandb-tags "${WANDB_TAGS[@]}" \
  "${EXTRA_ARGS[@]}"
