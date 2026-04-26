#!/bin/bash
#SBATCH --job-name=loss-ablate-idea1-1d1e-eucl
#SBATCH --array=0-3
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --exclude=node7,node9,node1,node13
#SBATCH --output=logs/slurm_logs/loss-ablate-Idea1-1d1e-eucl-%A_%a.out
#SBATCH --error=logs/slurm_logs/loss-ablate-Idea1-1d1e-eucl-%A_%a.err

# ╔════════════════════════════════════════════════════════════════════╗
# ║  Loss ablation: Idea-1 extensions 1d/1e × Euclidean Domain Loss    ║
# ║  Fills the missing cells in the results table (only instance cond).║
# ║                                                                    ║
# ║  Indexing: TASK_ID = 0..3 (EXP = TASK_ID)                          ║
# ║    0  1a+1d         — same-camera drift attraction (w=0.1)         ║
# ║    1  1a+1e         — cross-camera drift uniformity (w=0.1, t=2)   ║
# ║    2  1a+1d+1e      — both 1d and 1e combined                      ║
# ║    3  Unified + 1c  — Full Finsler triplet + drift L2 alignment    ║
# ╚════════════════════════════════════════════════════════════════════╝

set -euo pipefail
ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p "${ROOT}/logs/slurm_logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU
cd "${ROOT}"

IDX=${SLURM_ARRAY_TASK_ID}
EXP=${IDX}

# ── Fixed recipe (Euclidean Domain Loss) ───────────────────────────
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

DRT_CONDITIONING=instance
COND_TAG="driftInst"
SAMPLER="RandomIdentity"

IDENTITY_TRIPLET="--identity-triplet-only"

case "${EXP}" in
  0)
    RUN_NAME="00_1a_1d_sameCamAttract_w${DRIFT_SC_W}_${COND_TAG}_euclLdom"
    WANDB_TAGS=(sweep-loss-ablate-1d1e-eucl idea1d same-cam-attract "sc-w${DRIFT_SC_W}" "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    EXTRA_ARGS=(
      --use-drift-same-cam-attract
      --drift-same-cam-attract-weight "${DRIFT_SC_W}"
    )
    ;;
  1)
    RUN_NAME="01_1a_1e_crossCamUniform_w${DRIFT_CU_W}_t${DRIFT_CU_T}_${COND_TAG}_euclLdom"
    WANDB_TAGS=(sweep-loss-ablate-1d1e-eucl idea1e cross-cam-uniform "cu-w${DRIFT_CU_W}" "cu-t${DRIFT_CU_T}" "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    EXTRA_ARGS=(
      --use-drift-cross-cam-uniform
      --drift-cross-cam-uniform-weight "${DRIFT_CU_W}"
      --drift-cross-cam-uniform-t "${DRIFT_CU_T}"
    )
    ;;
  2)
    RUN_NAME="02_1a_1d1e_combined_scW${DRIFT_SC_W}_cuW${DRIFT_CU_W}_${COND_TAG}_euclLdom"
    WANDB_TAGS=(sweep-loss-ablate-1d1e-eucl idea1d idea1e combined "sc-w${DRIFT_SC_W}" "cu-w${DRIFT_CU_W}" "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    EXTRA_ARGS=(
      --use-drift-same-cam-attract
      --drift-same-cam-attract-weight "${DRIFT_SC_W}"
      --use-drift-cross-cam-uniform
      --drift-cross-cam-uniform-weight "${DRIFT_CU_W}"
      --drift-cross-cam-uniform-t "${DRIFT_CU_T}"
    )
    ;;
  3)
    RUN_NAME="03_UnifiedFinsler_1c_driftW${DRIFT_X_W}_${COND_TAG}_euclLdom"
    WANDB_TAGS=(sweep-loss-ablate-1d1e-eucl unified-finsler idea1c-drift-align "drift-w${DRIFT_X_W}" "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    IDENTITY_TRIPLET="" # No --identity-triplet-only for Unified Finsler
    EXTRA_ARGS=(
      --use-drift-only-cross-contrastive
      --drift-only-cross-contrastive-weight "${DRIFT_X_W}"
    )
    ;;  
  *)
    echo "Invalid EXP=${EXP}" >&2
    exit 1
    ;;
esac

LOGS_DIR="${ROOT}/logs/sweep_loss_ablation_Idea1_1d1e_EuclideanDom/job_${SLURM_ARRAY_JOB_ID}_${RUN_NAME}"
WANDB_NAME="lossAbI1Eucl1d1e_${RUN_NAME}"

echo "=== Euclidean L_domain 1d/1e/Unified task=${IDX} EXP=${EXP} (${DRT_CONDITIONING}) ==="
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
  --use-euclidean-domain-loss \
  --memory-bank-mode identity \
  --drift-method "${DRIFT_METHOD}" \
  --drift-conditioning "${DRT_CONDITIONING}" \
  --use-omega-reg \
  --omega-reg-weight "${OMEGA_REG_WEIGHT}" \
  --sampler "${SAMPLER}" \
  ${IDENTITY_TRIPLET} \
  --wandb-tags "${WANDB_TAGS[@]}" \
  "${EXTRA_ARGS[@]}"