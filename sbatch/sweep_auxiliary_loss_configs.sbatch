#!/bin/bash
#SBATCH --job-name=aux-loss-configs
#SBATCH --array=0-11
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --exclude=node7,node9,node1,node13
#SBATCH --output=logs/slurm_logs/aux-loss-configs-%A_%a.out
#SBATCH --error=logs/slurm_logs/aux-loss-configs-%A_%a.err

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Auxiliary loss configuration sweep × drift conditioning                ║
# ║                                                                        ║
# ║  Indexing: TASK_ID = 0..11,  EXP = TASK_ID / 2,  COND = TASK_ID % 2   ║
# ║    COND=0 → --drift-conditioning instance                              ║
# ║    COND=1 → --drift-conditioning domain                                ║
# ║                                                                        ║
# ║  EXP  Base     Auxiliaries          Bidir  Key variable                 ║
# ║   0   Unified  1d (w=0.1)           yes    Unified+1d                   ║
# ║   1   Unified  1c (w=0.1)           no     Unified+1c (best R1 aux)    ║
# ║   2   Unified  1c+1d (w=0.1 each)   no     complementary drift pair    ║
# ║   3   1a       1d (w=0.1)           yes    eval-drift+bidir on 1a+1d   ║
# ║   4   Unified  1d (w=0.3)           yes    weight sensitivity           ║
# ║   5   Unified  L_tri^dom (camera)   yes    interference thesis §3.7     ║
# ║                                                                        ║
# ║  All configs use --eval-drift true and --use-omega-reg (w=1.5).        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

set -euo pipefail
ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p "${ROOT}/logs/slurm_logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU
cd "${ROOT}"

IDX=${SLURM_ARRAY_TASK_ID}
EXP=$((IDX / 2))
COND=$((IDX % 2))

# ── Fixed recipe ──────────────────────────────────────────────────────────
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

# ── Per-experiment flags ──────────────────────────────────────────────────
# BASE_ARGS: flags that differ between Unified Finsler and 1a
# EXTRA_ARGS: auxiliary loss flags specific to each config
case "${EXP}" in
  0)
    # Unified Finsler + 1d (w=0.1) + bidirectional
    RUN_NAME="00_unified_1d_bidir_w${DRIFT_SC_W}_${COND_TAG}"
    WANDB_TAGS=(sweep-aux-configs unified idea1d bidirectional "sc-w${DRIFT_SC_W}" "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    BASE_ARGS=(
      --bidirectional-triplet
    )
    EXTRA_ARGS=(
      --use-drift-same-cam-attract
      --drift-same-cam-attract-weight "${DRIFT_SC_W}"
    )
    ;;
  1)
    # Unified Finsler + 1c (w=0.1)
    RUN_NAME="01_unified_1c_w${DRIFT_X_W}_${COND_TAG}"
    WANDB_TAGS=(sweep-aux-configs unified idea1c "drift-w${DRIFT_X_W}" "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    BASE_ARGS=( )
    EXTRA_ARGS=(
      --use-drift-only-cross-contrastive
      --drift-only-cross-contrastive-weight "${DRIFT_X_W}"
    )
    ;;
  2)
    # Unified Finsler + 1c (w=0.1) + 1d (w=0.1)
    RUN_NAME="02_unified_1c1d_xW${DRIFT_X_W}_scW${DRIFT_SC_W}_${COND_TAG}"
    WANDB_TAGS=(sweep-aux-configs unified idea1c idea1d complementary-pair "drift-w${DRIFT_X_W}" "sc-w${DRIFT_SC_W}" "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    BASE_ARGS=( )
    EXTRA_ARGS=(
      --use-drift-only-cross-contrastive
      --drift-only-cross-contrastive-weight "${DRIFT_X_W}"
      --use-drift-same-cam-attract
      --drift-same-cam-attract-weight "${DRIFT_SC_W}"
    )
    ;;
  3)
    # 1a + 1d (w=0.1) + bidirectional (identity-triplet-only kept)
    RUN_NAME="03_1a_1d_bidir_w${DRIFT_SC_W}_${COND_TAG}"
    WANDB_TAGS=(sweep-aux-configs idea1a idea1d bidirectional eval-drift-true "sc-w${DRIFT_SC_W}" "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    BASE_ARGS=(
      --identity-triplet-only
      --bidirectional-triplet
    )
    EXTRA_ARGS=(
      --use-drift-same-cam-attract
      --drift-same-cam-attract-weight "${DRIFT_SC_W}"
    )
    ;;
  4)
    # Unified Finsler + 1d (w=0.3) + bidirectional — weight sensitivity
    DRIFT_SC_W_HIGH=0.3
    RUN_NAME="04_unified_1d_bidir_w${DRIFT_SC_W_HIGH}_${COND_TAG}"
    WANDB_TAGS=(sweep-aux-configs unified idea1d bidirectional weight-sensitivity "sc-w${DRIFT_SC_W_HIGH}" "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    BASE_ARGS=(
      --bidirectional-triplet
    )
    EXTRA_ARGS=(
      --use-drift-same-cam-attract
      --drift-same-cam-attract-weight "${DRIFT_SC_W_HIGH}"
    )
    ;;
  5)
    # Unified Finsler + L_tri^dom (camera mining, w=1.0) + bidirectional — §3.7 interference test
    DOM_TRI_W=1.0
    RUN_NAME="05_unified_domTri_bidir_w${DOM_TRI_W}_${COND_TAG}"
    WANDB_TAGS=(sweep-aux-configs unified domain-triplet bidirectional interference-test "domtri-w${DOM_TRI_W}" "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    BASE_ARGS=(
      --bidirectional-triplet
    )
    EXTRA_ARGS=(
      --use-domain-triplet
      --domain-triplet-weight "${DOM_TRI_W}"
      --domain-triplet-mining-label camera
    )
    ;;
  *)
    echo "Invalid derived EXP=${EXP} from SLURM_ARRAY_TASK_ID=${IDX}" >&2
    exit 1
    ;;
esac

LOGS_DIR="${ROOT}/logs/sweep_auxiliary_loss_configs/job_${SLURM_ARRAY_JOB_ID}_${RUN_NAME}"
WANDB_NAME="auxCfg_${RUN_NAME}"

echo "=== Auxiliary loss configs task=${IDX} EXP=${EXP} COND=${COND} (${DRT_CONDITIONING}) ==="
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
  --eval-drift true \
  --memory-bank-mode full \
  --drift-method "${DRIFT_METHOD}" \
  --drift-conditioning "${DRT_CONDITIONING}" \
  --use-omega-reg \
  --omega-reg-weight "${OMEGA_REG_WEIGHT}" \
  --sampler "${SAMPLER}" \
  --wandb-tags "${WANDB_TAGS[@]}" \
  "${BASE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
