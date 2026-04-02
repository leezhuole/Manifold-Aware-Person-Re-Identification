#!/bin/bash
#SBATCH --job-name=loss-ablate-idea1-eucldom
#SBATCH --array=0-9
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --exclude=node7,node9,node1,node13
#SBATCH --output=logs/slurm_logs/loss-ablate-Idea1-eucldom-%A_%a.out
#SBATCH --error=logs/slurm_logs/loss-ablate-Idea1-eucldom-%A_%a.err

# ╔════════════════════════════════════════════════════════════════════╗
# ║  Same factorial as Idea-1 extensions, but L_domain = original BAU  ║
# ║  Euclidean repulsion on identity features + identity memory bank.  ║
# ║  --use-euclidean-domain-loss (no --use-omega-reg: see changelog).  ║
# ║                                                                    ║
# ║  Indexing: TASK_ID = 0..9,  EXP = TASK_ID / 2,  COND = TASK_ID % 2 ║
# ║    COND=0 → --drift-conditioning instance                          ║
# ║    COND=1 → --drift-conditioning domain                            ║
# ║  EXP 0..4: same arms as sweep_loss_ablation_Idea1_extensions.sbatch║
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

# ── Fixed recipe ───────────────────────────────────────────────────
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
XDOM_W=0.1
DOM_TRI_W=1.0
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
    RUN_NAME="00_baseline_1a_only_${COND_TAG}_euclLdom"
    WANDB_TAGS=(sweep-loss-ablate-eucldom baseline idea1a-only "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    EXTRA_ARGS=( )
    ;;
  1)
    RUN_NAME="01_cam_xdom_w${XDOM_W}_${COND_TAG}_euclLdom"
    WANDB_TAGS=(sweep-loss-ablate-eucldom cam-nuisance-contrast "xdom-w${XDOM_W}" "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    EXTRA_ARGS=(
      --use-cross-domain-contrastive
      --cross-domain-contrastive-nuisance camera
      --cross-domain-contrastive-weight "${XDOM_W}"
    )
    ;;
  2)
    RUN_NAME="02_domTri_cam_w${DOM_TRI_W}_noLdom_${COND_TAG}_euclLdom"
    WANDB_TAGS=(sweep-loss-ablate-eucldom domain-triplet camera-mining "tri-w${DOM_TRI_W}" no-domain-loss "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    EXTRA_ARGS=(
      --no-domain
      --use-domain-triplet
      --domain-triplet-mining-label camera
      --domain-triplet-weight "${DOM_TRI_W}"
    )
    ;;
  3)
    RUN_NAME="03_driftOnly_xcam_w${DRIFT_X_W}_${COND_TAG}_euclLdom"
    WANDB_TAGS=(sweep-loss-ablate-eucldom drift-only-contrast "drift-w${DRIFT_X_W}" "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    EXTRA_ARGS=(
      --use-drift-only-cross-contrastive
      --drift-only-cross-contrastive-weight "${DRIFT_X_W}"
    )
    ;;
  4)
    RUN_NAME="04_domTri_cam_w${DOM_TRI_W}_withLdom_${COND_TAG}_euclLdom"
    WANDB_TAGS=(sweep-loss-ablate-eucldom domain-triplet camera-mining "tri-w${DOM_TRI_W}" with-domain-loss "${COND_TAG}" "dt-${TARGET_DATASET}" "Sampler-${SAMPLER}")
    EXTRA_ARGS=(
      --use-domain-triplet
      --domain-triplet-mining-label camera
      --domain-triplet-weight "${DOM_TRI_W}"
    )
    ;;
  *)
    echo "Invalid derived EXP=${EXP} from SLURM_ARRAY_TASK_ID=${IDX}" >&2
    exit 1
    ;;
esac

LOGS_DIR="${ROOT}/logs/sweep_loss_ablation_Idea1_EuclideanDom_2/job_${SLURM_ARRAY_JOB_ID}_${RUN_NAME}"
WANDB_NAME="lossAbI1EuclDom_${RUN_NAME}"

echo "=== Euclidean L_domain ablation task=${IDX} EXP=${EXP} COND=${COND} (${DRT_CONDITIONING}) ==="
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
  --identity-triplet-only \
  --wandb-tags "${WANDB_TAGS[@]}" \
  "${EXTRA_ARGS[@]}"
