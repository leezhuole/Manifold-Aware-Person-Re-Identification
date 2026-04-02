#!/bin/bash
#SBATCH --job-name=bau-finsler-p2-sweep
#SBATCH --array=0-2
#SBATCH --gres=gpu:1,VRAM:32G                            # Request one 28GB-class GPU (reduce OOM risk)
#SBATCH --nodes=1                                        # Run all processes on a single node
#SBATCH --cpus-per-task=5                                # Number of CPU cores per task
#SBATCH --mem=32G                                        # Dataset + model + memory bank fits in 28 GB
#SBATCH --time=15:00:00                                  # Time limit hrs:min:sec per task
#SBATCH --exclude=node7,node9,node1,node13               # Exclude problematic nodes
# #SBATCH --constraint="GPU_MODEL:a40"                   # Ensure specific GPU model if available
#SBATCH --output=logs/slurm_logs/finsler-p2-%A_%a.out
#SBATCH --error=logs/slurm_logs/finsler-p2-%A_%a.err

# ╔════════════════════════════════════════════════════════════════════╗
# ║  Protocol 2 (BAU DG): sweep over the three source/target permutations║
# ║  from old_bau/BAU/train.sh — same Finsler recipe, different -ds/-dt. ║
# ║                                                                    ║
# ║  SLURM_ARRAY_TASK_ID:                                              ║
# ║    0  -ds market1501 msmt17 cuhksysu  -dt cuhk03                     ║
# ║    1  -ds market1501 cuhksysu cuhk03  -dt msmt17                     ║
# ║    2  -ds msmt17 cuhk03 cuhksysu    -dt market1501                   ║
# ║                                                                    ║
# ║  Original BAU train.sh used --iters 200 for split 1 only; this sweep ║
# ║  keeps ITERS=500 for all splits (matches split 0/2 in that script).  ║
# ╚════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# Activate the environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU

# Ensure local repo is on Python path for srun processes
export PYTHONPATH=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification

# Base hyper-parameters (shared across array tasks)
ARCH=resnet50_finsler
BATCH_SIZE=256
LAM=1.5
KNN=10
EPOCHS=60
OMEGA_REG_WEIGHT=1.5
DRIFT_METHOD=symmetric_trapezoidal
DRT_CONDITIONING=domain                 # instance or domain
SAMPLER="RandomMultipleGallery"

IDX=${SLURM_ARRAY_TASK_ID:-0}
case "${IDX}" in
  0)
    SOURCE_DATASET="market1501 msmt17 cuhksysu"
    TARGET_DATASET=cuhk03
    PROTOCOL_TAG="m-ms-cs_dt-cuhk03"
    ITERS=500
    ;;
  1)
    SOURCE_DATASET="market1501 cuhksysu cuhk03"
    TARGET_DATASET=msmt17
    PROTOCOL_TAG="m-cs-c3_dt-msmt17"
    ITERS=200
    ;;
  2)
    SOURCE_DATASET="msmt17 cuhk03 cuhksysu"
    TARGET_DATASET=market1501
    PROTOCOL_TAG="ms-c3-cs_dt-mkt1501"
    ITERS=500
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${IDX} (expected 0-2)" >&2
    exit 1
    ;;
esac

WANDB_TAGS=(finsler idea1a-identity-triplet "p2-split-${IDX}" "${PROTOCOL_TAG}" "drift-condition-${DRT_CONDITIONING}" "drift-method-${DRIFT_METHOD}" "OmegaReg${OMEGA_REG_WEIGHT}" "Sampler-${SAMPLER}")
RUN_NAME="Idea1a_Baseline_P2split${IDX}_${PROTOCOL_TAG}_DriftCond${DRT_CONDITIONING}_${SAMPLER}_BidirectTri"
WANDB_NAME="${PROTOCOL_TAG}_${RUN_NAME}"

LOGS_DIR="$(pwd)/logs/Protocol-2/job_${SLURM_JOB_ID}_${RUN_NAME}"
echo "Starting finsler Protocol-2 sweep task ${IDX}: -ds ${SOURCE_DATASET} -dt ${TARGET_DATASET}"
echo "Logs dir: ${LOGS_DIR}"

mkdir -p "${LOGS_DIR}"

# --- Snapshot code to isolate job from git branch changes ---
SNAP_DIR="${LOGS_DIR}/code_snapshot"
mkdir -p "${SNAP_DIR}"
cp -r bau examples "${SNAP_DIR}/"
echo "Snapshotted code to: ${SNAP_DIR}"
# -----------------------------------------------------------

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
  --identity-triplet-only \
  --wandb-tags "${WANDB_TAGS[@]}" \
  --bidirectional-triplet

# -- CLI INPUT GRAVEYARD --

  # --use-drift-align \
  # --drift-dim 32 \

  # --use-cross-domain-contrastive \
  # --cross-domain-contrastive-weight "${CROSS_DOMAIN_CONTRAST_WEIGHT}" \
  # --warmup-step "${WARMUP_STEP}" \
  # --milestones ${MILESTONES} \

  # -ds agreidv2_aerial agreidv2_wearable agreidv2_cctv \
  # -dt agreidv2_aerial_to_cctv \
