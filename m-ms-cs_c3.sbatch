#!/bin/bash
#SBATCH --job-name=bau-train                  # Job name
#SBATCH --gres=gpu:2,VRAM:48G                 # Request 2 GPUs with at least 48GB VRAM each                         
#SBATCH --nodes=1                             # Run all processes on a single node
#SBATCH --cpus-per-task=10                    # Number of CPU cores per task
#SBATCH --mem=64G                             # Total memory for the entire job
#SBATCH --time=24:00:00                       # Time limit hrs:min:sec
#SBATCH --constraint="GPU_MODEL:a40"          # Use only rtx5000 for compatibility
#SBATCH --output=logs/%x-%j.out               # Standard output and error log  (e.g., logs/bau-train-12345.out)
#SBATCH --error=logs/%x-%j.err                # Separate error log (e.g., logs/bau-train-12345.err)
# Optional if you need a specific GPU gen:
# SBATCH --constraint=<feature>               # PDF: Choosing nodes → By features/type (p.4–5)

set -euo pipefail
mkdir -p logs

# Activate the environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU   

# Define variables
ARCH=resnet50
TARGET_DATASET=cuhk03
CURVATURE=1.0

BATCH_SIZE=128
MANIFOLD_AWARE=False
CHUNK_SIZE=500

FINE_TUNING=True
CHECKPOINT_PATH="/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/examples/logs/m-ms-cs_c3/best.pth"
LEARNING_RATE=0.000035
EPOCHS=30
MILESTONES="10 20"
WARMUP_STEPS=0
ITERS=200

# Slurm already sets CUDA_VISIBLE_DEVICES for the allocated GPU
srun --unbuffered python examples/train_bau.py \
  -a ${ARCH} \
  -ds market1501 msmt17 cuhksysu \
  -dt ${TARGET_DATASET} \
  -b  ${BATCH_SIZE} \
  --lam 1.5 --k 10 --iters 500 \
  --logs-dir "/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/examples/logs/m-ms-cs_c3_bsize${BATCH_SIZE}_maware${MANIFOLD_AWARE}_ftune${FINE_TUNING}" \
  --manifold-aware ${MANIFOLD_AWARE} \
  --curvature ${CURVATURE} \
  --eval-epochs 5 10 15 20 25 30 35 40 45 50 55 59 \
  --manifold-chunk-size ${CHUNK_SIZE} \
  --fine-tuning ${FINE_TUNING} \
  --checkpoint-path ${CHECKPOINT_PATH} \
  --verbosity 2 \
  --wandb-name "${ARCH}_${TARGET_DATASET}_hyp${MANIFOLD_AWARE}_c${CURVATURE}_bsize${BATCH_SIZE}_ftune${FINE_TUNING}" \
  --lr ${LEARNING_RATE} \
  --epochs ${EPOCHS} \
  --milestones ${MILESTONES} \
  --warmup-step ${WARMUP_STEPS} \
  --iters ${ITERS}
