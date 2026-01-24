#!/bin/bash
#SBATCH --job-name=bau-finsler                # Job name
#SBATCH --gres=gpu:1,VRAM:28G                 # Request one 28GB-class GPU (reduce OOM risk)
#SBATCH --nodes=1                             # Run all processes on a single node
#SBATCH --cpus-per-task=5                     # Number of CPU cores per task
#SBATCH --mem=24G                             # Total memory for the entire job
#SBATCH --time=12:00:00                       # Time limit hrs:min:sec per task
#SBATCH --exclude=node7,node9,node1,node13    # Exclude problematic nodes
# #SBATCH --constraint="GPU_MODEL:a40"          # Ensure specific GPU model if available
#SBATCH --output=logs/fins_learnableAlpha/finsler-%j.out          # Standard output log
#SBATCH --error=logs/fins_learnableAlpha/finsler-%j.err           # Error log

set -euo pipefail
mkdir -p logs/learnableAlpha

# Activate the environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU

# Ensure local repo is on Python path for srun processes
export PYTHONPATH=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification

# Base hyper-parameters (shared across jobs)
ARCH=resnet50                   # Default
BATCH_SIZE=256                  # Default
LAM=1.5                         # Default
KNN=10                          # Default
ITERS=500                       # Default
WARMUP_STEP=10                  # Default
EPOCHS=60                       # Default

# Finsler alpha learnable configurations
ALPHA_INIT=0.0                 # Learnable alpha initial value
ALPHA_MAX=1.0                  # Learnable alpha max value
ALPHA_TEMP=1.0                 # Learnable alpha temperature

LOGS_DIR="$(pwd)/logs/learnableAlpha/job_${SLURM_JOB_ID}_alphainit${ALPHA_INIT}_alphamax${ALPHA_MAX}_alphatemp${ALPHA_TEMP}_monotonicsigmoid"
WANDB_NAME="learnableAlpha_alphainit${ALPHA_INIT}_alphamax${ALPHA_MAX}_alphatemp${ALPHA_TEMP}_monosigmoid_cuhk03"
echo "Starting finsler experimentation | Alpha Init ${ALPHA_INIT} | Alpha Max ${ALPHA_MAX} | Alpha Temp ${ALPHA_TEMP}"
echo "Logs dir: ${LOGS_DIR}"

mkdir -p "${LOGS_DIR}"

srun --unbuffered python examples/train_bau.py \
  -a "${ARCH}" \
  -ds market1501 msmt17 cuhksysu \
  -dt cuhk03 \
  -b "${BATCH_SIZE}" \
  --lam "${LAM}" --k "${KNN}" --iters "${ITERS}" \
  --epochs "${EPOCHS}" \
  --eval-epochs 5 10 15 20 25 30 35 40 45 50 55 59 \
  --warmup-step "${WARMUP_STEP}" \
  --logs-dir "${LOGS_DIR}" \
  --wandb-name "${WANDB_NAME}" \
  --verbosity 2 \
  --alpha-init "${ALPHA_INIT}" \
  --alpha-max "${ALPHA_MAX}" \
  --alpha-temp "${ALPHA_TEMP}"

