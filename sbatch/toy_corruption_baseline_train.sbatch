#!/bin/bash
#SBATCH --job-name=bau-toy-baseline
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_logs/bau-toy-baseline-%j.out
#SBATCH --error=logs/slurm_logs/bau-toy-baseline-%j.err
#SBATCH --exclude=node7,node9,node1,node13


# Euclidean BAU ResNet-50 on ToyCorruption (old_bau/BAU).
# Mirrors local usage in old_bau/BAU/train.sh (conda BAU, train_bau.py from examples/).
#
# Prerequisites:
#   - Generate v4 data under <repo>/examples/data/ToyCorruption/ (see scripts/generate_toy_dataset.py).
#   - If your folder is named ToyCorruptionv4, symlink or copy it to examples/data/ToyCorruption
#     so bau.datasets.toy_corruption.ToyCorruption finds dataset_dir.

set -euo pipefail

ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p "${ROOT}/logs/slurm_logs"
mkdir -p "${ROOT}/old_bau/BAU/logs/baseline_toy"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU

cd "${ROOT}/old_bau/BAU"

srun --unbuffered python examples/train_bau.py \
    --arch resnet50 \
    -ds toycorruption \
    -dt toycorruption \
    --data-dir ../../examples/data \
    --logs-dir logs/baseline_toy \
    --epochs 40 \
    --iters 200 \
    --eval-epochs 5 10 15 20 25 30 35 \
    --seed 1
