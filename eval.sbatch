#!/bin/bash
#SBATCH --job-name=bau-eval                 # Job name
#SBATCH --gres=gpu:1,VRAM:12G               # Request 2 GPUs with at least 12GB VRAM each                         
#SBATCH --nodes=1                           # Run all processes on a single node
#SBATCH --cpus-per-task=5                   # Number of CPU cores per task
#SBATCH --mem=32G                           # Total memory for the entire job
#SBATCH --time=24:00:00                     # Time limit hrs:min:sec
#SBATCH --constraint="GPU_MODEL:rtx_5000"   # Use only rtx5000 for compatibility
#SBATCH --output=logs/%x-%j.out             # Standard output and error log  (e.g., logs/bau-train-12345.out)
#SBATCH --error=logs/%x-%j.err              # Separate error log (e.g., logs/bau-train-12345.err)
# Optional if you need a specific GPU gen:
# SBATCH --constraint=<feature>              # PDF: Choosing nodes → By features/type (p.4–5)

set -euo pipefail
mkdir -p logs

# Activate your environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU   # or module load <python/conda>

# Target: Market1501
srun --unbuffered env python examples/test.py \
    -d market1501 \
    --resume "/home/stud/leez/reid/src/BAU/examples/logs/ms-cs-c3_m/best.pth"

# Target: CUHK-SYSU
srun --unbuffered env python examples/test.py \
    -d cuhk03 \
    --resume "/home/stud/leez/reid/src/BAU/examples/logs/m-ms-cs_c3/best.pth"

# Target: MSMT17
srun --unbuffered env python examples/test.py \
    -d msmt17 \
    --resume "/home/stud/leez/reid/src/BAU/examples/logs/m-cs-c3_ms/best.pth"