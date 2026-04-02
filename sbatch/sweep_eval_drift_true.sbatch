#!/bin/bash
#SBATCH --job-name=eval-drift-true
#SBATCH --array=0-27
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=node7,node9,node1,node13
#SBATCH --output=logs/slurm_logs/eval-drift-true-%A_%a.out
#SBATCH --error=logs/slurm_logs/eval-drift-true-%A_%a.err

# ╔════════════════════════════════════════════════════════════════════╗
# ║  Evaluation Sweep: --eval-drift true                               ║
# ║                                                                    ║
# ║  This script locates `model_best.pth.tar` across the three Idea-1  ║
# ║  sweeps, dynamically parses the required architectural hooks       ║
# ║  (e.g. drift conditioning, num domains) from the directory name,   ║
# ║  and evaluates the model with the Finsler drift explicitly         ║
# ║  activated (--eval-drift true).                                    ║
# ║                                                                    ║
# ║  It includes safety checks to abort if `copy_state_dict`           ║
# ║  reports missing or mismatched keys, preventing silent failures.   ║
# ╚════════════════════════════════════════════════════════════════════╝

set -euo pipefail
ROOT=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
mkdir -p "${ROOT}/logs/slurm_logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU
cd "${ROOT}"

# 1. Gather all checkpoints from the three Idea-1 sweeps
CHECKPOINTS=()
while IFS=  read -r -d $'\0'; do
    CHECKPOINTS+=("$REPLY")
done < <(find logs/sweep_loss_ablation_Idea1_2 logs/sweep_loss_ablation_Idea1_EuclideanDom_2 logs/sweep_unified_finsler_idea1_gap -name "model_best.pth.tar" -print0 | sort -z)

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#CHECKPOINTS[@]}" ]; then
    echo "Task ID ${SLURM_ARRAY_TASK_ID} exceeds available checkpoints (${#CHECKPOINTS[@]}). Exiting cleanly."
    exit 0
fi

CKPT="${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}"
LOG_DIR=$(dirname "$CKPT")
RUN_NAME=$(basename "$LOG_DIR")

echo "=== Evaluation Task ${SLURM_ARRAY_TASK_ID} ==="
echo "  Checkpoint: ${CKPT}"

# 2. Parse proper architecture hooks from the directory name
if [[ "$RUN_NAME" == *"driftInst"* ]]; then
    DRT_CONDITIONING="instance"
    NUM_DOMAINS=0
elif [[ "$RUN_NAME" == *"driftDom"* ]]; then
    DRT_CONDITIONING="domain"
    # Protocol-2 (Market, MSMT, CUHK-SYSU -> CUHK03) uses 3 source domains
    NUM_DOMAINS=3
else
    echo "WARNING: Could not parse drift conditioning from path $RUN_NAME! Defaulting to instance."
    DRT_CONDITIONING="instance"
    NUM_DOMAINS=0
fi

TARGET_DATASET="cuhk03"
OUTPUT_LOG="${LOG_DIR}/eval_drift_true.log"

echo "  Parsed Conditioning: ${DRT_CONDITIONING}"
echo "  Parsed Num Domains:  ${NUM_DOMAINS}"
echo "  Output Log:          ${OUTPUT_LOG}"
echo "----------------------------------------"

# 3. Execute Standalone Evaluation
srun --unbuffered python examples/test.py \
  -a resnet50_finsler \
  -d "$TARGET_DATASET" \
  --resume "$CKPT" \
  --eval-drift true \
  --drift-method symmetric_trapezoidal \
  --drift-conditioning "$DRT_CONDITIONING" \
  --num-domains "$NUM_DOMAINS" \
  > "$OUTPUT_LOG" 2>&1

# 4. Strict warnings/validation to prevent mis-loading
if grep -q -i "mismatch" "$OUTPUT_LOG" || grep -q -i "missing keys" "$OUTPUT_LOG"; then
    echo "❌ ERROR: Weight loading mismatch detected in $CKPT!"
    echo "This indicates the architectural arguments passed to examples/test.py do not match the checkpoint."
    echo "Review the log for details: $OUTPUT_LOG"
    
    # Extract the exact mismatch lines to show in the SLURM output
    grep -i "mismatch" "$OUTPUT_LOG" || true
    grep -i "missing keys" "$OUTPUT_LOG" || true
    
    exit 1
fi

echo "✅ Evaluation completed successfully."
echo "Results:"
# Print the final mAP and Rank-1 from the log
grep -A 5 "Mean AP" "$OUTPUT_LOG" || echo "Could not parse Mean AP from log."
