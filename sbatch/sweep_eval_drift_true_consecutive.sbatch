#!/bin/bash
#SBATCH --job-name=eval-drift-true-seq
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --exclude=node7,node9,node1,node13
#SBATCH --output=logs/slurm_logs/eval-drift-true-seq-%j.out
#SBATCH --error=logs/slurm_logs/eval-drift-true-seq-%j.err
# If checkpoints are under the repo but you submit from elsewhere, uncomment and set:
# #SBATCH --chdir=/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification

# ╔════════════════════════════════════════════════════════════════════╗
# ║  Evaluation Sweep: --eval-drift true (Consecutive execution)       ║
# ║                                                                    ║
# ║  This script locates training checkpoints across the three Idea-1  ║
# ║  sweeps: **`best.pth`** (written by `examples/train_bau.py`) first,   ║
# ║  then **`model_best.pth.tar`** if present (legacy naming).          ║
# ║  sweeps, dynamically parses the required architectural hooks       ║
# ║  (e.g. drift conditioning, num domains) from the directory name,   ║
# ║  and evaluates the model with the Finsler drift explicitly         ║
# ║  activated (--eval-drift true).                                    ║
# ║                                                                    ║
# ║  It evaluates all models sequentially in a single job to prevent   ║
# ║  exceeding the QOSMaxSubmitJobPerUserLimit limit. If one eval      ║
# ║  fails, it will log the error and continue to the next model.      ║
# ╚════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# Repo root: prefer explicit override, then Slurm submit dir (where you ran sbatch from),
# then directory containing this script's parent (../ from sbatch/).
if [ -n "${REPO_ROOT:-}" ]; then
  ROOT="${REPO_ROOT}"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
fi

mkdir -p "${ROOT}/logs/slurm_logs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate BAU
cd "${ROOT}"

echo "Using REPO_ROOT=${ROOT} (override with REPO_ROOT=... sbatch ... if wrong)."

# 1. Gather checkpoints: search each sweep tree separately so a missing dir does not
#    fail the whole find (and so we avoid empty results from wrong cwd).
SWEEP_DIRS=(
  "logs/sweep_loss_ablation_Idea1_2"
  "logs/sweep_loss_ablation_Idea1_EuclideanDom_2"
  "logs/sweep_unified_finsler_idea1_gap"
)
CHECKPOINTS=()
for d in "${SWEEP_DIRS[@]}"; do
  if [ ! -d "${ROOT}/${d}" ] && [ ! -d "${d}" ]; then
    echo "WARNING: sweep log directory missing (skipped): ${d}"
    continue
  fi
  # train_bau.py saves the tracked best weights as `best.pth` (not model_best.pth.tar)
  while IFS= read -r -d '' f; do
    CHECKPOINTS+=("$f")
  done < <(find "${ROOT}/${d}" -type f \( -name "best.pth" -o -name "model_best.pth.tar" \) \
    ! -path "*/code_snapshot/*" -print0 2>/dev/null || true)
done

# Stable order (path sort; paths here do not contain newlines)
if [ "${#CHECKPOINTS[@]}" -gt 0 ]; then
  mapfile -t CHECKPOINTS < <(printf '%s\n' "${CHECKPOINTS[@]}" | sort -u)
fi

if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
  echo "ERROR: Found 0 checkpoints (expected best.pth or model_best.pth.tar per run dir) under the Idea-1 sweep trees."
  echo "Diagnostics (pwd=$(pwd)):"
  for d in "${SWEEP_DIRS[@]}"; do
    abs="${ROOT}/${d}"
    if [ -d "$abs" ]; then
      echo "  --- ${abs} ---"
      find "$abs" -maxdepth 2 -type d 2>/dev/null | head -n 20 || true
      echo "  (best.pth / *.pth under maxdepth 4, excluding code_snapshot:)"
      find "$abs" -maxdepth 4 -type f \( -name "best.pth" -o -name "*.pth" -o -name "*.pth.tar" \) \
        ! -path "*/code_snapshot/*" 2>/dev/null | head -n 40 || true
    else
      echo "  --- MISSING: ${abs}"
    fi
  done
  echo ""
  echo "Fix: run sbatch from the repo root (so SLURM_SUBMIT_DIR points here), or set REPO_ROOT to your clone."
  echo "This fork's trainer writes best.pth; extend the find -name list if you use another filename."
  exit 1
fi

echo "Found ${#CHECKPOINTS[@]} checkpoints to evaluate."
echo "Starting consecutive evaluation..."
echo "========================================"

SUCCESS_COUNT=0
FAIL_COUNT=0

for i in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$i]}"
    LOG_DIR=$(dirname "$CKPT")
    RUN_NAME=$(basename "$LOG_DIR")

    echo "=== Evaluation Task ${i} / $((${#CHECKPOINTS[@]} - 1)) ==="
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
    # We use set +e to ensure that a failure in one eval doesn't crash the entire job
    set +e
    srun --unbuffered python examples/test.py \
      -a resnet50_finsler \
      -d "$TARGET_DATASET" \
      --resume "$CKPT" \
      --eval-drift true \
      --drift-method symmetric_trapezoidal \
      --drift-conditioning "$DRT_CONDITIONING" \
      --num-domains "$NUM_DOMAINS" \
      > "$OUTPUT_LOG" 2>&1
    EVAL_EXIT_CODE=$?
    set -e

    # 4. Strict warnings/validation to prevent mis-loading
    if [ $EVAL_EXIT_CODE -ne 0 ]; then
        echo "❌ ERROR: Evaluation failed with exit code $EVAL_EXIT_CODE!"
        echo "Review the log for details: $OUTPUT_LOG"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    elif grep -q -i "mismatch" "$OUTPUT_LOG" || grep -q -i "missing keys" "$OUTPUT_LOG"; then
        echo "❌ ERROR: Weight loading mismatch detected in $CKPT!"
        echo "This indicates the architectural arguments passed to examples/test.py do not match the checkpoint."
        echo "Review the log for details: $OUTPUT_LOG"
        
        # Extract the exact mismatch lines to show in the SLURM output
        grep -i "mismatch" "$OUTPUT_LOG" || true
        grep -i "missing keys" "$OUTPUT_LOG" || true
        
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        echo "✅ Evaluation completed successfully."
        echo "Results:"
        # Print the final mAP and Rank-1 from the log
        grep -A 5 "Mean AP" "$OUTPUT_LOG" || echo "Could not parse Mean AP from log."
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    fi
    echo "========================================"
    echo ""
done

echo "Finished consecutive evaluation!"
echo "Total Evaluated: ${#CHECKPOINTS[@]}"
echo "Success: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
