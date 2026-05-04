#!/usr/bin/env bash
# PLAN.md toy experiment: M1 zero-shot eval + M2a/M2b checkpoint eval (mAP, Spearman, D8).
# Run from repo root. Uses BAU conda env (adjust if needed).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CKPT_DIR="${CKPT_DIR:-$ROOT/archive/logs/AGReIDv2_sweep/job_1502418_dt_cuhk03_EucBaseline}"
CKPT="${CKPT:-$CKPT_DIR/best.pth}"

DATA_DIR="${DATA_DIR:-$ROOT/examples/data/ToyCorruption}"
OUT_DIR="${OUT_DIR:-$ROOT/logs/toy_lmono_runs}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/toy_lmono}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

PYTHON="${PYTHON:-conda run -n BAU python}"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: BAU backbone checkpoint not found: $CKPT" >&2
  exit 1
fi

M2A_CKPT="${M2A_CKPT:-$OUT_DIR/m2a_lambda0.1_seed1.pth}"
M2B_CKPT="${M2B_CKPT:-$OUT_DIR/m2b_lambda0_seed1.pth}"

# --- M1: Euclidean BAU zero-shot (no θ head, no training) ---
# $PYTHON "$ROOT/examples/eval_toy_m1.py" \
#   --data-dir "$DATA_DIR" \
#   --checkpoint "$CKPT" \
#   --log-dir "$LOG_DIR"

# # --- M2a / M2b: load train_toy_lmono checkpoint; bidirectional mAP + Spearman + D8 ---
# if [[ ! -f "$M2A_CKPT" ]]; then
#   echo "WARN: M2a checkpoint missing (train first or set M2A_CKPT): $M2A_CKPT" >&2
# else
#   $PYTHON "$ROOT/examples/eval_toy_checkpoint.py" \
#     --resume "$M2A_CKPT" \
#     --data-dir "$DATA_DIR" \
#     --log-dir "$LOG_DIR"
# fi

# if [[ ! -f "$M2B_CKPT" ]]; then
#   echo "WARN: M2b checkpoint missing (train first or set M2B_CKPT): $M2B_CKPT" >&2
# else
#   $PYTHON "$ROOT/examples/eval_toy_checkpoint.py" \
#     --resume "$M2B_CKPT" \
#     --data-dir "$DATA_DIR" \
#     --log-dir "$LOG_DIR"
# fi

# --- Balanced per-severity eval (Section 8): 50×50 symmetric protocol, isolates content asymmetry ---
# M1: --no-theta forces Euclidean-only path (no theta_head in backbone checkpoint)
$PYTHON "$ROOT/examples/eval_toy_balanced.py" \
  --resume "$CKPT" \
  --data-dir "$DATA_DIR" \
  --log-dir "$LOG_DIR" \
  --no-theta

if [[ ! -f "$M2A_CKPT" ]]; then
  echo "WARN: M2a checkpoint missing (balanced eval skipped): $M2A_CKPT" >&2
else
  $PYTHON "$ROOT/examples/eval_toy_balanced.py" \
    --resume "$M2A_CKPT" \
    --data-dir "$DATA_DIR" \
    --log-dir "$LOG_DIR"
fi

if [[ ! -f "$M2B_CKPT" ]]; then
  echo "WARN: M2b checkpoint missing (balanced eval skipped): $M2B_CKPT" >&2
else
  $PYTHON "$ROOT/examples/eval_toy_balanced.py" \
    --resume "$M2B_CKPT" \
    --data-dir "$DATA_DIR" \
    --log-dir "$LOG_DIR"
fi
