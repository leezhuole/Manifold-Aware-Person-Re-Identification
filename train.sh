#!/usr/bin/env bash
# PLAN.md toy experiment: M1 (Euclidean zero-shot), M2a (L_CE + λ L_mono), M2b (L_CE only).
# Run from repo root. Uses BAU conda env (adjust if needed).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Default: job directory you named (often only log.txt is archived; weights are usually best.pth there).
CKPT_DIR="${CKPT_DIR:-$ROOT/archive/logs/AGReIDv2_sweep/job_1502418_dt_cuhk03_EucBaseline}"
CKPT="${CKPT:-$CKPT_DIR/best.pth}"

DATA_DIR="${DATA_DIR:-$ROOT/examples/data/ToyCorruption}"
OUT_DIR="${OUT_DIR:-$ROOT/logs/toy_lmono_runs}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/toy_lmono}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

PYTHON="${PYTHON:-conda run -n BAU python}"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  echo "  The sweep log saved weights under logs_dir (see that job's log.txt)." >&2
  echo "  Copy best.pth (or model_best.pth.tar after conversion) here or set CKPT=..." >&2
  exit 1
fi

# --- M1: Euclidean BAU zero-shot (no θ head, no training) ---
# $PYTHON "$ROOT/examples/eval_toy_m1.py" \
#   --data-dir "$DATA_DIR" \
#   --checkpoint "$CKPT" \
#   --log-dir "$LOG_DIR"

# --- M2a: frozen backbone + θ head, L_CE + λ_mono * L_mono (λ > 0) ---
$PYTHON "$ROOT/examples/train_toy_lmono.py" \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CKPT" \
  --arch toy_resnet50 \
  --lambda-mono 1 \
  --margin 0.1 \
  --alpha "0.0,0.1,0.3,0.5,0.9" \
  --epochs 20 \
  --batch-size 32 \
  --num-instances 5 \
  --seed 1 \
  --log-dir "$LOG_DIR" \
  --save-path "$OUT_DIR/m2a_lambda1.0_seed1.pth"

$PYTHON "$ROOT/examples/train_toy_lmono.py" \
  --data-dir "$DATA_DIR" \
  --checkpoint "$CKPT" \
  --arch toy_resnet50 \
  --lambda-mono 2.0 \
  --margin 0.1 \
  --alpha "0.0,0.1,0.3,0.5,0.9" \
  --epochs 20 \
  --batch-size 32 \
  --num-instances 5 \
  --seed 1 \
  --log-dir "$LOG_DIR" \
  --save-path "$OUT_DIR/m2a_lambda2.0_seed1.pth"

# --- M2b: same as M2a but λ_mono = 0 (ablation: θ unconstrained by L_mono) ---
# $PYTHON "$ROOT/examples/train_toy_lmono.py" \
#   --data-dir "$DATA_DIR" \
#   --checkpoint "$CKPT" \
#   --arch toy_resnet50 \
#   --lambda-mono 0.0 \
#   --margin 0.1 \
#   --alpha "0.0,0.1,0.3,0.5,0.9" \
#   --epochs 20 \
#   --batch-size 32 \
#   --num-instances 5 \
#   --seed 1 \
#   --log-dir "$LOG_DIR" \
#   --save-path "$OUT_DIR/m2b_lambda0_seed1.pth"
