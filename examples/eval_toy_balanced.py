# encoding: utf-8
"""Balanced per-severity bidirectional eval (Section 8, toy_lmono_diagnostic_analysis.md).

For each severity k ∈ {0,1,2,3,4}:
  Direction A: query = source-1 σ=0  (50 items) → gallery = source-2 σ=k  (50 items)
  Direction B: query = source-2 σ=k  (50 items) → gallery = source-1 σ=0  (50 items)

Both directions: exactly 50 queries, 50 gallery items, 1 correct match per query.
k=0 is the σ=0 vs σ=0 sanity check; expected Δ≈0 under Euclidean distance.

NOTE ON D8 / Randers gap: ``_collect_d8_pairs`` matches by **PID** and severity (σ=0 vs σ=k);
cameras may differ. In balanced eval, clean queries are source-1 σ=0 and the per-k gallery
is source-2 σ=k, so gaps are **cross-camera PID-matched** pairs (one per identity per k when
both views exist). Logged on each Randers line as ``D8 mean_gap=…`` / ``by_severity``.

Usage examples
--------------
M2a/M2b checkpoint:
    python examples/eval_toy_balanced.py \
        --resume logs/toy_lmono_runs/m2a_lambda0.1_seed1.pth \
        --data-dir examples/data/ToyCorruption

M1 (Euclidean backbone, no θ head):
    python examples/eval_toy_balanced.py \
        --resume <path/to/best.pth> \
        --data-dir examples/data/ToyCorruption \
        --no-theta
"""

from __future__ import absolute_import, print_function

import argparse
import os.path as osp
import sys
import time
from datetime import datetime

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

from bau import datasets
from bau import models
from bau.evaluators import (
    bidirectional_evaluate,
    extract_features,
    spearman_rho_theta_severity,
)
from bau.utils.data.preprocessor import Preprocessor
from bau.utils.data import transforms as T
from bau.utils.osutils import mkdir_if_missing
from bau.utils.serialization import copy_state_dict, load_checkpoint
from bau.utils.stdout_tee import StdoutTee


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_alpha_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _state_dict_from_checkpoint(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj:
            return ckpt_obj["state_dict"]
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
    return ckpt_obj


def _strip_module_prefix(sd):
    if not isinstance(sd, dict):
        return sd
    if not any(k.startswith("module.") for k in sd):
        return sd
    return {k.replace("module.", "", 1): v for k, v in sd.items()}


def _resolve_log_path(args):
    if args.log_file and str(args.log_file).strip():
        p = osp.abspath(osp.expanduser(args.log_file.strip()))
        d = osp.dirname(p)
        if d:
            mkdir_if_missing(d)
        return p
    log_dir = osp.abspath(osp.expanduser(args.log_dir))
    mkdir_if_missing(log_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = osp.splitext(osp.basename(args.resume))[0]
    return osp.join(log_dir, "eval_toy_balanced_{}_{}.log".format(base, stamp))


def build_eval_loader(eval_rows, batch_size, workers, height, width):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_t = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])
    return DataLoader(
        Preprocessor(sorted(eval_rows), root=None, transform=test_t),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args):
    """Return (model, return_theta).

    If the checkpoint is a dict with a "model" key it is treated as an
    M2a/M2b checkpoint from train_toy_lmono.py (toy_resnet50 with theta_head).
    Everything else is treated as an M1-style state-dict (resnet50, no theta_head).
    --no-theta overrides and forces return_theta=False regardless of checkpoint format.
    """
    ckpt = load_checkpoint(args.resume)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model = models.create("toy_resnet50", num_classes=0, pretrained=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print("load_state_dict missing keys: {}".format(missing))
        if unexpected:
            print("load_state_dict unexpected keys: {}".format(unexpected))
        return_theta = not args.no_theta
    else:
        model = models.create(
            "resnet50", num_classes=0, pretrained=False, with_theta_head=False
        )
        sd = _strip_module_prefix(_state_dict_from_checkpoint(ckpt))
        copy_state_dict(sd, model, strip=None)
        return_theta = False

    for par in model.parameters():
        par.requires_grad_(False)
    model.eval()
    model.cuda()
    return model, return_theta


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _fmt_d8(d8):
    """Same layout as ``examples/eval_toy_checkpoint._fmt_d8`` for Randers gap lines."""
    if not d8:
        return "D8=<empty>"
    mg = d8.get("mean_gap")
    mg_txt = "nan" if mg is None or (isinstance(mg, float) and mg != mg) else "{:.6f}".format(mg)
    parts = ["mean_gap={}".format(mg_txt), "n_pairs={}".format(d8.get("n_pairs", 0))]
    by_sev = d8.get("mean_gap_by_severity") or {}
    if by_sev:
        sev_bits = []
        for k in sorted(by_sev.keys()):
            v = by_sev[k]
            if v is None or (isinstance(v, float) and v != v):
                sev_bits.append("{}:nan".format(k))
            else:
                sev_bits.append("{}:{:.6f}".format(k, v))
        parts.append("by_severity={" + ", ".join(sev_bits) + "}")
    return "D8 " + " ".join(parts)


def _fmt_metrics(label, metrics, alpha_values):
    """Format one severity-block of results."""
    eu = metrics["euclidean"]
    lines = [
        "  Euclidean:     mAP A={:.4f}  B={:.4f}  Δ={:+.4f}".format(
            eu["direction_A"]["mAP"], eu["direction_B"]["mAP"], eu["delta_mAP"]
        )
    ]
    for a in alpha_values:
        ra = metrics["randers"][a]
        d8 = ra.get("d8_mean_asymmetric_gap", {})
        lines.append(
            "  Randers α={:<4} mAP A={:.4f}  B={:.4f}  Δ={:+.4f} | {}".format(
                a,
                ra["direction_A"]["mAP"],
                ra["direction_B"]["mAP"],
                ra["delta_mAP"],
                _fmt_d8(d8),
            )
        )
    return "\n".join(lines)


def _run_severity(model, dataset, k, alpha_values, return_theta, args):
    query_A = dataset.by_source_severity[(1, 0)]
    gallery_A = dataset.by_source_severity[(2, k)]
    all_rows = sorted(set(query_A) | set(gallery_A))
    loader = build_eval_loader(
        all_rows, args.eval_batch_size, args.workers, args.height, args.width
    )
    metrics = bidirectional_evaluate(
        model,
        loader,
        query_clean=query_A,
        gallery_corrupted=gallery_A,
        query_corrupted=gallery_A,
        gallery_clean=query_A,
        alpha_values=alpha_values,
        return_theta=return_theta,
        print_freq=args.eval_print_freq,
        verbose=False,
    )
    metrics.pop("theta_by_fname", None)
    return metrics


def _compute_spearman(model, dataset, args):
    """Extract θ for all 500 bounding_box_test images and compute Spearman ρ(θ, severity)."""
    all_items = []
    for rows in dataset.by_source_severity.values():
        all_items.extend(rows)
    all_items = sorted(set(all_items))
    loader = build_eval_loader(
        all_items, args.eval_batch_size, args.workers, args.height, args.width
    )
    _, _, theta_dict = extract_features(
        model, loader, print_freq=args.eval_print_freq, return_theta=True
    )
    paths = [t[0] for t in all_items]
    return spearman_rho_theta_severity(theta_dict, paths)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA is required.", file=sys.stderr)
        sys.exit(1)

    log_path = _resolve_log_path(args)
    print("Logging to: {}".format(log_path), file=sys.stderr)
    log_fp = open(log_path, "a", buffering=1)
    old_stdout = sys.stdout
    sys.stdout = StdoutTee(old_stdout, log_fp)
    try:
        print("========== eval_toy_balanced ==========")
        print("time_utc={} log_file={}".format(
            time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), log_path
        ))
        print("argv={}".format(sys.argv))
        print("args={}".format(vars(args)))

        cudnn.benchmark = True

        dataset = datasets.create("toy_corruption", args.data_dir)
        model, return_theta = load_model(args)
        alpha_values = _parse_alpha_list(args.alpha)

        print()
        print(
            "NOTE: Randers gap (D8) pairs are PID-matched clean (source-1, σ=0) vs "
            "corrupted gallery (source-2, σ=k); cameras may differ."
        )
        print("return_theta={}".format(return_theta))
        print()

        severities = [0, 1, 2, 3, 4]
        results_per_k = {}

        for k in severities:
            if k == 0:
                label = "severity k=0 (σ=0 vs σ=0 sanity check — expect Δ≈0 under Euclidean)"
            else:
                label = "severity k={} (σ=0 vs σ={})".format(k, k)
            print("--- {} ---".format(label))
            metrics = _run_severity(
                model, dataset, k, alpha_values, return_theta, args
            )
            results_per_k[k] = metrics
            print(_fmt_metrics(label, metrics, alpha_values))
            print()

        # --- Summary table ---
        print("--- Summary: balanced per-severity Δ (mean over k=1..4) ---")
        main_ks = [1, 2, 3, 4]
        eu_deltas = [results_per_k[k]["euclidean"]["delta_mAP"] for k in main_ks]
        mean_eu_delta = sum(eu_deltas) / len(eu_deltas)
        per_k_eu = "  ".join("k{}={:+.4f}".format(k, d) for k, d in zip(main_ks, eu_deltas))
        print("  Euclidean:     mean_Δ={:+.4f}  [{}]".format(mean_eu_delta, per_k_eu))
        for a in alpha_values:
            r_deltas = [results_per_k[k]["randers"][a]["delta_mAP"] for k in main_ks]
            mean_r_delta = sum(r_deltas) / len(r_deltas)
            per_k_r = "  ".join("k{}={:+.4f}".format(k, d) for k, d in zip(main_ks, r_deltas))
            print("  Randers α={:<4} mean_Δ={:+.4f}  [{}]".format(a, mean_r_delta, per_k_r))

        # --- Spearman (M2a/M2b only) ---
        if return_theta:
            print()
            print("Computing Spearman ρ(θ, severity) over all 500 bounding_box_test images ...")
            rho = _compute_spearman(model, dataset, args)
            print("Spearman rho(theta, severity) = {:.4f}".format(rho))
            print("(L_mono target: strongly negative, e.g. ρ ≤ −0.8)")

        print()
        print("========== eval_toy_balanced finished ==========")
    finally:
        sys.stdout = old_stdout
        log_fp.close()
        print("Log closed: {}".format(log_path), file=old_stdout)


parser = argparse.ArgumentParser(
    description="Balanced per-severity toy eval (Section 8 of diagnostic analysis)"
)
parser.add_argument(
    "--resume", type=str, required=True,
    help="Checkpoint: dict with 'model' key (M2a/M2b) or state-dict (M1)."
)
parser.add_argument("--data-dir", type=str, required=True)
parser.add_argument("--eval-batch-size", type=int, default=64)
parser.add_argument(
    "--workers", type=int, default=0,
    help="DataLoader workers (0 avoids fork+CUDA issues).",
)
parser.add_argument("--height", type=int, default=256)
parser.add_argument("--width", type=int, default=128)
parser.add_argument(
    "--alpha", type=str, default="0.0,0.1,0.3,0.5,0.9",
    help="Comma-separated Randers α values.",
)
parser.add_argument("--eval-print-freq", type=int, default=100)
parser.add_argument(
    "--log-dir", type=str, default="logs/toy_lmono",
    help="Directory for auto-named log files when --log-file is not set.",
)
parser.add_argument(
    "--log-file", type=str, default="",
    help="Explicit log path (stdout tee'd here for the whole run).",
)
parser.add_argument(
    "--no-theta", action="store_true",
    help="Force return_theta=False (M1 mode). Auto-set when checkpoint has no 'model' key.",
)

if __name__ == "__main__":
    main()
