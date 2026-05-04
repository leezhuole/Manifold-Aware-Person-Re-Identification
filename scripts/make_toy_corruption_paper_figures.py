#!/usr/bin/env python3
"""
Camera-ready **individual** PDFs for ToyCorruption / L_mono (PLAN.md §2 + §4.4).

Each visual is its own file (no multi-panel PDFs); compose side-by-side in LaTeX if needed.

**Intro (§2):** from ``eval_toy_balanced*.log`` (default M1 via ``no_theta: True``)
  - ``fig_balanced_asymmetry_intro_strip.pdf`` — severity ladder only
  - ``fig_balanced_asymmetry_intro_delta.pdf`` — Euclidean Δ mAP[k] bars only

**Section 4.4:** mean Randers gap vs σ from **balanced** ``eval_toy_balanced*.log`` (default: newest
  M2a checkpoint with ``m2a_lambda0.1`` in the logged ``resume`` path). Parses the
  ``D8 mean_gap=…`` field on the ``Randers α=<alpha>`` line inside each ``--- severity k=…`` block
  (PID-matched clean σ=0 vs corrupted σ=k; cameras may differ).

  - ``fig_toy_mechanism_strip.pdf`` — same ladder (standalone)
  - ``fig_toy_mechanism_gap.pdf`` — mean Randers gap vs σ only

Usage (repo root):
    python scripts/make_toy_corruption_paper_figures.py all
    python scripts/make_toy_corruption_paper_figures.py intro sec44-mechanism
    python scripts/make_toy_corruption_paper_figures.py --log-dir logs/toy_lmono --out-dir paper
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import os.path as osp
import re
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_FONT_LABEL = 9
_FONT_TICK = 8

_SEVERITY_HEAD = re.compile(r"--- severity k=(\d+)")
_EUCL_DELTA = re.compile(
    r"^\s+Euclidean:\s+mAP A=[\d.]+\s+B=[\d.]+\s+Δ=([+-]?[\d.]+)"
)
_SEVERITY_BLOCK = re.compile(r"--- severity k=(\d+).*?(?=--- severity k=|\Z)", re.DOTALL)


def _repo_root() -> str:
    return osp.dirname(osp.dirname(osp.abspath(__file__)))


def list_balanced_eval_logs(log_dir: str) -> list[str]:
    pattern = osp.join(log_dir, "eval_toy_balanced*.log")
    return sorted(glob.glob(pattern))


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fp:
        return fp.read()


def _args_dict_from_log(text: str) -> dict | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("args=") and line.startswith("args={"):
            payload = line[len("args=") :]
            try:
                return ast.literal_eval(payload)
            except (SyntaxError, ValueError):
                return None
    return None


def pick_m1_balanced_log(log_dir: str) -> str:
    candidates = list_balanced_eval_logs(log_dir)
    if not candidates:
        raise FileNotFoundError(f"No eval_toy_balanced*.log under {log_dir!r}")
    m1: list[tuple[float, str]] = []
    for path in candidates:
        args = _args_dict_from_log(_read_text(path)[:16384])
        if args is not None and args.get("no_theta") is True:
            m1.append((os.path.getmtime(path), path))
    if not m1:
        raise RuntimeError(
            f"No M1 balanced log (no_theta: True) under {log_dir!r}. Use --intro-eval-log."
        )
    m1.sort(key=lambda x: x[0])
    return m1[-1][1]


def pick_balanced_by_resume_substr(log_dir: str, substr: str) -> str:
    """Newest balanced log whose ``args['resume']`` contains ``substr``."""
    hits: list[tuple[float, str]] = []
    for path in list_balanced_eval_logs(log_dir):
        args = _args_dict_from_log(_read_text(path)[:16384])
        if not args:
            continue
        if substr in str(args.get("resume", "")):
            hits.append((os.path.getmtime(path), path))
    if not hits:
        raise FileNotFoundError(
            f"No eval_toy_balanced*.log under {log_dir!r} with resume containing {substr!r}"
        )
    hits.sort(key=lambda x: x[0])
    return hits[-1][1]


def parse_euclidean_delta_per_k(log_text: str) -> dict[int, float]:
    deltas: dict[int, float] = {}
    pending_k: int | None = None
    for line in log_text.splitlines():
        m = _SEVERITY_HEAD.search(line)
        if m:
            pending_k = int(m.group(1))
            continue
        if pending_k is None:
            continue
        m2 = _EUCL_DELTA.match(line)
        if m2:
            deltas[pending_k] = float(m2.group(1))
            pending_k = None
    return deltas


def parse_balanced_randers_mean_gap_per_k(log_text: str, alpha: float) -> dict[int, float]:
    """
    From ``eval_toy_balanced`` output: for each ``--- severity k=σ`` block (σ≥1), read
    ``D8 mean_gap`` on the Randers line at the requested α (finite values only).
    """
    rx_line = re.compile(
        rf"^\s+Randers\s+[\u03b1a]={alpha:g}\s+.+\|\s*D8 mean_gap=([-+]?[\d.]+|nan)",
        re.MULTILINE,
    )
    out: dict[int, float] = {}
    for m in _SEVERITY_BLOCK.finditer(log_text):
        k = int(m.group(1))
        if k == 0:
            continue
        block = m.group(0)
        mm = rx_line.search(block)
        if not mm:
            continue
        val_s = mm.group(1)
        if val_s.lower() == "nan":
            continue
        out[k] = float(val_s)
    return out


def _strip_image_paths(dataset_dir: str) -> list[str]:
    bb_dir = osp.join(dataset_dir, "bounding_box_test")

    def _fname(sev: int) -> str:
        v3 = f"0001_c1s{sev + 1}_000001_01.jpg"
        v2 = f"0001_c{sev + 1}s1_000001_01.jpg"
        if osp.exists(osp.join(bb_dir, v3)):
            return v3
        return v2

    return [osp.join(bb_dir, _fname(s)) for s in range(5)]


def make_severity_strip_pdf(
    dataset_dir: str,
    output_path: str,
    *,
    panel_title: str = "Controlled ToyCorruption severity ladder",
) -> None:
    """Five-image strip only (same crops as ``toy_paper_figures.make_corruption_strip``)."""
    fig = plt.figure(figsize=(5.5, 2.0))
    gs = fig.add_gridspec(1, 5, wspace=0.06, left=0.02, right=0.98, top=0.82, bottom=0.06)
    paths = _strip_image_paths(dataset_dir)
    for i in range(5):
        ax = fig.add_subplot(gs[0, i])
        p = paths[i]
        if not osp.isfile(p):
            raise FileNotFoundError(f"ToyCorruption image missing: {p}")
        ax.imshow(Image.open(p).convert("RGB"))
        ax.axis("off")
        ax.set_title(f"$\\sigma={i}$", fontsize=_FONT_LABEL, pad=2)
    fig.suptitle(panel_title, fontsize=_FONT_LABEL, y=0.98)
    os.makedirs(osp.dirname(osp.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def make_intro_delta_bars_pdf(delta_by_k: dict[int, float], output_path: str) -> None:
    """Euclidean Δ mAP (Dir. A − Dir. B) vs k; uniform bar color; tight asymmetric y margins."""
    ks = list(range(5))
    deltas = [delta_by_k[k] for k in ks]
    bar_color = "#546e7a"

    fig, ax = plt.subplots(figsize=(3.15, 2.35))
    x = np.arange(len(ks))
    ax.bar(x, deltas, color=bar_color, width=0.72, edgecolor="0.2", linewidth=0.45)
    ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle="-", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks], fontsize=_FONT_TICK)
    ax.set_xlabel("Corruption severity $k$", fontsize=_FONT_LABEL)
    ax.set_ylabel(r"$\Delta$ mAP (Dir. A $-$ Dir. B)", fontsize=_FONT_LABEL)
    ax.tick_params(axis="y", labelsize=_FONT_TICK)

    lo, hi = min(deltas), max(deltas)
    span = max(hi - lo, 1e-6)
    pad = max(0.01, 0.06 * span)
    ax.set_ylim(lo - pad, hi + pad)

    os.makedirs(osp.dirname(osp.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def make_toy_mechanism_gap_pdf(
    gap_by_sigma: dict[int, float],
    output_path: str,
    *,
    alpha: float,
) -> None:
    sigmas = sorted(gap_by_sigma.keys())
    vals = [gap_by_sigma[s] for s in sigmas]

    fig, ax = plt.subplots(figsize=(3.15, 2.35))
    x = np.arange(len(sigmas))
    ax.bar(x, vals, color="#1565c0", width=0.72, edgecolor="0.25", linewidth=0.4)
    ax.axhline(0.0, color="0.35", linewidth=0.85, linestyle="-", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sigmas], fontsize=_FONT_TICK)
    ax.set_xlabel(r"Corruption severity $\sigma$", fontsize=_FONT_LABEL)
    ax.set_ylabel("Mean Randers gap ($\\alpha=%g$)" % alpha, fontsize=_FONT_LABEL)
    ax.tick_params(axis="y", labelsize=_FONT_TICK)
    lo, hi = min(vals), max(0.0, max(vals))
    span = max(hi - lo, 1e-6)
    pad = max(0.012, 0.06 * span)
    ax.set_ylim(lo - pad, hi + pad)

    os.makedirs(osp.dirname(osp.abspath(output_path)) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _ensure_balanced_log(path: str, role: str) -> None:
    base = osp.basename(path)
    if not base.startswith("eval_toy_balanced"):
        print(
            f"Warning: expected eval_toy_balanced*.log for {role}, got {base!r}",
            file=sys.stderr,
        )


def run_intro(args: argparse.Namespace) -> None:
    log_path = args.intro_eval_log.strip() or pick_m1_balanced_log(args.log_dir)
    _ensure_balanced_log(log_path, "intro")
    log_text = _read_text(log_path)
    delta_by_k = parse_euclidean_delta_per_k(log_text)
    if not set(range(5)).issubset(delta_by_k):
        raise ValueError(f"Missing k in Euclidean Δ from {log_path}: {delta_by_k!r}")

    out_strip = osp.join(args.out_dir, "fig_balanced_asymmetry_intro_strip.pdf")
    out_delta = osp.join(args.out_dir, "fig_balanced_asymmetry_intro_delta.pdf")
    make_severity_strip_pdf(args.dataset_dir, out_strip)
    make_intro_delta_bars_pdf(delta_by_k, out_delta)
    print(f"[intro] → {out_strip}")
    print(f"[intro] → {out_delta} (log {log_path})")


def run_sec44_mechanism(args: argparse.Namespace) -> None:
    log_path = args.mechanism_eval_log.strip() or pick_balanced_by_resume_substr(
        args.log_dir, "m2a_lambda0.1"
    )
    _ensure_balanced_log(log_path, "sec44-mechanism")
    log_text = _read_text(log_path)
    gap_by_sigma = parse_balanced_randers_mean_gap_per_k(log_text, args.randers_alpha)
    need = {1, 2, 3, 4}
    if not need.issubset(gap_by_sigma.keys()):
        raise ValueError(f"Missing σ in Randers gap parse from {log_path}: {gap_by_sigma!r}")

    out_strip = osp.join(args.out_dir, "fig_toy_mechanism_strip.pdf")
    out_gap = osp.join(args.out_dir, "fig_toy_mechanism_gap.pdf")
    make_severity_strip_pdf(args.dataset_dir, out_strip, panel_title="ToyCorruption severity ladder")
    make_toy_mechanism_gap_pdf(gap_by_sigma, out_gap, alpha=args.randers_alpha)
    print(f"[sec44-mechanism] → {out_strip}")
    print(f"[sec44-mechanism] → {out_gap} (balanced log: {log_path}, α={args.randers_alpha})")


def main() -> None:
    root = _repo_root()
    if os.getcwd() != root:
        os.chdir(root)

    fig_choices = ("intro", "sec44-mechanism", "all")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "figures",
        nargs="*",
        default=["all"],
        help=f"One or more of: {', '.join(fig_choices)} (default: all)",
    )
    parser.add_argument(
        "--log-dir",
        default=osp.join(root, "logs", "toy_lmono"),
        help="Directory containing eval_toy_balanced*.log",
    )
    parser.add_argument(
        "--out-dir",
        default=osp.join(root, "paper"),
        help="Output directory for PDFs",
    )
    parser.add_argument(
        "--dataset-dir",
        default=osp.join(root, "examples", "data", "ToyCorruption"),
        help="ToyCorruption root",
    )
    parser.add_argument(
        "--intro-eval-log",
        default="",
        help="Balanced eval log for intro Δ[k] bars (default: newest M1 / no_theta)",
    )
    parser.add_argument(
        "--mechanism-eval-log",
        default="",
        help="Balanced M2a eval log for Randers gap bars (default: newest resume containing m2a_lambda0.1)",
    )
    parser.add_argument(
        "--randers-alpha",
        type=float,
        default=0.9,
        help="α for Randers gap panel (default: 0.9)",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    targets: Iterable[str]
    if "all" in args.figures or not args.figures:
        targets = ("intro", "sec44-mechanism")
    else:
        for f in args.figures:
            if f not in fig_choices:
                parser.error(f"unknown figure target {f!r}; choose from {fig_choices}")
        targets = args.figures

    for name in targets:
        if name == "all":
            continue
        if name == "intro":
            run_intro(args)
        elif name == "sec44-mechanism":
            run_sec44_mechanism(args)


if __name__ == "__main__":
    main()
