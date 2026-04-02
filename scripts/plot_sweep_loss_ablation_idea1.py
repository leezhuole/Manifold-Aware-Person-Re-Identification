#!/usr/bin/env python3
"""
Publication-style figure for sweep ``sweep_loss_ablation_Idea1`` (job 1512947).

Metrics: final evaluation on CUHK03 (Market1501 + MSMT17 + CUHK-SYSU sources),
``resnet50_finsler``, identity-only eval (``--eval-drift false``), Euclidean
identity triplet (1a), 60 epochs, seed 1.

Outputs (default): ``results/plots/sweep_loss_ablation_Idea1.{pdf,png,svg}``
  (SVG for viewers that mishandle PDF font embedding; PNG uses opaque RGB.)

Example:
  python scripts/plot_sweep_loss_ablation_idea1.py --output-dir results/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Final metrics (best checkpoint, identity-only eval) — job_1512947_*
# ---------------------------------------------------------------------------
# Short x-labels for readability at slide size; full names in figure note / paper caption.
ARMS = (
    "Baseline\n(1a only)",
    "+ Camera\ncontrast (1b)",
    "Domain triplet\n(no mem. dom. loss)",
    "+ Drift-only\nL$_2$ (xcam)",
    "Domain triplet\n(+ mem. dom. loss)",
)

MAP_INST = (43.7, 42.2, 24.6, 43.0, 35.5)
MAP_DOM = (44.1, 42.8, 24.6, 42.8, 36.2)
R1_INST = (43.4, 42.2, 30.4, 43.1, 36.4)
R1_DOM = (45.0, 44.1, 30.4, 41.9, 36.1)


def setup_style() -> None:
    """Publication-style; fonts bundled with Matplotlib so exports open on any OS."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            # DejaVu ships with matplotlib — avoids missing Times on Linux / headless CI
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "DejaVu Sans"],
            "mathtext.fontset": "dejavuserif",
            # TrueType in PDF so Acrobat / PowerPoint do not substitute broken Type 3
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.45,
            "axes.axisbelow": True,
        }
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plots"),
        help="Directory for PDF/PNG output",
    )
    p.add_argument("--basename", type=str, default="sweep_loss_ablation_Idea1")
    p.add_argument(
        "--formats",
        type=str,
        default="pdf,png,svg",
        help="Comma-separated: pdf, png, svg",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Okabe–Ito blue / vermillion — distinguishable when printed grayscale + CVD
    c_inst = "#0072B2"
    c_dom = "#D55E00"
    n = len(ARMS)
    x = np.arange(n, dtype=float)
    w = 0.36

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(8.0, 3.6),
        constrained_layout=True,
        sharey=False,
    )
    fig.patch.set_facecolor("white")

    def grouped_bars(ax, y_inst, y_dom, ylabel: str, letter: str) -> None:
        ax.bar(
            x - w / 2,
            y_inst,
            width=w,
            color=c_inst,
            edgecolor="0.15",
            linewidth=0.5,
            zorder=3,
        )
        ax.bar(
            x + w / 2,
            y_dom,
            width=w,
            color=c_dom,
            edgecolor="0.15",
            linewidth=0.5,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(ARMS, ha="center")
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
        ax.grid(True, axis="y", linestyle="--", color="0.75")
        ax.set_axisbelow(True)
        ax.set_xlim(x.min() - 0.65, x.max() + 0.65)

        # Panel label
        ax.text(
            0.0,
            1.02,
            letter,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
    grouped_bars(ax0, MAP_INST, MAP_DOM, "mAP (\\%)", "(a)")
    grouped_bars(ax1, R1_INST, R1_DOM, "Rank-1 (\\%)", "(b)")

    ax0.set_ylim(20, 48)
    ax1.set_ylim(28, 48)

    handles = [
        mpl.patches.Patch(facecolor=c_inst, edgecolor="0.15", linewidth=0.5, label="Instance drift"),
        mpl.patches.Patch(facecolor=c_dom, edgecolor="0.15", linewidth=0.5, label="Domain-conditioned drift"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        columnspacing=1.8,
    )

    fig.suptitle(
        "Loss ablation: Idea-1 extensions ($N{=}1$ seed, CUHK03 target)",
        fontsize=11,
        y=1.09,
    )

    note = (
        "Sources: Market1501, MSMT17, CUHK-SYSU $\\rightarrow$ target CUHK03. "
        "Architecture: \\texttt{resnet50\\_finsler}. "
        "Euclidean identity triplet (Idea 1a); test-time ranking: identity features only. "
        "Single run per configuration (seed 1); bars are not error bars."
    )
    fig.text(0.5, -0.04, note, ha="center", va="top", fontsize=7.5, style="italic")

    base = args.output_dir / args.basename
    save_kw = dict(
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor="white",
        edgecolor="none",
    )
    written = []
    for fmt in [f.strip().lower() for f in args.formats.split(",") if f.strip()]:
        if fmt not in ("pdf", "png", "svg"):
            raise SystemExit(f"Unknown format {fmt!r}; use pdf, png, or svg")
        path = f"{base}.{fmt}"
        dpi = 300 if fmt == "png" else None
        fig.savefig(path, format=fmt, dpi=dpi, **save_kw)
        written.append(path)
    plt.close(fig)
    print("Wrote:\n  " + "\n  ".join(written))


if __name__ == "__main__":
    main()
