#!/usr/bin/env python3
"""Parse loss ablation logs and generate publication-style plots.

This script scans a loss-ablation log root (default:
``logs/finsler_duet_rng``), extracts the *last* recorded ``Mean AP`` and
``Rank-1/top-1`` values from each run directory, groups them by the loss-term
configuration found in the folder name, saves a CSV, and produces per-metric
plots (mAP and Rank-1) with mean±std error bars.

Example:

python scripts/generate_plot_mean_variance_lossterms.py \
    --log-root /home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/logs/ablationFinsler-P2 \
    --output-csv loss_ablation_results.csv \
    --output-plot loss_ablation_plot.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


# Default locations (override via CLI if needed)
DEFAULT_LOG_ROOT = Path(
    "/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/logs/finsler_duet_rng"
)
DEFAULT_CSV = Path("loss_ablation_results.csv")
DEFAULT_PLOT = Path("loss_ablation_plot.png")

# Ordering of loss configs (matches sbatch LOSS_NAMES ordering)
LOSS_ORDER: List[str] = ["L_ce", "L_triplet", "L_align", "L_uniform", "L_domain"]
BASELINE_NAMES: List[str] = ["alphaNone"]

# Directory naming pattern from sbatch: job_${JOBID}_${TASKID}_${LOSS}_run${RUN}
RUN_DIR_PATTERN = re.compile(r"job_\d+_\d+_(?P<loss>[A-Za-z0-9_]+)_run(?P<run>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan loss-ablation logs, export CSV, and plot mAP/Rank-1 versus loss term."
        )
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help="Root directory containing run folders (default: logs/ablationFinsler-P2)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to save extracted metrics CSV (default: loss_ablation_results.csv)",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=DEFAULT_PLOT,
        help="Path to save the plots (default: loss_ablation_plot.png)",
    )
    parser.add_argument(
        "--csv-decimals",
        type=int,
        default=3,
        help="Decimal places for floating values in the saved CSV (default: 3)",
    )
    return parser.parse_args()


def setup_plot_style() -> None:
    """Configure plotting aesthetics for publication-style figures."""

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["text.usetex"] = False


def _normalize_percentage(val: Optional[float], has_percent_token: bool) -> Optional[float]:
    """Return percentage value, respecting whether the log already had a % sign."""

    if val is None:
        return None
    if has_percent_token:
        return val
    if 0.0 <= val <= 1.0:
        return val * 100.0
    return val


def parse_metrics_from_file(log_file: Path) -> Tuple[Optional[float], Optional[float]]:
    """Return the last (mAP, Rank-1) seen in the given log file."""

    map_val: Optional[float] = None
    map_has_pct = False
    rank1_val: Optional[float] = None
    rank1_has_pct = False

    p_map_variants = [
        re.compile(r"Mean AP:\s*([0-9]+\.?[0-9]*)(%)?"),
        re.compile(r"mAP:\s*([0-9]+\.?[0-9]*)(%)?"),
        re.compile(r"current mAP:\s*([0-9]+\.?[0-9]*)(%)?"),
    ]

    p_rank_variants = [
        re.compile(r"Rank-1[:\s]+([0-9]+\.?[0-9]*)(%)?", re.IGNORECASE),
        re.compile(r"top-1\s+([0-9]+\.?[0-9]*)(%)?", re.IGNORECASE),
    ]

    try:
        with log_file.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                for pat in p_map_variants:
                    match = pat.search(line)
                    if match:
                        try:
                            map_val = float(match.group(1))
                            map_has_pct = bool(match.group(2))
                        except ValueError:
                            pass

                for pat in p_rank_variants:
                    match = pat.search(line)
                    if match:
                        try:
                            rank1_val = float(match.group(1))
                            rank1_has_pct = bool(match.group(2))
                        except ValueError:
                            pass
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Could not read {log_file}: {exc}")
        return None, None

    map_val = _normalize_percentage(map_val, map_has_pct)
    rank1_val = _normalize_percentage(rank1_val, rank1_has_pct)

    return map_val, rank1_val


def _iter_log_files(run_dir: Path) -> Iterable[Path]:
    """Yield candidate log files (sorted by mtime descending) inside run_dir."""

    patterns = ("*.txt", "*.log", "*.out", "*.err")
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(run_dir.glob(pattern))

    candidates = [p for p in candidates if p.is_file() and p.stat().st_size > 0]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def _ordered_loss_names(loss_names: Iterable[str]) -> List[str]:
    """Return loss names ordered by priority, numeric prefix, or alpha."""

    names = list(dict.fromkeys(loss_names))
    baseline_set = {name.lower() for name in BASELINE_NAMES}
    baseline = [name for name in names if name.lower() in baseline_set]
    remaining = [name for name in names if name not in baseline]

    if any(name in LOSS_ORDER for name in remaining):
        known = [name for name in LOSS_ORDER if name in remaining]
        extras = sorted(name for name in remaining if name not in known)
        return baseline + known + extras

    numeric_pairs = []
    non_numeric = []
    for name in remaining:
        match = re.match(r"^(\d+)_", name)
        if match:
            numeric_pairs.append((int(match.group(1)), name))
        else:
            non_numeric.append(name)

    if numeric_pairs:
        ordered_numeric = [name for _, name in sorted(numeric_pairs, key=lambda x: x[0])]
        return baseline + ordered_numeric + sorted(non_numeric)

    return baseline + sorted(remaining)


def scan_logs(root_dir: Path) -> pd.DataFrame:
    """Walk run directories and extract the latest metrics for each run."""

    if not root_dir.exists():
        print(f"[ERROR] Log directory not found: {root_dir}")
        return pd.DataFrame()

    rows = []

    for child in root_dir.iterdir():
        if not child.is_dir():
            continue

        match = RUN_DIR_PATTERN.search(child.name)
        if not match:
            continue

        loss_name = match.group("loss")

        log_files = list(_iter_log_files(child))
        if not log_files:
            # Likely still running or empty; skip quietly to allow partial sweeps.
            continue

        log_file = log_files[0]
        map_val, rank1_val = parse_metrics_from_file(log_file)

        # Skip runs that have not yet emitted metrics (incomplete runs are expected).
        if map_val is None or rank1_val is None:
            continue

        if not (0.0 <= map_val <= 100.0) or not (0.0 <= rank1_val <= 100.0):
            print(f"[WARN] Metrics out of bounds in {child.name}; skipping")
            continue

        rows.append(
            {
                "LossName": loss_name,
                "mAP": map_val,
                "Rank-1": rank1_val,
                "LogDir": child.name,
                "LogFile": log_file.name,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Ensure categorical order for downstream operations
    ordered = _ordered_loss_names(df["LossName"].unique())
    df["LossName"] = pd.Categorical(df["LossName"], categories=ordered, ordered=True)
    return df


def compute_loss_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-loss mean/std for mAP and Rank-1 (NaN std -> 0)."""

    if df.empty:
        return df

    agg = df.groupby("LossName", observed=True)[["mAP", "Rank-1"]].agg(["mean", "std"])
    agg.columns = [f"{col[0]}_{col[1]}" for col in agg.columns]

    # Preserve ordering and fill NaN std (single run) with 0 for plotting.
    agg = agg.reset_index()
    std_cols = ["mAP_std", "Rank-1_std"]
    agg[std_cols] = agg[std_cols].fillna(0)
    ordered = _ordered_loss_names(agg["LossName"].tolist())
    agg["LossName"] = pd.Categorical(agg["LossName"], categories=ordered, ordered=True)
    agg = agg.sort_values("LossName")
    return agg


def attach_loss_stats(df_runs: pd.DataFrame, agg_stats: pd.DataFrame) -> pd.DataFrame:
    """Attach per-loss mean/std columns; blank out repeats for readability."""

    if df_runs.empty:
        return df_runs

    cols_stats = ["mAP_mean", "mAP_std", "Rank-1_mean", "Rank-1_std"]
    merged = df_runs.merge(agg_stats, on="LossName", how="left")
    merged["LossName"] = pd.Categorical(
        merged["LossName"], categories=agg_stats["LossName"].cat.categories, ordered=True
    )
    merged = merged.sort_values(["LossName", "LogDir"])

    for _, idx in merged.groupby("LossName").groups.items():
        idx_list = list(idx)
        if len(idx_list) <= 1:
            continue
        non_first = idx_list[1:]
        merged.loc[non_first, cols_stats] = ""

    return merged


def _report_sanity(df: pd.DataFrame) -> None:
    """Emit simple sanity checks about coverage and ranges."""

    if df[["mAP", "Rank-1"]].isnull().any().any():
        print("[WARN] NaN detected in extracted metrics.")

    for col in ["mAP", "Rank-1"]:
        if (df[col] < 0).any() or (df[col] > 100).any():
            print(f"[WARN] Values in {col} outside [0, 100].")


def plot_metric(
    df_runs: pd.DataFrame,
    agg_stats: pd.DataFrame,
    metric_name: str,
    output_path: Path,
    color: str = "#c0392b",
) -> None:
    """Generate a categorical plot (mean ± std) for the given metric."""

    setup_plot_style()

    names = agg_stats["LossName"].astype(str).tolist()
    scores = agg_stats[f"{metric_name}_mean"].to_numpy()
    errors = agg_stats[f"{metric_name}_std"].to_numpy()

    positions = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        positions,
        scores,
        color=color,
        s=130,
        alpha=0.7,
        edgecolor="white",
        linewidth=1.2,
        zorder=4,
        label="Mean",
    )

    for idx, loss_name in enumerate(agg_stats["LossName"].astype(str)):
        group = df_runs[df_runs["LossName"].astype(str) == loss_name]
        if group.empty:
            continue
        y_vals = group[metric_name].to_numpy()
        x_offsets = np.zeros(len(y_vals))
        ax.scatter(
            idx + x_offsets,
            y_vals,
            color=color,
            alpha=0.8,
            s=45,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    ax.set_xlabel("Loss Configuration", fontweight="bold")
    ax.set_ylabel(f"{metric_name} (%)", color=color, fontweight="bold")

    ax.tick_params(axis="y", labelcolor=color)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_xlim(-0.5, len(names) - 0.5)

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_title(f"{metric_name} by Loss Term", pad=14, fontweight="bold")

    if len(scores) > 0:
        best_idx = int(scores.argmax())
        best_name = names[best_idx]
        best_score = scores[best_idx]
        best_pos = positions[best_idx]

        ax.scatter(
            [best_pos],
            [best_score],
            color=color,
            zorder=5,
            s=150,
            edgecolor="white",
            linewidth=2,
        )

        ax.annotate(
            f"Best: {best_name}\n{best_score:.1f}%",
            xy=(best_pos, best_score),
            xytext=(0.5, 0.92),
            textcoords="axes fraction",
            fontsize=10,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[GEN] Plot saved to {output_path}")
    if len(scores) > 0:
        print(f"[INFO] Best {metric_name}: {names[best_idx]} ({best_score:.2f}%)")


def plot_results(df_runs: pd.DataFrame, agg_stats: pd.DataFrame, output_path: Path) -> None:
    """Generate separate plots for mAP and Rank-1."""

    base_name = output_path.stem
    parent = output_path.parent
    extension = output_path.suffix

    map_path = parent / f"{base_name}_mAP{extension}"
    plot_metric(df_runs, agg_stats, "mAP", map_path, color="#c0392b")

    rank1_path = parent / f"{base_name}_Rank1{extension}"
    plot_metric(df_runs, agg_stats, "Rank-1", rank1_path, color="#1f618d")


def main() -> None:

    args = parse_args()

    print(f"[INFO] Scanning logs in: {args.log_root}")
    df_runs = scan_logs(args.log_root)
    if df_runs.empty:
        print("[ERROR] No valid metrics found; nothing to plot.")
        sys.exit(1)

    _report_sanity(df_runs)

    agg_stats = compute_loss_stats(df_runs)
    if agg_stats.empty:
        print("[ERROR] Could not compute per-loss statistics; aborting.")
        sys.exit(1)

    csv_df = attach_loss_stats(df_runs, agg_stats)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    float_format = f"%.{max(args.csv_decimals, 0)}f"
    csv_df.to_csv(args.output_csv, index=False, float_format=float_format)
    print(f"[GEN] CSV saved to {args.output_csv}")

    plot_results(df_runs, agg_stats, args.output_plot)


if __name__ == "__main__":
    main()
