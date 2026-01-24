#!/usr/bin/env python3
"""Parse Finsler alpha sweep logs and generate a publication-style plot.

This script scans a log root directory (default: ``logs/finsler``), extracts the
*last* recorded ``Mean AP`` and ``Rank-1/top-1`` values from each run directory,
stores them in a CSV, and produces a dual-axis plot (mAP on the left, Rank-1 on
the right) over alpha. The design follows the publication-style template
provided by the user.

python scripts/generate_plot_mean_variance_alpha.py \
    --log-root /home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/logs/finsler_alphaRNG \
    --output-csv finsler_sweepRNG_results.csv \
    --output-plot finsler_alpha_sweep.png   
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
DEFAULT_LOG_ROOT = Path("/home/stud/leez/reid/src/bau_finsler/logs/finsler_alphaRNG")
DEFAULT_CSV = Path("finsler_sweep_results.csv")
DEFAULT_PLOT = Path("finsler_alpha_sweep.png")

# Expected alpha sweep (from finslersweep.sbatch) for basic sanity checking
EXPECTED_ALPHAS: List[float] = [0.0, 0.1, 0.5]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Scan Finsler sweep logs, export CSV, and plot mAP/Rank-1 versus alpha."
	)
	parser.add_argument(
		"--log-root",
		type=Path,
		default=DEFAULT_LOG_ROOT,
		help="Root directory containing per-alpha run folders (default: logs/finsler)",
	)
	parser.add_argument(
		"--output-csv",
		type=Path,
		default=DEFAULT_CSV,
		help="Path to save extracted metrics CSV (default: finsler_sweep_results.csv)",
	)
	parser.add_argument(
		"--output-plot",
		type=Path,
		default=DEFAULT_PLOT,
		help="Path to save the dual-axis plot (default: finsler_alpha_sweep.png)",
	)
	parser.add_argument(
		"--opt-metric",
		choices=["mAP", "Rank-1"],
		default="mAP",
		help="Metric used to highlight the optimal alpha (default: mAP)",
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


def _normalize_percentage(val: Optional[float]) -> Optional[float]:
	if val is None:
		return None
	# Convert 0-1 to percent if clearly fractional
	if 0.0 <= val <= 1.0:
		return val * 100.0
	return val


def parse_metrics_from_file(log_file: Path) -> Tuple[Optional[float], Optional[float]]:
	"""Return the last (mAP, Rank-1) seen in the given log file.

	Supports patterns:
	- "Mean AP: 37.8%"
	- "current mAP: 36.4%" (falls back if Mean AP not found later)
	- "Rank-1: 78.5%" or "top-1          36.5%" (CMC block)
	"""

	map_val: Optional[float] = None
	rank1_val: Optional[float] = None

	p_map_variants = [
		re.compile(r"Mean AP:\s*([0-9]+\.?[0-9]*)"),
		re.compile(r"mAP:\s*([0-9]+\.?[0-9]*)"),
		re.compile(r"current mAP:\s*([0-9]+\.?[0-9]*)"),
	]

	p_rank_variants = [
		re.compile(r"Rank-1[:\s]+([0-9]+\.?[0-9]*)", re.IGNORECASE),
		re.compile(r"top-1\s+([0-9]+\.?[0-9]*)", re.IGNORECASE),
	]

	try:
		with log_file.open("r", encoding="utf-8", errors="ignore") as fh:
			for line in fh:
				for pat in p_map_variants:
					m = pat.search(line)
					if m:
						try:
							map_val = float(m.group(1))
						except ValueError:
							pass

				for pat in p_rank_variants:
					m = pat.search(line)
					if m:
						try:
							rank1_val = float(m.group(1))
						except ValueError:
							pass
	except Exception as exc:  # pragma: no cover - defensive
		print(f"[WARN] Could not read {log_file}: {exc}")
		return None, None

	map_val = _normalize_percentage(map_val)
	rank1_val = _normalize_percentage(rank1_val)

	return map_val, rank1_val


def _iter_log_files(run_dir: Path) -> Iterable[Path]:
	"""Yield candidate log files (sorted by mtime descending) inside run_dir."""

	candidates = list(run_dir.glob("*.txt")) + list(run_dir.glob("*.log"))
	candidates = [p for p in candidates if p.is_file() and p.stat().st_size > 0]
	candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
	return candidates


def scan_logs(root_dir: Path) -> pd.DataFrame:
	"""Walk run directories, extract metrics, return DataFrame of runs."""

	if not root_dir.exists():
		print(f"[ERROR] Log directory not found: {root_dir}")
		return pd.DataFrame()

	dir_pattern = re.compile(r"alpha([0-9.]+)")
	rows = []

	for child in root_dir.iterdir():
		if not child.is_dir():
			continue

		# Handle Euclidean baseline runs (alphaNone_run*)
		if "alphaNone" in child.name:
			log_files = list(_iter_log_files(child))
			if not log_files:
				print(f"[WARN] No non-empty log file found in {child.name}; skipping baseline")
				continue

			log_file = log_files[0]
			map_val, rank1_val = parse_metrics_from_file(log_file)
			if map_val is None or rank1_val is None:
				print(f"[WARN] Missing metrics in baseline {child.name}/{log_file.name}; skipping")
				continue

			rows.append(
				{
					"Alpha": None,
					"AlphaLabel": "None",
					"AlphaOrder": -1.0,
					"mAP": map_val,
					"Rank-1": rank1_val,
					"LogDir": child.name,
					"LogFile": log_file.name,
				}
			)
			continue

		match = dir_pattern.search(child.name)
		if not match:
			continue

		try:
			alpha = float(match.group(1))
		except ValueError:
			print(f"[WARN] Skipping directory with unparseable alpha: {child.name}")
			continue

		log_files = list(_iter_log_files(child))
		if not log_files:
			print(f"[WARN] No non-empty log file found in {child.name}; skipping")
			continue

		log_file = log_files[0]
		map_val, rank1_val = parse_metrics_from_file(log_file)

		if map_val is None or rank1_val is None:
			print(f"[WARN] Missing metrics in {child.name}/{log_file.name}; skipping")
			continue

		# Sanity bounds
		if not (0.0 <= map_val <= 100.0):
			print(f"[WARN] mAP out of bounds ({map_val}) in {child.name}; skipping")
			continue
		if not (0.0 <= rank1_val <= 100.0):
			print(f"[WARN] Rank-1 out of bounds ({rank1_val}) in {child.name}; skipping")
			continue

		rows.append(
			{
				"Alpha": alpha,
				"AlphaLabel": f"{alpha:g}",
				"AlphaOrder": alpha,
				"mAP": map_val,
				"Rank-1": rank1_val,
				"LogDir": child.name,
				"LogFile": log_file.name,
			}
		)

	df = pd.DataFrame(rows)
	if df.empty:
		return df

	_report_sanity(df)

	return df.sort_values(["AlphaOrder", "LogDir"]).reset_index(drop=True)


def compute_alpha_stats(df: pd.DataFrame) -> pd.DataFrame:
	"""Compute per-alpha mean/std for mAP and Rank-1 (NaN std -> 0)."""

	if df.empty:
		return df

	agg = df.groupby(["AlphaOrder", "AlphaLabel"])[["mAP", "Rank-1"]].agg(["mean", "std"])
	agg.columns = [f"{col[0]}_{col[1]}" for col in agg.columns]
	agg = agg.reset_index().sort_values("AlphaOrder").fillna(0)
	return agg


def attach_alpha_stats(df_runs: pd.DataFrame, agg_stats: pd.DataFrame) -> pd.DataFrame:
	"""Attach per-alpha mean/std columns; values shown on the first row of each alpha group."""

	if df_runs.empty:
		return df_runs

	cols_stats = ["mAP_mean", "mAP_std", "Rank-1_mean", "Rank-1_std"]
	merged = df_runs.merge(agg_stats, on=["AlphaOrder", "AlphaLabel"], how="left")
	merged = merged.sort_values(["AlphaOrder", "LogDir"])

	for _, idx in merged.groupby(["AlphaOrder", "AlphaLabel"]).groups.items():
		idx_list = list(idx)
		if len(idx_list) <= 1:
			continue
		non_first = idx_list[1:]
		merged.loc[non_first, cols_stats] = np.nan

	return merged


def _report_sanity(df: pd.DataFrame) -> None:
	"""Emit simple sanity checks about coverage and ranges."""

	numeric = df[df["Alpha"].notna()]
	present = set(numeric["Alpha"].round(4).tolist())
	expected = set(round(a, 4) for a in EXPECTED_ALPHAS)
	missing = sorted(expected - present)
	if missing:
		print(f"[INFO] Missing expected alphas: {missing}")
	else:
		print("[INFO] All expected alphas present.")

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
	"""Generate a single-axis plot for the given metric."""

	setup_plot_style()

	labels = agg_stats["AlphaLabel"].tolist()
	positions = np.arange(len(labels))
	means = agg_stats[f"{metric_name}_mean"].to_numpy()

	fig, ax = plt.subplots(figsize=(8, 6))

	# Plot individual runs (no error bars)
	for idx, (alpha_order, alpha_label) in enumerate(
		zip(agg_stats["AlphaOrder"], agg_stats["AlphaLabel"])
	):
		group = df_runs[(df_runs["AlphaOrder"] == alpha_order) & (df_runs["AlphaLabel"] == alpha_label)]
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

	# Plot mean line
	ax.plot(
		positions,
		means,
		marker="o",
		linestyle="-.",
		color=color,
		linewidth=2.2,
		markersize=7,
		zorder=4,
	)

	ax.set_xlabel(r"Finsler $\alpha$ (norm of $\omega$)", fontweight="bold")
	ax.set_ylabel(f"{metric_name} (%)", color=color, fontweight="bold")

	ax.tick_params(axis="y", labelcolor=color)
	ax.set_xticks(positions)
	ax.set_xticklabels(labels)
	ax.set_xlim(-0.5, len(labels) - 0.5)

	ax.grid(True, linestyle=":", alpha=0.6)
	ax.set_title(
		f"Effect of Finsler alpha on {metric_name} (Runs + Mean)",
		pad=14,
		fontweight="bold",
	)

	# Highlight optimal
	best_idx = means.argmax()
	best_alpha = labels[best_idx]
	best_score = means[best_idx]
	best_pos = positions[best_idx]

	ax.axvline(best_pos, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
	ax.scatter([best_pos], [best_score], color=color, zorder=5, s=100, edgecolor='white')

	ax.annotate(
		f"opt @ {best_alpha}\n{metric_name}={best_score:.1f}%",
		xy=(best_pos, best_score),
		xytext=(0.95, 0.9),
		textcoords="axes fraction",
		fontsize=11,
		ha="right",
		va="top",
		bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9),
		arrowprops=None,
	)

	plt.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, bbox_inches="tight")
	plt.close(fig)

	print(f"[GEN] Plot saved to {output_path}")
	print(f"[INFO] Optimal alpha ({metric_name}): {best_alpha} | {metric_name}={best_score:.2f}%")


def plot_results(
	df_runs: pd.DataFrame,
	agg_stats: pd.DataFrame,
	output_path: Path,
	opt_metric: str = "mAP",
) -> None:
	"""Generate separate plots for mAP and Rank-1."""

	base_name = output_path.stem
	parent = output_path.parent
	extension = output_path.suffix

	# mAP Plot
	map_path = parent / f"{base_name}_mAP{extension}"
	plot_metric(df_runs, agg_stats, "mAP", map_path, color="#c0392b")

	# Rank-1 Plot
	rank1_path = parent / f"{base_name}_Rank1{extension}"
	plot_metric(df_runs, agg_stats, "Rank-1", rank1_path, color="#1f618d")


def main() -> None:
	args = parse_args()

	df = scan_logs(args.log_root)
	if df.empty:
		print("[ERROR] No valid metrics found; nothing to plot.")
		sys.exit(1)

	# Compute per-alpha stats for plotting and tabular display
	agg_stats = compute_alpha_stats(df)
	if agg_stats.empty:
		print("[ERROR] Could not compute per-alpha statistics; aborting.")
		sys.exit(1)

	csv_df = attach_alpha_stats(df, agg_stats)

	args.output_csv.parent.mkdir(parents=True, exist_ok=True)
	float_format = f"%.{max(args.csv_decimals, 0)}f"
	csv_df.to_csv(args.output_csv, index=False, float_format=float_format)
	print(f"[GEN] CSV saved to {args.output_csv}")

	plot_results(df, agg_stats, args.output_plot, opt_metric=args.opt_metric)


if __name__ == "__main__":
	main()
