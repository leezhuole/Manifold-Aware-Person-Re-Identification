#!/usr/bin/env python3
"""Parse learnable-alpha / alpha-sweep logs and generate a publication-style plot.

This script scans a log root directory, extracts the *last* recorded ``Mean AP``
and ``Rank-1/top-1`` values from each run directory, stores them in a CSV, and
produces plots for mAP and Rank-1.

It supports two common layouts:

1) Learnable alpha runs (your case)
	- Run dir contains: ``..._alphainit0.1_...``
	- Log contains: ``Args:Namespace(... alpha_init=0.1, ...)``
	- Log contains per-epoch: ``Logged alpha value: 0.0976`` (we take the last one)
	- X-axis uses *initial* alpha (alpha_init)
	- Each point can show *final* alpha (alpha_end) via annotation or colorbar

2) Fixed alpha sweep runs
	- Run dir contains: ``...alpha0.1...``
	- X-axis uses alpha from the folder name

python scripts/generate_plot_mean_variance_alpha.py \
	--log-root logs/learnableAlpha \
	--output-csv learnable_alpha_results.csv \
	--output-plot learnable_alpha.png
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
from matplotlib import transforms


# Default locations (override via CLI if needed)
DEFAULT_LOG_ROOT = Path("logs/fins_learnableAlpha")
DEFAULT_LOG_ROOT_FALLBACK = Path("logs/learnableAlpha")
DEFAULT_CSV = Path("learnable_alpha_results.csv")
DEFAULT_PLOT = Path("learnable_alpha.png")

# Optional expected values for sanity checking (disabled by default)
EXPECTED_ALPHAS: List[float] = []


def _default_log_root() -> Path:
	if DEFAULT_LOG_ROOT.exists():
		return DEFAULT_LOG_ROOT
	if DEFAULT_LOG_ROOT_FALLBACK.exists():
		return DEFAULT_LOG_ROOT_FALLBACK
	return DEFAULT_LOG_ROOT


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Scan logs, export CSV, and plot mAP/Rank-1 versus initial alpha (optionally showing final alpha)."
	)
	parser.add_argument(
		"--log-root",
		type=Path,
		default=_default_log_root(),
		help="Root directory containing per-run folders (default: logs/fins_learnableAlpha or logs/learnableAlpha)",
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
		"--alpha-end-display",
		choices=["annotate", "colorbar", "none"],
		default="annotate",
		help="How to display final alpha per run (default: annotate)",
	)
	parser.add_argument(
		"--alpha-end-decimals",
		type=int,
		default=3,
		help="Decimal places for alpha_end labels when annotating (default: 3)",
	)
	parser.add_argument(
		"--baseline-decimals",
		type=int,
		default=3,
		help="Decimal places for baseline label (default: 3)",
	)
	parser.add_argument(
		"--jitter",
		type=float,
		default=0.0,
		help="Horizontal jitter added to points within the same alpha_init (default: 0.0; try 0.005)",
	)
	parser.add_argument(
		"--csv-decimals",
		type=int,
		default=3,
		help="Decimal places for floating values in the saved CSV (default: 3)",
	)
	return parser.parse_args()


def _parse_float(text: str) -> Optional[float]:
	try:
		return float(text)
	except Exception:
		return None


def parse_alphas_from_file(log_file: Path) -> Tuple[Optional[float], Optional[float]]:
	"""Return (alpha_init, alpha_end) from a log file.

	- alpha_init: parsed from the Args line (alpha_init=...) if present
	- alpha_end: last occurrence of 'Logged alpha value: ...'
	"""
	alpha_init: Optional[float] = None
	alpha_end: Optional[float] = None

	p_init = re.compile(r"alpha_init=([0-9]+\.?[0-9]*)")
	p_end = re.compile(r"Logged alpha value:\s*([0-9]+\.?[0-9]*)")

	try:
		with log_file.open("r", encoding="utf-8", errors="ignore") as fh:
			for line in fh:
				m_init = p_init.search(line)
				if m_init and alpha_init is None:
					alpha_init = _parse_float(m_init.group(1))

				m_end = p_end.search(line)
				if m_end:
					parsed = _parse_float(m_end.group(1))
					if parsed is not None:
						alpha_end = parsed
	except Exception as exc:  # pragma: no cover
		print(f"[WARN] Could not read {log_file}: {exc}")
		return None, None

	return alpha_init, alpha_end


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

	# Two possible naming conventions
	dir_pattern_alpha = re.compile(r"alpha([0-9.]+)")
	dir_pattern_alphainit = re.compile(r"alphainit([0-9.]+)")
	rows = []

	for child in root_dir.iterdir():
		if not child.is_dir():
			continue

		# Handle baseline runs (no alpha)
		if "alphaNone" in child.name or "alphainitNone" in child.name:
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
					"AlphaInit": None,
					"AlphaInitLabel": "None",
					"AlphaInitOrder": -1.0,
					"AlphaEnd": None,
					"mAP": map_val,
					"Rank-1": rank1_val,
					"LogDir": child.name,
					"LogFile": log_file.name,
				}
			)
			continue

		alpha_init: Optional[float] = None
		alpha_init_from_dir = None
		m_init_dir = dir_pattern_alphainit.search(child.name)
		if m_init_dir:
			alpha_init_from_dir = _parse_float(m_init_dir.group(1))

		alpha_from_dir = None
		m_alpha_dir = dir_pattern_alpha.search(child.name)
		if m_alpha_dir:
			alpha_from_dir = _parse_float(m_alpha_dir.group(1))

		log_files = list(_iter_log_files(child))
		if not log_files:
			print(f"[WARN] No non-empty log file found in {child.name}; skipping")
			continue

		log_file = log_files[0]
		map_val, rank1_val = parse_metrics_from_file(log_file)
		file_alpha_init, file_alpha_end = parse_alphas_from_file(log_file)

		# Prefer learnable-alpha init if present; otherwise fall back to fixed-alpha folder naming
		alpha_init = alpha_init_from_dir if alpha_init_from_dir is not None else file_alpha_init
		if alpha_init is None:
			alpha_init = alpha_from_dir

		alpha_end = file_alpha_end
		if alpha_end is None and alpha_from_dir is not None:
			# For fixed-alpha sweeps, alpha_end is just the fixed value
			alpha_end = alpha_from_dir

		if alpha_init is None:
			print(f"[WARN] Could not determine alpha_init for {child.name}; skipping")
			continue

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
				"AlphaInit": alpha_init,
				"AlphaInitLabel": f"{alpha_init:g}",
				"AlphaInitOrder": alpha_init,
				"AlphaEnd": alpha_end,
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

	return df.sort_values(["AlphaInitOrder", "LogDir"]).reset_index(drop=True)


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
	"""Backward/forward compatible schema handling."""
	if df.empty:
		return df
	# Older schema (Alpha/AlphaLabel/AlphaOrder) -> new schema
	if "AlphaInit" not in df.columns and "Alpha" in df.columns:
		df = df.rename(
			columns={
				"Alpha": "AlphaInit",
				"AlphaLabel": "AlphaInitLabel",
				"AlphaOrder": "AlphaInitOrder",
			}
		)
	if "AlphaEnd" not in df.columns:
		df["AlphaEnd"] = np.nan
	return df


def compute_alpha_stats(df: pd.DataFrame) -> pd.DataFrame:
	"""Compute per-alpha mean/std for mAP and Rank-1 (NaN std -> 0)."""

	if df.empty:
		return df

	# Exclude baseline rows (alphaNone) from the alpha_init aggregation
	df_alpha = df[df["AlphaInit"].notna()].copy()
	if df_alpha.empty:
		return pd.DataFrame()

	agg = df_alpha.groupby(["AlphaInitOrder", "AlphaInitLabel"])[["mAP", "Rank-1"]].agg(["mean", "std"])
	agg.columns = [f"{col[0]}_{col[1]}" for col in agg.columns]
	agg = agg.reset_index().sort_values("AlphaInitOrder").fillna(0)
	return agg


def compute_baseline_means(df: pd.DataFrame) -> dict[str, float]:
	"""Compute baseline (alphaNone) mean metrics across all baseline runs."""
	if df.empty:
		return {}
	baseline = df[df["AlphaInit"].isna()]
	if baseline.empty:
		return {}

	result: dict[str, float] = {}
	for metric in ["mAP", "Rank-1"]:
		vals = baseline[metric].dropna().to_numpy(dtype=float)
		if vals.size:
			result[metric] = float(np.mean(vals))
	return result


def attach_alpha_stats(df_runs: pd.DataFrame, agg_stats: pd.DataFrame) -> pd.DataFrame:
	"""Attach per-alpha mean/std columns; values shown on the first row of each alpha group."""

	if df_runs.empty:
		return df_runs

	cols_stats = ["mAP_mean", "mAP_std", "Rank-1_mean", "Rank-1_std"]
	merged = df_runs.merge(agg_stats, on=["AlphaInitOrder", "AlphaInitLabel"], how="left")
	merged = merged.sort_values(["AlphaInitOrder", "LogDir"])

	for _, idx in merged.groupby(["AlphaInitOrder", "AlphaInitLabel"]).groups.items():
		idx_list = list(idx)
		if len(idx_list) <= 1:
			continue
		non_first = idx_list[1:]
		merged.loc[non_first, cols_stats] = np.nan

	return merged


def _report_sanity(df: pd.DataFrame) -> None:
	"""Emit simple sanity checks about coverage and ranges."""

	if EXPECTED_ALPHAS:
		numeric = df[df["AlphaInit"].notna()]
		present = set(numeric["AlphaInit"].round(4).tolist())
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
	alpha_end_display: str = "annotate",
	alpha_end_decimals: int = 3,
	jitter: float = 0.0,
	baseline_mean: Optional[float] = None,
	baseline_decimals: int = 3,
) -> None:
	"""Generate a single-axis plot for the given metric."""

	setup_plot_style()

	labels = agg_stats["AlphaInitLabel"].tolist()
	x_vals = agg_stats["AlphaInitOrder"].to_numpy(dtype=float)
	means = agg_stats[f"{metric_name}_mean"].to_numpy()

	fig, ax = plt.subplots(figsize=(8, 6))

	# Plot individual runs
	all_end = df_runs["AlphaEnd"].to_numpy(dtype=float)
	use_color = alpha_end_display == "colorbar" and np.isfinite(all_end).any()
	sc = None

	for alpha_init, alpha_label in zip(x_vals, labels):
		group = df_runs[df_runs["AlphaInitLabel"] == alpha_label]
		if group.empty:
			continue

		y_vals = group[metric_name].to_numpy(dtype=float)
		m = len(y_vals)
		offsets = np.zeros(m, dtype=float)
		if jitter and m > 1:
			offsets = np.linspace(-jitter, jitter, m)

		x_plot = alpha_init + offsets

		if use_color:
			c_vals = group["AlphaEnd"].to_numpy(dtype=float)
			if np.isfinite(c_vals).any():
				sc = ax.scatter(
					x_plot,
					y_vals,
					c=c_vals,
					cmap="viridis",
					alpha=0.85,
					s=55,
					edgecolor="white",
					linewidth=0.6,
					zorder=3,
				)
			else:
				ax.scatter(
					x_plot,
					y_vals,
					color=color,
					alpha=0.8,
					s=55,
					edgecolor="white",
					linewidth=0.6,
					zorder=3,
				)
		else:
			ax.scatter(
				x_plot,
				y_vals,
				color=color,
				alpha=0.8,
				s=55,
				edgecolor="white",
				linewidth=0.6,
				zorder=3,
			)

		if alpha_end_display == "annotate":
			for x_i, y_i, end_i in zip(x_plot, y_vals, group["AlphaEnd"].to_numpy(dtype=float)):
				if not np.isfinite(end_i):
					continue
				label = f"{end_i:.{max(alpha_end_decimals,0)}f}"
				ax.annotate(
					label,
					xy=(x_i, y_i),
					xytext=(3, 3),
					textcoords="offset points",
					fontsize=8,
					color="black",
					alpha=0.85,
					zorder=4,
				)

	# Plot mean line
	ax.plot(
		x_vals,
		means,
		marker="o",
		linestyle="-.",
		color=color,
		linewidth=2.2,
		markersize=7,
		zorder=4,
	)

	ax.set_xlabel(r"Initial $\alpha$", fontweight="bold")
	ax.set_ylabel(f"{metric_name} (%)", color=color, fontweight="bold")

	ax.tick_params(axis="y", labelcolor=color)
	ax.set_xticks(x_vals)
	ax.set_xticklabels(labels)
	if len(x_vals) > 0:
		pad = max(0.01, 0.05 * float(np.nanmax(x_vals) - np.nanmin(x_vals)))
		ax.set_xlim(float(np.nanmin(x_vals)) - pad, float(np.nanmax(x_vals)) + pad)

	ax.grid(True, linestyle=":", alpha=0.6)

	# Baseline dotted horizontal line (alphaNone mean)
	if baseline_mean is not None and np.isfinite(baseline_mean):
		dec = max(int(baseline_decimals), 0)
		ax.axhline(
			y=baseline_mean,
			color="gray",
			linestyle=":",
			linewidth=1.8,
			alpha=0.8,
			zorder=1,
		)
		# Put the numeric baseline value on the y-axis at the dotted line.
		blended = transforms.blended_transform_factory(ax.transAxes, ax.transData)
		ax.text(
			-0.02,
			baseline_mean,
			f"{baseline_mean:.{dec}f}%",
			transform=blended,
			ha="right",
			va="center",
			fontsize=10,
			color="gray",
			clip_on=False,
			zorder=5,
		)
	ax.set_title(
		f"Effect of initial alpha on {metric_name} (Runs + Mean)",
		pad=14,
		fontweight="bold",
	)

	# Colorbar (if requested)
	if use_color and sc is not None:
		cbar = plt.colorbar(sc, ax=ax, pad=0.02)
		cbar.set_label(r"Final $\alpha$", fontweight="bold")

	# Highlight optimal (mean)
	best_idx = means.argmax()
	best_alpha = labels[best_idx]
	best_score = means[best_idx]
	best_pos = x_vals[best_idx]

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
	alpha_end_display: str = "annotate",
	alpha_end_decimals: int = 3,
	jitter: float = 0.0,
	baseline_means: Optional[dict[str, float]] = None,
	baseline_decimals: int = 3,
) -> None:
	"""Generate separate plots for mAP and Rank-1."""

	base_name = output_path.stem
	parent = output_path.parent
	extension = output_path.suffix

	# mAP Plot
	map_path = parent / f"{base_name}_mAP{extension}"
	plot_metric(
		df_runs,
		agg_stats,
		"mAP",
		map_path,
		color="#c0392b",
		alpha_end_display=alpha_end_display,
		alpha_end_decimals=alpha_end_decimals,
		jitter=jitter,
		baseline_mean=(baseline_means or {}).get("mAP"),
		baseline_decimals=baseline_decimals,
	)

	# Rank-1 Plot
	rank1_path = parent / f"{base_name}_Rank1{extension}"
	plot_metric(
		df_runs,
		agg_stats,
		"Rank-1",
		rank1_path,
		color="#1f618d",
		alpha_end_display=alpha_end_display,
		alpha_end_decimals=alpha_end_decimals,
		jitter=jitter,
		baseline_mean=(baseline_means or {}).get("Rank-1"),
		baseline_decimals=baseline_decimals,
	)


def main() -> None:
	args = parse_args()

	df = scan_logs(args.log_root)
	df = _coerce_schema(df)
	if df.empty:
		print("[ERROR] No valid metrics found; nothing to plot.")
		sys.exit(1)

	# Compute per-alpha stats for plotting and tabular display
	baseline_means = compute_baseline_means(df)
	agg_stats = compute_alpha_stats(df)
	if agg_stats.empty:
		print("[ERROR] Could not compute per-alpha statistics; aborting.")
		sys.exit(1)

	csv_df = attach_alpha_stats(df, agg_stats)

	args.output_csv.parent.mkdir(parents=True, exist_ok=True)
	float_format = f"%.{max(args.csv_decimals, 0)}f"
	csv_df.to_csv(args.output_csv, index=False, float_format=float_format)
	print(f"[GEN] CSV saved to {args.output_csv}")

	plot_results(
		df,
		agg_stats,
		args.output_plot,
		opt_metric=args.opt_metric,
		alpha_end_display=args.alpha_end_display,
		alpha_end_decimals=args.alpha_end_decimals,
		jitter=args.jitter,
		baseline_means=baseline_means,
		baseline_decimals=args.baseline_decimals,
	)


if __name__ == "__main__":
	main()
