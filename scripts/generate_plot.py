#!/usr/bin/env python3
"""Parse Finsler alpha sweep logs and generate a publication-style plot.

This script scans a log root directory (default: ``logs/finsler``), extracts the
*last* recorded ``Mean AP`` and ``Rank-1/top-1`` values from each run directory,
stores them in a CSV, and produces a dual-axis plot (mAP on the left, Rank-1 on
the right) over alpha. The design follows the publication-style template
provided by the user.

python scripts/generate_plot.py \
    --log-root /home/stud/leez/reid/src/bau_finsler/logs/finsler \
    --output-csv finsler_sweep_results.csv \
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
DEFAULT_LOG_ROOT = Path("/home/stud/leez/reid/src/bau_finsler/logs/finsler")
DEFAULT_CSV = Path("finsler_sweep_results.csv")
DEFAULT_PLOT = Path("finsler_alpha_sweep.png")

# Expected alpha sweep (from finsler.sbatch) for basic sanity checking
EXPECTED_ALPHAS: List[float] = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]


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


def scan_logs(root_dir: Path) -> Tuple[pd.DataFrame, Optional[Tuple[float, float]]]:
	"""Walk run directories, extract metrics, return DataFrame and Euclidean baseline."""

	if not root_dir.exists():
		print(f"[ERROR] Log directory not found: {root_dir}")
		return pd.DataFrame(), None

	dir_pattern = re.compile(r"alpha([0-9.]+)")
	rows = []
	baseline: Optional[Tuple[float, float]] = None
	baseline_mtime: Optional[float] = None

	for child in root_dir.iterdir():
		if not child.is_dir():
			continue

		# Handle Euclidean baseline (alphaNone) separately
		if child.name.endswith("alphaNone"):
			log_files = list(_iter_log_files(child))
			if not log_files:
				print(f"[WARN] No non-empty log file found in {child.name}; skipping baseline")
				continue

			log_file = log_files[0]
			map_val, rank1_val = parse_metrics_from_file(log_file)
			if map_val is None or rank1_val is None:
				print(f"[WARN] Missing metrics in baseline {child.name}/{log_file.name}; skipping")
				continue

			mtime = log_file.stat().st_mtime
			if baseline is None or (baseline_mtime is not None and mtime > baseline_mtime):
				baseline = (map_val, rank1_val)
				baseline_mtime = mtime
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
				"mAP": map_val,
				"Rank-1": rank1_val,
				"LogDir": child.name,
				"LogFile": log_file.name,
			}
		)

	df = pd.DataFrame(rows)
	if df.empty:
		return df, baseline

	# Deduplicate by alpha: keep the row with the newest log file (mtime)
	df["_mtime"] = df.apply(lambda r: (root_dir / r["LogDir"] / r["LogFile"]).stat().st_mtime, axis=1)
	df = df.sort_values(["Alpha", "_mtime"], ascending=[True, False])
	df = df.drop_duplicates(subset=["Alpha"], keep="first")
	df = df.sort_values("Alpha").reset_index(drop=True)
	df = df.drop(columns=["_mtime"])

	_report_sanity(df)
	return df, baseline


def _report_sanity(df: pd.DataFrame) -> None:
	"""Emit simple sanity checks about coverage and ranges."""

	present = set(df["Alpha"].round(4).tolist())
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


def plot_results(
	df: pd.DataFrame,
	output_path: Path,
	opt_metric: str = "mAP",
	baseline: Optional[Tuple[float, float]] = None,
) -> None:
	"""Generate dual-axis plot with alpha on x, mAP and Rank-1 on y-axes."""

	setup_plot_style()

	alphas = df["Alpha"].to_numpy()
	map_scores = df["mAP"].to_numpy()
	rank1_scores = df["Rank-1"].to_numpy()

	positions = np.arange(len(alphas))  # evenly spaced positions for better edge visibility

	fig, ax_left = plt.subplots(figsize=(10, 6))
	ax_right = ax_left.twinx()

	color_map = "#c0392b"  # deep red
	color_rank1 = "#1f618d"  # muted blue

	ax_left.plot(
		positions,
		map_scores,
		marker="o",
		linestyle="-",
		color=color_map,
	)

	ax_right.plot(
		positions,
		rank1_scores,
		marker="s",
		linestyle="--",
		color=color_rank1,
	)

	ax_left.set_xlabel(r"Finsler $\alpha$ (norm of $\omega$)", fontweight="bold")
	ax_left.set_ylabel("mAP (%)", color=color_map, fontweight="bold")
	ax_right.set_ylabel("Rank-1 (%)", color=color_rank1, fontweight="bold")

	ax_left.tick_params(axis="y", labelcolor=color_map)
	ax_right.tick_params(axis="y", labelcolor=color_rank1)
	ax_left.set_xticks(positions)
	ax_left.set_xticklabels([f"{a:g}" for a in alphas])
	ax_left.set_xlim(-0.5, len(alphas) - 0.5)

	ax_left.grid(True, linestyle=":", alpha=0.6)
	ax_left.set_title("Effect of Finsler alpha on Re-ID performance", pad=14, fontweight="bold")

	# Euclidean baseline (alphaNone) as dual horizontal guides, same color with style cues
	if baseline is not None:
		base_map, base_rank1 = baseline
		baseline_color = "#4d5656"
		ax_left.axhline(
			base_map,
			color=color_map,
			linestyle="-.",
			linewidth=1.6,
			alpha=0.85,
			zorder=1,
		)
		ax_right.axhline(
			base_rank1,
			color=color_rank1,
			linestyle=":",
			linewidth=1.6,
			alpha=0.85,
			zorder=1,
		)
		ax_left.text(
			len(alphas) - 0.45,
			base_map,
			"Euclidean mAP",
			color=baseline_color,
			fontsize=10,
			ha="right",
			va="bottom",
		)
		ax_right.text(
			len(alphas) - 0.45,
			base_rank1,
			"Euclidean Rank-1",
			color=baseline_color,
			fontsize=10,
			ha="right",
			va="bottom",
		)

	# Highlight optimal point based on chosen metric
	series = map_scores if opt_metric == "mAP" else rank1_scores
	best_idx = series.argmax()
	best_alpha = alphas[best_idx]
	best_map = map_scores[best_idx]
	best_rank1 = rank1_scores[best_idx]
	best_pos = positions[best_idx]

	ax_left.axvline(best_pos, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
	ax_left.scatter([best_pos], [best_map], color=color_map, zorder=5)
	ax_right.scatter([best_pos], [best_rank1], color=color_rank1, zorder=5)

	ax_left.annotate(
		f"opt @ {best_alpha:g}\n{opt_metric}={series[best_idx]:.1f}%",
		xy=(best_pos, best_map),
		xytext=(0.95, 0.9),
		textcoords="axes fraction",
		fontsize=11,
		ha="right",
		va="top",
		bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9),
		arrowprops=None,
	)

	# Legend is intentionally omitted per requirements; color-coded axes differentiate series.
	plt.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, bbox_inches="tight")
	plt.close(fig)

	print(f"[GEN] Plot saved to {output_path}")
	print(f"[INFO] Optimal alpha ({opt_metric}): {best_alpha} | mAP={best_map:.2f}% | Rank-1={best_rank1:.2f}%")


def main() -> None:
	args = parse_args()

	df, baseline = scan_logs(args.log_root)
	if df.empty:
		print("[ERROR] No valid metrics found; nothing to plot.")
		sys.exit(1)

	args.output_csv.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(args.output_csv, index=False)
	print(f"[GEN] CSV saved to {args.output_csv}")

	if baseline is None:
		print("[WARN] Euclidean baseline (alphaNone) not found; baseline lines omitted.")

	plot_results(df, args.output_plot, opt_metric=args.opt_metric, baseline=baseline)


if __name__ == "__main__":
	main()
