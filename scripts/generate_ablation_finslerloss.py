#!/usr/bin/env python3
"""Parse ablation logs and generate publication-quality visualizations.

This script scans experiment log folders for the ablation sweeps and extracts
the final evaluation metrics (mAP and Rank-1/top-1). For the Finsler sweep with
learnable alpha, it also plots the final alpha value recorded in the log.

Example usage:
	python scripts/generate_ablation_finslerloss.py \
		--finsler-dir /home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/logs/ablationFinsler-P2-learnableAlpha \
		--euclidean-dir /home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/logs/ablationEuclideanP3_2501 \
		--output-dir /home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/results/plots/ablation_comparison
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Ordered list of loss configurations from the SBATCH sweep
CONFIG_ORDER: List[str] = [
	"1_Abs_Baseline",
	"2_Pure_Triplet",
	"3_Std_Baseline",
	"4_CE_Align",
	"5_Base_Align",
	"6_Base_Uniform",
	"7_Base_Domain",
	"8_Base_Align_Uni",
	"9_Base_Align_Dom",
	"10_Base_Uni_Dom",
	"11_Full_Method",
]

PRETTY_LABELS: Dict[str, str] = {
	"1_Abs_Baseline": "Absolute Baseline",
	"2_Pure_Triplet": "Pure Triplet",
	"3_Std_Baseline": "Std Baseline",
	"4_CE_Align": "CE + Ali.",
	"5_Base_Align": "Base + Ali.",
	"6_Base_Uniform": "Base + Uni.",
	"7_Base_Domain": "Base + Dom.",
	"8_Base_Align_Uni": "Base + Ali. + Uni.",
	"9_Base_Align_Dom": "Base + Ali. + Dom.",
	"10_Base_Uni_Dom": "Base + Uni. + Dom.",
	"11_Full_Method": "Full Method",
}

MAP_PATTERN = re.compile(r"Mean AP:\s*([0-9.]+)%")
TOP1_PATTERN = re.compile(r"top-1\s+([0-9.]+)%")
ALPHA_BEST_PATTERN = re.compile(r"Best model alpha value:\s*([0-9.]+)")
ALPHA_LOG_PATTERN = re.compile(r"Logged alpha values:\s*alpha=([0-9.]+)")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate ablation plots from log directories.")
	parser.add_argument(
		"--finsler-dir",
		type=Path,
		required=True,
		help="Path to Finsler (learnable alpha) ablation log directory",
	)
	parser.add_argument(
		"--euclidean-dir",
		type=Path,
		required=True,
		help="Path to Euclidean ablation log directory",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("results/plots/ablation_comparison"),
		help="Directory to store generated plot images",
	)
	parser.add_argument(
		"--log-name",
		type=str,
		default="log.txt",
		help="Log filename to parse within each run directory",
	)
	parser.add_argument(
		"--alpha-init",
		type=float,
		default=0.1,
		help="Initial alpha value for reference line in alpha plot",
	)
	parser.add_argument(
		"--dpi",
		type=int,
		default=300,
		help="Output image DPI (default: 300)",
	)
	return parser.parse_args()


def setup_plot_style() -> None:
	"""Configure plotting aesthetics for publication-style figures."""

	sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
	plt.rcParams.update(
		{
			"font.family": "STIXGeneral",  # clean, publication-ready serif
			"mathtext.fontset": "stix",
			"axes.titlesize": 16,
			"axes.labelsize": 13,
			"legend.fontsize": 12,
			"lines.linewidth": 2.5,
			"lines.markersize": 8,
		}
	)
	plt.rcParams["figure.dpi"] = 150
	plt.rcParams["savefig.dpi"] = 300
	plt.rcParams["text.usetex"] = False  # avoid external LaTeX dependency while keeping mathtext


@dataclass
class RunMetrics:
	config_key: str
	model: str
	map_score: Optional[float]
	rank1_score: Optional[float]
	final_alpha: Optional[float]

	@property
	def label(self) -> str:
		return PRETTY_LABELS.get(self.config_key, self.config_key)


def _last_float(matches: Iterable[str]) -> Optional[float]:
	values = [float(value) for value in matches]
	return values[-1] if values else None


def parse_log_file(log_path: Path) -> RunMetrics:
	"""Parse a single log file for final mAP, Rank-1, and alpha values."""
	text = log_path.read_text(encoding="utf-8", errors="ignore")

	map_score = _last_float(MAP_PATTERN.findall(text))
	rank1_score = _last_float(TOP1_PATTERN.findall(text))

	alpha_best = _last_float(ALPHA_BEST_PATTERN.findall(text))
	alpha_last = _last_float(ALPHA_LOG_PATTERN.findall(text))
	final_alpha = alpha_best if alpha_best is not None else alpha_last

	return RunMetrics(
		config_key="",
		model="",
		map_score=map_score,
		rank1_score=rank1_score,
		final_alpha=final_alpha,
	)


def find_run_dir(base_dir: Path, config_key: str) -> Optional[Path]:
	if not base_dir.exists():
		return None
	matches = sorted([path for path in base_dir.iterdir() if path.is_dir() and config_key in path.name])
	return matches[0] if matches else None


def collect_runs(base_dir: Path, model_label: str, log_name: str) -> List[RunMetrics]:
	rows: List[RunMetrics] = []
	for config_key in CONFIG_ORDER:
		run_dir = find_run_dir(base_dir, config_key)
		if run_dir is None:
			print(f"[WARN] Missing run directory for {model_label}: {config_key}")
			continue
		log_path = run_dir / log_name
		if not log_path.exists():
			print(f"[WARN] Missing log file: {log_path}")
			continue
		metrics = parse_log_file(log_path)
		metrics.config_key = config_key
		metrics.model = model_label
		if metrics.map_score is None or metrics.rank1_score is None:
			print(f"[WARN] Incomplete metrics for {model_label} {config_key}: {log_path}")
			continue
		rows.append(metrics)
	return rows


def _ordered_dataframe(rows: List[RunMetrics]) -> pd.DataFrame:
	data = [
		{
			"Model": row.model,
			"ConfigKey": row.config_key,
			"ConfigLabel": row.label,
			"mAP": row.map_score,
			"Rank-1": row.rank1_score,
			"FinalAlpha": row.final_alpha,
		}
		for row in rows
	]
	frame = pd.DataFrame(data)
	frame["ConfigKey"] = pd.Categorical(frame["ConfigKey"], CONFIG_ORDER, ordered=True)
	return frame.sort_values(["ConfigKey", "Model"]).reset_index(drop=True)


def plot_performance(df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
	metric_labels = {"mAP": "mAP (%)", "Rank-1": "Rank-1 (%)"}
	palette = {"Euclidean": "#c0392b", "Finsler (Learnable α)": "#1f618d"}

	fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
	for ax, metric in zip(axes, ["mAP", "Rank-1"]):
		sns.lineplot(
			data=df,
			x="ConfigLabel",
			y=metric,
			hue="Model",
			style="Model",
			markers=True,
			dashes=False,
			palette=palette,
			ax=ax,
		)
		ax.set_title(metric_labels[metric])
		ax.set_xlabel("")
		ax.set_ylabel(metric_labels[metric])
		ax.tick_params(axis="x", rotation=30)
		ax.grid(True, linestyle=":", alpha=0.6)
		ax.legend(title="Model")

	fig.suptitle("Loss Ablation: Euclidean vs Finsler (Learnable α)", fontsize=16, fontweight="bold")
	fig.tight_layout()
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "ablation_performance.png"
	fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)
	print(f"[GEN] Saved plot -> {output_path}")


def plot_alpha(df: pd.DataFrame, output_dir: Path, dpi: int, alpha_init: float) -> None:
	finsler_df = df[df["Model"] == "Finsler (Learnable α)"].copy()
	if finsler_df.empty:
		print("[WARN] No Finsler rows found; skipping alpha plot.")
		return
	if finsler_df["FinalAlpha"].isnull().all():
		print("[WARN] No alpha values found; skipping alpha plot.")
		return

	fig, ax = plt.subplots(figsize=(12, 6))
	sns.lineplot(
		data=finsler_df,
		x="ConfigLabel",
		y="FinalAlpha",
		marker="o",
		color="#8e44ad",
		ax=ax,
	)
	if alpha_init is not None:
		ax.axhline(alpha_init, linestyle="--", color="gray", label=f"Init α = {alpha_init}")
		ax.legend()
	ax.set_title("Final α per Loss Configuration", fontsize=15, fontweight="bold")
	ax.set_xlabel("")
	ax.set_ylabel("Final α")
	ax.tick_params(axis="x", rotation=30)
	ax.grid(True, linestyle=":", alpha=0.6)
	fig.tight_layout()
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "ablation_alpha.png"
	fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)
	print(f"[GEN] Saved plot -> {output_path}")


def main() -> None:
	args = parse_args()
	setup_plot_style()

	finsler_rows = collect_runs(args.finsler_dir, "Finsler (Learnable α)", args.log_name)
	euclidean_rows = collect_runs(args.euclidean_dir, "Euclidean", args.log_name)
	rows = finsler_rows + euclidean_rows
	if not rows:
		print("[ERROR] No valid runs found. Check the log directories and filenames.")
		return

	df = _ordered_dataframe(rows)
	args.output_dir.mkdir(parents=True, exist_ok=True)
	output_csv = args.output_dir / "ablation_summary.csv"
	df.to_csv(output_csv, index=False)
	print(f"[GEN] Saved summary -> {output_csv}")

	plot_performance(df, args.output_dir, args.dpi)
	plot_alpha(df, args.output_dir, args.dpi, args.alpha_init)


if __name__ == "__main__":
	main()
