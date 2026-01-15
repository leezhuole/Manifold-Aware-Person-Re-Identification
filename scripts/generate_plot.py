#!/usr/bin/env python3
"""Generate publication-quality plots from ablation CSV files.

The CSV files are expected to come from ``parse_ablation_results.py`` and
contain the columns: ``MethodKey, Method, Transfer, mAP, Rank-1``.

For each dataset transfer configuration (``Transfer``), a plot is produced
showing two lines (mAP and Rank-1) on a shared y-axis spanning 0–100. A plot
is only rendered when *all* required method rows for that CSV are present for
the given transfer, ensuring no partial or broken lines.

python generate_plot.py \
	--input /home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/results/ablationEuclidean-P3.csv \
    --output-dir /home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/results/plots 

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

# Ordered list of known methods to stabilise x-axis ordering
CANONICAL_ORDER: List[str] = [
	"baseline",
	"l_ce",
	"l_align",
	"l_align_uniform",
	"l_full",
]

# Hard requirement for CSV schema
REQUIRED_COLUMNS = {"MethodKey", "Method", "Transfer", "mAP", "Rank-1"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate line plots from ablation CSV files.")
	parser.add_argument(
		"--input",
		type=Path,
		nargs="+",
		required=True,
		help="Path(s) to CSV files (e.g., results/ablationEuclidean.csv)",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("results/plots"),
		help="Directory to store generated plot images (default: results/plots)",
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


def determine_required_keys(df: pd.DataFrame) -> List[str]:
	"""Return ordered method keys required for a complete plot.

	We treat all unique ``MethodKey`` values present in the CSV (and known to
	``CANONICAL_ORDER``) as required. Each transfer plot is rendered only if it
	contains *all* of these keys.
	"""

	keys_in_file = [key for key in df["MethodKey"].unique().tolist() if key in CANONICAL_ORDER]
	# Preserve canonical ordering while keeping only keys that actually appear
	ordered_keys = [key for key in CANONICAL_ORDER if key in keys_in_file]
	return ordered_keys


def sanitize_filename(name: str) -> str:
	"""Convert a transfer label into a safe filename fragment."""

	return (
		name.replace("->", "to")
		.replace("+", "_")
		.replace("/", "-")
		.replace(" ", "")
	)


def plot_transfer(
	df_transfer: pd.DataFrame,
	transfer_name: str,
	output_path: Path,
	required_keys: List[str],
	dpi: int,
) -> bool:
	"""Plot mAP and Rank-1 for a single transfer configuration.

	Returns True if the plot was generated, False if skipped due to incomplete
	data (missing methods or NaN metrics).
	"""

	# Ensure all required methods are present
	present_keys = set(df_transfer["MethodKey"].dropna().unique().tolist())
	missing = set(required_keys) - present_keys
	if missing:
		print(f"[SKIP] {transfer_name}: missing methods {sorted(missing)}")
		return False

	# Remove any stray methods not in required_keys to stabilise ordering
	df_transfer = df_transfer[df_transfer["MethodKey"].isin(required_keys)].copy()
	df_transfer["MethodKey"] = pd.Categorical(df_transfer["MethodKey"], required_keys, ordered=True)
	df_transfer = df_transfer.sort_values("MethodKey")

	# Validate metric availability
	if df_transfer[["mAP", "Rank-1"]].isnull().any().any():
		print(f"[SKIP] {transfer_name}: contains NaN in metrics")
		return False

	x_positions = range(len(required_keys))
	x_labels = df_transfer["Method"].tolist()
	map_scores = df_transfer["mAP"].to_numpy()
	rank1_scores = df_transfer["Rank-1"].to_numpy()

	def _limits(series: pd.Series, pad_frac: float = 0.05) -> tuple[float, float]:
		min_val = float(series.min())
		max_val = float(series.max())
		span = max_val - min_val
		pad = span * pad_frac if span > 0 else max_val * pad_frac or 1.0
		lower = min_val - pad
		upper = max_val + pad
		# keep within sensible bounds
		return max(0.0, lower), upper

	map_min, map_max = _limits(df_transfer["mAP"])
	r1_min, r1_max = _limits(df_transfer["Rank-1"])

	fig, ax_left = plt.subplots(figsize=(10, 6))
	ax_right = ax_left.twinx()

	map_line: Line2D = ax_left.plot(
		x_positions,
		map_scores,
		marker="o",
		linestyle="-",
		color="#c0392b",
		label="mAP",
	)[0]
	r1_line: Line2D = ax_right.plot(
		x_positions,
		rank1_scores,
		marker="s",
		linestyle="--",
		color="#1f618d",
		label="Rank-1",
	)[0]

	ax_left.set_title(f"Transfer: {transfer_name}", pad=16, fontweight="bold")
	ax_left.set_xlabel("Method")
	ax_left.set_ylabel("mAP (%)", color=map_line.get_color())
	ax_right.set_ylabel("Rank-1 (%)", color=r1_line.get_color())

	ax_left.set_ylim(map_min, map_max)
	ax_right.set_ylim(r1_min, r1_max)

	ax_left.set_xticks(list(x_positions))
	ax_left.set_xticklabels(x_labels, rotation=20, ha="right")

	# Tick colors to match series
	ax_left.tick_params(axis="y", colors=map_line.get_color())
	ax_right.tick_params(axis="y", colors=r1_line.get_color())

	# Build a combined legend
	lines: List[Line2D] = [map_line, r1_line]
	labels: List[str] = [str(line.get_label()) for line in lines]
	legend = ax_left.legend(lines, labels, loc="upper left", frameon=True, framealpha=0.9, edgecolor="gray")
	legend.get_frame().set_linewidth(0.8)

	ax_left.grid(True, linestyle=":", alpha=0.6)

	plt.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)
	print(f"[GEN] Saved plot -> {output_path}")
	return True


def process_file(csv_path: Path, output_root: Path, dpi: int) -> None:
	"""Process a single CSV file and generate plots per transfer configuration."""

	if not csv_path.exists():
		print(f"[WARN] CSV not found: {csv_path}")
		return

	try:
		df = pd.read_csv(csv_path)
	except Exception as exc:  # pragma: no cover - defensive logging
		print(f"[WARN] Failed to read {csv_path}: {exc}")
		return

	missing_cols = REQUIRED_COLUMNS - set(df.columns)
	if missing_cols:
		print(f"[WARN] {csv_path} missing required columns: {sorted(missing_cols)}; skipping.")
		return

	required_keys = determine_required_keys(df)
	if not required_keys:
		print(f"[WARN] No recognised MethodKey values in {csv_path}; skipping.")
		return

	output_dir = output_root / csv_path.stem
	generated = 0
	for transfer_name, group in df.groupby("Transfer"):
		filename = sanitize_filename(str(transfer_name)) or "plot"
		plot_path = output_dir / f"{filename}.png"
		if plot_transfer(group, str(transfer_name), plot_path, required_keys, dpi):
			generated += 1

	if generated == 0:
		print(f"[INFO] No plots generated for {csv_path} (incomplete data).")
	else:
		print(f"[DONE] Generated {generated} plot(s) for {csv_path}.")


def main() -> None:
	args = parse_args()
	setup_plot_style()

	for csv_file in args.input:
		process_file(csv_file, args.output_dir, args.dpi)


if __name__ == "__main__":
	main()
