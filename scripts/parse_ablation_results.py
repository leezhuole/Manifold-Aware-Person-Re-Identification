#!/usr/bin/env python3
"""Aggregate ablation results and render summary tables.

This script scans per-run `log.txt` files produced by `train_bau.py`, extracts
final mAP / Rank-1 scores, groups them by loss configuration and
source->target dataset transfer, and emits both Markdown and LaTeX tables that
mirror the ablation study shown in the BAU paper (Table 5).

parse_ablation_results.py \
    --logs-root /home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/logs/ablationEuclidean-P3 \
    --markdown-output /home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/results/ablationEuclidean-P3.md \
    --csv-output /home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/results/ablationEuclidean-P3.csv


"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Dataset transfer identifiers -> pretty column headers
DATASET_COLUMNS = [
    (("market1501", "msmt17", "cuhksysu"), "cuhk03", "M+MS+CS -> C3"),
    (("market1501dg", "msmt17dg", "cuhksysu"), "cuhk03", "M+MS+CS -> C3"),
    (("market1501", "cuhksysu", "cuhk03"), "msmt17", "M+CS+C3 -> MS"),
    (("msmt17", "cuhksysu", "cuhk03"), "market1501", "MS+CS+C3 -> M"),
]

# Loss configurations, ordered to match the paper's rows
LOSS_ROWS = [
    ("baseline", "Baseline (w/o augmented images)", r"Baseline (w/o augmented images)"),
    ("l_ce", r"$\mathcal{L}_{ce}$", r"$\mathcal{L}_{ce}$"),
    ("l_align", r"$\mathcal{L}_{align}$", r"$\mathcal{L}_{align}$"),
    ("l_align_uniform", r"$\mathcal{L}_{align}+\mathcal{L}_{uniform}$", r"$\mathcal{L}_{align}+\mathcal{L}_{uniform}$"),
    ("l_full", r"$\mathcal{L}_{align}+\mathcal{L}_{uniform}+\mathcal{L}_{domain}$", r"$\mathcal{L}_{align}+\mathcal{L}_{uniform}+\mathcal{L}_{domain}$"),
]

ARGS_PATTERN = re.compile(r"Args:(Namespace\(.*\))")
MEAN_AP_PATTERN = re.compile(r"Mean AP:\s*([0-9.]+)%")
RANK1_PATTERN = re.compile(r"top-1\s+([0-9.]+)%")


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse BAU ablation logs and render tables.")
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs/ablation"),
        help="Root directory containing per-run log folders (default: logs/ablation)",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=None,
        help="Optional path to write the LaTeX table (omit to skip writing)",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional path to write the Markdown table (omit to skip writing)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to dump the raw metrics as JSON",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional path to save metrics as CSV for plotting",
    )
    return parser.parse_args()


def parse_namespace(namespace_str: str) -> Dict[str, object]:
    """Convert the string representation of argparse.Namespace to a dict."""
    # We trust the training script output; restrict builtins for safety.
    namespace_obj = eval(namespace_str, {"__builtins__": {}}, {"Namespace": argparse.Namespace})  # noqa: S307
    return vars(namespace_obj)


def extract_metrics(log_path: Path) -> Tuple[Dict[str, object], float, float]:
    args_dict: Optional[Dict[str, object]] = None
    mean_ap: Optional[float] = None
    rank1: Optional[float] = None

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            if args_dict is None:
                match = ARGS_PATTERN.search(line)
                if match:
                    args_dict = parse_namespace(match.group(1))
                    continue

            ap_match = MEAN_AP_PATTERN.search(line)
            if ap_match:
                mean_ap = float(ap_match.group(1))

            rank_match = RANK1_PATTERN.search(line)
            if rank_match:
                rank1 = float(rank_match.group(1))

    if args_dict is None:
        raise ValueError(f"Args block not found in {log_path}")
    if mean_ap is None or rank1 is None:
        raise ValueError(f"Final metrics not found in {log_path}")

    return args_dict, mean_ap, rank1


def determine_loss_key(args_dict: Dict[str, object]) -> str:
    use_align = not bool(args_dict.get("no_align", False))
    use_uniform = not bool(args_dict.get("no_uniform", False))
    use_domain = not bool(args_dict.get("no_domain", False))
    use_aug_ce = bool(args_dict.get("use_aug_ce", False))

    if not use_align and not use_uniform and not use_domain:
        return "l_ce" if use_aug_ce else "baseline"
    if use_align and not use_uniform and not use_domain and not use_aug_ce:
        return "l_align"
    if use_align and use_uniform and not use_domain and not use_aug_ce:
        return "l_align_uniform"
    if use_align and use_uniform and use_domain and not use_aug_ce:
        return "l_full"

    raise ValueError(
        "Encountered unsupported loss configuration: "
        f"use_align={use_align}, use_uniform={use_uniform}, use_domain={use_domain}, use_aug_ce={use_aug_ce}"
    )


def determine_dataset_key(args_dict: Dict[str, object]) -> str:
    sources_value = args_dict.get("source_dataset", [])
    target_value = args_dict.get("target_dataset")

    if not isinstance(sources_value, (list, tuple)):
        raise ValueError(f"Unexpected source_dataset format: {sources_value!r}")
    if not isinstance(target_value, str):
        raise ValueError(f"Unexpected target_dataset format: {target_value!r}")

    sources = tuple(sources_value)
    target = target_value

    for source_tuple, target_name, header in DATASET_COLUMNS:
        if sources == source_tuple and target == target_name:
            return header

    raise ValueError(f"Unrecognised dataset configuration: sources={sources}, target={target}")


def collect_log_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Logs root '{root}' does not exist")
    return sorted(root.glob("**/log.txt"))


def build_tables(
    results: Dict[str, Dict[str, Tuple[float, float]]]
) -> Tuple[str, str, Dict[str, object], List[Dict[str, object]]]:
    # Prepare Markdown header
    headers: List[str] = ["Method"]
    for _, _, header in DATASET_COLUMNS:
        headers.extend([f"{header} mAP", f"{header} Rank-1"])
    headers.extend(["Average mAP", "Average Rank-1"])

    md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]

    latex_lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{lcccccccc}",
        "\\hline",
        "Method & \\multicolumn{2}{c}{M+MS+CS \\rightarrow C3} & \\multicolumn{2}{c}{M+CS+C3 \\rightarrow MS} & \\multicolumn{2}{c}{MS+CS+C3 \\rightarrow M} & \\multicolumn{2}{c}{Average} \\",
        " & mAP & Rank-1 & mAP & Rank-1 & mAP & Rank-1 & mAP & Rank-1 \\",
        "\\hline",
    ]

    raw_payload: Dict[str, object] = {}
    csv_rows: List[Dict[str, object]] = []

    for key, md_label, latex_label in LOSS_ROWS:
        row_metrics = results.get(key, {})
        row_values: List[str] = [md_label]
        latex_cells: List[str] = [latex_label]
        averages: List[Tuple[float, float]] = []

        for _, _, column_header in DATASET_COLUMNS:
            metrics = row_metrics.get(column_header)
            if metrics is None:
                row_values.extend(["-", "-"])
                latex_cells.extend(["-", "-"])
                continue
            map_val, rank_val = metrics

            # Collect long-form rows to make plotting straightforward.
            csv_rows.append(
                {
                    "MethodKey": key,
                    "Method": md_label,
                    "Transfer": column_header,
                    "mAP": map_val,
                    "Rank-1": rank_val,
                }
            )
            row_values.extend([f"{map_val:.1f}", f"{rank_val:.1f}"])
            latex_cells.extend([f"{map_val:.1f}", f"{rank_val:.1f}"])
            averages.append(metrics)

        if averages:
            avg_map = sum(metric[0] for metric in averages) / len(averages)
            avg_rank = sum(metric[1] for metric in averages) / len(averages)
            row_values.extend([f"{avg_map:.1f}", f"{avg_rank:.1f}"])
            latex_cells.extend([f"{avg_map:.1f}", f"{avg_rank:.1f}"])
        else:
            row_values.extend(["-", "-"])
            latex_cells.extend(["-", "-"])

        md_lines.append("| " + " | ".join(row_values) + " |")
        latex_lines.append(" & ".join(latex_cells) + r" \\")
        raw_payload[key] = {
            column: {"mAP": values[0], "rank1": values[1]} for column, values in row_metrics.items()
        }

    md_table = "\n".join(md_lines)
    latex_lines.extend(["\\hline", "\\end{tabular}", "\\caption{Ablation study of loss functions for augmented images.}", "\\end{table}"])
    latex_table = "\n".join(latex_lines)

    return md_table, latex_table, raw_payload, csv_rows


def main() -> None:
    args = parse_cli()

    results: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)

    for log_path in collect_log_files(args.logs_root):
        try:
            args_dict, map_val, rank_val = extract_metrics(log_path)
        except ValueError as exc:
            print(f"[WARN] Skipping {log_path}: {exc}")
            continue

        loss_key = determine_loss_key(args_dict)
        dataset_key = determine_dataset_key(args_dict)

        if dataset_key in results[loss_key]:
            print(f"[WARN] Duplicate entry for ({loss_key}, {dataset_key}) found in {log_path}; overwriting previous value.")

        results[loss_key][dataset_key] = (map_val, rank_val)

    if not results:
        raise SystemExit("No results parsed; ensure that training logs exist under the specified root.")

    md_table, latex_table, payload, csv_rows = build_tables(results)

    print("\nMarkdown Table:\n")
    print(md_table)
    print("\nLaTeX Table:\n")
    print(latex_table)

    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(md_table + "\n", encoding="utf-8")
        print(f"[INFO] Markdown table written to {args.markdown_output}")

    if args.latex_output:
        args.latex_output.parent.mkdir(parents=True, exist_ok=True)
        args.latex_output.write_text(latex_table + "\n", encoding="utf-8")
        print(f"[INFO] LaTeX table written to {args.latex_output}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[INFO] JSON payload written to {args.json_output}")

    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["MethodKey", "Method", "Transfer", "mAP", "Rank-1"],
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[INFO] CSV table written to {args.csv_output}")


if __name__ == "__main__":
    main()
