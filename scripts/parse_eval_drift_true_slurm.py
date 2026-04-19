#!/usr/bin/env python3
"""
Parse Slurm stdout from sweep_eval_drift_true_consecutive.sbatch (or per-run
eval_drift_true.log) and emit TSV / markdown-friendly rows.

Canonical per-run log path (written by the sbatch script):
  <run_log_dir>/eval_drift_true.log

Slurm aggregate log:
  logs/slurm_logs/eval-drift-true-seq-<JOBID>.out
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

Cell = Dict[str, Any]


def parse_slurm_out(text: str) -> List[Dict[str, object]]:
    """Extract checkpoint path, Mean AP, top-1 / top-5 / top-10 from each task block."""
    blocks = re.split(r"=== Evaluation Task \d+ / \d+ ===", text)
    rows: List[Dict[str, object]] = []
    for block in blocks:
        if "Checkpoint:" not in block or "Mean AP:" not in block:
            continue
        m_ckpt = re.search(r"Checkpoint:\s*(\S+)", block)
        m_map = re.search(r"Mean AP:\s*([\d.]+)%", block)
        m_r1 = re.search(r"top-1\s+([\d.]+)%", block)
        m_r5 = re.search(r"top-5\s+([\d.]+)%", block)
        m_r10 = re.search(r"top-10\s+([\d.]+)%", block)
        if not (m_ckpt and m_map and m_r1):
            continue
        row: Dict[str, object] = {
            "checkpoint": m_ckpt.group(1).strip(),
            "map_pct": float(m_map.group(1)),
            "r1_pct": float(m_r1.group(1)),
        }
        if m_r5 and m_r10:
            row["r5_pct"] = float(m_r5.group(1))
            row["r10_pct"] = float(m_r10.group(1))
        rows.append(row)
    return rows


def classify_row(checkpoint: str) -> Tuple[str, str, str, str]:
    """
    Returns (short_name, drift_cond, domain_column, ldom_slot).

    domain_column: 'finsler_ldom' (Idea1_2 or gap LdomFins) vs 'euclidean_ldom' (EuclideanDom_2 or gap LdomEucl)
    ldom_slot: 'main' | 'LdomFins' | 'LdomEucl' for gap runs
    """
    p = checkpoint.replace("\\", "/")
    if "sweep_loss_ablation_Idea1_2/" in p or "/sweep_loss_ablation_Idea1_2/" in p:
        dom_col = "finsler_ldom"
        ldom = "main"
        name = Path(p).parent.name
    elif "sweep_loss_ablation_Idea1_EuclideanDom_2/" in p or "/sweep_loss_ablation_Idea1_EuclideanDom_2/" in p:
        dom_col = "euclidean_ldom"
        ldom = "main"
        name = Path(p).parent.name
    elif "sweep_unified_finsler_idea1_gap/" in p or "/sweep_unified_finsler_idea1_gap/" in p:
        name = Path(p).parent.name
        if "LdomFins" in name:
            dom_col, ldom = "finsler_ldom", "LdomFins"
        elif "LdomEucl" in name:
            dom_col, ldom = "euclidean_ldom", "LdomEucl"
        else:
            dom_col, ldom = "unknown", "unknown"
    else:
        return ("?", "?", "unknown", "?")

    if "driftInst" in name:
        drift = "instance"
    elif "driftDom" in name:
        drift = "domain"
    else:
        drift = "?"

    if "00_baseline_1a_only" in name:
        short = "1a"
    elif "02_domTri_cam" in name and "noLdom" in name:
        short = "1a + 1b (no L_dom)"
    elif "04_domTri_cam" in name and "withLdom" in name:
        short = "1a + 1b"
    elif "01_cam_xdom" in name:
        short = "1a + Refined 1b"
    elif "03_driftOnly_xcam" in name:
        short = "1a + 1c"
    elif "idea1bOnly_domTriW" in name:
        short = "1b"
    elif "unifiedFin_onlyTri" in name:
        short = "Unified Finsler"
    else:
        short = name

    return (short, drift, dom_col, ldom)


EUCLIDEAN_EVAL_TRAINING: Dict[Tuple[str, str, str], Tuple[float, float]] = {
    # (short, drift, finsler_or_euclidean_ldom) -> (mAP, R1) from training log, eval-drift false
    # Finsler L_dom training = Idea1_2 + gap LdomFins; Euclidean L_dom = EuclideanDom_2 + gap LdomEucl
    ("1a", "instance", "finsler"): (44.1, 43.6),
    ("1a", "domain", "finsler"): (43.2, 43.7),
    ("1a", "instance", "euclidean"): (43.9, 43.9),
    ("1a", "domain", "euclidean"): (43.8, 43.9),
    ("1b", "instance", "finsler"): (35.1, 33.9),
    ("1b", "domain", "finsler"): (33.2, 34.1),
    ("1b", "instance", "euclidean"): (30.6, 30.6),
    ("1b", "domain", "euclidean"): (25.6, 27.0),
    ("1a + 1b", "instance", "finsler"): (36.4, 36.2),
    ("1a + 1b", "domain", "finsler"): (34.9, 36.1),
    ("1a + 1b", "instance", "euclidean"): (36.9, 36.2),
    ("1a + 1b", "domain", "euclidean"): (37.8, 38.9),
    ("1a + 1b (no L_dom)", "instance", "finsler"): (34.6, 36.4),
    ("1a + 1b (no L_dom)", "domain", "finsler"): (24.9, 28.5),
    ("1a + 1b (no L_dom)", "instance", "euclidean"): (33.6, 35.0),
    ("1a + 1b (no L_dom)", "domain", "euclidean"): (25.5, 26.4),
    ("1a + Refined 1b", "instance", "finsler"): (43.1, 43.7),
    ("1a + Refined 1b", "domain", "finsler"): (42.1, 43.2),
    ("1a + Refined 1b", "instance", "euclidean"): (42.8, 43.4),
    ("1a + Refined 1b", "domain", "euclidean"): (42.4, 43.1),
    ("1a + 1c", "instance", "finsler"): (43.0, 43.4),
    ("1a + 1c", "domain", "finsler"): (43.8, 44.9),
    ("1a + 1c", "instance", "euclidean"): (43.8, 42.9),
    ("1a + 1c", "domain", "euclidean"): (43.7, 42.9),
    ("Unified Finsler", "instance", "finsler"): (44.1, 44.0),
    ("Unified Finsler", "domain", "finsler"): (44.1, 44.3),
    ("Unified Finsler", "instance", "euclidean"): (43.4, 43.9),
    ("Unified Finsler", "domain", "euclidean"): (43.4, 43.0),
}


def fill_cells(rows: List[Dict[str, object]]) -> Tuple[List[str], List[str], Dict[Tuple[str, str], Cell]]:
    keys_order = [
        "1a",
        "1b",
        "1a + 1b",
        "1a + 1b (no L_dom)",
        "1a + Refined 1b",
        "1a + 1c",
        "Unified Finsler",
    ]
    drifts = ["instance", "domain"]

    cell: Dict[Tuple[str, str], Cell] = {}
    for s in keys_order:
        for d in drifts:
            cell[(s, d)] = {
                "FD_E": EUCLIDEAN_EVAL_TRAINING.get((s, d, "finsler")),
                "ED_E": EUCLIDEAN_EVAL_TRAINING.get((s, d, "euclidean")),
                "FD_F": None,
                "ED_F": None,
                "FD_F_ranks": None,  # (r5, r10)
                "ED_F_ranks": None,
            }

    for r in rows:
        ck = str(r["checkpoint"])
        short, drift, dom_col, ldom = classify_row(ck)
        if short not in keys_order or drift not in drifts:
            continue
        m = float(r["map_pct"])
        r1 = float(r["r1_pct"])
        tup = (m, r1)
        ranks = None
        if "r5_pct" in r and "r10_pct" in r:
            ranks = (float(r["r5_pct"]), float(r["r10_pct"]))
        c = cell[(short, drift)]
        if dom_col == "finsler_ldom" and ldom == "main":
            c["FD_F"] = tup
            if ranks:
                c["FD_F_ranks"] = ranks
        elif dom_col == "euclidean_ldom" and ldom == "main":
            c["ED_F"] = tup
            if ranks:
                c["ED_F_ranks"] = ranks
        elif dom_col == "finsler_ldom" and ldom == "LdomFins":
            c["FD_F"] = tup
            if ranks:
                c["FD_F_ranks"] = ranks
        elif dom_col == "euclidean_ldom" and ldom == "LdomEucl":
            c["ED_F"] = tup
            if ranks:
                c["ED_F_ranks"] = ranks

    return keys_order, drifts, cell


def _fmt_pair(t: Optional[Tuple[float, float]]) -> Tuple[str, str]:
    if t is None:
        return ("", "")
    return (f"{t[0]:.1f}", f"{t[1]:.1f}")


def _fmt_delta(
    fins: Optional[Tuple[float, float]], eucl: Optional[Tuple[float, float]]
) -> Tuple[str, str]:
    if fins is None or eucl is None:
        return ("", "")
    return (f"{fins[0] - eucl[0]:+.1f}", f"{fins[1] - eucl[1]:+.1f}")


def build_markdown_document(rows: List[Dict[str, object]], slurm_name: str) -> str:
    """Full results markdown: header docs + main table + appendix Δ + appendix ranks."""
    keys_order, drifts, cell = fill_cells(rows)

    intro = f"""# Asymmetric ranking at test (`--eval-drift true`) — merged table

## Where Slurm and per-run logs go (`sweep_eval_drift_true_consecutive.sbatch`)

| Output | Path |
|--------|------|
| **Slurm stdout** | `logs/slurm_logs/eval-drift-true-seq-<JOBID>.out` |
| **Slurm stderr** | `logs/slurm_logs/eval-drift-true-seq-<JOBID>.err` |
| **Canonical eval log (per training run)** | `<run_logs_dir>/eval_drift_true.log` next to `best.pth` (same directory as `log.txt` for that job) |

The Slurm `.out` file repeats the mAP / top-1 block after each checkpoint; the per-run `eval_drift_true.log` is the full `examples/test.py` stdout/stderr for that checkpoint only.

**Table source:** `{slurm_name}` ({len(rows)} checkpoints). Regenerate:

```bash
python scripts/parse_eval_drift_true_slurm.py logs/slurm_logs/eval-drift-true-seq-<JOBID>.out -o results/eval_drift_true_finsler_ranking_table.md
```

---

## Metric definitions

- **Eucl. eval** (training split): metrics from the original training run with **`--eval-drift false`** (identity slice / Euclidean ranking at test), tabulated in `results/sweep_loss_ablation_Idea1_2_metrics.md`.
- **Finsler eval**: **`examples/test.py`** on **`best.pth`** with **`--eval-drift true`** (full embedding + model Finsler distance).

**Column groups:** *Finsler-domain training* = `logs/sweep_loss_ablation_Idea1_2/` runs plus gap sweep rows with **`LdomFins`**. *Euclidean-domain training* = `logs/sweep_loss_ablation_Idea1_EuclideanDom_2/` plus gap rows with **`LdomEucl`**.
"""

    main_lines = [
        "",
        "## Main table (mAP / Rank-1)",
        "",
        "| Short Name | Drift Cond. | **Finsler-domain training** ||| **Euclidean-domain training** |||",
        "| | | Eucl. eval mAP | Eucl. eval R1 | Finsler eval mAP | Finsler eval R1 | Eucl. eval mAP | Eucl. eval R1 | Finsler eval mAP | Finsler eval R1 |",
        "|------------|-------------|------|------|------|------|------|------|------|------|",
    ]
    for s in keys_order:
        for i, d in enumerate(drifts):
            c = cell[(s, d)]
            fe_m, fe_r1 = _fmt_pair(c["FD_E"])
            ff_m, ff_r1 = _fmt_pair(c["FD_F"])
            ee_m, ee_r1 = _fmt_pair(c["ED_E"])
            ef_m, ef_r1 = _fmt_pair(c["ED_F"])
            name_cell = s if i == 0 else ""
            main_lines.append(
                f"| {name_cell} | {d} | {fe_m} | {fe_r1} | {ff_m} | {ff_r1} | {ee_m} | {ee_r1} | {ef_m} | {ef_r1} |"
            )

    delta_lines = [
        "",
        "---",
        "",
        "## Appendix A — Δ when using Finsler ranking at test",
        "",
        "Defined as **Finsler eval − Eucl. eval** (same checkpoint; only the evaluation distance / embedding slice changes). Units: percentage points on mAP and Rank-1.",
        "",
        "| Short Name | Drift Cond. | ΔmAP (Fins. train) | ΔR1 (Fins. train) | ΔmAP (Euc. train) | ΔR1 (Euc. train) |",
        "|------------|-------------|------|------|------|------|",
    ]
    for s in keys_order:
        for i, d in enumerate(drifts):
            c = cell[(s, d)]
            dm_f, dr_f = _fmt_delta(c["FD_F"], c["FD_E"])
            dm_e, dr_e = _fmt_delta(c["ED_F"], c["ED_E"])
            name_cell = s if i == 0 else ""
            delta_lines.append(f"| {name_cell} | {d} | {dm_f} | {dr_f} | {dm_e} | {dr_e} |")

    rank_lines = [
        "",
        "---",
        "",
        "## Appendix B — Finsler-eval Rank-5 / Rank-10",
        "",
        "From the same standalone **`--eval-drift true`** run as the Finsler eval mAP/R1 columns (CUHK03 test split).",
        "",
        "| Short Name | Drift Cond. | R5 (Fins. train) | R10 (Fins. train) | R5 (Euc. train) | R10 (Euc. train) |",
        "|------------|-------------|------|------|------|------|",
    ]
    for s in keys_order:
        for i, d in enumerate(drifts):
            c = cell[(s, d)]
            fr = c.get("FD_F_ranks")
            er = c.get("ED_F_ranks")
            f5, f10 = _fmt_pair(fr) if fr else ("", "")
            e5, e10 = _fmt_pair(er) if er else ("", "")
            name_cell = s if i == 0 else ""
            rank_lines.append(f"| {name_cell} | {d} | {f5} | {f10} | {e5} | {e10} |")

    return intro + "\n".join(main_lines + delta_lines + rank_lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "slurm_out",
        type=Path,
        nargs="?",
        default=Path("logs/slurm_logs/eval-drift-true-seq-1518384.out"),
        help="Slurm .out from consecutive eval job",
    )
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()
    text = args.slurm_out.read_text(encoding="utf-8", errors="replace")
    rows = parse_slurm_out(text)
    md = build_markdown_document(rows, slurm_name=str(args.slurm_out))
    if args.output:
        args.output.write_text(md, encoding="utf-8")
        print(f"Wrote {args.output} ({len(rows)} parsed checkpoints)")
    else:
        print(md)


if __name__ == "__main__":
    main()
