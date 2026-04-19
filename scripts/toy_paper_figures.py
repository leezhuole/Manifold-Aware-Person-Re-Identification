"""
Shared, camera-ready toy figures for the paper (CVPR single-column width).

Used by ``regenerate_toy_figures.py`` and ``toy_dataset_analysis.py`` so strip + mAP
stay typographically aligned.  No figure titles/suptitles — LaTeX captions only.

Axis sizing targets ~\\linewidth ≈ 3.31 in; mAP panel uses a compact aspect ratio.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import shutil
from typing import Any, Mapping, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Style constants — keep in sync across strip + mAP
# ---------------------------------------------------------------------------
FONT_LABEL = 9
FONT_TICK = 8
FONT_ANNOT = 7.5
FONT_LEGEND = 8

LINE_W = 1.8
MARKER_S = 7

COLOR_FINS = "#1565c0"
COLOR_EUCL = "#c62828"

# mAP figure size (inches) — matches regenerate_toy_figures historical sizing
FIGMAP_W = 3.5
FIGMAP_H = 2.6

# Extra offset points (dx, dy) added to annotation xytext for specific printed values
# (fixes labels clipping past the axes border). Keys are one-decimal strings.
VALUE_TEXT_NUDGE_POINTS: dict[str, tuple[float, float]] = {
    "100.0": (5.0, -3.0),  # slightly right, slightly down
    "68.8": (0.0, 4.0),  # slightly up (Euclidean above marker)
    "62.3": (-8.0, 5.0),  # slightly left, more up (Finsler below marker)
    "62.6": (-8.0, 5.0),  # alias if rounding differs
}


def _norm_severity_dict(raw: Mapping[Any, Any]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for k, v in raw.items():
        out[int(k)] = v
    return out


def _series_percent_mAP(eucl_raw: Mapping, fins_raw: Mapping) -> tuple[list[int], list[float], list[float]]:
    e = _norm_severity_dict(eucl_raw)
    f = _norm_severity_dict(fins_raw)
    sevs = sorted(set(e.keys()) & set(f.keys()))
    eucl_map = [e[s]["mAP"] * 100 for s in sevs]
    fins_map = [f[s]["mAP"] * 100 for s in sevs]
    return sevs, eucl_map, fins_map


def make_corruption_strip(dataset_dir: str, output_path: str) -> None:
    """One-row strip: five crops for PID 1 at severities 0..4; only ``$s=k$`` panel titles."""
    bb_dir = osp.join(dataset_dir, "bounding_box_test")
    n_sev = 5
    fnames = [f"0001_c{sev + 1}s1_000001_01.jpg" for sev in range(n_sev)]

    fig, axes = plt.subplots(1, n_sev, figsize=(5.5, 1.80))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.01, wspace=0.05)

    for sev_idx, (ax, fname) in enumerate(zip(axes, fnames)):
        img_path = osp.join(bb_dir, fname)
        if not osp.exists(img_path):
            raise FileNotFoundError(
                f"Image not found: {img_path}\nRun scripts/generate_toy_dataset.py first."
            )
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"$s={sev_idx}$", fontsize=FONT_LABEL, pad=3)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_retrieval_mAP_plot(
    eucl_raw: Union[str, Mapping[Any, Any]],
    fins_raw: Union[None, Mapping[Any, Any]],
    output_path: str,
) -> None:
    """mAP vs gallery corruption severity: Euclidean labels above markers, Finsler below.

    If ``eucl_raw`` is a path string, load ``retrieval_metrics.json`` and ignore ``fins_raw``.
    Otherwise ``eucl_raw`` / ``fins_raw`` are dicts like ``{severity: {\"mAP\": float}}``
    (keys int or str), matching ``toy_dataset_analysis`` / JSON exports.
    """
    if isinstance(eucl_raw, str):
        with open(eucl_raw) as fp:
            data = json.load(fp)
        if "finsler_dF_ranking" in data:
            eucl_d = data["euclidean_model_ranking"]
            fins_d = data["finsler_dF_ranking"]
        elif "finsler_bau_dF" in data:
            eucl_d = data["euclidean_bau_resnet50_dE"]
            fins_d = data["finsler_bau_dF"]
        else:
            raise KeyError(f"Unrecognised retrieval_metrics.json schema in {eucl_raw}")
    else:
        eucl_d = eucl_raw
        fins_d = fins_raw
        if fins_d is None:
            raise ValueError("fins_raw is required when eucl_raw is not a path")

    sevs, eucl_map, fins_map = _series_percent_mAP(eucl_d, fins_d)

    all_y = eucl_map + fins_map
    y_lo = min(all_y)
    y_hi = max(all_y)
    span = max(y_hi - y_lo, 1.0)

    pad_lo = max(4.0, span * 0.14)
    pad_hi = max(6.0, span * 0.10)
    ylim_lo = y_lo - pad_lo
    ylim_hi = y_hi + pad_hi

    fig, ax = plt.subplots(figsize=(FIGMAP_W, FIGMAP_H))

    ax.plot(
        sevs,
        fins_map,
        "o-",
        color=COLOR_FINS,
        linewidth=LINE_W,
        markersize=MARKER_S,
        label=r"Finsler $d_F$",
        zorder=3,
        clip_on=False,
    )
    ax.plot(
        sevs,
        eucl_map,
        "s--",
        color=COLOR_EUCL,
        linewidth=LINE_W,
        markersize=MARKER_S,
        label=r"Euclidean $d_E$",
        zorder=3,
        clip_on=False,
    )

    ax.set_ylim(ylim_lo, ylim_hi)

    tol = 0.1
    for s, fm, em in zip(sevs, fins_map, eucl_map):
        same = abs(fm - em) < tol
        fs = f"{fm:.1f}"
        es = f"{em:.1f}"
        nudge_f = VALUE_TEXT_NUDGE_POINTS.get(fs, (0.0, 0.0))
        nudge_e = VALUE_TEXT_NUDGE_POINTS.get(es, (0.0, 0.0))

        # Euclidean: above marker — offset points up, va='bottom'
        # Finsler: below marker — offset points down, va='top'
        base_eucl = (0.0, 10.0)
        base_fins = (0.0, -12.0)

        if same:
            ax.annotate(
                fs,
                (s, fm),
                textcoords="offset points",
                xytext=(nudge_e[0], max(base_eucl[1], 8.0) + nudge_e[1]),
                ha="center",
                va="bottom",
                fontsize=FONT_ANNOT,
                color="0.30",
            )
        else:
            ax.annotate(
                es,
                (s, em),
                textcoords="offset points",
                xytext=(base_eucl[0] + nudge_e[0], base_eucl[1] + nudge_e[1]),
                ha="center",
                va="bottom",
                fontsize=FONT_ANNOT,
                color=COLOR_EUCL,
            )
            ax.annotate(
                fs,
                (s, fm),
                textcoords="offset points",
                xytext=(base_fins[0] + nudge_f[0], base_fins[1] + nudge_f[1]),
                ha="center",
                va="top",
                fontsize=FONT_ANNOT,
                color=COLOR_FINS,
            )

    ax.set_xlabel("Gallery corruption severity", fontsize=FONT_LABEL)
    ax.set_ylabel("mAP (%)", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_xticks(sevs)
    ax.legend(fontsize=FONT_LEGEND, loc="lower left", framealpha=0.92)

    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def copy_paper_outputs(
    output_dir: str,
    paper_fig_dir: str,
    *,
    mirror_dir: str | None = None,
) -> None:
    """Copy hook PDFs into ``paper_fig_dir``; optionally mirror full ``output_dir`` to ``mirror_dir``."""
    os.makedirs(paper_fig_dir, exist_ok=True)
    for name in ("fig_corruption_strip.pdf", "fig_retrieval_mAP.pdf"):
        src = osp.join(output_dir, name)
        if osp.isfile(src):
            shutil.copy2(src, osp.join(paper_fig_dir, name))

    if mirror_dir:
        os.makedirs(mirror_dir, exist_ok=True)
        for fn in sorted(os.listdir(output_dir)):
            p = osp.join(output_dir, fn)
            if osp.isfile(p):
                shutil.copy2(p, osp.join(mirror_dir, fn))
