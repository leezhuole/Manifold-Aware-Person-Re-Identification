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
    """One-row strip: five crops for PID 1, source 1 at severities 0..4; only ``$s=k$`` titles.

    v3.0 layout: ``0001_c1s{sev+1}_...jpg`` (cam=source_idx, seq=severity+1). Falls back
    to the v2.0 layout ``0001_c{sev+1}s1_...jpg`` if the v3 filename is missing.
    """
    bb_dir = osp.join(dataset_dir, "bounding_box_test")
    n_sev = 5

    def _pick(sev: int) -> str:
        v3 = f"0001_c1s{sev + 1}_000001_01.jpg"
        v2 = f"0001_c{sev + 1}s1_000001_01.jpg"
        if osp.exists(osp.join(bb_dir, v3)):
            return v3
        return v2

    fnames = [_pick(sev) for sev in range(n_sev)]

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


def make_cross_severity_heatmap(
    matrix,
    output_path: str,
    *,
    severities=(0, 1, 2, 3, 4),
    cbar_label: str = "mAP (%)",
    cmap: str = "viridis",
    value_scale: float = 100.0,
    vmin=None,
    vmax=None,
    diverging: bool = False,
) -> None:
    """Render a 5x5 (sigma_q, sigma_g) heatmap. Diverging=True centres the colormap at 0."""
    arr = np.array(matrix, dtype=np.float64) * value_scale
    n = len(severities)

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    if diverging:
        lim = np.nanmax(np.abs(arr)) if np.any(~np.isnan(arr)) else 1.0
        lim = max(lim, 1e-6)
        im = ax.imshow(arr, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="equal")
    else:
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([str(s) for s in severities], fontsize=FONT_TICK)
    ax.set_yticklabels([str(s) for s in severities], fontsize=FONT_TICK)
    ax.set_xlabel(r"Gallery severity $\sigma_g$", fontsize=FONT_LABEL)
    ax.set_ylabel(r"Query severity $\sigma_q$", fontsize=FONT_LABEL)

    for i in range(n):
        for j in range(n):
            v = arr[i, j]
            if np.isnan(v):
                txt = "—"
                color = "0.3"
            else:
                txt = f"{v:.1f}"
                # Text color flips on dark backgrounds
                if diverging:
                    color = "black"
                else:
                    norm_v = (v - np.nanmin(arr)) / max(np.nanmax(arr) - np.nanmin(arr), 1e-9)
                    color = "white" if norm_v < 0.45 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=FONT_ANNOT, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=FONT_TICK)
    cbar.set_label(cbar_label, fontsize=FONT_LABEL)
    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def make_cosine_block_bar(
    block_means: dict,
    output_path: str,
) -> None:
    """Bar chart of the 8-way (pid x source x severity) block means of the drift cosine."""
    order = []
    for P in ("same_pid", "diff_pid"):
        for S in ("same_src", "diff_src"):
            for Sig in ("same_sev", "diff_sev"):
                order.append(f"{P}__{S}__{Sig}")
    means = [block_means.get(k, {}).get("mean", float("nan")) for k in order]
    stds = [block_means.get(k, {}).get("std", float("nan")) for k in order]
    labels = [
        "pid=\npid=\nsev=",  # placeholder, overridden below
    ]
    short = []
    for k in order:
        p, s, sig = k.split("__")
        short.append(
            f"{'P' if p=='same_pid' else 'p'}/"
            f"{'S' if s=='same_src' else 's'}/"
            f"{'Σ' if sig=='same_sev' else 'σ'}"
        )

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    xs = np.arange(len(order))
    colors = [COLOR_FINS if k.startswith("same_pid") else COLOR_EUCL for k in order]
    ax.bar(xs, means, yerr=stds, color=colors, alpha=0.85, capsize=3, edgecolor="black", linewidth=0.4)
    ax.axhline(0.0, color="0.5", linewidth=0.6, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(short, fontsize=FONT_TICK, rotation=0)
    ax.set_ylabel(r"Mean drift cosine $\bar{C}$", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.text(
        0.02, 0.97,
        r"Upper-case $=$ same; lower-case $=$ diff. " + "\n" + r"Axes: P(id), S(ource), $\Sigma$(ev).",
        transform=ax.transAxes, fontsize=FONT_ANNOT, va="top", ha="left", color="0.3",
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def make_drift_absorption_plot(
    per_severity: dict,
    analytic_null_B: float,
    analytic_null_C: float,
    shuffle_null: dict,
    output_path: str,
) -> None:
    """Plot eta_B and eta_C vs severity k, with analytic and shuffle nulls as reference bands."""
    ks = sorted(int(k) for k in per_severity.keys() if per_severity.get(k, {}).get("n", 0) > 0)
    if not ks:
        return
    etaB = [per_severity[k]["eta_B_mean"] for k in ks]
    etaB_std = [per_severity[k].get("eta_B_std", 0.0) for k in ks]
    etaC = [per_severity[k]["eta_C_mean"] for k in ks]
    etaC_std = [per_severity[k].get("eta_C_std", 0.0) for k in ks]

    fig, ax = plt.subplots(figsize=(FIGMAP_W, FIGMAP_H))
    ax.errorbar(ks, etaB, yerr=etaB_std, color=COLOR_FINS, marker="o", linewidth=LINE_W,
                markersize=MARKER_S, capsize=3, label=r"$\bar{\eta}_B$ (midpoint)")
    ax.errorbar(ks, etaC, yerr=etaC_std, color=COLOR_EUCL, marker="s", linewidth=LINE_W,
                markersize=MARKER_S, linestyle="--", capsize=3, label=r"$\bar{\eta}_C$ (2-D GS)")

    # Analytic nulls as horizontal lines
    ax.axhline(analytic_null_B, color=COLOR_FINS, linewidth=0.6, linestyle=":",
               label=r"$m/d_{\mathrm{id}}$ isotropic null (B)")
    ax.axhline(analytic_null_C, color=COLOR_EUCL, linewidth=0.6, linestyle=":",
               label=r"$m/d_{\mathrm{id}}$ isotropic null (C)")

    # Shuffle null bands (mean ± std) per severity
    shuffle_xs, shuffle_B_mean, shuffle_B_std = [], [], []
    shuffle_C_mean, shuffle_C_std = [], []
    for k in ks:
        d = shuffle_null.get(k) or shuffle_null.get(str(k)) or {}
        if d.get("n_reps", 0) > 0:
            shuffle_xs.append(k)
            shuffle_B_mean.append(d.get("eta_B_null_mean", float("nan")))
            shuffle_B_std.append(d.get("eta_B_null_std", 0.0))
            shuffle_C_mean.append(d.get("eta_C_null_mean", float("nan")))
            shuffle_C_std.append(d.get("eta_C_null_std", 0.0))
    if shuffle_xs:
        sb_m = np.array(shuffle_B_mean); sb_s = np.array(shuffle_B_std)
        sc_m = np.array(shuffle_C_mean); sc_s = np.array(shuffle_C_std)
        ax.fill_between(shuffle_xs, sb_m - sb_s, sb_m + sb_s, color=COLOR_FINS, alpha=0.12,
                        label="shuffle null (B)")
        ax.fill_between(shuffle_xs, sc_m - sc_s, sc_m + sc_s, color=COLOR_EUCL, alpha=0.12,
                        label="shuffle null (C)")

    ax.set_yscale("log")
    ax.set_xlabel(r"Severity $k$", fontsize=FONT_LABEL)
    ax.set_ylabel(r"Absorption $\bar{\eta}$", fontsize=FONT_LABEL)
    ax.set_xticks(ks)
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_ANNOT, loc="best", framealpha=0.9, ncol=2)
    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def make_drift_alignment_plot(
    per_severity: dict,
    shuffle_null: dict,
    output_path: str,
) -> None:
    """Two-row panel: (top) mean drift-alignment cosine $\\bar{c}_k = \\overline{\\cos(\\omega_0, \\omega_k)}$
    as the validity gauge; (bottom) $\\bar{\\eta}_{\\mathrm{ref}}$ and $\\bar{\\eta}_B$ on log-scale
    with shuffle-null bands. See changelogs/toy_dataset_reference_projector_supplement.md.
    """
    ks = sorted(int(k) for k in per_severity.keys() if per_severity.get(k, {}).get("n", 0) > 0)
    if not ks:
        return

    def _get(k, key, default=float("nan")):
        row = per_severity.get(k) or per_severity.get(str(k)) or {}
        v = row.get(key, default)
        return float(v) if v is not None else default

    cos_mean = [_get(k, "cos_omega_0_k_mean") for k in ks]
    cos_std = [_get(k, "cos_omega_0_k_std", 0.0) for k in ks]
    eta_ref = [_get(k, "eta_ref_mean") for k in ks]
    eta_ref_std = [_get(k, "eta_ref_std", 0.0) for k in ks]
    eta_B = [_get(k, "eta_B_mean") for k in ks]
    eta_B_std = [_get(k, "eta_B_std", 0.0) for k in ks]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(FIGMAP_W, FIGMAP_H * 1.55),
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )

    ax_top.errorbar(ks, cos_mean, yerr=cos_std, color=COLOR_FINS, marker="o",
                    linewidth=LINE_W, markersize=MARKER_S, capsize=3,
                    label=r"$\bar{c}_k = \overline{\cos(\omega_0,\omega_k)}$")
    ax_top.axhline(1.0, color="0.6", linewidth=0.5, linestyle=":")
    ax_top.axhline(0.0, color="0.6", linewidth=0.5, linestyle=":")
    ax_top.set_ylabel(r"$\bar{c}_k$", fontsize=FONT_LABEL)
    ax_top.tick_params(labelsize=FONT_TICK)
    ax_top.set_ylim(-0.05, 1.05)
    ax_top.legend(fontsize=FONT_ANNOT, loc="lower left", framealpha=0.9)

    ax_bot.errorbar(ks, eta_ref, yerr=eta_ref_std, color=COLOR_FINS, marker="o",
                    linewidth=LINE_W, markersize=MARKER_S, capsize=3,
                    label=r"$\bar{\eta}_{\mathrm{ref}}$ (reference-point)")
    ax_bot.errorbar(ks, eta_B, yerr=eta_B_std, color=COLOR_EUCL, marker="s",
                    linewidth=LINE_W, markersize=MARKER_S, capsize=3, linestyle="--",
                    label=r"$\bar{\eta}_B$ (midpoint)")

    sh_xs, sh_refm, sh_refs, sh_Bm, sh_Bs = [], [], [], [], []
    for k in ks:
        d = shuffle_null.get(k) or shuffle_null.get(str(k)) or {}
        if d.get("n_reps", 0) > 0:
            sh_xs.append(k)
            sh_refm.append(d.get("eta_ref_null_mean", float("nan")))
            sh_refs.append(d.get("eta_ref_null_std", 0.0))
            sh_Bm.append(d.get("eta_B_null_mean", float("nan")))
            sh_Bs.append(d.get("eta_B_null_std", 0.0))
    if sh_xs:
        m = np.array(sh_refm); s = np.array(sh_refs)
        ax_bot.fill_between(sh_xs, np.clip(m - s, 1e-9, None), m + s, color=COLOR_FINS,
                            alpha=0.12, label="shuffle null (ref)")
        m = np.array(sh_Bm); s = np.array(sh_Bs)
        ax_bot.fill_between(sh_xs, np.clip(m - s, 1e-9, None), m + s, color=COLOR_EUCL,
                            alpha=0.12, label="shuffle null (B)")

    ax_bot.set_yscale("log")
    ax_bot.set_xlabel(r"Severity $k$", fontsize=FONT_LABEL)
    ax_bot.set_ylabel(r"Absorption $\bar{\eta}$", fontsize=FONT_LABEL)
    ax_bot.set_xticks(ks)
    ax_bot.tick_params(labelsize=FONT_TICK)
    ax_bot.legend(fontsize=FONT_ANNOT, loc="best", framealpha=0.9, ncol=2)

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
