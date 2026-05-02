"""
Generate a controlled toy dataset from Market-1501 test images for drift vector validation.

Corruptions follow the **ImageNet-C** taxonomy (Hendrycks & Dietterich, ICLR 2019;
arXiv:1903.12261, https://arxiv.org/abs/1903.12261; reference code
https://github.com/hendrycks/robustness): we compose benchmark-aligned proxies for
**Defocus Blur**, **Motion Blur**, **Pixelate** (bilinear down/up vs. block pixelation),
**Brightness** (gamma tone curve in sRGB-like space), **Gaussian** + **Shot** noise,
    and **JPEG** compression. This is a stylized severity axis for drift-head sanity checks,
    not a calibrated ISP simulation.

    Pipeline order (capture → tone/noise → encode):
    defocus blur → horizontal motion blur → resolution loss (down/up) → gamma (exposure-like)
  → sensor noise → JPEG

See docs/toy_dataset_synthesis.md for the full literature mapping and limitations.

Usage:
    python scripts/generate_toy_dataset.py \
        --eval-source-dir /home/stud/leez/storage/user/reid/data/Market-1501-v15.09.15/bounding_box_test \
        --train-source-dir /home/stud/leez/storage/user/reid/data/Market-1501-v15.09.15/bounding_box_train \
        --output-dir examples/data/ToyCorruptionv4 \
        --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import random
import re
from io import BytesIO
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageFilter

try:
    _RESAMPLE = Image.Resampling.BILINEAR  # Pillow >= 9.1
except AttributeError:
    _RESAMPLE = Image.BILINEAR  # type: ignore[attr-defined]

# Bump when corruption definitions or metadata schema change.
# v3.0 — two source crops per PID (distinct Market-1501 cameras) x 5 severities.
#        filename: {pid:04d}_c{source}s{severity+1}_000001_01.jpg
#        cam_id now encodes SOURCE INDEX, severity encoded via the seq field.
# v4.0 — adds an optional training split written to `bounding_box_train/`
#        sourced from Market-1501 *bounding_box_train*; PIDs are disjoint
#        from the eval split by construction (different Market source pools).
#        metadata.json gains an `eval` and a `train` block under `dataset`,
#        and each entry under `images` carries a `split` field.
TOY_DATASET_VERSION = "4.0"

# Number of distinct source crops picked per identity for the EVAL split
# (from distinct Market-1501 cameras, sourced from bounding_box_test).
NUM_SOURCES_PER_PID = 2

# Number of distinct source crops picked per identity for the TRAIN split
# (sourced from Market-1501 bounding_box_train; train identities typically have >= 4 cameras).
NUM_TRAIN_SOURCES_PER_PID = 4

# Default number of training identities drawn from Market-1501 bounding_box_train.
NUM_TRAIN_IDENTITIES = 750

# Severity 0 = clean. Each row: ImageNet-C-style proxies (see module docstring).
# Keys:
#   blur_sigma     — defocus (Gaussian), analogue to ImageNet-C "Defocus Blur"
#   motion_k       — odd kernel length for horizontal motion blur; 0 = skip ("Motion Blur")
#   downscale      — bilinear down then up; analogue to "Pixelate" (resolution loss)
#   gamma          — >1.0 darkens midtones; analogue to "Brightness" / tone curve
#   poisson_L      — shot-noise scale (0 = skip); analogue to "Shot Noise"
#   gauss_std      — additive noise in [0,1]; analogue to "Gaussian Noise"
#   jpeg_quality   — "JPEG compression"
CORRUPTION_TABLE: Dict[int, Dict[str, Any]] = {
    0: {
        "blur_sigma": 0.0,
        "motion_k": 0,
        "jpeg_quality": 100,
        "downscale": 1.0,
        "gamma": 1.0,
        "gauss_std": 0.0,
        "poisson_L": 0,
    },
    1: {
        "blur_sigma": 1.0,
        "motion_k": 3,
        "jpeg_quality": 80,
        "downscale": 0.75,
        "gamma": 1.07,
        "gauss_std": 0.01,
        "poisson_L": 120,
    },
    2: {
        "blur_sigma": 2.0,
        "motion_k": 5,
        "jpeg_quality": 60,
        "downscale": 0.50,
        "gamma": 1.14,
        "gauss_std": 0.018,
        "poisson_L": 90,
    },
    3: {
        "blur_sigma": 4.0,
        "motion_k": 7,
        "jpeg_quality": 40,
        "downscale": 0.35,
        "gamma": 1.24,
        "gauss_std": 0.028,
        "poisson_L": 60,
    },
    4: {
        "blur_sigma": 8.0,
        "motion_k": 9,
        "jpeg_quality": 20,
        "downscale": 0.25,
        "gamma": 1.36,
        "gauss_std": 0.042,
        "poisson_L": 40,
    },
}

PIPELINE_ORDER = [
    "defocus_gaussian_blur",
    "horizontal_motion_blur",
    "bilinear_down_up",
    "gamma_tone",
    "shot_and_gaussian_noise",
    "jpeg_recompress",
]

LITERATURE_ANCHOR = {
    "primary": "Hendrycks & Dietterich (ICLR 2019), ImageNet-C / common corruptions",
    "arxiv": "https://arxiv.org/abs/1903.12261",
    "code": "https://github.com/hendrycks/robustness",
    "operator_mapping": {
        "defocus_gaussian_blur": "Defocus Blur (blur group)",
        "horizontal_motion_blur": "Motion Blur (blur group)",
        "bilinear_down_up": "Pixelate (digital group; we use bilinear resize, not block pixelation)",
        "gamma_tone": "Brightness (weather group; gamma on encoded RGB)",
        "shot_and_gaussian_noise": "Shot Noise + Gaussian Noise (noise group)",
        "jpeg_recompress": "JPEG compression (digital group)",
    },
}


def _rng_for_image(seed: int, orig_pid: int, level: int,
                   source_idx: int = 1) -> np.random.Generator:
    """Deterministic RNG for noise so reruns match (same seed, pid, source_idx, level).

    ``source_idx`` defaults to 1 for backward compatibility with v2.0 single-source runs.
    """
    h = hashlib.sha256(f"{seed}:{orig_pid}:{source_idx}:{level}".encode()).digest()
    s = int.from_bytes(h[:8], "little", signed=False)
    return np.random.default_rng(s)


def _horizontal_motion_blur_rgb(arr: np.ndarray, kernel_len: int) -> np.ndarray:
    """Horizontal averaging blur along width (axis=1), per ImageNet-C-style motion blur.

    NumPy-only (no SciPy) so the script runs in minimal environments; complexity O(H W k).
    """
    if kernel_len <= 1:
        return arr
    pad = kernel_len // 2
    x = np.pad(arr.astype(np.float64), ((0, 0), (pad, pad), (0, 0)), mode="edge")
    h, w0, c = arr.shape
    k = float(kernel_len)
    out = np.empty((h, w0, c), dtype=np.float64)
    for j in range(w0):
        out[:, j, :] = x[:, j : j + kernel_len, :].sum(axis=1) / k
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _apply_gamma(arr_u8: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-9:
        return arr_u8
    x = arr_u8.astype(np.float64) / 255.0
    x = np.power(np.clip(x, 0.0, 1.0), gamma)
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def _apply_sensor_noise(
    arr_u8: np.ndarray,
    poisson_L: int,
    gauss_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x = arr_u8.astype(np.float64) / 255.0
    if poisson_L and poisson_L > 0:
        lam = np.clip(x * float(poisson_L), 0.0, None)
        x = rng.poisson(lam).astype(np.float64) / float(poisson_L)
        x = np.clip(x, 0.0, 1.0)
    if gauss_std and gauss_std > 0:
        x = x + rng.normal(0.0, gauss_std, size=x.shape)
    x = np.clip(x, 0.0, 1.0)
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def apply_corruption_from_params(
    img: Image.Image,
    params: Dict[str, Any],
    rng: np.random.Generator,
) -> Image.Image:
    """Lower-level helper: apply the six-operator pipeline given an explicit
    params dict (used by both ``apply_corruption`` and the toy-diagnostics
    scripts which need to ablate operators independently).

    Pipeline order (must match ``PIPELINE_ORDER``):
      defocus blur → motion blur → bilinear down/up → gamma → sensor noise → JPEG.
    """
    out = img.copy()

    if params.get("blur_sigma", 0.0) and params["blur_sigma"] > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(params["blur_sigma"])))

    arr = np.array(out, dtype=np.uint8)

    mk = int(params.get("motion_k", 0) or 0)
    if mk > 1:
        arr = _horizontal_motion_blur_rgb(arr, mk)

    out = Image.fromarray(arr, mode="RGB")

    if params.get("downscale", 1.0) < 1.0:
        orig_size = out.size
        ds = float(params["downscale"])
        new_w = max(1, int(orig_size[0] * ds))
        new_h = max(1, int(orig_size[1] * ds))
        out = out.resize((new_w, new_h), _RESAMPLE)
        out = out.resize(orig_size, _RESAMPLE)

    arr = np.array(out, dtype=np.uint8)

    gm = float(params.get("gamma", 1.0))
    if gm != 1.0:
        arr = _apply_gamma(arr, gm)

    pL = int(params.get("poisson_L", 0) or 0)
    gs = float(params.get("gauss_std", 0.0) or 0.0)
    if pL > 0 or gs > 0:
        arr = _apply_sensor_noise(arr, pL, gs, rng)

    out = Image.fromarray(arr, mode="RGB")

    jq = int(params.get("jpeg_quality", 100))
    if jq < 100:
        buf = BytesIO()
        out.save(buf, format="JPEG", quality=jq)
        buf.seek(0)
        out = Image.open(buf).copy()

    return out


def apply_corruption(
    img: Image.Image,
    level: int,
    *,
    orig_pid: int,
    seed: int,
    source_idx: int = 1,
) -> Image.Image:
    """Apply composite corruption at the given severity level (0 = identity)."""
    if level == 0:
        return img.copy()

    params = CORRUPTION_TABLE[level]
    rng = _rng_for_image(seed, orig_pid, level, source_idx=source_idx)
    return apply_corruption_from_params(img, params, rng)


def select_identities(source_dir: str, num_identities: int, seed: int,
                      num_sources: int = NUM_SOURCES_PER_PID):
    """Select identities with >=num_sources distinct cameras and pick one image per camera.

    Returns a dict ``{pid: [fname_source_1, fname_source_2, ...]}`` of length ``num_sources``.
    Source crops are drawn from DISTINCT Market-1501 cameras whenever available, so that the
    two renderings of the same identity capture genuinely different appearance conditions
    (pose, viewpoint, illumination) on top of the synthetic corruption ladder.
    """
    pattern = re.compile(r"([-\d]+)_c(\d)")
    pid_images = {}  # {pid: {cam: [fnames]}}

    for fname in sorted(os.listdir(source_dir)):
        m = pattern.search(fname)
        if not m:
            continue
        pid = int(m.group(1))
        if pid <= 0:
            continue
        cam = int(m.group(2))
        pid_images.setdefault(pid, {}).setdefault(cam, []).append(fname)

    multi_cam_pids = {
        pid: cam_dict for pid, cam_dict in pid_images.items() if len(cam_dict) >= num_sources
    }

    rng = random.Random(seed)
    available = sorted(multi_cam_pids.keys())
    selected = rng.sample(available, min(num_identities, len(available)))
    selected.sort()

    result = {}
    for pid in selected:
        cam_dict = multi_cam_pids[pid]
        # Pick the two (or num_sources) lexicographically-smallest cameras; from each
        # camera pick the lex-smallest filename. Deterministic given the sorted source_dir.
        chosen_cams = sorted(cam_dict.keys())[:num_sources]
        result[pid] = [sorted(cam_dict[c])[0] for c in chosen_cams]

    return result


def generate_eval_split(source_dir: str, output_dir: str, num_identities: int, seed: int,
                        num_sources: int = NUM_SOURCES_PER_PID):
    """Generate the *evaluation* toy corruption split (num_sources crops per PID x 5 severities).

    Output subtree (mirrors prior v3.0 layout for backward compatibility):
      - ``bounding_box_test/`` — all num_sources * 5 crops per PID.
      - ``query_s1/`` — source 1, severity 0 (one clean anchor per PID).
      - ``query_s2/`` — source 2, severity 0 (one clean anchor per PID).
      - ``gallery/``  — everything else.

    Returns ``(images_meta, dataset_info)``; does NOT write metadata.json (the
    orchestrator merges train + eval blocks and writes once).
    """
    selected = select_identities(source_dir, num_identities, seed, num_sources=num_sources)
    print(f"[eval]  Selected {len(selected)} identities from {source_dir} (num_sources={num_sources})")

    pid_map = {orig: new + 1 for new, orig in enumerate(sorted(selected.keys()))}

    dirs = {
        "all": osp.join(output_dir, "bounding_box_test"),
        "query_s1": osp.join(output_dir, "query_s1"),
        "query_s2": osp.join(output_dir, "query_s2"),
        "gallery": osp.join(output_dir, "gallery"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    images_meta = {}
    num_levels = len(CORRUPTION_TABLE)

    for orig_pid, fnames in sorted(selected.items()):
        new_pid = pid_map[orig_pid]
        assert len(fnames) == num_sources, (
            f"select_identities returned {len(fnames)} crops for pid {orig_pid}; expected {num_sources}"
        )

        for source_idx, fname in enumerate(fnames, start=1):
            src_path = osp.join(source_dir, fname)
            img = Image.open(src_path).convert("RGB")

            for level in range(num_levels):
                corrupted = apply_corruption(
                    img, level, orig_pid=orig_pid, seed=seed, source_idx=source_idx
                )

                seq_id = level + 1
                out_name = f"{new_pid:04d}_c{source_idx}s{seq_id}_000001_01.jpg"

                corrupted.save(osp.join(dirs["all"], out_name), quality=95)

                if level == 0 and source_idx in (1, 2):
                    corrupted.save(osp.join(dirs[f"query_s{source_idx}"], out_name), quality=95)
                else:
                    corrupted.save(osp.join(dirs["gallery"], out_name), quality=95)

                images_meta[out_name] = {
                    "split": "eval",
                    "original_pid": orig_pid,
                    "new_pid": new_pid,
                    "source_idx": source_idx,
                    "severity": level,
                    "cam_id": source_idx,
                    "seq_id": seq_id,
                    "source_file": fname,
                }

    dataset_info = {
        "source_dir": source_dir,
        "output_subtree": "bounding_box_test/, query_s1/, query_s2/, gallery/",
        "num_identities": num_identities,
        "num_sources_per_pid": num_sources,
        "source_crop_selection": "lex_sort_distinct_camera_topN",
        "filename_convention": "{pid:04d}_c{source_idx}s{severity+1}_000001_01.jpg",
        "seed": seed,
    }

    total_per_pid = num_sources * num_levels
    total = len(selected) * total_per_pid
    n_query_per_source = len(selected)
    n_gallery = total - num_sources * n_query_per_source
    print(f"[eval]  Generated {total} images ({len(selected)} identities x {num_sources} sources x {num_levels} severities)")
    print(f"[eval]    Query S1 (clean, s=0, source 1): {n_query_per_source} images in {dirs['query_s1']}")
    print(f"[eval]    Query S2 (clean, s=0, source 2): {n_query_per_source} images in {dirs['query_s2']}")
    print(f"[eval]    Gallery (rest):                  {n_gallery} images in {dirs['gallery']}")
    print(f"[eval]    All:                             {total} images in {dirs['all']}")

    return images_meta, dataset_info


def generate_train_split(source_dir: str, output_dir: str, num_identities: int, seed: int,
                         num_sources: int = NUM_TRAIN_SOURCES_PER_PID):
    """Generate the *training* toy corruption split.

    Train PIDs are drawn from a *different* Market-1501 source folder
    (typically ``bounding_box_train``) than the eval PIDs (``bounding_box_test``),
    so the two splits are disjoint by construction at the Market level.

    Output subtree:
      - ``bounding_box_train/`` — all num_sources * 5 crops per train PID.

    No query/gallery sub-split is written — the train pool is consumed
    holistically by the BAU training loop (relabel=True). Returns
    ``(images_meta, dataset_info)``.
    """
    selected = select_identities(source_dir, num_identities, seed, num_sources=num_sources)
    print(f"[train] Selected {len(selected)} identities from {source_dir} (num_sources={num_sources})")

    pid_map = {orig: new + 1 for new, orig in enumerate(sorted(selected.keys()))}

    train_dir = osp.join(output_dir, "bounding_box_train")
    os.makedirs(train_dir, exist_ok=True)

    images_meta = {}
    num_levels = len(CORRUPTION_TABLE)

    for orig_pid, fnames in sorted(selected.items()):
        new_pid = pid_map[orig_pid]
        assert len(fnames) == num_sources, (
            f"select_identities returned {len(fnames)} crops for pid {orig_pid}; expected {num_sources}"
        )

        for source_idx, fname in enumerate(fnames, start=1):
            src_path = osp.join(source_dir, fname)
            img = Image.open(src_path).convert("RGB")

            for level in range(num_levels):
                corrupted = apply_corruption(
                    img, level, orig_pid=orig_pid, seed=seed, source_idx=source_idx
                )

                seq_id = level + 1
                out_name = f"{new_pid:04d}_c{source_idx}s{seq_id}_000001_01.jpg"

                corrupted.save(osp.join(train_dir, out_name), quality=95)

                images_meta[out_name] = {
                    "split": "train",
                    "original_pid": orig_pid,
                    "new_pid": new_pid,
                    "source_idx": source_idx,
                    "severity": level,
                    "cam_id": source_idx,
                    "seq_id": seq_id,
                    "source_file": fname,
                }

    dataset_info = {
        "source_dir": source_dir,
        "output_subtree": "bounding_box_train/",
        "num_identities": num_identities,
        "num_sources_per_pid": num_sources,
        "source_crop_selection": "lex_sort_distinct_camera_topN",
        "filename_convention": "{pid:04d}_c{source_idx}s{severity+1}_000001_01.jpg",
        "seed": seed,
    }

    total = len(selected) * num_sources * num_levels
    print(f"[train] Generated {total} images ({len(selected)} identities x {num_sources} sources x {num_levels} severities) in {train_dir}")

    return images_meta, dataset_info


def generate_dataset(eval_source_dir: str, train_source_dir: str | None,
                     output_dir: str, num_identities: int, num_train_identities: int,
                     seed: int,
                     num_sources: int = NUM_SOURCES_PER_PID,
                     num_train_sources: int = NUM_TRAIN_SOURCES_PER_PID,
                     skip_eval: bool = False, skip_train: bool = False):
    """Orchestrator: emits eval and/or train splits and writes a single merged metadata.json.

    Train and eval filenames may collide (PID integers from different Market source
    folders); ``images`` is keyed by ``"<split>/<filename>"`` for global uniqueness.
    """
    images_meta = {}
    eval_info = None
    train_info = None

    if not skip_eval:
        eval_meta, eval_info = generate_eval_split(
            eval_source_dir, output_dir, num_identities, seed, num_sources=num_sources
        )
        for fname, entry in eval_meta.items():
            images_meta[f"eval/{fname}"] = entry

    if not skip_train:
        if train_source_dir is None:
            raise ValueError("--train-source-dir is required when generating the train split (omit --skip-train to disable).")
        train_meta, train_info = generate_train_split(
            train_source_dir, output_dir, num_train_identities, seed, num_sources=num_train_sources
        )
        for fname, entry in train_meta.items():
            images_meta[f"train/{fname}"] = entry

    dataset_info = {
        "toy_dataset_version": TOY_DATASET_VERSION,
        "schema_version": 4,
        "output_dir": output_dir,
        "pipeline_order": PIPELINE_ORDER,
        "literature_anchor": LITERATURE_ANCHOR,
        "corruption_table": {str(k): v for k, v in CORRUPTION_TABLE.items()},
        "synthesis_doc": "changelogs/toy_dataset_synthesis.md",
        "diagnostics_doc": "changelogs/toy_dataset_asymmetry_diagnostics.md",
    }
    if eval_info is not None:
        dataset_info["eval"] = eval_info
    if train_info is not None:
        dataset_info["train"] = train_info

    metadata = {"dataset": dataset_info, "images": images_meta}

    meta_path = osp.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata: {meta_path}")

    print("\nCorruption levels (toy_dataset_version=%s):" % TOY_DATASET_VERSION)
    for lvl, params in CORRUPTION_TABLE.items():
        print(
            f"  Level {lvl}: blur_σ={params['blur_sigma']}, motion_k={params['motion_k']}, "
            f"jpeg_q={params['jpeg_quality']}, downscale={params['downscale']}, "
            f"gamma={params['gamma']}, poisson_L={params['poisson_L']}, "
            f"gauss_std={params['gauss_std']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate toy corruption dataset for drift validation")
    parser.add_argument(
        "--eval-source-dir",
        type=str,
        default="/home/stud/leez/storage/user/reid/data/Market-1501-v15.09.15/bounding_box_test",
        help="Path to Market-1501 *bounding_box_test* directory (eval-split source).",
    )
    parser.add_argument(
        "--train-source-dir",
        type=str,
        default="/home/stud/leez/storage/user/reid/data/Market-1501-v15.09.15/bounding_box_train",
        help="Path to Market-1501 *bounding_box_train* directory (train-split source).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/examples/data/ToyCorruption",
        help="Output directory for the toy dataset",
    )
    parser.add_argument("--num-eval-identities", type=int, default=50,
                        help="Eval-split identity count (default: 50).")
    parser.add_argument("--num-train-identities", type=int, default=NUM_TRAIN_IDENTITIES,
                        help="Train-split identity count (default: 750).")
    parser.add_argument("--num-eval-sources", type=int, default=NUM_SOURCES_PER_PID,
                        help="Number of distinct source crops per eval PID (default: 2).")
    parser.add_argument("--num-train-sources", type=int, default=NUM_TRAIN_SOURCES_PER_PID,
                        help="Number of distinct source crops per train PID (default: 4).")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip generating the eval split (bounding_box_test/, query_s*/, gallery/).")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip generating the train split (bounding_box_train/).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        eval_source_dir=args.eval_source_dir,
        train_source_dir=args.train_source_dir,
        output_dir=args.output_dir,
        num_identities=args.num_eval_identities,
        num_train_identities=args.num_train_identities,
        seed=args.seed,
        num_sources=args.num_eval_sources,
        num_train_sources=args.num_train_sources,
        skip_eval=args.skip_eval,
        skip_train=args.skip_train,
    )
