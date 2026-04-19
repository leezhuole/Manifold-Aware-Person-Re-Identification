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
        --source-dir /home/stud/leez/storage/user/reid/data/Market-1501-v15.09.15/bounding_box_test \
        --output-dir examples/data/ToyCorruption \
        --num-identities 50 \
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
TOY_DATASET_VERSION = "2.0"

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


def _rng_for_image(seed: int, orig_pid: int, level: int) -> np.random.Generator:
    """Deterministic RNG for noise so reruns match (same seed, pid, level)."""
    h = hashlib.sha256(f"{seed}:{orig_pid}:{level}".encode()).digest()
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


def apply_corruption(
    img: Image.Image,
    level: int,
    *,
    orig_pid: int,
    seed: int,
) -> Image.Image:
    """Apply composite corruption at the given severity level (0 = identity)."""
    if level == 0:
        return img.copy()

    params = CORRUPTION_TABLE[level]
    rng = _rng_for_image(seed, orig_pid, level)
    out = img.copy()

    # 1. Defocus blur (ImageNet-C: Defocus Blur)
    if params["blur_sigma"] and params["blur_sigma"] > 0:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(params["blur_sigma"])))

    arr = np.array(out, dtype=np.uint8)

    # 2. Motion blur (ImageNet-C: Motion Blur)
    mk = int(params.get("motion_k", 0) or 0)
    if mk > 1:
        arr = _horizontal_motion_blur_rgb(arr, mk)

    out = Image.fromarray(arr, mode="RGB")

    # 3. Resolution loss (ImageNet-C: Pixelate analogue)
    if params["downscale"] < 1.0:
        orig_size = out.size
        ds = float(params["downscale"])
        new_w = max(1, int(orig_size[0] * ds))
        new_h = max(1, int(orig_size[1] * ds))
        out = out.resize((new_w, new_h), _RESAMPLE)
        out = out.resize(orig_size, _RESAMPLE)

    arr = np.array(out, dtype=np.uint8)

    # 4. Gamma / exposure-like tone (ImageNet-C: Brightness family)
    gm = float(params.get("gamma", 1.0))
    if gm != 1.0:
        arr = _apply_gamma(arr, gm)

    # 5. Sensor noise (ImageNet-C: Shot + Gaussian)
    pL = int(params.get("poisson_L", 0) or 0)
    gs = float(params.get("gauss_std", 0.0) or 0.0)
    if pL > 0 or gs > 0:
        arr = _apply_sensor_noise(arr, pL, gs, rng)

    out = Image.fromarray(arr, mode="RGB")

    # 6. JPEG (ImageNet-C: JPEG compression)
    jq = int(params["jpeg_quality"])
    if jq < 100:
        buf = BytesIO()
        out.save(buf, format="JPEG", quality=jq)
        buf.seek(0)
        out = Image.open(buf).copy()

    return out


def select_identities(source_dir: str, num_identities: int, seed: int):
    """Select identities with at least 2 cameras and pick one image per identity."""
    pattern = re.compile(r"([-\d]+)_c(\d)")
    pid_images = {}

    for fname in sorted(os.listdir(source_dir)):
        m = pattern.search(fname)
        if not m:
            continue
        pid = int(m.group(1))
        if pid <= 0:
            continue
        if pid not in pid_images:
            pid_images[pid] = []
        pid_images[pid].append(fname)

    multi_cam_pids = {}
    for pid, fnames in pid_images.items():
        cams = set()
        for f in fnames:
            m = pattern.search(f)
            if m is not None:
                cams.add(int(m.group(2)))
        if len(cams) >= 2:
            multi_cam_pids[pid] = fnames

    rng = random.Random(seed)
    available = sorted(multi_cam_pids.keys())
    selected = rng.sample(available, min(num_identities, len(available)))
    selected.sort()

    result = {}
    for pid in selected:
        result[pid] = sorted(multi_cam_pids[pid])[0]

    return result


def generate_dataset(source_dir: str, output_dir: str, num_identities: int, seed: int):
    """Generate the full toy corruption dataset."""
    selected = select_identities(source_dir, num_identities, seed)
    print(f"Selected {len(selected)} identities from {source_dir}")

    pid_map = {orig: new + 1 for new, orig in enumerate(sorted(selected.keys()))}

    dirs = {
        "all": osp.join(output_dir, "bounding_box_test"),
        "query": osp.join(output_dir, "query"),
        "gallery": osp.join(output_dir, "gallery"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    images_meta = {}
    num_levels = len(CORRUPTION_TABLE)

    for orig_pid, fname in sorted(selected.items()):
        new_pid = pid_map[orig_pid]
        src_path = osp.join(source_dir, fname)
        img = Image.open(src_path).convert("RGB")

        for level in range(num_levels):
            corrupted = apply_corruption(img, level, orig_pid=orig_pid, seed=seed)

            cam_id = level + 1
            out_name = f"{new_pid:04d}_c{cam_id}s1_000001_01.jpg"

            out_path = osp.join(dirs["all"], out_name)
            corrupted.save(out_path, quality=95)

            if level == 0:
                corrupted.save(osp.join(dirs["query"], out_name), quality=95)
            else:
                corrupted.save(osp.join(dirs["gallery"], out_name), quality=95)

            images_meta[out_name] = {
                "original_pid": orig_pid,
                "new_pid": new_pid,
                "severity": level,
                "cam_id": cam_id,
                "source_file": fname,
            }

    dataset_info = {
        "toy_dataset_version": TOY_DATASET_VERSION,
        "schema_version": 2,
        "source_dir": source_dir,
        "output_dir": output_dir,
        "num_identities": num_identities,
        "seed": seed,
        "pipeline_order": PIPELINE_ORDER,
        "literature_anchor": LITERATURE_ANCHOR,
        "corruption_table": {str(k): v for k, v in CORRUPTION_TABLE.items()},
        "synthesis_doc": "docs/toy_dataset_synthesis.md",
    }

    metadata = {"dataset": dataset_info, "images": images_meta}

    meta_path = osp.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    total = len(selected) * num_levels
    print(f"Generated {total} images ({len(selected)} identities x {num_levels} severity levels)")
    print(f"  Query (clean):   {len(selected)} images in {dirs['query']}")
    print(f"  Gallery (corr.): {len(selected) * (num_levels - 1)} images in {dirs['gallery']}")
    print(f"  All:             {total} images in {dirs['all']}")
    print(f"  Metadata:        {meta_path}")

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
        "--source-dir",
        type=str,
        default="/home/stud/leez/storage/user/reid/data/Market-1501-v15.09.15/bounding_box_test",
        help="Path to Market-1501 bounding_box_test directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification/examples/data/ToyCorruption",
        help="Output directory for the toy dataset",
    )
    parser.add_argument("--num-identities", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(args.source_dir, args.output_dir, args.num_identities, args.seed)
