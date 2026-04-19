# ToyCorruption synthetic dataset — design synthesis

This document describes **ToyCorruption**, a small controlled dataset derived from Market-1501 test crops. It is used in this repository to **sanity-check the drift branch** of `resnet50_finsler` and the asymmetric distance `finsler_drift_dist` under **graded synthetic degradations**. It is **not** a replacement for real multi-camera benchmarks (AG-ReIDv2, Protocol-2, etc.).

---

## 1. Purpose and relation to drift validation (claims C1–C3)

The analysis script [`scripts/toy_dataset_analysis.py`](../scripts/toy_dataset_analysis.py) evaluates three claims on ToyCorruption:

| Claim | Content |
|-------|--------|
| **C1** | The **identity** slice of the embedding should remain relatively stable as synthetic “camera severity” increases (identity distance from the clean reference should not explode erratically). |
| **C2** | The **drift** vector magnitude (or norm) should **correlate monotonically** with corruption severity (Spearman ρ vs. per-image severity). |
| **C3** | **Retrieval**: clean query vs. corrupted gallery at each severity, comparing Euclidean BAU (`resnet50`, \(d_E\)) vs. Finsler BAU (`resnet50_finsler`, \(d_F\) on the full `[identity \| drift]` vector). |

ToyCorruption exists to make C1–C3 **measurable on a fully controlled severity axis** where **identity labels are exact** and **each severity is assigned a distinct synthetic camera ID** (`cam_id = severity + 1`).

---

## 2. Source data and identity selection

- **Source:** Market-1501 `bounding_box_test` directory (filename pattern `PID_cCAMID_...`).
- **Filter:** Only person IDs with images from **at least two distinct camera indices** in the test split are kept (so the sampling logic is consistent with multi-camera structure, even though the toy uses a **single** crop per ID for generation).
- **Sampling:** `num_identities` PIDs are drawn with a fixed `--seed` (reproducible).
- **Representative crop:** For each chosen PID, the **lexicographically first** filename among its images is used as the clean reference (deterministic).

Person IDs are **reindexed** to contiguous IDs `0001`, `0002`, … for the toy output layout.

---

## 3. Severity axis and directory layout

For each identity, the generator writes **five** images at severity levels `0 … 4`:

- **Severity 0:** uncorrupted (reference).
- **Severities 1–4:** progressively stronger composite corruptions.

**Market-1501–style naming** (compatible with standard ReID loaders):

```text
{new_pid:04d}_c{cam_id}s1_000001_01.jpg
```

where **`cam_id = severity + 1`** (so `c1` = clean, `c2`–`c5` = corruption grades).

**Splits:**

- `bounding_box_test/` — all five severities (full set for feature extraction).
- `query/` — severity 0 only (clean probes).
- `gallery/` — severities 1–4 (corrupted gallery entries).

---

## 4. Corruption pipeline (ordered steps)

Implementations live in [`scripts/generate_toy_dataset.py`](../scripts/generate_toy_dataset.py). The **intended** processing order follows a coarse **capture → tone/noise → encode** narrative:

1. **Defocus blur** — isotropic Gaussian blur (`PIL.ImageFilter.GaussianBlur`), parameterized by `blur_sigma`.
2. **Horizontal motion blur** — 1D box average along image width (NumPy sliding window; ImageNet-C **Motion Blur** analogue), kernel length `motion_k` (odd length; `0` skips).
3. **Resolution loss** — bilinear downscale by factor `downscale`, then bilinear upscale to the **original** height and width (fixed crop size for the ReID network).
4. **Gamma tone curve** — per-channel gamma on **8-bit encoded** RGB: \(v' = \mathrm{round}(\mathrm{clip}( (v/255)^{\gamma} \cdot 255))\). Values **`gamma > 1`** darken midtones (exposure-like appearance in sRGB-like space; **not** a full radiometric exposure model).
5. **Sensor noise** — **Poisson (shot)** noise on scaled intensities plus **additive Gaussian** noise in \([0,1]\) space; strengths `poisson_L` and `gauss_std`. Randomness uses a **deterministic** `numpy.random.Generator` seeded from `(dataset seed, original PID, severity)` via SHA-256 (same command line → same images).
6. **JPEG recompression** — save to in-memory buffer with quality `jpeg_quality`, reload (lossy round-trip).

Final files are written as JPEG at quality **95** for the on-disk dataset (additional mild compression).

**Version tag:** `TOY_DATASET_VERSION` in the script (currently `"2.0"`) bumps when the **mathematical definition** of corruptions or **metadata schema** changes.

---

## 5. Literature anchoring (ImageNet-C)

We justify operators by alignment with the **common corruptions** benchmark:

- **Hendrycks & Dietterich**, *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*, **ICLR 2019**.  
  - Paper: [https://arxiv.org/abs/1903.12261](https://arxiv.org/abs/1903.12261)  
  - Reference implementation: [https://github.com/hendrycks/robustness](https://github.com/hendrycks/robustness)

That work defines **fifteen** corruption **types** × **five** severity levels on ImageNet validation images. Our toy uses **fewer** operators but maps each step to the **same taxonomy families**:

| Step in ToyCorruption | ImageNet-C analogue | Deviation (explicit) |
|----------------------|---------------------|----------------------|
| Gaussian defocus blur | **Defocus Blur** | Isotropic 2D Gaussian; ImageNet-C uses its own discretization and severity scaling. |
| Horizontal motion blur | **Motion Blur** | Separable 1D kernel along **x**; full benchmark uses richer kernels. |
| Bilinear down/up | **Pixelate** | Same *intent* (loss of spatial detail); ImageNet-C **pixelate** uses block downsampling—we use bilinear resize pairs. |
| Gamma on RGB | **Brightness** (and related weather / digital tone) | Not identical to their brightness transform; we document **gamma** as tone mapping in encoded space. |
| Poisson + Gaussian | **Shot Noise**, **Gaussian Noise** | Classical split of photon and read noise; strengths are **hand-tuned** to the toy severity ladder, not ImageNet-C severity indices. |
| JPEG | **JPEG compression** | Direct analogue; quality schedule is custom. |

**Robustness leaderboards** (e.g. RobustBench) often report **mCE** on ImageNet-C variants; they reinforce **evaluation norms**, not the exact ToyCorruption recipe.

---

## 6. Severity parameter table (v2.0)

Exact `CORRUPTION_TABLE` (also embedded under `dataset.corruption_table` in `metadata.json`):

| Level | `blur_sigma` | `motion_k` | `jpeg_quality` | `downscale` | `gamma` | `poisson_L` | `gauss_std` |
|------|--------------|------------|----------------|-------------|---------|-------------|-------------|
| 0 | 0.0 | 0 | 100 | 1.0 | 1.0 | 0 | 0.0 |
| 1 | 1.0 | 3 | 80 | 0.75 | 1.07 | 120 | 0.01 |
| 2 | 2.0 | 5 | 60 | 0.50 | 1.14 | 90 | 0.018 |
| 3 | 4.0 | 7 | 40 | 0.35 | 1.24 | 60 | 0.028 |
| 4 | 8.0 | 9 | 20 | 0.25 | 1.36 | 40 | 0.042 |

**Validation snapshot (50 identities, seed 42, Finsler/Euclidean checkpoints passed to `toy_dataset_analysis.py` on the same machine):** per-image Spearman correlation between corruption severity and drift norm was **ρ ≈ 0.44** (p ≪ 0.05) on one Finsler checkpoint run—sufficient to reject flat noise, but **not** a guarantee of pointwise monotonicity for every future checkpoint. Use **≥ ~30–50 identities** for stable correlation estimates.

---

## 7. Metadata (`metadata.json`)

Each generation run writes a single JSON file with:

- **`dataset`** — `toy_dataset_version`, `schema_version`, paths, `seed`, `num_identities`, `pipeline_order`, `literature_anchor` (arXiv + GitHub + operator mapping), embedded `corruption_table`, pointer to this doc.
- **`images`** — per-filename records: `original_pid`, `new_pid`, `severity`, `cam_id`, `source_file`.

Downstream tools should read **`dataset.toy_dataset_version`** when comparing experiments.

---

## 8. Limitations (what we do *not* claim)

1. **Not an ISP model** — No demosaicing, color matrix, CCM, local tone mapping, or HDR pipeline.
2. **8-bit approximations** — Shot noise is applied after gamma on **display-referred** RGB, not linear scene radiance.
3. **Surveillance realism** — CCTV failure modes are **qualitatively** similar (blur, compression, resolution, noise, exposure); **quantitative** definitions follow the **benchmark** literature, not a specific vendor camera.
4. **Monotonicity** — C2 is an **empirical** check; a change in corruption schedule or checkpoint can reduce Spearman correlation.

---

## 9. How to regenerate

**Stale files:** The generator **overwrites** filenames it writes but does **not** delete extra JPGs left from a previous run (e.g. after lowering `--num-identities`). Remove old crops before regenerating, e.g.:

```bash
rm -rf examples/data/ToyCorruption/bounding_box_test/* \
       examples/data/ToyCorruption/query/* \
       examples/data/ToyCorruption/gallery/*
```

From the repository root (adjust paths):

```bash
python scripts/generate_toy_dataset.py \
  --source-dir /path/to/Market-1501/bounding_box_test \
  --output-dir examples/data/ToyCorruption \
  --num-identities 50 \
  --seed 42
```

Then run drift / retrieval analysis (requires trained checkpoints):

```bash
python scripts/toy_dataset_analysis.py \
  --finsler-checkpoint /path/to/resnet50_finsler_best.pth \
  --euclidean-checkpoint /path/to/resnet50_bau_best.pth \
  --dataset-dir examples/data/ToyCorruption \
  --output-dir results/toy_analysis
```

Outputs typically include `drift_identity_metrics.json`, `retrieval_metrics.json`, and PDF figures (`fig_drift_monotonicity.pdf`, `fig_retrieval_mAP.pdf` with per-marker mAP labels, `fig_paper_corruption_strip.pdf`, optional `fig_omega_identity_pca.pdf`, etc.).

---

## 10. References (BibTeX-friendly)

```bibtex
@inproceedings{hendrycks2019benchmarking,
  title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author={Hendrycks, Dan and Dietterich, Thomas},
  booktitle={ICLR},
  year={2019}
}
```
e
Preprint: [https://arxiv.org/abs/1903.12261](https://arxiv.org/abs/1903.12261)

---

## 11. Changelog pointer

Repository-level changes to the generator are logged in [`changelog.md`](../changelog.md) at the project root.
