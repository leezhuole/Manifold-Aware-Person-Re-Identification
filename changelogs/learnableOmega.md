## Finsler Drift Extension (Randers Distance) | 24.01.2026

This repository now includes a Finsler-aware backbone and standalone distance functions to enable asymmetric retrieval in Euclidean space while keeping the existing BAU training pipeline intact.

### What Changed (Summary)

- **New backbone**: `resnet50_finsler` is implemented directly in `bau/models/model.py`.
- **Distance injection**: The model instance owns a `dist_func` (no runtime branching inside hot distance loops).
- **Drift-aware distance**: `finsler_drift_dist` is defined in `bau/loss/triplet.py` and can be attached to any model.
- **Modular evaluation**: Evaluation can include or exclude drift features, controlled by a model flag.
- **Memory bank sizing**: Memory bank can store full concatenated features or only identity features.

---

## Finsler Training Stabilisation & Multi-Domain AG-ReIDv2 | 10.02.2026

This update addresses the performance gap between the Finsler (`resnet50_finsler`) model
and the Euclidean baseline on AG-ReIDv2 cross-view re-identification.

### Root Cause Analysis

Three issues were identified:

1. **Alignment loss bug** — the fallback check `f_w.size(1) == 2048` was *never* true
   for Finsler embeddings (which are 4096-d), so alignment was computed over the full
   `[identity|drift]` concatenation using plain Euclidean distance.  This created
   conflicting gradients: alignment tried to make augmentations of the same image
   produce identical drift vectors, while domain/uniform losses relied on drift
   asymmetry.

2. **Single-domain training** — AG-ReIDv2 was treated as one domain despite having
   three fundamentally different viewpoints (UAV, Wearable, CCTV).  The domain centroid
   in the memory bank averaged features from all viewpoints, producing an uninformative
   centroid that confused the domain loss.

3. **Drift head overfitting (shortcut)** — the drift head received the same learning
   rate as the backbone.  Because the repulsion losses (uniform, domain) can be cheaply
   reduced by scaling drift magnitudes, the drift head "took the shortcut" without
   improving identity discrimination.

### Changes

#### 1. View-filtered AG-ReIDv2 dataset splits (`bau/datasets/agreidv2.py`, `__init__.py`)

A new `view_filter` parameter in `AG_ReID_v2.__init__` filters the `train_all`
directory by camera ID.  Pre-registered dataset keys:

| Key                  | Cameras | Description                      |
|----------------------|---------|----------------------------------|
| `agreidv2_aerial`    | C0      | UAV only (~21k images, 590 PIDs) |
| `agreidv2_wearable`  | C2      | Wearable only (~21k, 507 PIDs)   |
| `agreidv2_cctv`      | C3      | CCTV only (~9k, 419 PIDs)        |
| `agreidv2_ground`    | C2, C3  | Both ground cameras (~30k)       |

PIDs are re-labelled to be contiguous after filtering.

**Usage** — treat each view as a separate source to get per-view domain IDs:

```bash
-ds agreidv2_aerial agreidv2_cctv agreidv2_wearable
```

This gives `MultiSourceTrainDataset` three domain IDs (0, 1, 2), enabling the
domain and uniformity losses to operate per-viewpoint.

#### 2. Identity-only alignment loss (`bau/trainers.py`)

For Finsler models, `align_loss` now slices embeddings to identity-only
(`f[:, :identity_dim]`) and uses `euclidean_dist`.  The `identity_dim` is read
from the model at trainer init time, replacing the broken `f.size(1) == 2048` check.

Rationale: alignment enforces augmentation invariance.  The drift vector is structural
(models camera/domain shift) and should *not* be invariant to colour jitter or random
erasing.  By isolating the identity part, the alignment gradient no longer conflicts
with the drift learning signal.

#### 3. Drift head gradient control (`bau/trainers.py`, `examples/train_bau.py`)

Two mechanisms prevent the drift shortcut:

- **Reduced LR**: drift head parameters get `lr × drift_lr_mult` (default 0.1).
  New CLI flag: `--drift-lr-mult <float>`.
- **Gradient clipping**: `clip_grad_norm_(drift_head.parameters(), max_norm=1.0)`
  is applied after `scaler.unscale_()` and before `scaler.step()`.

#### 4. Stop-gradient identity in repulsion losses (`bau/trainers.py`)

The `finsler_drift_dist` asymmetry term $\omega_x \cdot y_\text{id}$ back-propagates
through both the drift head **and** the backbone (via $y_\text{id}$).  As drift norms
grew during training, the identity-gradient magnitude from uniform/domain losses
increased proportionally, overpowering the alignment signal and causing the alignment
loss to rise — eventually collapsing mAP.

Fix: before computing uniform and domain losses, the identity portion of `f_w` / `f_s`
is **detached** from the computation graph:

```python
f_w_repulse = torch.cat([f_w[:, :identity_dim].detach(), f_w[:, identity_dim:]], dim=1)
```

This creates a clean gradient separation:

| Component | Receives gradients from |
|-----------|------------------------|
| Backbone (identity) | CE, Triplet, Alignment |
| Drift head (ω) | CE (via logits), Triplet, Uniform, Domain |

The backbone is trained purely for identity discrimination; the drift head alone
learns the asymmetric repulsion structure.

#### 5. Cosmetic: view-filtered dataset stats (`bau/datasets/agreidv2.py`)

View-filtered source datasets (e.g. `agreidv2_aerial`) now only print training
statistics.  The inherited query/gallery from the full dataset were misleading
because evaluation uses the target dataset's query/gallery, not the source's.

### CLI Additions

```
--drift-lr-mult FLOAT   LR multiplier for drift_head params (default: 0.1)
```

### Gradient flow summary (Finsler model)

| Loss | Input features | Backbone grads | Drift grads |
|------|---------------|---------------|-------------|
| CE | logits (from identity) | ✓ | ✗ |
| Triplet | `[identity\|drift]` | ✓ | ✓ |
| Alignment | identity-only slice | ✓ | ✗ |
| Uniform | identity-detached `[id.detach()\|drift]` | ✗ | ✓ |
| Domain | identity-detached `[id.detach()\|drift]` | ✗ | ✓ |

- Triplet loss still uses the combined embedding via `finsler_drift_dist`.
- Evaluation pipeline unchanged.

### New Model: `resnet50_finsler`

The backbone splits into two branches after pooling:

- **Identity branch**: standard `bn_neck` feature for class discrimination.
- **Drift branch**: predicts a drift vector $
\omega$
 constrained to $
\lVert \omega \rVert_2 < 1$
 (Randers requirement). The drift is concatenated with the identity feature and used by the Finsler distance.

The output embedding is:

```
[identity (2048) | drift (2048)]
```

By default, the drift branch is used during evaluation ranking, but this is modular and can be toggled (see CLI options below).

### Standalone Distance Functions

`bau/loss/triplet.py` now contains:

- `euclidean_dist(x, y, alpha=None)`
- `finsler_drift_dist(x, y, alpha=None)`

The model chooses its distance function at instantiation (`model.dist_func`). Trainers and evaluators simply call the model’s `dist_func`, avoiding conditionals inside the distance computation.

### Key Integration Points

- **Trainer** uses `model.dist_func` for triplet, alignment, uniformity, and domain losses.
- **Evaluator** uses `model.dist_func` for ranking and re-ranking.
- **Memory Bank** uses `model.memory_bank_dim` to determine feature dimensionality.

### CLI Usage

Select the model via `--arch`. New options:

```
--arch resnet50_finsler
--eval-drift true|false
--memory-bank-mode full|identity
```

- `--eval-drift`: toggles whether drift is included during evaluation ranking.
- `--memory-bank-mode`:
    - `full`: store concatenated `[identity|drift]` (default).
    - `identity`: store only identity features.

### Manifold Compatibility

`resnet50_finsler` is designed for Euclidean feature space with Finsler distance.
If `--arch resnet50_finsler` is used, `--manifold-aware` is forced to Euclidean.

### Sanity Checks and Verbosity

`examples/train_bau.py` logs the following (verbosity level ≥ 2):

- `dist_func` name detected from the model
- `embedding_dim` and `memory_bank_dim`

Additionally, incompatible configurations are explicitly rejected:

- Finsler model with a manifold-aware backbone triggers a user-facing error.