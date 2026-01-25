## Finsler Drift Extension (Randers Distance) | 24.01.2026

This repository now includes a Finsler-aware backbone and standalone distance functions to enable asymmetric retrieval in Euclidean space while keeping the existing BAU training pipeline intact.

### What Changed (Summary)

- **New backbone**: `resnet50_finsler` is implemented directly in `bau/models/model.py`.
- **Distance injection**: The model instance owns a `dist_func` (no runtime branching inside hot distance loops).
- **Drift-aware distance**: `finsler_drift_dist` is defined in `bau/loss/triplet.py` and can be attached to any model.
- **Modular evaluation**: Evaluation can include or exclude drift features, controlled by a model flag.
- **Memory bank sizing**: Memory bank can store full concatenated features or only identity features.

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