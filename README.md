# Manifold-Aware Person Re-Identification

This project extends the [BAU (Between-class Alignment and Uniformity)](https://github.com/yoonkicho/BAU) Domain-Generalizable (DG) Person Re-ID framework with **asymmetric Finsler geometry** to model the directional nature of cross-view retrieval in surveillance networks.

Standard ReID models assume a symmetric, Euclidean feature space. This assumption breaks down in real-world deployments where probe-gallery matching is inherently direction-dependent (e.g., low-resolution UAV probe vs. high-resolution CCTV gallery). This project replaces the symmetric Euclidean distance with an asymmetric **Randers metric** from Finsler geometry, capturing camera-topology-aware directional relationships without leaving the Euclidean embedding space.

---

## Contributions

### 1. Finsler Drift Extension — `resnet50_finsler` backbone

A split-head ResNet-50 that produces a structured embedding:

```
output = [identity (2048-d) | drift/omega (variable-d)]
```

- **Identity branch**: standard BN-neck feature for class discrimination.
- **Drift branch** (`FinslerDriftHead`): predicts a drift vector omega constrained to `||omega||_2 < 1` (Randers admissibility). It models the asymmetric displacement induced by camera/domain shift.

The Randers distance between two embeddings `x` and `y` is:

```
d_F(x, y) = d_E(x_id, y_id) + omega_x · y_id
```

where `d_E` is Euclidean distance and `omega_x` is the drift vector of the query. The distance function is injected into the model at instantiation (`model.dist_func`) and used uniformly by trainers and evaluators with no runtime branching.

### 2. Learnable Randers alpha — `AlphaParameter`

A global scalar alpha in [0, 1] that continuously interpolates between the Euclidean baseline (alpha=0) and the full Randers metric (alpha=1), implemented as a learnable parameter with a centered sigmoid parameterization. This guarantees alpha=0 at initialization and keeps alpha within [0, alpha_max] throughout training. The value is logged per epoch via W&B.

**CLI flags:** `--alpha`, `--alpha-init`, `--alpha-max`, `--alpha-temp`

### 3. Gradient-Isolated Training for the Drift Head

A careful gradient flow design prevents the drift head from taking shortcuts at the expense of identity discrimination:

| Loss | Backbone grads | Drift head grads |
|------|:--------------:|:----------------:|
| Cross-Entropy | yes | no |
| Triplet (Finsler) | yes | yes |
| Alignment (identity-only slice) | yes | no |
| Uniformity (identity detached) | no | yes |
| Domain (identity detached) | no | yes |

Additional controls: reduced LR for drift head (`--drift-lr-mult`, default `0.1`) and gradient clipping (`max_norm=1.0`).

### 4. Multi-View AG-ReIDv2 Dataset Splits

AG-ReIDv2 is decomposed into per-viewpoint source domains so the BAU domain loss operates on semantically coherent centroids:

| Dataset key | Camera | Description |
|-------------|--------|-------------|
| `agreidv2_aerial` | C0 | UAV (~21k images, 590 PIDs) |
| `agreidv2_wearable` | C2 | Wearable (~21k, 507 PIDs) |
| `agreidv2_cctv` | C3 | CCTV (~9k, 419 PIDs) |
| `agreidv2_ground` | C2+C3 | Both ground cameras (~30k) |

PIDs are re-labelled to be contiguous after filtering.

---

## Architecture Overview

```
Input image
    |
ResNet-50 backbone
    |-- BN-neck ---------> identity features (2048-d)
    |-- FinslerDriftHead -> drift vector omega (N-d, ||omega|| < 1)

output embedding = [identity | drift]

Distance: d_F(x, y) = d_E(x_id, y_id) + alpha * (omega_x . y_id)
```

---

## Baselines Supported

| `--arch` | Distance | Description |
|----------|----------|-------------|
| `resnet50` | Euclidean | Standard BAU baseline |
| `resnet50_finsler` | Randers (Finsler) | Asymmetric drift-based distance |
| `mobilenetv2` | Euclidean | Lightweight backbone |
| `vit_base_patch16` | Euclidean | Vision Transformer backbone |

---

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Training

**Euclidean baseline:**
```bash
python examples/train_bau.py --arch resnet50 \
    -ds market1501 msmt17 cuhk03 \
    --logs-dir logs/baseline
```

**Finsler model with learnable drift:**
```bash
python examples/train_bau.py --arch resnet50_finsler \
    -ds agreidv2_aerial agreidv2_cctv agreidv2_wearable \
    --drift-lr-mult 0.1 \
    --logs-dir logs/finsler
```

**Finsler model with learnable alpha:**
```bash
python examples/train_bau.py --arch resnet50_finsler \
    --alpha-init 0.0 --alpha-max 1.0 --alpha-temp 1.0 \
    --logs-dir logs/finsler_alpha
```

**Key CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--arch` | `resnet50` | Backbone architecture |
| `--eval-drift` | `true` | Include drift in evaluation ranking |
| `--memory-bank-mode` | `full` | `full` = [id\|drift], `identity` = id only |
| `--drift-lr-mult` | `0.1` | LR multiplier for drift head |
| `--alpha-init` | — | Initial alpha value (0 = Euclidean start) |
| `--alpha-max` | `1.0` | Upper bound for learnable alpha |

---

## Evaluation

```bash
python examples/test.py --arch resnet50_finsler \
    --resume path/to/checkpoint.pth.tar \
    -t market1501
```

Primary metrics: **mAP** and **Rank-1** on unseen target domains (DG protocol).

---

## Project Structure

```
bau/
├── models/
│   └── model.py           # resnet50, resnet50_finsler, mobilenetv2, vit
├── loss/
│   ├── triplet.py         # euclidean_dist, finsler_drift_dist, AlphaParameter
│   └── crossentropy.py
├── datasets/
│   └── agreidv2.py        # view-filtered AG-ReIDv2 splits
├── trainers.py            # BAUTrainer with gradient-isolated drift training
└── evaluators.py          # rank evaluation with pluggable dist_func
examples/
├── train_bau.py
└── test.py
changelogs/
├── learnableAlpha.md      # Learnable alpha implementation notes
└── learnableOmega.md      # Finsler drift extension and stabilisation
```
