---
name: Finsler Port to ReNorm2 — Comprehensive Modification Analysis
overview: >
  Rigorous documentation and critical analysis of all modifications made to the ReNorm2 codebase
  to port the Finsler/Randers asymmetric distance from the BAU (Manifold-Aware-Person-Re-Identification)
  framework. Covers every file touched, every function altered, mathematical correspondence to the
  BAU source, discrepancies, evaluation-time geometric concerns, and open issues that may confound
  the experimental comparison.
date: 2026-03-19
base_commit: ReNorm2 clean clone of https://github.com/3699nr/ReNorm (Euclidean baseline reproduced)
source_repo: /home/stud/leez/reid/src/Manifold-Aware-Person-Re-Identification
target_repo: /home/stud/leez/reid/src/ReNorm2
---

# Finsler Port to ReNorm2 — Comprehensive Modification Analysis

---

## Part 0: Motivation and Framework Selection

### Why port at all?

The Finsler/Randers asymmetric distance hypothesis — that directional retrieval bias in person re-identification can be modelled by an asymmetric metric in the embedding space — was developed and tested exclusively within BAU's loss landscape (alignment + uniformity + domain-specific uniformity). The [analysis of 2026-03-18](analysis_18_03.md) identified that BAU's uniform loss actively suppresses the asymmetric component (drives alpha → 0), creating a confound: does the Finsler signal fail because the hypothesis is wrong, or because BAU's loss landscape is hostile to asymmetry?

Porting to a framework with a fundamentally different training regime isolates the Finsler hypothesis from BAU's loss-specific pathology.

### Why ReNorm?

A systematic comparison of DG-ReID SOTA methods identified ReNorm [Lu et al., "Rethinking Normalization Layers for Domain Generalizable Person Re-identification," ECCV 2024] as the optimal target for the following reasons:

1. **Orthogonality of contributions.** ReNorm's novelty is in normalization layers (Remix Normalization blending IN/BN, Emulation Normalization simulating domain shift). BAU's novelty is in loss functions (alignment/uniformity/domain). Finsler's novelty is in distance geometry. These three axes are largely independent, enabling a clean factorial analysis.
2. **Shared backbone.** Both use ResNet50 with stride-1 layer4, enabling parameter-count-matched comparison.
3. **Protocol-2 benchmarks.** Both report on the same leave-one-out Protocol-2 DG-ReID evaluation (Market-1501, MSMT17, CUHK03-NP).
4. **Simple loss landscape.** ReNorm uses only CE + triplet (Euclidean). There is no alignment loss, no uniformity loss, and no domain-specific uniformity. This eliminates the primary mechanism identified as hostile to Finsler asymmetry in BAU.

Alternatives considered and rejected: ACL (unstable codebase), MetaBIN (meta-learning complicates attribution), META (second-order methods), QAConv (attention-based, architecturally distant), GDNorm (normalization-based but inferior results).

---

## Part 1: Inventory of All Modifications

### Files Created (New)

| File | Purpose |
|------|---------|
| `fastreid/modeling/heads/finsler_head.py` | `FinslerDriftHead`, `scale_drift_vector`, `FinslerEmbeddingHead` |
| `configs/ReNorm_Finsler.yml` | Full training config for Finsler-enabled ReNorm |
| `finsler_single.sbatch` | SLURM array job for 3 Protocol-2 leave-one-out runs |
| `changelog.md` | Port documentation |

### Files Modified (Existing)

| File | Functions/Sections Altered |
|------|----------------------------|
| `fastreid/modeling/heads/__init__.py` | Added import of `FinslerEmbeddingHead` |
| `fastreid/modeling/losses/triplet_loss.py` | Added `finsler_drift_dist`, `FinslerTripletLoss` |
| `fastreid/modeling/meta_arch/ReNorm.py` | `ARCH_RENORM.__init__`, `ARCH_RENORM.losses` |
| `fastreid/config/defaults.py` | Added `MODEL.FINSLER` config group |
| `fastreid/utils/compute_dist.py` | `build_dist` assertion, added `compute_finsler_distance` |
| `fastreid/evaluation/reid_evaluation.py` | `ReidEvaluator.evaluate` — `identity_dim` kwarg |
| `tools/train_net.py` | `setup()` — config file argument parsing |

### Files NOT Modified (Verified Unchanged)

- `fastreid/modeling/backbones/resnet_renorm.py` — backbone architecture
- `fastreid/modeling/backbones/batch_norm.py` — RN/EN normalization modules
- `fastreid/modeling/losses/cross_entroy_loss.py` — CE loss
- `fastreid/data/` — data pipeline and augmentation
- `fastreid/engine/` — training loop, optimizer, scheduler
- `fastreid/evaluation/evaluator.py` — evaluation orchestration (RN+EN fusion logic)
- `configs/ReNorm.yml` — original Euclidean config

---

## Part 2: Detailed Modification Analysis

### Modification 1: `fastreid/modeling/heads/finsler_head.py` (New File)

**Source:** `bau/models/model.py` lines 120–172 (`FinslerDriftHead`, `scale_drift_vector`), lines 229–384 (`resnet50_finsler` forward path).

#### 1a: `scale_drift_vector(drift, max_norm=0.95, eps=1e-6)`

**BAU original** (`bau/models/model.py` lines 151–172):
```python
def scale_drift_vector(drift, max_norm=0.95, eps=1e-6):
    norms = torch.norm(drift, p=2, dim=1, keepdim=True)
    scaling_factor = torch.sigmoid(norms)
    unit_drift = drift / (norms + eps)
    drift = unit_drift * (scaling_factor * max_norm)
    return drift
```

**ReNorm2 port** (`fastreid/modeling/heads/finsler_head.py` lines 21–25):
```python
def scale_drift_vector(drift, max_norm=0.95, eps=1e-6):
    norms = torch.norm(drift, p=2, dim=1, keepdim=True)
    scaling_factor = torch.sigmoid(norms)
    unit_drift = drift / (norms + eps)
    return unit_drift * (scaling_factor * max_norm)
```

**Correspondence:** Exact. The function is copied verbatim. The only change is stylistic (direct return vs. intermediate variable).

**Mathematical specification:**

$$
\text{scale}(\boldsymbol{\omega}) = \frac{\boldsymbol{\omega}}{\|\boldsymbol{\omega}\|_2 + \epsilon} \cdot \sigma(\|\boldsymbol{\omega}\|_2) \cdot \omega_{\max}
$$

where $\sigma$ is the sigmoid function. The output norm is bounded: $\|\text{scale}(\boldsymbol{\omega})\| \in [0, \omega_{\max})$ with $\omega_{\max} = 0.95$ by default.

**Critical note (inherited from BAU):** At initialization, when drift norms are small due to `init.normal_(weight, std=0.001)`, the effective drift norm is approximately $\sigma(\|\boldsymbol{\omega}\|) \cdot 0.95 \approx 0.5 \cdot 0.95 = 0.475$. This is not near-zero. However, because the identity features are L2-normalized (norm exactly 1.0), a drift norm of 0.475 represents ~47.5% of the identity scale, which is substantial at initialization. The sigmoid gating means drift can never start from zero and must always overcome the $\sigma(0) = 0.5$ floor.

#### 1b: `FinslerDriftHead(nn.Module)`

**BAU original** (`bau/models/model.py` lines 120–148):
```python
class FinslerDriftHead(nn.Module):
    def __init__(self, input_dim, output_dim, max_norm=0.95):
        super().__init__()
        hidden_dim = max(1, input_dim // 2)
        self.max_norm = max_norm
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )
        init.normal_(self.block[-1].weight, std=0.001)
```

**ReNorm2 port** (`fastreid/modeling/heads/finsler_head.py` lines 28–42): Exact copy. Architecture: `Linear(2048 → 1024, no bias) → ReLU → Linear(1024 → 2048, no bias)`. Last layer initialized to near-zero.

**What was NOT ported:** `DomainConditionedDriftHead` (`bau/models/model.py` lines 175–227). This is deliberate — the domain-conditioned path has no empirical validation in BAU and would introduce confounds (domain embedding, domain classifier, domain gate) that violate the minimality constraint of this port.

**Parameter count:** `Linear(2048, 1024)` = 2,097,152 params. `Linear(1024, 2048)` = 2,097,152 params. Total drift head: **4,194,304 parameters** (4.19M). For reference, the ResNet50 backbone has ~23.5M parameters. The drift head adds ~17.8% additional parameters.

#### 1c: `FinslerEmbeddingHead(nn.Module)`

**This is a new component** with no direct analogue in BAU. In BAU, the drift branch is integrated directly into the `resnet50_finsler` model class. In ReNorm2, the drift branch is encapsulated within a registered head (`REID_HEADS_REGISTRY`) to conform to fast-reid's modular architecture.

**Structure.** `FinslerEmbeddingHead` replicates the original `EmbeddingHead` (pool → BN neck → classifier) and adds a parallel drift branch:

```
                       ┌─ BN neck [cur_idx] → classifier → cls_outputs
                       │
features → pool → flat ─┤─ BN neck → F.normalize → identity_norm ─┬─ concat → finsler_features
                       │                                            │
                       └─ FinslerDriftHead(global_feat) → drift ─────┘
                                                   (orthogonalized)
```

**Discrepancy H1 — Eval output differs from `EmbeddingHead`.** The original `EmbeddingHead` (line 77) returns `F.normalize(global_feat)` at evaluation — a 2048-d L2-normalized vector **without** applying the BN neck to the returned tensor. `FinslerEmbeddingHead` (post-2026-03-20 fix) uses `identity_norm = F.normalize(bn_feat)` for BAU parity; eval returns either that slice alone or `[identity_norm | drift]`. The identity slice is unit-norm, but the drift slice is NOT L2-normalized. The concatenated vector has norm $\sqrt{1 + \|\boldsymbol{\omega}^\perp\|^2}$ which varies per sample. This matters for evaluation fusion (see Part 3).

**Discrepancy H2 — Orthogonalization placement.** In BAU (`bau/models/model.py` lines 366–368), orthogonalization occurs after drift generation and before concatenation:

```python
inner_product = torch.sum(drift * identity_norm, dim=1, keepdim=True)
drift = drift - inner_product * identity_norm
```

The ReNorm2 port (`finsler_head.py` lines 107–108) uses the identical formulation:

```python
inner_product = torch.sum(drift * identity_norm, dim=1, keepdim=True)
drift = drift - inner_product * identity_norm
```

**Exact correspondence.** This is a Gram-Schmidt projection removing the component of drift along the normalized identity vector. As noted in [analysis_18_03.md Part 1, Modification 3], this removes 1 direction out of 2048, so the orthogonalization is near-vacuous at these dimensions.

**Discrepancy H3 — Drift input source.** In BAU, drift receives pre-BN features `emb` (pooled backbone output before BatchNorm). In ReNorm2, drift receives `global_feat` (GeM-pooled features before BN neck). These are functionally identical — both are the raw 2048-d features after spatial pooling, before any normalization. **No discrepancy.**

**Discrepancy H4 — Classifier input.** In both BAU and ReNorm2, the classifier operates on BN-neck features (post-BatchNorm), NOT on identity_norm or drift. CE loss is confined to the identity subspace. **Exact correspondence with BAU design.**

---

### Modification 2: `finsler_drift_dist` and `FinslerTripletLoss`

**File:** `fastreid/modeling/losses/triplet_loss.py` lines 178–249.

**Source:** `bau/loss/triplet.py` lines 132–273 (`finsler_drift_dist`, `symmetric_trapezoidal` method only).

#### 2a: `finsler_drift_dist(x, y, identity_dim=None)`

**What was ported:** Only the `symmetric_trapezoidal` method. The `constant_drift`, `slerp`, and `analytical` methods from BAU were deliberately omitted.

**BAU original** (`bau/loss/triplet.py` lines 132–273, `symmetric_trapezoidal` branch at line 201–203):

```python
asymmetry = 0.5 * ((term_xy - term_xx) + (term_yy - term_yx))
scaling_factor = identity_x.size(1) / shared_dim
asymmetry = asymmetry * scaling_factor
return dist + asymmetry
```

**ReNorm2 port** (`fastreid/modeling/losses/triplet_loss.py` lines 178–224):

```python
asymmetry = 0.5 * ((term_xy - term_xx) + (term_yy - term_yx))
scaling_factor = identity_x.size(1) / shared_dim
asymmetry = asymmetry * scaling_factor
return dist + asymmetry
```

**Exact correspondence.** All intermediate computations — `term_xy`, `term_xx`, `term_yy`, `term_yx`, the Euclidean base distance, the clamping at `1e-12`, and the D/S scaling — match the BAU source line-for-line.

**Mathematical specification (repeated from analysis_18_03.md § 5.1):**

Given concatenated features $\mathbf{x} = [\mathbf{x}_{\text{id}} \mid \mathbf{x}_\omega]$ and $\mathbf{y} = [\mathbf{y}_{\text{id}} \mid \mathbf{y}_\omega]$:

$$
d_F(\mathbf{x}, \mathbf{y}) = \underbrace{\sqrt{\|\mathbf{x}_{\text{id}} - \mathbf{y}_{\text{id}}\|^2}}_{\text{Euclidean base}} + \frac{D}{S} \cdot \frac{1}{2}\bigl[(\mathbf{x}_\omega^\top \mathbf{y}_{\text{id}} - \mathbf{x}_\omega^\top \mathbf{x}_{\text{id}}) + (\mathbf{y}_\omega^\top \mathbf{y}_{\text{id}} - \mathbf{y}_\omega^\top \mathbf{x}_{\text{id}})\bigr]
$$

where $D = \text{identity\_dim}$ and $S = \min(D_\omega, D)$. When $D_\omega = D = 2048$ (default), $S = D$ and the scaling factor is 1.0.

**Discrepancy FD1 — No alpha parameter.** The BAU `finsler_drift_dist` accepts an optional `alpha` parameter, but as documented in [analysis_18_03.md § 5.3, Discrepancy T4], **alpha is dead code** in the Finsler drift distance path — it has no effect on the asymmetry computation. The ReNorm2 port correctly omits it. This is not a discrepancy; it is a cleanup.

**Discrepancy FD2 — Method restriction.** BAU exposes 4 methods (`constant_drift`, `symmetric_trapezoidal`, `slerp`, `analytical`). The ReNorm2 port hardcodes `symmetric_trapezoidal`. This is deliberate: BAU's triplet loss itself forces `symmetric_trapezoidal` regardless of the global method setting (see [analysis_18_03.md § 5.3, Discrepancy T3]). Porting only this method ensures exact training-time correspondence.

**Inherited issue (from BAU) — Non-negativity violation.** $d_F$ can be negative because the asymmetry term $A$ can be negative with magnitude exceeding $d_E$. This violates the non-negativity axiom of a distance metric. The implemented function is a signed cost function, not a valid Randers metric. This is unchanged from BAU.

**Inherited issue (from BAU) — Asymmetry property.** $d_F(\mathbf{x}, \mathbf{y}) - d_F(\mathbf{y}, \mathbf{x}) = \frac{1}{2}[(\mathbf{x}_\omega - \mathbf{y}_\omega)^\top(\mathbf{y}_{\text{id}} - \mathbf{x}_{\text{id}})] \cdot \frac{D}{S}$. Asymmetry is non-zero if and only if both drift and identity differ between the pair. This is the mechanism by which the distance is supposed to model directional retrieval bias.

#### 2b: `FinslerTripletLoss(nn.Module)`

**ReNorm2 port** (`fastreid/modeling/losses/triplet_loss.py` lines 227–249):

```python
class FinslerTripletLoss(nn.Module):
    def __init__(self, margin, identity_dim, normalize_feature=False):
        super().__init__()
        self.margin = margin
        self.identity_dim = identity_dim
        self.normalize_feature = normalize_feature
        self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()

    def forward(self, emb, label):
        mat_dist = finsler_drift_dist(emb, emb, identity_dim=self.identity_dim)
        N = mat_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
        dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
        y = torch.ones_like(dist_ap)
        loss = self.margin_loss(dist_an, dist_ap, y)
        return loss
```

**Correspondence with BAU:** BAU's `TripletLoss` (`bau/loss/triplet.py` lines 310–406) is significantly more complex — it accepts pluggable `dist_func`, optional `alpha`, `bidirectional` mode, and `alpha_param`. The ReNorm2 version is a clean simplification that hardcodes `finsler_drift_dist` as the distance function and uses standard (non-bidirectional) batch-hard mining.

**Discrepancy TL1 — No bidirectional mode.** BAU optionally computes $\mathcal{L}_{tri} = 0.5(\mathcal{L}(\tilde{D}) + \mathcal{L}(\tilde{D}^\top))$. The ReNorm2 port uses only $\mathcal{L}(\tilde{D})$ (forward direction). This is a deliberate simplification — bidirectional triplet was an experimental option in BAU, not the default.

**Discrepancy TL2 — Batch-hard mining on asymmetric matrix.** The `_batch_hard` function (original ReNorm code, line 137–146) uses:
```python
sorted_mat_distance, positive_indices = torch.sort(
    mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
hard_p = sorted_mat_distance[:, 0]
```

This finds the **hardest positive** as `max over positives of d(anchor, positive)` and **hardest negative** as `min over negatives of d(anchor, negative)`. Since $\tilde{D}$ is asymmetric, the hardest positive for anchor $i$ searches row $i$ of $\tilde{D}$ — the "query $i$ → gallery $j$" direction. Reciprocity breaks: $i$'s hardest positive may not be $j$'s hardest positive. **This is the intended asymmetric behavior** and the core mechanism by which Finsler distance should aid directional retrieval.

**Discrepancy TL3 — Shared `_batch_hard` with Euclidean `TripletLoss`.** Both `TripletLoss` and `FinslerTripletLoss` use the same `_batch_hard` function. The only difference is the input distance matrix. This ensures the mining logic is identical and the comparison isolates the distance metric change only.

---

### Modification 3: `ARCH_RENORM` Wiring

**File:** `fastreid/modeling/meta_arch/ReNorm.py`

#### 3a: `__init__` Changes (lines 25–57)

**Added:**
```python
self.finsler_enabled = cfg.MODEL.FINSLER.ENABLED
self.finsler_triplet = cfg.MODEL.FINSLER.TRIPLET_USE_DRIFT if self.finsler_enabled else False
```

**Modified:** Triplet loss instantiation (lines 43–57) is now conditional:

```python
if self.finsler_triplet:
    identity_dim = cfg.MODEL.BACKBONE.FEAT_DIM  # 2048
    self.criterion_triple = FinslerTripletLoss(
        margin=cfg.MODEL.LOSSES.TRI_RN.MARGIN,
        identity_dim=identity_dim, ...)
    self.criterion_triple_EN = FinslerTripletLoss(
        margin=cfg.MODEL.LOSSES.TRI_EN.MARGIN,
        identity_dim=identity_dim, ...)
else:
    self.criterion_triple = TripletLoss(margin=..., ...)
    self.criterion_triple_EN = TripletLoss(margin=..., ...)
```

**Key design decision:** Two separate `FinslerTripletLoss` instances are created — one for the RN path (margin 0.5) and one for the EN path (margin 1.0). Both share the same `identity_dim = 2048` but use different margins, matching the original ReNorm design where RN and EN have separate triplet margins.

**What is preserved when `FINSLER.ENABLED = False`:** All code paths fall through to the original `TripletLoss` instantiation. The `finsler_enabled` and `finsler_triplet` flags are both `False`. No additional parameters are created. The model is architecturally identical to original ReNorm.

#### 3b: `losses` Method Changes (lines 128–162)

**Original ReNorm:**
```python
pred_features = outputs['features']
pred_features_EN = outputs_EN['features']
loss_dict["loss_triplet"] = self.criterion_triple(pred_features, gt_labels)
loss_dict["loss_triplet_EN"] = self.criterion_triple_EN(pred_features_EN, gt_labels)
```

**Modified:**
```python
if self.finsler_triplet:
    tri_feat = outputs['finsler_features']       # [identity_norm | drift], 4096-d
    tri_feat_EN = outputs_EN['finsler_features']  # [identity_norm | drift], 4096-d
else:
    tri_feat = pred_features                       # pre-BN global_feat, 2048-d
    tri_feat_EN = pred_features_EN
loss_dict["loss_triplet"] = self.criterion_triple(tri_feat, gt_labels)
loss_dict["loss_triplet_EN"] = self.criterion_triple_EN(tri_feat_EN, gt_labels)
```

**Discrepancy M1 — Feature space for triplet.** In the Euclidean baseline, triplet operates on `pred_features` which, with `NECK_FEAT = "before"`, is `global_feat` — the raw 2048-d GeM-pooled features, **not L2-normalized**. In the Finsler path, triplet operates on `finsler_features = [F.normalize(global_feat) | drift]` — where the identity slice **is L2-normalized**.

This is a **confound in the experimental comparison**. Any performance difference between Euclidean and Finsler ReNorm could be attributed to:
(a) the asymmetric distance geometry (the intended signal), OR
(b) L2-normalization of the identity features in the triplet loss (an unintended confound).

Euclidean triplet on unnormalized features allows the network to satisfy the margin by scaling magnitudes. Finsler triplet on normalized identity features constrains the identity component to the unit hypersphere, forcing angular separation instead. This is a fundamentally different optimization landscape.

**Severity: HIGH.** This must be controlled for. Options:
1. Add `normalize_feature=True` to the baseline `TripletLoss` to match.
2. Use unnormalized identity in `finsler_features` (i.e., `[global_feat | drift]`).
3. Report both configurations and note the confound explicitly.

**Discrepancy M2 — CE loss is unchanged.** `cls_outputs` comes from `self.classifier_list[cur_idx](bn_feat)` in both `EmbeddingHead` and `FinslerEmbeddingHead`. The CE loss path is identical. **No discrepancy.**

**Discrepancy M3 — Dual-head architecture.** ReNorm constructs `self.heads` (RN path) and `self.headsEN` (EN path) as two **independent** head instances. When the head is `FinslerEmbeddingHead`, this means two independent `FinslerDriftHead` instances are created — one for RN and one for EN. Each has its own weights (4.19M parameters each, 8.39M total). The RN drift head processes features from the RN normalization path; the EN drift head processes features from the EN normalization path. **The drift heads are not shared across paths.** This differs from the stated plan in the conversation summary, which specified "the drift head is shared between ReNorm's RN and EN paths." In the actual implementation, each path has its own drift head due to fast-reid's `build_heads(cfg)` creating independent instances.

**Impact of Discrepancy M3:** Having separate drift heads per-path is defensible — RN and EN features have different statistical properties (RN uses domain-specific running stats; EN uses cross-domain mixed stats), so separate heads allow path-specific drift learning. However, this doubles the drift parameter count and may make the capacity comparison less clean.

---

### Modification 4: Config Defaults

**File:** `fastreid/config/defaults.py` lines 128–134.

```python
_C.MODEL.FINSLER = CN()
_C.MODEL.FINSLER.ENABLED = False
_C.MODEL.FINSLER.DRIFT_DIM = 2048
_C.MODEL.FINSLER.MAX_DRIFT_NORM = 0.95
_C.MODEL.FINSLER.EVAL_USE_DRIFT = True
_C.MODEL.FINSLER.TRIPLET_USE_DRIFT = True
```

**Default behavior:** `ENABLED = False` ensures that the original ReNorm codebase is functionally unchanged unless explicitly overridden. All downstream conditionals check `cfg.MODEL.FINSLER.ENABLED` before activating any Finsler code path.

**Config options semantics:**

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `ENABLED` | bool | False | Master switch for all Finsler modifications |
| `DRIFT_DIM` | int | 2048 | Output dimension of `FinslerDriftHead` |
| `MAX_DRIFT_NORM` | float | 0.95 | Upper bound for sigmoid-gated drift norm |
| `EVAL_USE_DRIFT` | bool | True | If True, eval returns `[identity | drift]`; if False, returns `identity_norm` only |
| `TRIPLET_USE_DRIFT` | bool | True | If True, triplet uses `FinslerTripletLoss` on `finsler_features`; if False, uses standard `TripletLoss` on `features` |

**Discrepancy C1 — `EVAL_USE_DRIFT = True` by default.** When evaluation uses drift, the RN+EN fusion (line 119 of `evaluator.py`) adds 4096-d vectors. This raises geometric concerns (see Part 3).

**Discrepancy C2 — No `EVAL_METRIC` coupling.** `EVAL_USE_DRIFT` and `TEST.METRIC` are independent settings. It is possible to set `EVAL_USE_DRIFT = True` with `TEST.METRIC = cosine`, which would apply cosine distance to `[identity | drift]` features. This is a user error but not prevented by the config.

---

### Modification 5: Evaluation Pipeline

#### 5a: `compute_finsler_distance`

**File:** `fastreid/utils/compute_dist.py` lines 207–214.

```python
@torch.no_grad()
def compute_finsler_distance(features, others, identity_dim=None):
    from fastreid.modeling.losses.triplet_loss import finsler_drift_dist
    features = features.cuda()
    others = others.cuda()
    dist_m = finsler_drift_dist(features, others, identity_dim=identity_dim)
    return dist_m.cpu().numpy()
```

This is a thin wrapper that moves features to GPU, calls `finsler_drift_dist`, and returns a NumPy array matching the interface of `compute_euclidean_distance` and `compute_cosine_distance`.

**Discrepancy E1 — No normalization.** `compute_cosine_distance` applies `F.normalize(features, p=2, dim=1)` before computing distances. `compute_finsler_distance` does NOT normalize. The identity slice is already L2-normalized by `FinslerEmbeddingHead`, so re-normalization would distort the drift component. However, after the RN+EN fusion (which adds features), the identity slice is the sum of multiple L2-normalized vectors and is NOT L2-normalized. See Part 3 for the implications.

#### 5b: `build_dist` Dispatch

**File:** `fastreid/utils/compute_dist.py` lines 31–53.

The assertion on line 42 was extended from `["cosine", "euclidean", "jaccard"]` to include `"finsler"`. A new `elif` branch dispatches to `compute_finsler_distance` with optional `identity_dim` from kwargs.

#### 5c: `ReidEvaluator.evaluate`

**File:** `fastreid/evaluation/reid_evaluation.py` lines 85–88.

```python
metric_kwargs = {}
if self.cfg.TEST.METRIC == "finsler":
    metric_kwargs["identity_dim"] = self.cfg.MODEL.BACKBONE.FEAT_DIM
dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC, **metric_kwargs)
```

This passes `identity_dim = 2048` to `build_dist` when the metric is `"finsler"`, enabling `finsler_drift_dist` to split the concatenated features correctly.

---

### Modification 6: `tools/train_net.py`

**File:** `tools/train_net.py` line 25.

**Original:**
```python
cfg.merge_from_file("./configs/ReNorm.yml")
```

**Modified:**
```python
config_file = getattr(args, "config_file", "") or "./configs/ReNorm.yml"
cfg.merge_from_file(config_file)
```

This allows `--config-file configs/ReNorm_Finsler.yml` to be passed from the command line. Without this fix, the training script always loaded `ReNorm.yml` regardless of CLI arguments, making any alternative config unusable.

**Impact on baseline:** When `--config-file` is not provided, the default `"./configs/ReNorm.yml"` is used, preserving original behavior.

---

### Modification 7: `configs/ReNorm_Finsler.yml`

**File:** `configs/ReNorm_Finsler.yml` (new, 116 lines).

This is a copy of `configs/ReNorm.yml` with the following overrides:

| Key | ReNorm.yml Value | ReNorm_Finsler.yml Value |
|-----|------------------|--------------------------|
| `MODEL.HEADS.NAME` | `EmbeddingHead` | `FinslerEmbeddingHead` |
| `MODEL.FINSLER.ENABLED` | (absent, defaults to False) | `True` |
| `MODEL.FINSLER.DRIFT_DIM` | (absent, defaults to 2048) | `2048` |
| `MODEL.FINSLER.MAX_DRIFT_NORM` | (absent, defaults to 0.95) | `0.95` |
| `MODEL.FINSLER.EVAL_USE_DRIFT` | (absent, defaults to True) | `True` |
| `MODEL.FINSLER.TRIPLET_USE_DRIFT` | (absent, defaults to True) | `True` |
| `TEST.METRIC` | `cosine` | `finsler` |

**All other settings are identical.** Optimizer (Adam, LR 3.5e-4), scheduler (MultiStep [30, 60, 100], γ=0.1), warmup (10 epochs, linear), batch size (64), augmentations (color jitter, autoaug, flip, padding), backbone (ResNet50, stride-1 layer4, GeM pool, pretrained), loss weights — all unchanged.

**Discrepancy CF1 — Baseline uses cosine eval, Finsler uses finsler eval.** The original ReNorm paper evaluates with cosine distance on L2-normalized features. The Finsler config evaluates with finsler distance on `[identity_norm | drift]` features. This is a necessary change (cosine distance on the concatenated vector is semantically wrong), but it means the evaluation metrics are not directly comparable unless a matched baseline with cosine distance on the same 2048-d identity features is also reported.

---

### Modification 8: `finsler_single.sbatch`

**File:** `finsler_single.sbatch` (new, 105 lines).

SLURM job array with 3 tasks for Protocol-2 leave-one-out evaluation:

| Task ID | Train Sources | Test Target | `SOLVER.ITERS` |
|---------|--------------|-------------|----------------|
| 1 | cuhkSYSU + Market1501 + MSMT17 | CUHK03 | 2000 |
| 2 | cuhkSYSU + Market1501 + CUHK03 | MSMT17 | 1000 |
| 3 | cuhkSYSU + MSMT17 + CUHK03 | Market1501 | 2000 |

Environment: `conda activate ReNorm`. Config: `configs/ReNorm_Finsler.yml`. W&B logging enabled with `finsler` tags.

---

### Modification 9: `fastreid/modeling/heads/__init__.py`

**File:** `fastreid/modeling/heads/__init__.py` line 11.

Added: `from .finsler_head import FinslerEmbeddingHead`

This registers `FinslerEmbeddingHead` in `REID_HEADS_REGISTRY`, making it accessible via `cfg.MODEL.HEADS.NAME = "FinslerEmbeddingHead"`.

---

## Part 3: Critical Analysis — Evaluation-Time Feature Fusion

This is the most critical issue with the port. The evaluation pipeline in `fastreid/evaluation/evaluator.py` (lines 95–123, **unmodified**) fuses features from 4 forward passes:

$$
\mathbf{o} = \mathbf{o}_{\text{RN}} + \frac{1}{3}(\mathbf{o}_{\text{EN}_1} + \mathbf{o}_{\text{EN}_2} + \mathbf{o}_{\text{EN}_3})
$$

With flip test enabled (default), each output is itself an average of original and flipped:

$$
\mathbf{o}_{\text{RN}} = \frac{1}{2}(\mathbf{o}_{\text{RN}}^{\text{orig}} + \mathbf{o}_{\text{RN}}^{\text{flip}})
$$

so the final output is an average of up to 8 forward passes.

### Issue F1: Fusion of [identity | drift] vectors across normalization paths

When the head is `FinslerEmbeddingHead`, each output $\mathbf{o}$ is a 4096-d vector `[identity_norm | drift]`. The fusion adds these:

$$
\mathbf{o}_{\text{fused}} = [\hat{\mathbf{f}}_{\text{RN}} + \frac{1}{3}\sum_k \hat{\mathbf{f}}_{\text{EN}_k} \mid \boldsymbol{\omega}_{\text{RN}} + \frac{1}{3}\sum_k \boldsymbol{\omega}_{\text{EN}_k}]
$$

**Problem with the identity slice:** Each $\hat{\mathbf{f}}_*$ is L2-normalized (norm = 1). Their sum is NOT L2-normalized. The norm of the fused identity is:

$$
\|\hat{\mathbf{f}}_{\text{fused}}\|_2 = \|\hat{\mathbf{f}}_{\text{RN}} + \frac{1}{3}\sum_k \hat{\mathbf{f}}_{\text{EN}_k}\|_2
$$

This depends on the cosine similarity between RN and EN features. If they agree (high cos-sim), the norm approaches 2.0. If they disagree, it can be much smaller. The fused identity norm varies per sample and is unconstrained.

**Problem with the drift slice:** Different normalization paths (RN vs EN) produce features with different statistical distributions. The drift heads (separate instances) process these different distributions and produce drift vectors in potentially different orientations. Adding drift vectors from different normalization regimes is geometrically questionable — in Randers/Finsler geometry, drift vectors are tangent vectors at specific points on the manifold, and adding tangent vectors from different base points requires parallel transport [Absil et al., "Optimization Algorithms on Matrix Manifolds," Princeton UP, 2008].

**Problem with the Finsler distance on fused features:** After fusion, `finsler_drift_dist` treats the first 2048 dims as identity and the last 2048 as drift. The identity part has variable, sample-dependent norm (not 1.0). The asymmetry term:

$$
A(\mathbf{x}, \mathbf{y}) = 0.5[(\mathbf{x}_\omega^\top \mathbf{y}_{\text{id}} - \mathbf{x}_\omega^\top \mathbf{x}_{\text{id}}) + (\mathbf{y}_\omega^\top \mathbf{y}_{\text{id}} - \mathbf{y}_\omega^\top \mathbf{x}_{\text{id}})]
$$

scales with $\|\mathbf{y}_{\text{id}}\|$ and $\|\mathbf{x}_{\text{id}}\|$. Since these norms are no longer unit, the magnitude of the asymmetry term varies with the RN-EN agreement level per sample. Samples with high RN-EN agreement (large identity norm) will have disproportionately large asymmetry terms, creating a bias toward penalizing well-agreed features.

**Severity: HIGH.** This is a potential source of pathological ranking behavior. The finsler distance was designed for unit-normed identity features. Applying it to summed, non-unit-normed features breaks the intended scale relationship between the Euclidean base and the asymmetry term.

### Issue F2: The Euclidean baseline does not have this problem

In the Euclidean baseline, `EmbeddingHead` returns `F.normalize(global_feat)` — always L2-normalized. The fusion adds L2-normalized vectors, producing non-unit vectors, and cosine distance is applied. `compute_cosine_distance` re-normalizes before computing distance (line 201: `features = F.normalize(features, p=2, dim=1)`), so the variable norms from fusion are corrected. In the Finsler path, `compute_finsler_distance` does NOT re-normalize, so the variable norms propagate into the distance computation. This is another **confound** in the comparison.

### Issue F3: Possible mitigations

1. **Re-normalize the identity slice after fusion:** Before `finsler_drift_dist`, split the fused features, L2-normalize the identity slice, and re-concatenate. This preserves the intended geometry.
2. **Evaluate without fusion:** Use RN-only or EN-only features to eliminate the fusion confound entirely. This provides a cleaner (but weaker) baseline comparison.
3. **Evaluate with cosine on identity-only:** Set `EVAL_USE_DRIFT = False`, which returns only `identity_norm` (2048-d, L2-normalized) at eval. Apply cosine distance. This tests whether the Finsler training signal improves identity features even when evaluated with Euclidean metrics. This is the cleanest comparison but does not test the asymmetric distance at evaluation.

---

## Part 4: Critical Analysis — Training Dynamics

### Issue T1: Will drift survive under CE + Triplet?

In BAU, [analysis_18_03.md Part 4] identified that the uniform loss actively suppresses drift. ReNorm has no uniform loss. The loss landscape is:

$$
\mathcal{L} = \mathcal{L}_{\text{CE}}^{\text{RN}} + \mathcal{L}_{\text{CE}}^{\text{EN}} + \mathcal{L}_{\text{tri}}^{\text{RN}} + \mathcal{L}_{\text{tri}}^{\text{EN}}
$$

- $\mathcal{L}_{\text{CE}}$ operates on identity features only (BN-neck → classifier). It provides no gradient signal to the drift head at all.
- $\mathcal{L}_{\text{tri}}$ operates on `[identity_norm | drift]`. The gradient flows to both the backbone (through `identity_norm`) and the drift head (through `drift`).

The triplet loss is the **sole source of gradient** for the drift head. Whether drift survives depends on whether the asymmetry term provides a useful training signal for batch-hard mining. If the asymmetry helps find harder positives or easier negatives, drift will be reinforced. If it adds noise to the mining, the optimizer may suppress drift weights to reduce the noise.

**Theoretical expectation:** Without uniformity loss actively fighting asymmetry, and with no alignment loss ignoring drift, the drift head has a cleaner gradient landscape than in BAU. The triplet loss alone should allow drift to express itself IF the asymmetry is genuinely useful. This makes ReNorm a more honest test of the Finsler hypothesis.

**However:** The CE loss provides the dominant gradient to the backbone. If CE alone produces sufficient identity-discriminative features, the triplet loss (and hence the drift head) may receive weak gradients. The drift head may collapse to near-zero simply because the backbone features are already well-separated by CE, making the triplet margin easy to satisfy without asymmetry.

### Issue T2: Gradient flow through orthogonalization

The orthogonalization step:

$$
\boldsymbol{\omega}^\perp = \boldsymbol{\omega} - (\boldsymbol{\omega} \cdot \hat{\mathbf{f}}) \hat{\mathbf{f}}
$$

creates a gradient dependency between the drift head and the backbone features (through $\hat{\mathbf{f}} = \text{normalize}(\text{global\_feat})$). Gradients from the triplet loss flow through $\boldsymbol{\omega}^\perp$ back to both the drift head ($\partial / \partial \boldsymbol{\omega}$) and the backbone ($\partial / \partial \hat{\mathbf{f}}$). This means the triplet loss with Finsler distance influences backbone optimization differently from the triplet loss with Euclidean distance, creating another potential confound.

### Issue T3: RN and EN drift head independence

With separate drift heads for RN and EN paths (Discrepancy M3), each drift head receives gradients only from its respective triplet loss. The RN drift head is trained by $\mathcal{L}_{\text{tri}}^{\text{RN}}$ (margin 0.5) and the EN drift head by $\mathcal{L}_{\text{tri}}^{\text{EN}}$ (margin 1.0). The different margins create different optimization pressures. The EN drift head, trained with a larger margin, may produce larger drift vectors to help satisfy the wider margin, leading to a systematic scale difference between RN and EN drift at evaluation. This interacts with Issue F1 (fusion of differently-scaled drift vectors).

---

## Part 5: Summary of Discrepancies and Open Issues

### Confounds in the Experimental Comparison

| ID | Issue | Severity | Affects |
|----|-------|----------|---------|
| M1 | Finsler triplet uses L2-normalized identity; Euclidean uses unnormalized | HIGH | Training comparison |
| F1 | RN+EN fusion produces non-unit-normed identity for Finsler distance | HIGH | Evaluation comparison |
| F2 | Cosine baseline re-normalizes after fusion; Finsler does not | MEDIUM | Evaluation comparison |
| M3 | Separate drift heads for RN/EN (8.39M extra params vs 4.19M stated) | MEDIUM | Capacity comparison |
| CF1 | Baseline uses cosine eval; Finsler uses finsler eval | MEDIUM | Metric comparison |

### Inherited Issues from BAU (Unchanged)

| ID | Issue | Severity |
|----|-------|----------|
| D2 (BAU) | $d_F$ can be negative (violates metric non-negativity) | LOW (for ranking) |
| H4 (BAU) | Orthogonalization removes 1/2048 dims (near-vacuous) | LOW |
| FD (BAU) | D/S scaling is ad-hoc, not canonical Randers | LOW (D/S = 1 when D_ω = D) |

### Recommended Pre-Experiment Fixes

1. **Fix M1 (Critical):** Either set `normalize_feature=True` in baseline `TripletLoss`, or pass unnormalized `[global_feat | drift]` to `FinslerTripletLoss`. This eliminates the normalization confound.
2. **Fix F1 (Critical):** Re-normalize the identity slice after RN+EN fusion, before passing to `finsler_drift_dist` at evaluation. Alternatively, evaluate with `EVAL_USE_DRIFT = False` (identity-only, cosine) as a confound-free comparison.
3. **Document F2:** Explicitly note that the Euclidean baseline uses cosine distance (with re-normalization) while the Finsler path uses finsler distance (without re-normalization) at evaluation.

### What This Port Tests (and What It Does Not)

**Tests:** Whether the Finsler asymmetric distance provides a useful training signal (through triplet loss) under a loss landscape that does not actively suppress asymmetry. If drift collapses to near-zero under CE + triplet only, the Finsler hypothesis is more convincingly falsified than under BAU (where uniform loss is the confound).

**Does not test:** Domain-conditioned drift (omitted from port). Alignment/uniformity compatibility with asymmetry. Cross-view directional retrieval (ReNorm uses standard Protocol-2, not AG-ReIDv2).

---

## Part 6: Smoke Test Results

The smoke test (run 2026-03-19) verified the following:

1. **Model instantiation:** `FinslerEmbeddingHead` builds correctly, `FinslerTripletLoss` replaces `TripletLoss`.
2. **Training forward pass:** Produces all 4 loss terms (`loss_cls`, `loss_cls_EN`, `loss_triplet`, `loss_triplet_EN`).
3. **Backward pass:** Gradients flow through the drift head. Observed gradient norms: `block.0.weight` = 0.078, `block.2.weight` = 1.038.
4. **Evaluation output:** 4096-d features (`[identity_norm(2048) | drift(2048)]`).
5. **Asymmetry:** Mean $|d_F(\mathbf{x}, \mathbf{y}) - d_F(\mathbf{y}, \mathbf{x})| = 0.000873$ on random features. Non-zero confirms the asymmetry mechanism is active.
6. **Baseline preservation:** With `FINSLER.ENABLED = False`, the model produces 2048-d eval features and uses `TripletLoss`. The Euclidean baseline is unaffected.
