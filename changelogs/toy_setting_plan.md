# Master Plan: Toy Dataset L_mono / Randers Experiment

## Context

This implements the controlled toy-dataset experiment described in `changelogs/toy_dataset_Lmono_critical_evaluation.md`. The goal is to train and evaluate three model variants on the existing `ToyCorruption` dataset (750 train PIDs, 50 eval PIDs, 5 severity levels) to validate whether L_mono can enforce a monotone scalar dimension θ and whether the resulting Randers metric produces a measurable asymmetric retrieval gap.

**Models:**
- **M1**: Euclidean BAU zero-shot — frozen backbone, no θ head, no training, standard L2 ranking. Establishes that task asymmetry exists before Randers.
- **M2a**: Frozen BAU + θ head, trained on toy train split with `L_CE + λ_mono * L_mono` (λ_mono > 0). Randers ranking at eval.
- **M2b**: Same as M2a but λ_mono = 0 (L_CE only). Ablation: confirms θ stays random without L_mono signal.

**Key architectural constraints** (from design doc):
- θ = w_new^T h where h = `emb` (GeM-pooled features, **pre-BN**). θ must bypass BatchNorm entirely.
- Only `theta_head` and `toy_classifier` are trainable. Everything else frozen.
- α (Randers weight) is a fixed constant, not learned; swept over {0.1, 0.3, 0.5, 0.9}.
- Batch must contain ≥2 severity levels per PID for L_mono to have non-zero gradient.

---

## Critical Files

| File | Role |
|---|---|
| `bau/models/model.py:14-53` | `resnet50` class — forward returns `(emb, F.normalize(f), logits)` in train |
| `bau/loss/crossentropy.py` | `CrossEntropyLabelSmooth` — reuse for L_CE |
| `bau/utils/data/sampler.py:19-84` | `RandomIdentitySampler` — extend for severity stratification |
| `bau/utils/data/preprocessor.py:18-42` | `Preprocessor.__getitem__` — extend to return severity |
| `bau/evaluators.py:56-73` | `pairwise_distance` — extend for Randers |
| `bau/utils/serialization.py:28-48` | `copy_state_dict` — reuse for frozen checkpoint loading |
| `examples/data/ToyCorruption/metadata.json` | Dataset manifest (severity, pid, split, cam info) |
| `scripts/generate_toy_dataset.py:77-123` | Corruption table reference |
| `train.sh` | Documented CLIs: M1 `eval_toy_m1.py`, M2a/M2b `train_toy_lmono.py`; defaults `LOG_DIR=logs/toy_lmono`, `OUT_DIR=logs/toy_lmono_runs` |
| `eval.sh` | Post-hoc only: same env as `train.sh`, then `eval_toy_checkpoint.py` on `m2a`/`m2b` `.pth` if present |
| `examples/eval_toy_m1.py` | M1 zero-shot bidirectional eval (tee logs) |
| `examples/eval_toy_checkpoint.py` | Load `train_toy_lmono` checkpoint; log mAP, Spearman, **printed D8** per α |
| `examples/train_toy_lmono.py` | M2a/M2b training + per-epoch eval (D8 computed but not printed in training logs) |
| `bau/evaluators.py` | `bidirectional_evaluate`, `compute_mean_asymmetric_gap_d8`, Spearman helper |

---

## Step-by-Step Implementation

---

### Step 1 — Toy Dataset Loader
**New file:** `bau/datasets/toy_corruption.py`

**What to build:**
- Class `ToyCorruption` with `train`, `query_s1`, `query_s2`, `gallery` splits.
- Parse `examples/data/ToyCorruption/metadata.json` once at init.
- Filename convention: `{pid:04d}_c{cam}s{seq}_000001_01.jpg` where `seq = severity + 1`.
- Return format:
  - Train: `(img_path, pid, camid, severity)` — 4-tuple, severity ∈ {0,1,2,3,4}
  - Eval (query/gallery): `(img_path, pid, camid)` — standard 3-tuple
- Expose `num_train_pids`, `num_train_imgs` attributes.
- Register in `bau/datasets/__init__.py` factory so `create("toy_corruption", root)` works.

**Watch out for:**
- PID re-labeling: metadata has `new_pid` (1-indexed). Re-label to 0-indexed for CE loss (`pid - 1`).
- `camid` in training encodes `source_idx` (1-4); fine to use directly.
- Train split has 750 PIDs × 4 sources × 5 severities = 15,000 images.
- Eval split: 50 PIDs × 2 sources × 5 severities = 500 total; query_s1 = 50 images (clean, source 1); query_s2 = 50 images (clean, source 2); gallery = 400 images.

---

### Step 2 — Severity-Stratified Sampler
**Modify:** `bau/utils/data/sampler.py` — add class `SeverityStratifiedSampler`

**What to build:**
- Same P×K identity sampling as `RandomIdentitySampler`, but add constraint that K instances per PID span ≥2 distinct severity levels.
- At init: build index `{pid: {severity: [indices]}}` from dataset (4-tuples, reading `data_source[i][3]` for severity).
- In `__iter__`: for each sampled PID, sample at least one index from each severity bucket (or at least 2 distinct buckets), then fill remaining slots randomly. If a PID has < 2 severity buckets in the dataset (shouldn't happen in toy data), fall back to standard sampling.
- K recommendation: set K=5 (one per severity) or K≥2 with severity-aware selection.

**Watch out for:**
- `RandomIdentitySampler` uses `data_source[i][1]` for pid. The new sampler uses `data_source[i][1]` for pid and `data_source[i][3]` for severity. The 4-tuple format is additive and doesn't break downstream code that ignores index [3].
- The sampler is only used during training of M2a/M2b, not M1.

---

### Step 3 — Extended ResNet50 with θ Head (the 2049th dimension)
**Modify:** `bau/models/model.py` — extend `resnet50.__init__` and `forward`

**Architectural note:** The actual `resnet50` in `model.py` has no `nn.Linear(2048, 2048)` before the classifier — the identity embedding is produced by `bn_neck = BatchNorm1d(2048)`, which is affine but not a general linear layer. "Extending to 2049" therefore means: keep `bn_neck` for the 2048-dim identity path (loaded from the pretrained checkpoint), and add a single new `nn.Linear(2048, 1, bias=False)` — the `theta_head` — that reads from `emb` (pre-BN). The combined eval-time embedding is `[F.normalize(bn_neck(emb)), theta_head(emb)] ∈ R^{2049}`. The pretrained checkpoint covers all parameters up to and including `bn_neck`; `theta_head` is not in the checkpoint and is therefore the only new weight being introduced. This is mechanically equivalent to "extending the final layer by one output dimension."

**What to build:**
- Add optional `with_theta_head=False` constructor parameter to `resnet50`.
- When `True`: instantiate `self.theta_head = nn.Linear(2048, 1, bias=False)`.
- Initialize `theta_head.weight` near zero (e.g., `nn.init.normal_(self.theta_head.weight, std=1e-4)`).
- In `forward`:
  - Compute `emb` (GeM-pooled, pre-BN) as before.
  - Compute `f = self.bn_neck(emb)` → `f_norm = F.normalize(f)` as before.
  - If `with_theta_head`: `theta = self.theta_head(emb)` — reads from pre-BN `emb`, NOT from `f` or `f_norm`.
  - Training: return `(emb, f_norm, theta)` — drop old `logits` since toy CE uses a separate `toy_classifier`.
  - Eval: return `(f_norm, theta)` — the 2049-dim embedding is `torch.cat([f_norm, theta], dim=1)` for downstream Randers distance.
- Add `freeze_pretrained()` method: sets `requires_grad_(False)` on `self.base`, `self.pool`, `self.bn_neck`, and `self.classifier` (old Market+MS+CS head). Leaves `self.theta_head` trainable.

**Checkpoint loading (in the training script, Step 7):**
- Load the pretrained BAU `best.pth` via `copy_state_dict`.
- The checkpoint will have keys for `base.*`, `pool.*`, `bn_neck.*`, and `classifier.*`.
- `copy_state_dict` will silently skip `theta_head.*` (not in checkpoint) — this is correct behaviour; log the skipped keys to confirm.
- The old `classifier` weights (Market+MS+CS) are loaded but immediately frozen and never used for toy CE loss.
- For toy CE loss: add a separate `toy_classifier = nn.Linear(2048, num_toy_pids, bias=False)` in the training script (NOT inside the model class), initialized with kaiming uniform. This reads from `f_norm`.

**Watch out for:**
- **θ must NOT pass through `bn_neck`**. Confirm `theta = self.theta_head(emb)` uses `emb` (the variable assigned before the `self.bn_neck(emb)` call).
- The old `self.classifier` maps to Market+MS+CS classes and is irrelevant for toy training — freeze it and ignore its outputs. Do not confuse it with `toy_classifier`.
- At eval time, the full 2049-dim embedding is `torch.cat([f_norm, theta], dim=1)`. The Randers distance function (Step 5) must split this into `f_norm` (first 2048) and `theta` (last 1) — document this split contract clearly.
- `freeze_pretrained()` should be called BEFORE constructing the optimizer, so the optimizer only receives `theta_head.parameters()` (and `toy_classifier.parameters()` from the training script).

---

### Step 4 — L_mono Loss
**New file:** `bau/loss/mono.py`

**What to build:**
- Class `MonotonicityLoss(margin=0.1)`.
- `forward(theta, pids, severities)`:
  - `theta`: (B,) scalar values from the batch.
  - `pids`: (B,) integer PID labels.
  - `severities`: (B,) integer severity levels 0-4.
  - For each pair (i, j) in batch where `pids[i] == pids[j]` and `severities[i] < severities[j]`: compute hinge `[theta[i] - theta[j] + margin]_+`. Note: cleaner image (lower severity) should have higher θ, so θ^{k_a} > θ^{k_b} when k_a < k_b. The violation is when `theta[i] - theta[j] < margin`.
  - Return mean over all valid pairs. Return 0.0 tensor if no valid pairs exist (guards against zero-gradient batches).

**Formula (from design doc):**
```
L_mono = mean_{(i,j): pid_i==pid_j, sev_i < sev_j} [ theta_i - theta_j + margin ]_+
```

**Watch out for:**
- The sign convention: lower severity → higher θ. Make sure the hinge direction is correct: penalize when `theta[lower_sev] < theta[higher_sev] + margin`, i.e., `[theta_i - theta_j + margin]_+` where i has lower severity.
- If the sampler is working correctly, most batches will have valid pairs. Log the fraction of zero-pair batches during training.
- No source-index grouping needed at loss level (the sampler handles co-occurrence; the loss pairs by PID alone since each PID has one source per corruption in train set).

---

### Step 5 — Randers Distance
**New file:** `bau/utils/randers.py`

**What to build:**
- Function `randers_distance_matrix(f_q, theta_q, f_g, theta_g, alpha)`:
  - `f_q`: (Nq, D) L2-normalized identity embeddings for queries.
  - `theta_q`: (Nq,) scalar θ values for queries.
  - `f_g`: (Ng, D) L2-normalized identity embeddings for gallery.
  - `theta_g`: (Ng,) scalar θ values for gallery.
  - `alpha`: scalar Randers weight ∈ (0, 1).
  - Returns: `dist` (Nq, Ng) where `dist[i,j] = ||f_g[j] - f_q[i]||_2 + alpha * (theta_g[j] - theta_q[i])`.
  - Note: `- alpha * theta_q[i]` is constant over j; can be factored for efficiency.
- This matrix is **asymmetric**: swapping query/gallery gives a different matrix.

**Watch out for:**
- Euclidean term: use the same formula as `pairwise_distance` in `evaluators.py:68-72` (`||x||^2 + ||y||^2 - 2x^T y`, clamped to ≥0, then sqrt).
- The `alpha * theta_g` broadcast: shape `(1, Ng)` added to `(Nq, Ng)` euclidean term.
- The `alpha * theta_q` broadcast: shape `(Nq, 1)` subtracted; constant per row.
- GPU-safe: keep all operations on torch tensors.

---

### Step 6 — Evaluation Extensions
**Modify:** `bau/evaluators.py`

**What to build:**
- Extend `extract_features` to optionally return θ values alongside embeddings:
  - Add `return_theta=False` parameter.
  - When model returns `(f_norm, theta)` in eval mode (with theta head), collect both.
  - Return `features_dict, theta_dict` where `theta_dict[fname] = scalar`.
- Add function `bidirectional_evaluate(model, query_loader, gallery_loader, alpha_values, ...)`:
  - Direction A: query=clean (severity 0), gallery=corrupted (severity 1-4).
  - Direction B: query=corrupted, gallery=clean.
  - For each direction and each alpha: compute Randers distance matrix, then mAP and Rank-1 using existing `mean_ap()` and `cmc()` from `bau/evaluation_metrics/ranking.py`.
  - Δ = mAP(A) - mAP(B); report as signed gap.
  - Also compute Euclidean mAP for comparison (α=0 special case).

**Watch out for:**
- The existing `pairwise_distance` is symmetric and does not support θ. Do not modify it; add a new function.
- For direction B (corrupted→clean), the gallery must be the clean split. The toy dataset has `query_s1` (clean, source 1) which can serve as the "clean gallery" for direction B, and the corrupted images serve as queries. Verify eval split composition with metadata.
- `mean_ap` and `cmc` expect `(distmat, query_ids, gallery_ids, query_cams, gallery_cams)`. The distance matrix shape determines query/gallery assignment. Ensure correct orientation.

---

### Step 7 — Training Script
**New file:** `examples/train_toy_lmono.py`

**What to build:**
- Stripped-down training script (no memory bank, no domain loss, no alignment loss, no two-view augmentation).
- **Arguments**:
  - `--data-dir`: path to `examples/data/ToyCorruption`
  - `--checkpoint`: path to pretrained BAU checkpoint (`best.pth` from Market+MS+CS run)
  - `--lambda-mono`: scalar multiplier for L_mono (0.0 for M2b, >0 for M2a)
  - `--margin`: L_mono hinge margin (default 0.1, sweep {0.05, 0.1, 0.2})
  - `--alpha`: Randers α for evaluation (sweep {0.1, 0.3, 0.5, 0.9})
  - `--epochs`, `--batch-size`, `--lr`, `--seed`
  - `--arch toy_resnet50` (new arch variant with theta head)
- **Setup**:
  1. Load `ToyCorruption` dataset.
  2. Build `resnet50(with_theta_head=True)`.
  3. Load pretrained BAU checkpoint via `copy_state_dict` (strips `module.` prefix if needed).
  4. Call `model.freeze_pretrained()`.
  5. Add `toy_classifier = nn.Linear(2048, num_train_pids, bias=False)` — trainable.
  6. Optimizer: Adam on `theta_head.parameters()` + `toy_classifier.parameters()` only (verify `requires_grad`).
- **Train loop per epoch**:
  1. Forward: `emb, f_norm, _, theta = model(imgs)` in train mode.
  2. `logits = toy_classifier(f_norm)` (CE on normalized features, standard practice).
  3. `loss_ce = CrossEntropyLabelSmooth(logits, pids)`.
  4. `loss_mono = MonotonicityLoss(theta.squeeze(), pids, severities)`.
  5. `loss = loss_ce + args.lambda_mono * loss_mono`.
  6. Backward + step.
  7. Log: `loss_ce`, `loss_mono`, `theta.std()`, fraction of zero-pair batches.
- **Eval**: After each epoch, run `bidirectional_evaluate` for all α values. Log Spearman ρ(θ, severity), mean Δ, mAP per direction.
- **Stopping criterion**: pre-specified epoch count (e.g., 20 epochs); monitor `loss_mono < margin` and **Spearman ρ(θ, severity) ≤ −0.8** as secondary indicators (severity increases with corruption; `MonotonicityLoss` enforces higher θ for lower severity, so ρ should be **negative** with large magnitude — equivalently **|ρ| ≥ 0.8** with the expected sign).

**Watch out for:**
- Use `SeverityStratifiedSampler`, not `RandomIdentitySampler`.
- The `Preprocessor` must return severity. Create `SeverityPreprocessor` (see Step 8).
- The toy classifier reads from `f_norm` (post-BN, L2-normalized). The θ head reads from `emb` (pre-BN). These are different — confirm in forward pass.
- Seeds: run ≥3 independent seeds for M2a; report mean ± std of all metrics.

---

### Step 8 — Severity Preprocessor
**Modify:** `bau/utils/data/preprocessor.py` — add class `SeverityPreprocessor`

**What to build:**
- Class `SeverityPreprocessor(Preprocessor)`:
  - Expects `data_source` as list of 4-tuples `(img_path, pid, camid, severity)`.
  - `__getitem__` returns `(img_tensor, pid, camid, severity)`.
- This is the only change needed to the preprocessor; keep `Preprocessor` untouched for eval.

---

## Diagnostics per Model

### M1: Euclidean Zero-Shot

| Diagnostic | Metric | Target / Interpretation |
|---|---|---|
| **D1: Bidirectional mAP** | `mAP(clean→corrupted)`, `mAP(corrupted→clean)`, Δ=difference | Δ ≠ 0 confirms task asymmetry exists without Randers |
| **D2: Per-severity mAP table** | 5×5 matrix: rows=query severity, cols=gallery severity | Off-diagonal reveals how ranking degrades cross-severity |
| **D3: Feature collinearity** | PCA on frozen `emb` grouped by severity; Spearman ρ(PC1, severity) | If ρ > 0.5, corruption signal exists in h → L_mono has signal to exploit |

**How to run D1**: Use `examples/data/ToyCorruption` eval split with `query_s1` as queries and `gallery` as gallery (standard direction). For reverse direction, use corrupted images as queries and the clean images as gallery.

**How to run D3**: Extract `emb` features from frozen backbone for all train images. Fit PCA. Compute Spearman correlation of first PC score against severity label. If ρ is near zero, the backbone has suppressed corruption — L_mono will have near-zero gradient.

---

### M2a: Frozen Backbone + L_CE + L_mono (λ_mono > 0)

| Diagnostic | Metric | Target |
|---|---|---|
| **D4: Training convergence** | `loss_mono` per epoch | Should decrease toward < margin |
| **D5: θ monotonicity** | Spearman ρ(θ, severity) on eval set (severity 0 = clean … 4 = worst) | Target **ρ ≤ −0.8** (same sign convention as `MonotonicityLoss`: monotone decreasing θ vs increasing severity) |
| **D6: θ scale** | std(θ) across severity groups | If near zero after 10 epochs → backbone suppressed signal; see D3 |
| **D7: Bidirectional mAP (Randers vs Euclidean)** | Δ for Randers vs Δ for Euclidean (M1) | Does Randers widen or narrow the gap? |
| **D8: Mean asymmetric gap** | mean[d_R(z^0, z^k) − d_R(z^k, z^0)] for k=1..4 | Target < 0 (clean queries reach corrupted gallery closer than reverse) |
| **D9: α sensitivity** | Rank-change count (vs Euclidean ranking) at each α ∈ {0.1, 0.3, 0.5, 0.9} | Quantifies how much Randers perturbation affects the ranked list |
| **D10: Zero-pair fraction** | Fraction of batches where L_mono has 0 valid pairs | Should be near 0 with severity-stratified sampler |

---

### M2b: Frozen Backbone + L_CE Only (λ_mono = 0)

| Diagnostic | Metric | Interpretation |
|---|---|---|
| **D11: θ monotonicity (ablation)** | Spearman ρ(θ, severity); std(θ) on eval | Prefer **std(θ) ≈ 0** and Randers ≡ Euclidean for all α; Spearman near 0 when θ varies, else noisy when θ collapses |
| **D12: Bidirectional mAP (Randers)** | Compare M2b+Randers vs M2b+Euclidean | Should be equal (random θ adds noise, not signal) |
| **D13: θ scale (ablation)** | std(θ) across severities | Random θ: no systematic variation by severity |

**Comparison table to report (Table 1):**

| Model | Ranking | mAP(clean→corr) | mAP(corr→clean) | Δ | Spearman ρ |
|---|---|---|---|---|---|
| M1 | Euclidean | ... | ... | ≈ (task asymmetry only) | N/A |
| M2b | Euclidean | ... | ... | ≈ M1 | ≈ 0 (or ill-defined if θ flat) |
| M2b | Randers | ... | ... | ≈ M1 (random θ) | ≈ 0 |
| M2a | Euclidean | ... | ... | ≈ M1 (BN running stats may drift vs M1) | **ρ ≤ −0.8** |
| M2a | Randers | ... | ... | target: wider gap | **ρ ≤ −0.8** |

---

## Implementation Order

1. **Step 1** (ToyCorruption dataset) — prerequisite for everything.
2. **Step 3** (Extended ResNet50) — needed for M2a/M2b training and M1 eval (M1 uses base model, no changes needed for M1).
3. **Step 4** (L_mono loss) + **Step 5** (Randers distance) — independent, do in parallel.
4. **Step 2** (SeverityStratifiedSampler) + **Step 8** (SeverityPreprocessor) — needed for training script.
5. **Step 6** (Evaluation extensions) — needed before training to validate diagnostics.
6. **Step 7** (Training script) — integrates all of the above.

**M1 can be run after Steps 1 and 5-6 only.** Run D3 (collinearity) immediately after M1 to decide whether to proceed.

---

## Verification Checklist

- [ ] `ToyCorruption` dataset loads without error; `len(dataset.train) == 15000`.
- [ ] `SeverityStratifiedSampler`: confirm each batch contains ≥2 distinct severity values per sampled PID.
- [ ] BN bypass: confirm `theta = self.theta_head(emb)` is called before `bn_neck` in forward; add assertion `not emb.requires_grad` (if frozen).
- [ ] `randers_distance_matrix(f_q, theta_q, f_g, theta_g, alpha=0.0)` matches `pairwise_distance` output (Euclidean special case).
- [ ] `MonotonicityLoss` with a synthetic batch where severity ordering is correct returns 0.0; with reversed ordering returns positive.
- [ ] M1 eval: bidirectional mAP numbers differ (Δ ≠ 0), confirming task asymmetry.
- [ ] M2b after training: Spearman ρ(θ, severity) near 0 when θ varies; else std(θ)≈0 and Randers ≡ Euclidean (ablation validates no L_mono signal).
- [ ] M2a after training: Spearman ρ(θ, severity) ≤ −0.8 and L_mono < margin (sign matches `MonotonicityLoss`: higher θ for cleaner / lower severity index).

---

## Agent handoff — operational decisions (keep in sync with `changelog.md`)

Use this section so later work on the toy setting does not re-derive protocol from scattered logs.

### Shell layout and artifacts

| Script | Purpose |
|---|---|
| [`train.sh`](train.sh) | **Train + inline eval each epoch.** Writes training logs to `LOG_DIR` (default `logs/toy_lmono/`) and optional weights to `OUT_DIR` (default `logs/toy_lmono_runs/`), e.g. `m2a_lambda0.1_seed1.pth`, `m2b_lambda0_seed1.pth`. |
| [`eval.sh`](eval.sh) | **Eval only:** runs [`examples/eval_toy_m1.py`](examples/eval_toy_m1.py) then [`examples/eval_toy_checkpoint.py`](examples/eval_toy_checkpoint.py) for the M2a/M2b checkpoints if files exist. Same `CKPT`, `DATA_DIR`, `LOG_DIR`, `OUT_DIR` / `M2A_CKPT` / `M2B_CKPT` defaults as `train.sh`. Does **not** retrain. |

Do not confuse repo-root [`eval.sh`](eval.sh) with [`scripts/eval.sh`](scripts/eval.sh) (legacy Market `test.py` invocations).

### Post-hoc checkpoint eval and D8 logging

- **`examples/eval_toy_checkpoint.py`** loads a dict checkpoint `{"model", "toy_classifier", "args"}` from `train_toy_lmono.py` and runs the same bidirectional protocol as training. It **prints D8** (`d8_mean_asymmetric_gap`: `mean_gap`, `mean_gap_by_severity`, `n_pairs`) on every Randers line — this is the canonical place to copy **geometric** asymmetric-gap numbers into papers or changelogs.
- **`train_toy_lmono.py`** still calls `bidirectional_evaluate` each epoch (so D8 exists in the Python return) but **only logs mAP and Spearman to stdout**; for D8 text, run `eval.sh` or `eval_toy_checkpoint.py` after training.

### Spearman sign convention (single source of truth)

- Severity index in the toy loader: **0 = clean … 4 = heaviest corruption.**
- `MonotonicityLoss` enforces **higher θ for lower severity** → Spearman **ρ(θ, severity)** should be **strongly negative** when L_mono works. Success criterion: **ρ ≤ −0.8** (equivalently anti-monotone with \|ρ\| ≥ 0.8). Do not use “ρ > 0.8” unless you explicitly flip sign or redefine severity.

### M1 vs M2a Euclidean bidirectional Δ (comparability)

- **M1** (`eval_toy_m1.py`): backbone + `bn_neck` **never** run in training mode on toy data; BN uses checkpoint **running_mean / running_var** as saved.
- **M2a/M2b**: each training epoch uses `model.train()`; **`BatchNorm1d` running stats still update** even when affine parameters are frozen (`freeze_pretrained` only sets `requires_grad=False`). After training, **Euclidean (α=0) mAP/Δ can differ from M1** for the same nominal backbone weights — this is expected and is **not** purely “task asymmetry vs Randers”; document BN drift when comparing M2a Euclidean row to M1. For a strict M1-aligned frozen-BN eval, a separate protocol would be needed (e.g. eval-only BN or snapshot running stats); not implemented unless added later.

### `toy_resnet50` implementation note

- Registry factory only: [`bau/models/__init__.py`](bau/models/__init__.py) (`toy_resnet50` → `resnet50(..., with_theta_head=True)`). Actual θ path in [`bau/models/model.py`](bau/models/model.py) `resnet50`.

### Changelog cross-reference

Toy eval wiring and PLAN Spearman edits are recorded under **2026-05-03 18:45:00 UTC** in [`changelog.md`](changelog.md). Update this handoff section when the toy protocol or scripts change materially.
