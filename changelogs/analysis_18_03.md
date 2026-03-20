---
name: Critical Finsler Modifications Analysis
overview: Comprehensive critical analysis of all modifications to the BAU repository for asymmetric Finsler/Randers person re-identification, evaluating architectural changes, loss modifications, and proposing ablation studies grounded in empirical results and literature.
todos:
  - id: verify-diagram
    content: Verify and correct the 4 architecture diagram discrepancies identified (domain gate dims, classifier output, drift head input source, Lp label)
    status: completed
  - id: ablation-capacity
    content: "Design and run Ablation A: capacity-controlled comparison (Euclidean 4096-d vs Finsler 2048+2048)"
    status: in_progress
  - id: ablation-uniform
    content: "Design and run Ablation B: test uniform loss incompatibility with asymmetric distance (with/without uniform, Euclidean uniform on identity-only)"
    status: pending
  - id: ablation-domain-drift
    content: "Design and run Ablation C: domain-conditioned vs instance-conditioned drift vs no drift baseline"
    status: pending
  - id: ablation-crossview
    content: "Design and run Ablation F: directional cross-view protocol on AG-ReIDv2 (aerial_to_cctv vs cctv_to_aerial)"
    status: pending
  - id: fix-uniform-loss
    content: Consider restricting uniform loss to identity-only features with Euclidean distance, matching alignment loss routing
    status: completed
  - id: validate-domain-pivot
    content: Run first domain-conditioned drift experiments with the corrected loss routing and report results
    status: pending
isProject: false
---

# Critical Analysis of Manifold-Aware Person Re-Identification Modifications

---

## Part 0: Architecture Diagram Verification

The attached diagram is **largely correct** but has the following discrepancies against the actual code in `[bau/models/model.py](bau/models/model.py)`:

**Confirmed correct:**

- Input: weak+strong augmentation concatenated along batch dim, fed through modified ResNet50 (stride-1 layer4, InstanceNorm after layers 1-3)
- GeM pooling -> BN neck -> identity features -> L2 normalization
- Domain Token Classifier reading **detached** BN-neck identity features
- Training: one-hot domain lookup; Eval: softmax with temperature -> soft mixture of domain embeddings
- Domain embedding (num_domains, 64) -> projection to drift space (Linear 64 -> 2048, no bias)
- Domain gate: sigmoid-gated modulation of domain drift prior using pre-BN features
- Residual drift block: Linear(2048->1024, no bias) -> ReLU -> Linear(1024->2048, no bias)
- Residual scaling factor = 0.1
- Sigmoid gated norm scaling on final drift
- Orthogonal projection of drift against normalized identity
- Concatenation of [identity_norm | drift] and [emb | drift]
- Loss routing: CE on identity logits, Alignment on identity-only, Triplet/Uniform/Domain on full embeddings
- Domain-token CE loss on domain logits vs source domain IDs

**Corrections needed:**

1. **Domain Gate dimensions:** The diagram labels the domain gate as `Linear (No Bias) 2048 -> 1024`. In code, `self.domain_gate = nn.Linear(input_dim, output_dim, bias=False)` where `input_dim=identity_dim=2048` and `output_dim=drift_dim`. When `drift_dim=2048` (default), this is **2048 -> 2048**, not 2048 -> 1024.
2. **Domain classifier output:** The diagram shows "Domain logits (1, #domain)" which is correct, but the block labeled "Query Embedding: Linear 2048 -> 64" is misleading. The domain classifier is `nn.Linear(2048, num_domains)` (e.g., 2048 -> 3 for three source domains), not 2048 -> 64. The 64-d embedding is the `context_dim` of the domain embedding table, which is a separate path.
3. **Drift head input source:** The diagram correctly shows the residual drift head and domain gate receiving pre-BN features (`emb`), but should make explicit that the domain-conditioned path (`DomainConditionedDriftHead.forward`) receives `emb` as `x`, not `identity`. This is confirmed by `model.py` line 357: `drift = self.drift_head(emb, domain_probs=resolved_domain_probs)`.
4. **Batch Normalisation Neck vs Lp Normalisation Neck:** The diagram shows these as two separate named blocks, which is technically correct (BN neck is `nn.BatchNorm1d(2048)`, Lp norm is `F.normalize`). However, the "Lp" label should specify p=2 explicitly.

---

## Part 1: Critical Analysis of Model Modifications (`resnet50_finsler`)

### Modification 1: InstanceNorm2d after backbone layers 1-3

**What changed:** InstanceNorm2d(256/512/1024) inserted after ResNet layers 1, 2, 3.

**Code:** `[bau/models/model.py` lines 279-291](bau/models/model.py)

**Present in old BAU?** Yes. The original `[old_bau/BAU/bau/models/model.py](old_bau/BAU/bau/models/model.py)` lines 25-31 already includes the same InstanceNorm layers. **This is not a new modification.**

**Justification:** InstanceNorm for style/domain normalization is well-established. [Pan et al., "Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net," ECCV 2018] showed that IN strips domain-specific style statistics while preserving content, aiding domain generalization. [Jia et al., "Instance Normalization for Domain Generalization," NeurIPS 2021 workshop] provides further support.

**Verdict:** Sound. Inherited from BAU.

---

### Modification 2: Split-head architecture [identity(2048-d) | drift(N-d)]

**What changed:** Instead of a single 2048-d embedding, the model produces a concatenated `[identity_norm | drift]` vector. Identity is L2-normalized post-BN features. Drift is produced by a dedicated `FinslerDriftHead` or `DomainConditionedDriftHead`.

**Code:** `[bau/models/model.py` lines 225-384](bau/models/model.py), absent from `[old_bau/BAU/bau/models/model.py](old_bau/BAU/bau/models/model.py)`.

**Justification:**

- Disentangling identity-discriminative features from domain/view nuisance factors is supported by [Zheng et al., "Joint Discriminative and Generative Learning for Person Re-identification," CVPR 2019] (DG-Net) and [Eom & Ham, "Learning Disentangled Representation for Robust Person Re-identification," NeurIPS 2019].
- However, these works disentangle through generative reconstruction or adversarial training, not through a concatenated auxiliary branch with a novel distance metric.

**Critical issue:** The drift branch is 2048-d by default, **doubling the embedding dimensionality**. This alone provides additional representational capacity. Any marginal improvement could be an artifact of increased capacity rather than asymmetric geometry. This must be controlled for in ablation (see Part 3).

**Verdict:** The disentanglement motivation is sound, but the specific realization (concatenation with a custom distance) is novel and unproven. A capacity-controlled ablation is essential.

---

### Modification 3: Orthogonalization of drift against identity

**What changed:** Before concatenation, drift is projected to be orthogonal to the L2-normalized identity vector: $\texttt{drift} = \texttt{drift} - (\texttt{drift} \cdot \hat{\mathbf{f}})\,\hat{\mathbf{f}}$.

**Code:** `[bau/models/model.py` lines 366-368](bau/models/model.py)

**Justification:**

- This enforces that identity and drift occupy complementary subspaces, preventing the drift from "leaking" identity information.
- Orthogonal regularization of feature subspaces appears in [Bousmalis et al., "Domain Separation Networks," NeurIPS 2016], which orthogonalizes shared and private feature components for domain adaptation.

**Critical issue:** This is a Gram-Schmidt projection in a single direction. It only removes the component of drift along the identity vector, not along the full identity subspace. For 2048-d drift in a 2048-d space, removing one direction is negligible. The orthogonalization is therefore **extremely weak** — it only removes `1/2048` of the potential alignment. If the intent is genuine decorrelation, a stronger mechanism (e.g., soft orthogonality regularization on the full covariance) would be needed.

**Verdict:** Theoretically motivated but practically near-vacuous at these dimensions. Unlikely to be a meaningful contributor or hindrance.

---

### Modification 4: Sigmoid-gated norm scaling for drift vectors

**What changed:** `scale_drift_vector()` uses `sigmoid(||drift||) * max_norm * (drift / ||drift||)` instead of simple clipping.

**Code:** `[bau/models/model.py` lines 146-167](bau/models/model.py)

**Justification:**

- Smooth, differentiable norm bounding avoids the gradient discontinuity of hard clipping.
- The sigmoid gate acts like a soft "confidence" in the drift magnitude.
- Similar smooth norm constraints appear in hyperbolic embeddings [Nickel & Kiela, "Poincaré Embeddings for Learning Hierarchical Representations," NeurIPS 2017] where embedding norms must be bounded.

**Critical issue:** The initialization `init.normal_(self.block[-1].weight, std=0.001)` already starts drift near zero. Combined with sigmoid gating (which maps small norms to ~~0.5), the effective initial drift norm is `~~0.5 * 0.95 ≈ 0.475`. This is not near-zero at initialization — it starts at roughly half the maximum. This may cause early training instability if the optimizer has to simultaneously learn identity features and suppress an unexpectedly large drift.

**Verdict:** Engineering choice. The smooth bounding is reasonable, but the effective initialization norm should be verified against training dynamics.

---

### Modification 5: Learnable alpha (AlphaParameter)

**What changed:** A bounded, learnable scalar alpha interpolates between Euclidean and Finsler distances.

**Code:** `[bau/loss/triplet.py` lines 11-64](bau/loss/triplet.py)

**Empirical evidence from results:**

- `learnable_alpha_results.csv`: Regardless of alpha_init (0.1 to 0.9), alpha converges to **0.098-0.150** after training. mAP = 42.0-42.4, matching Euclidean baseline (42.18 +/- 0.45).
- `ablation_summary.csv`: Full Method alpha converges to **0.0**. When uniform loss is present with alignment, alpha saturates to 1.0 and mAP collapses.
- Config 9 (Base+Align+Dom): the ONLY config where Finsler marginally beats Euclidean (43.4 vs 42.7), alpha = 0.027.

**Critical issue:** The empirical evidence overwhelmingly shows that **the optimization landscape actively suppresses the asymmetric component**. When alpha is free, it converges to near-zero, recovering the Euclidean baseline. This is not a failure of the parameterization — it is the optimizer's answer to the question "does asymmetry help?", and the answer is consistently "no" under the current loss landscape.

**Verdict:** The learnable alpha is itself a useful diagnostic tool — it reveals that the asymmetric distance is not beneficial under the current training regime. But it should not be presented as a contribution; it is evidence against the core hypothesis.

---

### Modification 6: Domain-conditioned drift head (DomainConditionedDriftHead)

**What changed:** Instead of predicting drift purely from the instance features, drift is now composed of: (a) a domain embedding looked up/mixed from source-domain labels, (b) projected and sigmoid-gated by instance features, (c) plus a small residual instance correction (scaled by 0.1).

**Code:** `[bau/models/model.py` lines 170-222](bau/models/model.py)

**Justification:**

- [Zhuang et al., "Camera-Based Batch Normalization," CVPR 2020] demonstrates that camera/domain-conditioned feature normalization is effective for ReID.
- [Dai et al., "Generalizable Person Re-identification with Relevance-aware Mixture of Experts," CVPR 2021] uses domain-specific expert branches.
- The pivot from instance to domain-level asymmetry is motivated by the observation (documented in README) that per-instance drift is suppressed by BAU's alignment/uniformity objectives.

**Critical issues:**

1. The domain embedding has only 64 dimensions, projected to 2048. This is a rank-64 bottleneck — the domain-conditioned drift can only span a 64-dimensional subspace. Whether 64 dimensions suffice to capture meaningful domain shifts is an open question.
2. At evaluation time, the soft domain prediction relies on the identity features (detached). If the domain classifier is poor, the soft mixture degenerates toward uniform weights, reducing the domain-conditioned path to a fixed average drift — essentially a learnable global bias term.
3. **No empirical results yet exist for the domain-conditioned path.** The pivot is documented but untested.

**Verdict:** This is the most theoretically promising modification and aligns with the literature on camera/domain-conditioned representations. However, it is entirely unvalidated empirically. The 64-d bottleneck and evaluation-time soft prediction require careful ablation.

---

### Modification 7: Classifier operates on identity features only (not drift)

**What changed:** `self.classifier = nn.Linear(self.identity_dim, self.num_classes, bias=False)`, applied to `identity` (BN-neck features), not the concatenated embedding.

**Code:** `[bau/models/model.py` line 316, line 375](bau/models/model.py)

**Justification:** Standard in disentangled representation learning: the identity classifier should only see identity-discriminative features, not domain nuisance factors. If the classifier also saw the drift, it could exploit domain-specific shortcuts, undermining generalization.

**Verdict:** Sound and well-motivated.

---

## Part 2: Critical Analysis of Loss Modifications (`[bau/trainers.py](bau/trainers.py)`)

### Loss 1: Cross-Entropy Loss (L_CE)

**Old BAU:** `criterion_ce(logits_w, pids)` — CE on weak-augmented logits only.

**New code:** Same default behavior. Optional `--use-aug-ce` adds CE on strong-augmented logits. Applied to identity logits only.

**Verdict:** No meaningful change. The aug-CE option is a standard practice.

---

### Loss 2: Triplet Loss (L_Tri)

**Old BAU:** `TripletLoss(margin=margin)` with `euclidean_dist` on `emb_w`. Hardcoded Euclidean.

**New code:** `TripletLoss` accepts pluggable `dist_func`, optional `alpha`, and `bidirectional` mode. When `finsler_drift_dist` is passed, the triplet loss **forces `symmetric_trapezoidal` method** regardless of the global drift method.

**Code:** `[bau/loss/triplet.py` lines 310-406](bau/loss/triplet.py), `[bau/trainers.py` line 190](bau/trainers.py)

**Critical issues:**

1. The triplet loss operates on `emb_w` (pre-BN, pre-normalization embeddings concatenated with drift). In old BAU, `emb` was 2048-d pre-BN features. Now it is `[emb(2048) | drift(2048)]` = 4096-d. The Euclidean component of finsler_drift_dist operates on the first 2048 dims, and the asymmetry term is added. **However, batch-hard mining is done on the full asymmetric distance matrix, which is no longer symmetric.** This means `hard_p` and `hard_n` can differ depending on the direction of the pair, which is the intended asymmetric behavior.
2. The forced `symmetric_trapezoidal` override for triplet is an engineering stability choice, not a principled decision. It means the triplet loss always sees a symmetric approximation of the asymmetric distance — contradicting the stated goal of asymmetric training.
3. Bidirectional triplet: `loss = 0.5 * (loss_forward + loss_backward)` using `mat_dist` and `mat_dist.t()`. This is correct for asymmetric distances where `d(x,y) != d(y,x)`.

**Verdict:** The asymmetric triplet is the correct implementation for the stated hypothesis. The `symmetric_trapezoidal` override is problematic — it partially symmetrizes what should be the main asymmetric training signal.

---

### Loss 3: Alignment Loss (L_Align)

**Old BAU:** Alignment between weak and strong augmented features using squared Euclidean distance, weighted by reciprocal k-NN Jaccard similarity. Applied to full normalized features.

**New code:** Alignment restricted to **identity-only features** (`f_w_align = f_w[:, :identity_dim]`). Uses `euclidean_dist` unconditionally (alpha forced to None). Optional drift alignment (`--use-drift-align`) adds a separate alignment loss on drift features.

**Code:** `[bau/trainers.py` lines 121-128, 192-200, 302-331](bau/trainers.py)

**Justification:**

- Alignment on identity-only is motivated by the principle that augmentation invariance should be enforced on identity features, not on domain/view-dependent drift features. Different augmentations of the same image should have the same identity but may legitimately differ in drift response.
- [Wang & Isola, "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere," ICML 2020] defines alignment as bringing positive pairs (same identity, different augmentation) closer. If drift encodes view-dependent information, forcing drift alignment across augmentations would collapse view-discriminative information.

**Verdict:** This is the most well-justified loss modification. It correctly isolates the alignment objective to the feature subspace that should be augmentation-invariant.

---

### Loss 4: Uniform Loss (L_Uniform)

**Old BAU:** `torch.pdist(f, p=2).pow(2).mul(-2).exp().mean().log()` — simple pairwise repulsion on full features.

**New code:** Uses the model's `dist_func` (Finsler if applicable) on full `[identity|drift]` features. Falls back to Euclidean if features are identity-only.

**Code:** `[bau/trainers.py` lines 333-371](bau/trainers.py)

**Critical issue:** This is the most problematic loss modification.

- Uniformity assumes a **symmetric distance** to push embeddings toward uniform coverage of the hypersphere. Applying it with an asymmetric distance creates a fundamentally ill-defined optimization objective. The log-sum-exp formulation $\log(\mathbb{E}[\exp(-2\,d(x,y)^2)])$ requires $d$ to be a metric (symmetric, triangle inequality). Finsler distances satisfy the triangle inequality but are not symmetric, so $d(x,y)^2 \neq d(y,x)^2$. The uniformity loss only computes upper-triangular pairs, which implicitly selects one direction.
- **This is the likely root cause of alpha collapse.** When uniform loss is active with asymmetric distance, the optimizer faces contradictory gradients: the asymmetric component pushes different pairs in different directions, while uniformity demands isotropic repulsion. The optimizer resolves this by driving alpha to zero.

**Ablation evidence:** In `ablation_summary.csv`, Config 9 (Base+Align+Dom, **no uniformity**) is the only config where Finsler marginally outperforms Euclidean (43.4 vs 42.7, alpha=0.027). Config 11 (Full Method, **with uniformity**) alpha = 0.0.

**Verdict:** Applying asymmetric distance to the uniform loss is theoretically inconsistent and empirically destructive. The uniformity loss should either operate on identity-only features with Euclidean distance (matching alignment), or a principled asymmetric uniformity formulation is needed.

---

### Loss 5: Domain Loss (L_Domain)

**Old BAU:** Nearest same-domain repulsion using squared Euclidean distance against the memory bank.

**New code:** Uses the model's `dist_func` (Finsler if applicable) on full features against memory bank features. Falls back to Euclidean for identity-only features.

**Code:** `[bau/trainers.py` lines 373-425](bau/trainers.py)

**Justification:** Domain repulsion in an asymmetric space makes more sense than uniform repulsion — it pushes apart same-domain pairs, which is directionally meaningful (camera A features should be pushed away from other camera A features in the memory bank). The directionality of the asymmetric distance could encode "which direction to push."

**Critical issue:** The memory bank stores features from the **weak augmentation path only** (`memory_bank.momentum_update(f_w, pids)`). If memory bank mode is "full", it stores `[identity|drift]`. The momentum update averages past and current features, which may blur the drift component over time as the drift head changes.

**Verdict:** More defensible than asymmetric uniformity, but the memory bank staleness for drift features needs investigation.

---

### Loss 6: Omega (Drift Norm) Regularization

**What changed:** Log-barrier penalty: `-log(max_norm - ||drift||)`. Penalizes drift norms approaching the maximum.

**Code:** `[bau/trainers.py` lines 428-456](bau/trainers.py)

**Justification:** Prevents drift from saturating at the maximum norm, which would make the sigmoid gating ineffective and cause gradient issues.

**Verdict:** Engineering safeguard. Reasonable but indicative of instability in the drift branch. If drift naturally remains small (as empirical results show), this regularizer is inactive and unnecessary.

---

### Loss 7: Domain-Token CE Loss

**What changed:** Auxiliary cross-entropy loss training the domain classifier to predict source domain from identity features.

**Code:** `[bau/trainers.py` lines 212-215](bau/trainers.py)

**Justification:** Standard auxiliary supervision for conditioning heads. Without it, the domain classifier has no training signal and the soft domain prediction at evaluation degenerates.

**Verdict:** Necessary for the domain-conditioned path. Well-motivated.

---

### Stabilization modifications: Gradient clipping, reduced LR, NaN guard

- **Grad clipping** (drift head, max_norm=1.0): `[bau/trainers.py` lines 232-236](bau/trainers.py)
- **Reduced LR** (0.05x drift LR multiplier): configured in `[examples/train_bau.py](examples/train_bau.py)`
- **NaN loss detection**: `[bau/trainers.py` lines 222-225](bau/trainers.py)

**Verdict:** These are symptoms of an unstable optimization landscape. They are necessary engineering safeguards but should be acknowledged as such in any paper. They are not contributions.

---

## Part 3: Ablation Study Design

The following ablation experiments are necessary to substantiate each modification. All should use 5+ seeds and report mean +/- std on at least two target domains.

### Ablation A: Capacity Control (Critical)

**Hypothesis to test:** Are marginal Finsler gains from increased embedding dimensionality (4096 vs 2048)?

- **A1:** Euclidean ResNet50 with 2048-d embedding (baseline)
- **A2:** Euclidean ResNet50 with 4096-d embedding (add 2048 random features, same distance)
- **A3:** Finsler ResNet50 with 2048-d identity + 2048-d drift

If A2 matches or exceeds A3, the Finsler geometry adds nothing beyond extra parameters.

### Ablation B: Loss-Geometry Compatibility (Critical)

**Hypothesis to test:** Does uniform loss suppress asymmetry?

- **B1:** Finsler with full BAU losses (CE + Tri + Align + Uniform + Domain)
- **B2:** Finsler with (CE + Tri + Align + Domain), no uniform loss
- **B3:** Finsler with Euclidean uniform loss on identity-only + Finsler triplet/domain on full embedding
- **B4:** Euclidean baseline with same loss configs as B1-B3 for fair comparison

Track alpha convergence in all runs. If alpha remains non-trivial in B2/B3 but collapses in B1, the uniform-asymmetry incompatibility is confirmed.

### Ablation C: Drift Conditioning Level

**Hypothesis to test:** Is domain-level drift more tractable than instance-level?

- **C1:** Instance-conditioned drift (current FinslerDriftHead)
- **C2:** Domain-conditioned drift (new DomainConditionedDriftHead)
- **C3:** Domain-conditioned drift without residual (residual_scale=0)
- **C4:** Euclidean baseline (no drift)

Report both identity-only and full-embedding evaluation metrics. Track drift norm evolution and domain classifier accuracy.

### Ablation D: Orthogonalization

**Hypothesis to test:** Does orthogonalization between identity and drift matter?

- **D1:** With orthogonalization (current)
- **D2:** Without orthogonalization (direct concatenation)

Given the analysis above (projection removes 1/2048 dimensions), the difference is expected to be negligible.

### Ablation E: Distance Method Comparison

**Hypothesis to test:** Does the integration method for the asymmetric term matter?

- **E1:** constant_drift
- **E2:** symmetric_trapezoidal
- **E3:** slerp
- **E4:** analytical

All with fixed alpha (not learnable) at several values {0.1, 0.3, 0.5}. This isolates the mathematical formulation from the learning dynamics.

### Ablation F: Cross-View Protocol (Critical for AG-ReIDv2)

**Hypothesis to test:** Does asymmetry help specifically in directional cross-view retrieval?

- **F1:** Euclidean, aerial_to_cctv
- **F2:** Finsler, aerial_to_cctv
- **F3:** Euclidean, cctv_to_aerial
- **F4:** Finsler, cctv_to_aerial

If Finsler helps F2 but not F4 (or vice versa), there is evidence of directional asymmetry. If both or neither, the asymmetric hypothesis is not supported even in the most favorable setting.

---

## Part 4: Summary Assessment

**Strongest modifications (well-justified):**

1. Identity-only alignment loss — clear theoretical motivation, consistent with disentanglement principles
2. Domain-conditioned drift pivot — aligns with camera-aware ReID literature, but needs empirical validation
3. Cross-view AG-ReIDv2 protocol — creates the experimental setting where asymmetry is most plausible

**Weakest modifications (insufficient justification):**

1. Applying Finsler distance to uniform loss — theoretically inconsistent, empirically causes alpha collapse
2. Learnable alpha — useful as a diagnostic, but evidence shows it converges to near-zero (i.e., it diagnoses the failure of the asymmetric hypothesis under current training)
3. Orthogonalization — near-vacuous at the implemented dimensionality

**Key finding from empirical data:** The `ablation_summary.csv` results are the most telling. Across all loss configurations with uniformity present, the optimizer drives alpha to 1.0 (causing collapse) or 0.0 (recovering Euclidean). The **only** configuration where Finsler marginally helps is Base+Align+Dom (no uniformity), with alpha=0.027 — essentially near-Euclidean. The 5-run comparison (alphaNone 42.18+/-0.45 vs Full Finsler 42.20+/-0.42) shows **no statistically significant difference** between Finsler and Euclidean under the full BAU loss.

**Implication for the research narrative:** The per-instance Finsler hypothesis has been empirically falsified under the current training regime. The domain-conditioned pivot is the correct research direction, but it must be validated with the ablations above before any claims can be made. The paper should frame the learnable-alpha experiments as evidence of the failure mode (uniformity-asymmetry incompatibility) rather than as a contribution.

---

## Part 5: Mathematically Rigorous Loss Function Analysis

This section states each loss function exactly as implemented in original BAU (`[old_bau/BAU/bau/trainers.py](old_bau/BAU/bau/trainers.py)`, `[old_bau/BAU/bau/loss/triplet.py](old_bau/BAU/bau/loss/triplet.py)`), then re-derives the current fork implementation (`[bau/trainers.py](bau/trainers.py)`, `[bau/loss/triplet.py](bau/loss/triplet.py)`), and catalogues every mathematical discrepancy.

### Notation

Let a training batch consist of B identities. After weak and strong augmentation and concatenation, the model processes 2B images. Define:

- $\mathbf{e}_i \in \mathbb{R}^D$: pooled pre-BN embedding for image i (D = 2048)
- $\mathbf{f}_i = \text{BN}(\mathbf{e}_i)$: batch-normalized embedding
- $\hat{\mathbf{f}}_i = \mathbf{f}_i / \|\mathbf{f}_i\|_2$: L2-normalized identity feature
- $y_i \in \{1, \ldots, C\}$: identity label for image i
- $d_i \in \{1, \ldots, S\}$: source-domain label for image i
- Superscripts $w$, $s$ denote weak and strong augmentation respectively

For the Finsler fork, additionally define:

- $\boldsymbol{\omega}_i \in \mathbb{R}^{D_\omega}$: drift vector for image i (default $D_\omega = D = 2048$)
- $\boldsymbol{\omega}_i^\perp = \boldsymbol{\omega}_i - (\boldsymbol{\omega}_i \cdot \hat{\mathbf{f}}_i)\hat{\mathbf{f}}_i$: drift orthogonalized against identity
- $\mathbf{z}_i = [\hat{\mathbf{f}}_i \mid \boldsymbol{\omega}_i^\perp] \in \mathbb{R}^{D + D_\omega}$: combined normalized feature
- $\mathbf{u}_i = [\mathbf{e}_i \mid \boldsymbol{\omega}_i^\perp] \in \mathbb{R}^{D + D_\omega}$: combined pre-BN embedding

---

### 5.1 Distance Functions

**Old BAU — Euclidean distance:**

$$
d_E(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\mathbf{x}^\top\mathbf{y}}
$$

Applied identically to all losses. Code: `[old_bau/BAU/bau/loss/triplet.py](old_bau/BAU/bau/loss/triplet.py)` lines 8-15.

**New fork — Finsler drift distance** (`finsler_drift_dist`, code: `[bau/loss/triplet.py](bau/loss/triplet.py)` lines 132-273):

Given concatenated features $\mathbf{x} = [\mathbf{x}_{\text{id}} \mid \mathbf{x}_\omega]$ and $\mathbf{y} = [\mathbf{y}_{\text{id}} \mid \mathbf{y}_\omega]$, let $S = \min(D_\omega, D)$ be the shared dimension, and restrict both identity and drift to the first $S$ coordinates for the asymmetry term. Then:

$$
d_F(\mathbf{x}, \mathbf{y}) = \underbrace{d_E(\mathbf{x}_{\text{id}}, \mathbf{y}_{\text{id}})}_{\text{Euclidean base}} + \underbrace{\frac{D}{S} \cdot A(\mathbf{x}, \mathbf{y})}_{\text{scaled asymmetry}}
$$

where the asymmetry term $A$ depends on the method:

**Method: `constant_drift`**

$$
A(\mathbf{x}, \mathbf{y}) = \mathbf{x}_\omega^\top \mathbf{y}_{\text{id}} - \mathbf{x}_\omega^\top \mathbf{x}_{\text{id}}
$$

**Method: `symmetric_trapezoidal`** (default for triplet, forced override)

$$
A(\mathbf{x}, \mathbf{y}) = \frac{1}{2}\bigl[(\mathbf{x}_\omega^\top \mathbf{y}_{\text{id}} - \mathbf{x}_\omega^\top \mathbf{x}_{\text{id}}) + (\mathbf{y}_\omega^\top \mathbf{y}_{\text{id}} - \mathbf{y}_\omega^\top \mathbf{x}_{\text{id}})\bigr]
$$

**Method: `analytical`**

$$
A(\mathbf{x}, \mathbf{y}) = \frac{1}{2}\frac{\theta}{\sin\theta}(\mathbf{x}_\omega^\top \mathbf{y}_{\text{id}} - \mathbf{y}_\omega^\top \mathbf{x}_{\text{id}}), \quad \theta = \arccos(\mathbf{x}_{\text{id}}^\top \mathbf{y}_{\text{id}})
$$

**Method: `slerp`** (K-step Riemann sum along geodesic)

$$
A(\mathbf{x}, \mathbf{y}) = \frac{1}{K}\sum_{k=1}^{K} \left[(1-t_k)A_kT_{xx} + (1-t_k)B_kT_{xy} + t_kA_kT_{yx} + t_kB_kT_{yy}\right]
$$

where $A_k = -\frac{\theta}{\sin\theta}\cos((1-t_k)\theta)$, $B_k = \frac{\theta}{\sin\theta}\cos(t_k\theta)$, and $T_{xy} = \mathbf{x}_\omega^\top \mathbf{y}_{\text{id}}$ etc.

**Discrepancy D0 — Scaling factor:** The asymmetry is multiplied by D/S. When D_\omega = D (default), S = D, so the factor is 1.0. When D_\omega < D, the factor scales up the asymmetry to compensate for the dimensionality mismatch. This scaling has no analogue in Randers geometry; it is an engineering normalization. **Impact on narrative:** Acceptable if documented, but it means the "Finsler distance" is not a canonical Randers metric — it includes an ad-hoc dimensional correction. Must be stated explicitly.

**Discrepancy D1 — Asymmetry of $d_F$:** $d_F(\mathbf{x}, \mathbf{y}) \neq d_F(\mathbf{y}, \mathbf{x})$ in general. For `symmetric_trapezoidal`:

$$
d_F(\mathbf{x}, \mathbf{y}) - d_F(\mathbf{y}, \mathbf{x}) = \frac{1}{2}\bigl[(\mathbf{x}_\omega - \mathbf{y}_\omega)^\top(\mathbf{y}_{\text{id}} - \mathbf{x}_{\text{id}})\bigr] \cdot \frac{D}{S}
$$

So the asymmetry vanishes when $\mathbf{x}_\omega = \mathbf{y}_\omega$ (identical drift) or $\mathbf{x}_{\text{id}} = \mathbf{y}_{\text{id}}$ (identical identity). This is geometrically sensible: asymmetry arises only when both identity and drift differ.

**Discrepancy D2 — Sign of $d_F$:** Since $A$ can be negative and its magnitude can exceed $d_E$, we can have $d_F < 0$. This violates the non-negativity axiom of a distance/metric. A Randers metric $F = \alpha + \beta$ requires $\|\beta\|_\alpha < 1$ pointwise, which the implementation does not enforce. **Impact on narrative:** Technically, the implemented function is not a valid distance metric. It is a signed cost function. This must be acknowledged.

---

### 5.2 Cross-Entropy Loss

**Old BAU** (`[old_bau/BAU/bau/trainers.py](old_bau/BAU/bau/trainers.py)` line 72):

$$
\mathcal{L}_{ce}^{\text{old}} = \text{CELS}(W \cdot \mathbf{f}_i^w, y_i), \quad W \in \mathbb{R}^{C \times D}
$$

where CELS is cross-entropy with label smoothing ($\epsilon = 0.1$):

$$
\mathcal{L}_{ce} = -\frac{1}{B}\sum_{i=1}^{B}\sum_{c=1}^{C} \tilde{y}_{i,c}\log\text{softmax}(W\mathbf{f}_i)_c, \quad \tilde{y}_{i,c} = (1-\epsilon)\mathbb{1}[c = y_i] + \frac{\epsilon}{C}
$$

**New fork** (`[bau/trainers.py](bau/trainers.py)` line 183):

$$
\mathcal{L}_{ce}^{\text{new}} = \text{CELS}(W \cdot \mathbf{f}_i^w, y_i), \quad W \in \mathbb{R}^{C \times D}
$$

**Discrepancy CE1 — Classifier input:** In old BAU, the classifier receives \mathbf{f} = \text{BN}(\mathbf{e}). In the new Finsler model, the classifier receives `identity` = \text{BN}(\mathbf{e}) — identical. The drift is not passed to the classifier. **Functionally identical.**

**Discrepancy CE2 — Optional aug-CE:** New code optionally adds $\text{CELS}(W \cdot \mathbf{f}_i^s, y_i)$. This doubles the CE signal. Not present in old BAU. **Impact on narrative:** Minor regularization enhancement, orthogonal to the Finsler hypothesis.

**Discrepancy CE3 — Double-normalization removal:** Old BAU trainer applies `f = F.normalize(f)` after the model already returns `F.normalize(f)` (idempotent). New code comments this out. **Numerically identical.**

---

### 5.3 Triplet Loss

**Old BAU** (`[old_bau/BAU/bau/loss/triplet.py](old_bau/BAU/bau/loss/triplet.py)` lines 51-84, `[old_bau/BAU/bau/trainers.py](old_bau/BAU/bau/trainers.py)` line 73):

Let $D_{ij} = d_E(\mathbf{e}_i^w, \mathbf{e}_j^w)$. Batch-hard mining:

$$
a_i^+ = \max_{j:y_j = y_i} D_{ij}, \quad a_i^- = \min_{j:y_j \neq y_i} D_{ij}
$$

$$
\mathcal{L}_{tri}^{\text{old}} = \frac{1}{B}\sum_{i=1}^{B} \max(0, a_i^+ - a_i^- + m)
$$

where $m$ is the margin. Input features are pre-BN embeddings $\mathbf{e}^w \in \mathbb{R}^D$ (weak augmentation only).

**New fork** (`[bau/loss/triplet.py](bau/loss/triplet.py)` lines 310-406, `[bau/trainers.py](bau/trainers.py)` line 190):

Let $\tilde{D}_{ij} = d_F(\mathbf{u}_i^w, \mathbf{u}_j^w)$, where $\mathbf{u} = [\mathbf{e} \mid \boldsymbol{\omega}^\perp]$. Batch-hard mining on the **asymmetric** matrix $\tilde{D}$:

$$
a_i^+ = \max_{j:y_j = y_i} \tilde{D}_{ij}, \quad a_i^- = \min_{j:y_j \neq y_i} \tilde{D}_{ij}
$$

$$
\mathcal{L}_{tri}^{\text{new}} = \frac{1}{B}\sum_{i=1}^{B} \max(0, a_i^+ - a_i^- + m)
$$

Optional bidirectional:

$$
\mathcal{L}_{tri}^{\text{bidir}} = \frac{1}{2}\left(\mathcal{L}_{tri}(\tilde{D}) + \mathcal{L}_{tri}(\tilde{D}^\top)\right)
$$

**Discrepancy T1 — Feature space:** Old uses $\mathbf{e} \in \mathbb{R}^D$. New uses $\mathbf{u} \in \mathbb{R}^{D + D_\omega}$. Even with $\alpha = 0$ (pure Euclidean in the Finsler distance), the base Euclidean distance operates on the identity slice only ($\mathbb{R}^D$), NOT the full $\mathbb{R}^{D+D_\omega}$. So the triplet loss with $\alpha = 0$ computes $d_E(\mathbf{e}_i, \mathbf{e}_j)$ on the identity dimensions — matching old BAU if and only if the pre-BN features are unchanged. **The drift dimensions do not enter the base distance even at $\alpha = 0$.**

**Discrepancy T2 — Asymmetric mining:** Since $\tilde{D}_{ij} \neq \tilde{D}_{ji}$, the hardest positive for anchor $i$ (row $i$ of $\tilde{D}$) is found in the "query i → gallery j" direction. This means different anchors are compared against different "views" of the distance matrix. In the symmetric case, $a_i^+ = a_j^+$ when $y_i = y_j$ and $i$, $j$ are each other's hardest positive. In the asymmetric case, this reciprocity breaks. **Impact on narrative:** This is the core mechanism by which the Finsler distance is supposed to aid cross-view retrieval. If probe-to-gallery distance differs from gallery-to-probe, the triplet loss can learn direction-dependent margins.

**Discrepancy T3 — Forced `symmetric_trapezoidal` for triplet:** Inside `TripletLoss.__init__`, if `finsler_drift_dist` is detected, the method is overridden to `symmetric_trapezoidal` regardless of the global `--drift-method`. This means even if the evaluator uses `analytical`, the triplet training always uses:

$$
A^{\text{tri}}(\mathbf{x}, \mathbf{y}) = \frac{1}{2}\bigl[(\mathbf{x}_\omega^\top\mathbf{y}_{\text{id}} - \mathbf{x}_\omega^\top\mathbf{x}_{\text{id}}) + (\mathbf{y}_\omega^\top\mathbf{y}_{\text{id}} - \mathbf{y}_\omega^\top\mathbf{x}_{\text{id}})\bigr]
$$

**Impact on narrative:** This creates a train/eval mismatch for the asymmetry term. The triplet loss optimizes one approximation of the path integral; the evaluator ranks with a different one. Any improvement from a better integration method (slerp, analytical) only helps at evaluation, not during training. This undermines claims about the geometric fidelity of the training objective.

**Discrepancy T4 — Alpha injection:** The new triplet loss accepts a per-step $\alpha$ override:

$$
\tilde{D}_{ij}(\alpha) = d_E(\mathbf{e}_i, \mathbf{e}_j) + \alpha \cdot \frac{D}{S} \cdot A(\mathbf{u}_i, \mathbf{u}_j)
$$

Wait — examining the code more carefully, alpha is passed to the `dist_func` as a keyword argument. In `finsler_drift_dist`, alpha is accepted but **not used** in the asymmetry computation. The asymmetry is always added with coefficient 1.0 (after the D/S scaling). Alpha is only used inside `euclidean_dist` (the old Randers formulation with `y[-1] - x[-1]`). So for the Finsler drift distance, **alpha has no effect**. The asymmetry magnitude is controlled entirely by the drift norms and the D/S scaling. This is a significant discrepancy between the stated narrative (learnable alpha interpolates Euclidean and Finsler) and the actual code behavior when `finsler_drift_dist` is the active distance function.

**Correction:** Re-reading `euclidean_dist`: alpha adds `(y_last - x_last) * alpha` where `y_last = y[:, -1]`. This is only active when `euclidean_dist` is called with alpha != None. But `finsler_drift_dist` computes its own Euclidean base distance internally and does not call `euclidean_dist`. So **alpha is dead code in the Finsler distance path**. The asymmetry is entirely determined by the drift vectors, not by alpha. Alpha only affects the old single-embedding Randers distance (last-dimension trick in `euclidean_dist`).

This means the learnable alpha results showing convergence to ~0 may not be measuring what is claimed. If the Finsler model uses `finsler_drift_dist`, alpha is ignored. The alpha would only matter if the evaluator or some loss path falls back to `euclidean_dist` with alpha.

**Impact on narrative:** This must be verified against the actual training configuration. If alpha is indeed dead in the Finsler path, the entire learnable-alpha analysis in the README is misleading.

---

### 5.4 Alignment Loss

**Old BAU** (`[old_bau/BAU/bau/trainers.py](old_bau/BAU/bau/trainers.py)` lines 116-125):

Let $\hat{\mathbf{f}}_i^w$, $\hat{\mathbf{f}}_j^s$ be L2-normalized BN features. Weight matrix $W \in \mathbb{R}^{B \times B}$ from reciprocal k-NN Jaccard similarity. Identity mask $M_{ij} = \mathbb{1}[y_i = y_j]$.

$$
\mathcal{L}_{align}^{\text{old}} = \frac{\sum_{i,j} M_{ij} W_{ij} \|\hat{\mathbf{f}}_i^s - \hat{\mathbf{f}}_j^w\|^2}{\sum_{i,j} M_{ij} W_{ij}}
$$

Note: the squared distance is computed **inline** as $\|\hat{\mathbf{f}}_i^s\|^2 + \|\hat{\mathbf{f}}_j^w\|^2 - 2(\hat{\mathbf{f}}_i^s)^\top \hat{\mathbf{f}}_j^w$. Since features are L2-normalized, this equals $2 - 2\cos(\hat{\mathbf{f}}_i^s, \hat{\mathbf{f}}_j^w)$. This is exactly the alignment metric from [Wang & Isola, ICML 2020] weighted by the BAU-specific Jaccard weights.

**New fork** (`[bau/trainers.py](bau/trainers.py)` lines 302-331):

Let $\hat{\mathbf{f}}_i^w = \mathbf{z}_i^w[:D]$ be the identity slice of the combined feature (already L2-normalized in the model). The alignment loss uses `euclidean_dist` unconditionally with `alpha=None`:

$$
\mathcal{L}_{align}^{\text{new}} = \frac{\sum_{i,j} M_{ij} W_{ij} \left[d_E(\hat{\mathbf{f}}_i^s, \hat{\mathbf{f}}_j^w)\right]^2}{\sum_{i,j} M_{ij} W_{ij}}
$$

where $d_E(\cdot, \cdot)$ is computed via `euclidean_dist()` which returns $\sqrt{\max(\cdot^2, 10^{-12})}$, then squared.

**Discrepancy A1 — Numerical path:** Old computes $\|\mathbf{a} - \mathbf{b}\|^2$ directly (no sqrt). New computes $\sqrt{\max(\|\mathbf{a} - \mathbf{b}\|^2, 10^{-12})}$ then squares. The sqrt-then-square round-trip introduces a clamp at $10^{-12}$ on the squared distance, which changes the gradient at near-zero distances:

Old gradient: $\nabla_{\mathbf{f}} \|\mathbf{f}_s - \mathbf{f}_w\|^2 = 2(\mathbf{f}_s - \mathbf{f}_w)$

New gradient: $\nabla_{\mathbf{f}} [d_E]^2 = 2(\mathbf{f}_s - \mathbf{f}_w)$ when $d_E > 0$, but the backward through sqrt introduces a $1/(2d_E)$ factor that is then cancelled by the chain rule of squaring. However, the clamp changes the gradient near zero: when the raw squared distance is below $10^{-12}$, the clamped sqrt returns $10^{-6}$, and the squared result is $10^{-12}$ with zero gradient. **Impact:** Negligible for any non-degenerate pair. This is a numerical stability improvement, not a semantic change.

**Discrepancy A2 — Feature space restriction:** Old operates on full $\hat{\mathbf{f}} \in \mathbb{R}^D$. New operates on the identity slice $\hat{\mathbf{f}}[:D] \in \mathbb{R}^D$, which is identical to old BAU's $\hat{\mathbf{f}}$ as long as the BN neck and L2 normalization are unchanged. **Impact on narrative:** This is the most principled modification — it correctly restricts augmentation-invariance enforcement to the identity subspace. The drift should not be aligned across augmentations because different augmentations of the same image may legitimately induce different domain-conditioned drifts.

**Discrepancy A3 — k-NN weight computation uses full features:** The Jaccard weight matrix $W$ is computed from similarities $\text{sim}_{ij} = \mathbf{z}_i^\top \mathbf{z}_j$ where $\mathbf{z} = [\hat{\mathbf{f}} \mid \boldsymbol{\omega}^\perp]$. In old BAU, $\text{sim}_{ij} = \hat{\mathbf{f}}_i^\top \hat{\mathbf{f}}_j$ (cosine similarity, bounded in $[-1, 1]$).

In the new code:

$$
\text{sim}_{ij}^{\text{new}} = \hat{\mathbf{f}}_i^\top \hat{\mathbf{f}}_j + (\boldsymbol{\omega}_i^\perp)^\top \boldsymbol{\omega}_j^\perp
$$

This is **NOT bounded** in $[-1, 1]$ because $\boldsymbol{\omega}^\perp$ is not L2-normalized. The drift-drift inner product can dominate the similarity if drift norms are large, biasing the k-NN graph toward drift similarity rather than identity similarity. This means the alignment weights W_{ij} may not reflect true identity-level neighborhood structure.

**Impact on narrative:** This is a subtle but potentially consequential bug. The alignment weights should reflect identity-level similarity (since alignment is applied to identity features), but they are contaminated by drift similarity. The fix would be to compute similarities on identity-only features: `sims = torch.matmul(f_w_align_combined, f_w_align_combined.t())` where the combined features exclude drift. Alternatively, compute weights before concatenation.

---

### 5.5 Uniform Loss

**Old BAU** (`[old_bau/BAU/bau/trainers.py](old_bau/BAU/bau/trainers.py)` lines 127-128):

$$
\mathcal{L}_{uniform}^{\text{old}} = \frac{1}{2}\left[\ell_u(\hat{\mathbf{f}}^w) + \ell_u(\hat{\mathbf{f}}^s)\right]
$$

where:

$$
\ell_u(\hat{\mathbf{f}}) = \log\left(\frac{2}{B(B-1)}\sum_{i < j} \exp\bigl(-2\|\hat{\mathbf{f}}_i - \hat{\mathbf{f}}_j\|^2\bigr)\right)
$$

This is the uniformity loss from [Wang & Isola, ICML 2020], operating on L2-normalized features. Since $\|\hat{\mathbf{f}}_i - \hat{\mathbf{f}}_j\|^2 = 2 - 2\cos(\hat{\mathbf{f}}_i, \hat{\mathbf{f}}_j)$, this penalizes angular clustering and promotes uniform coverage of the unit hypersphere $\mathbb{S}^{D-1}$.

**New fork** (`[bau/trainers.py](bau/trainers.py)` lines 333-371):

$$
\mathcal{L}_{uniform}^{\text{new}} = \frac{1}{2}\left[\ell_u(\mathbf{z}^w) + \ell_u(\mathbf{z}^s)\right]
$$

where:

$$
\ell_u(\mathbf{z}) = \log\left(\frac{2}{B(B-1)}\sum_{i < j} \exp\bigl(-2d_F(\mathbf{z}_i, \mathbf{z}_j)^2\bigr)\right)
$$

**Discrepancy U1 — Feature normalization:** Old BAU operates on $\hat{\mathbf{f}} \in \mathbb{S}^{D-1}$ (unit hypersphere). New code operates on $\mathbf{z} = [\hat{\mathbf{f}} \mid \boldsymbol{\omega}^\perp]$, which is NOT on any unit hypersphere because the drift is not normalized. The total norm $\|\mathbf{z}\|^2 = 1 + \|\boldsymbol{\omega}^\perp\|^2$. The uniformity loss from [Wang & Isola] is defined and analyzed specifically for features on $\mathbb{S}^{D-1}$. Applying it to features off the sphere invalidates the theoretical guarantees of the loss (the uniform distribution on the sphere is no longer the minimizer). **Impact on narrative:** Mathematically, the uniform loss is being applied to a non-spherical feature space. The optimal distribution under this loss is no longer the uniform distribution on the sphere; it depends on the drift norm distribution.

**Discrepancy U2 — Asymmetric distance in a symmetric loss:** The original uniform loss uses the symmetric $\|\hat{\mathbf{f}}_i - \hat{\mathbf{f}}_j\|^2$. The new loss uses $d_F(\mathbf{z}_i, \mathbf{z}_j)^2$ where $d_F$ is asymmetric. But the sum is over $i < j$ only, which means it only considers $d_F(\mathbf{z}_i, \mathbf{z}_j)$ for the "forward" direction. This creates a one-sided repulsion:

$$
\ell_u^{\text{new}} = \log\left(\frac{2}{B(B-1)}\sum_{i < j} \exp\bigl(-2[d_E(id_i, id_j) + A_{ij}]^2\bigr)\right)
$$

Expanding the square:

$$
[d_E + A]^2 = d_E^2 + 2d_EA + A^2
$$

The gradient with respect to the drift contains:

$$
\frac{\partial}{\partial \boldsymbol{\omega}} [d_E + A]^2 = 2(d_E + A)\frac{\partial A}{\partial \boldsymbol{\omega}}
$$

When $A > 0$ (forward direction increases distance), the gradient pushes to **reduce** $A$ (reducing the effective distance, weakening repulsion). When $A < 0$ (forward direction decreases distance), the gradient pushes to **increase** $A$ (increasing the effective distance, strengthening repulsion). This creates an asymmetric gradient landscape that interacts unpredictably with the triplet and domain losses that may want opposite drift directions.

**Impact on narrative:** This is the mathematical mechanism behind the empirically observed alpha collapse. The uniform loss creates gradient pressure to minimize |A|, i.e., to make the distance symmetric. Combined with the alignment loss (which ignores drift entirely), the optimizer receives no positive signal to maintain non-zero drift, and active pressure from uniformity to suppress it.

**Discrepancy U3 — `torch.pdist` vs explicit distance matrix:** Old uses `torch.pdist(f, p=2)` which directly computes pairwise L2 distances for all $i < j$. New computes a full $B \times B$ distance matrix then extracts upper-triangular indices. When the distance is symmetric (Euclidean fallback), these are equivalent. Under Finsler distance, the full matrix is asymmetric, and extracting the upper triangle gives only $d_F(\mathbf{z}_i, \mathbf{z}_j)$ for $i < j$. The lower triangle $d_F(\mathbf{z}_j, \mathbf{z}_i)$ is discarded.

---

### 5.6 Domain Loss

**Old BAU** (`[old_bau/BAU/bau/trainers.py](old_bau/BAU/bau/trainers.py)` lines 130-139):

Let $\mathbf{c}_k$ be memory bank features. For each sample $\hat{\mathbf{f}}_i$ with domain $d_i$, find the $B$ nearest same-domain memory bank entries (excluding self):

$$
\mathcal{N}_i = \text{argsort}_{k:d_k = d_i, k \neq i} \|\hat{\mathbf{f}}_i - \mathbf{c}_k\|^2, \quad \text{take first } B
$$

$$
\mathcal{L}_{domain}^{\text{old}} = \frac{1}{2}\left[\ell_d(\hat{\mathbf{f}}^w) + \ell_d(\hat{\mathbf{f}}^s)\right]
$$

$$
\ell_d(\hat{\mathbf{f}}) = \log\left(\frac{1}{B^2}\sum_i\sum_{k \in \mathcal{N}_i} \exp\bigl(-2\|\hat{\mathbf{f}}_i - \mathbf{c}_k\|^2\bigr)\right)
$$

This is a domain-conditioned uniformity: it pushes each sample away from its nearest **same-domain** memory-bank neighbors. The intuition from BAU is that same-domain features tend to cluster (due to shared camera/environment statistics), and this loss counteracts that clustering to improve domain generalization.

**New fork** (`[bau/trainers.py](bau/trainers.py)` lines 373-425):

$$
\ell_d^{\text{new}}(\mathbf{z}) = \log\left(\frac{1}{B^2}\sum_i\sum_{k \in \mathcal{N}_i} \exp\bigl(-2d_F(\mathbf{z}_i, \mathbf{c}_k)^2\bigr)\right)
$$

where $\mathcal{N}_i$ is now determined by sorting $d_F(\mathbf{z}_i, \mathbf{c}_k)^2$ (Finsler squared).

**Discrepancy D1 — Squared Finsler distance in sorting:** Old sorts by $\|\hat{\mathbf{f}}_i - \mathbf{c}_k\|^2 \geq 0$. New sorts by $d_F(\mathbf{z}_i, \mathbf{c}_k)^2 \geq 0$. Since $d_F$ can be negative, $d_F^2$ maps both $d_F = +\delta$ and $d_F = -\delta$ to the same value $\delta^2$. This means pairs with **strongly negative** Finsler distance (reverse-direction close) are sorted as equivalently "near" to pairs with small positive distance. The neighborhood $\mathcal{N}_i$ may include entries that are "close" only because the asymmetry term is large and negative. **Impact on narrative:** This is a subtle issue. The domain loss intends to repel same-domain neighbors. If the nearest neighbors include pairs with large negative $d_F$, the loss repels in a direction that may not correspond to meaningful domain clustering. In practice, if drift norms are small (as observed), this discrepancy is minor.

**Discrepancy D2 — Memory bank feature type:** In old BAU, the memory bank stores L2-normalized BN features $\hat{\mathbf{f}}$. In the new code with `memory_bank_mode="full"`, it stores $\mathbf{z} = [\hat{\mathbf{f}} \mid \boldsymbol{\omega}^\perp]$ with momentum updates. The drift portion of stored features becomes stale as the drift head evolves. With momentum coefficient $\mu$ (typically 0.2-0.5):

$$
\mathbf{c}_k^{(t)} = \mu\mathbf{z}_k^{(t)} + (1-\mu)\mathbf{c}_k^{(t-1)}
$$

The identity component converges because the backbone changes slowly. The drift component may oscillate because the drift head is more volatile (lower LR, but higher relative gradient variance due to the small parameter count). **Impact:** Stale drift in the memory bank could introduce noise into the domain loss neighborhood computation.

---

### 5.7 Omega (Drift Norm) Regularization

**Old BAU:** Not present.

**New fork** (`[bau/trainers.py](bau/trainers.py)` lines 428-456):

$$
\mathcal{L}_\omega = \lambda_\omega \cdot \frac{1}{2}\left[\ell_\Omega(\boldsymbol{\omega}^{w,\perp}) + \ell_\Omega(\boldsymbol{\omega}^{s,\perp})\right]
$$

$$
\ell_\Omega(\boldsymbol{\omega}) = -\frac{1}{B}\sum_{i=1}^B \log\bigl(\max(\omega_{\max} - \|\boldsymbol{\omega}_i\|_2, \epsilon)\bigr)
$$

This is a log-barrier interior point penalty. The gradient:

$$
\frac{\partial \ell_\Omega}{\partial \boldsymbol{\omega}_i} = \frac{1}{B}\cdot\frac{\boldsymbol{\omega}_i}{\|\boldsymbol{\omega}_i\|(\omega_{\max} - \|\boldsymbol{\omega}_i\|)}
$$

This diverges as $\|\boldsymbol{\omega}_i\| \to \omega_{\max}$, creating an infinite penalty barrier. At small norms, the gradient is approximately $\boldsymbol{\omega}_i / (B \cdot \|\boldsymbol{\omega}_i\| \cdot \omega_{\max})$, which gently pushes drift toward zero.

**Impact on narrative:** Combined with the sigmoid gating in `scale_drift_vector` (which already bounds drift to $[0, \omega_{\max}]$), this is a belt-and-suspenders approach. The log-barrier adds a small bias toward zero-drift that further suppresses the asymmetric component. For the domain-conditioned pivot, this regularizer may need to be relaxed or disabled to allow domain-level drift to express itself.

---

### 5.8 Domain-Token CE Loss

**Old BAU:** Not present.

**New fork** (`[bau/trainers.py](bau/trainers.py)` lines 212-215):

$$
\mathcal{L}_{dt} = \lambda_{dt} \cdot \text{CE}\bigl(g(\text{sg}[\mathbf{f}]), d\bigr)
$$

where $g: \mathbb{R}^D \to \mathbb{R}^S$ is a linear classifier, $\text{sg}[\cdot]$ denotes stop-gradient (detach), $d$ is the source-domain label, and CE is standard (unsmoothed) cross-entropy.

**Impact on narrative:** This is an auxiliary loss that trains the domain classifier to predict the source domain from identity features. The stop-gradient ensures this does not distort the identity representation. At evaluation time, the trained classifier produces soft domain probabilities for the domain-conditioned drift head. This is well-motivated and standard practice for auxiliary conditioning heads.

---

### 5.9 Total Loss Comparison

**Old BAU:**

$$
\mathcal{L}^{\text{old}} = \mathcal{L}_{ce} + \mathcal{L}_{tri} + \lambda\mathcal{L}_{align} + \mathcal{L}_{uniform} + \mathcal{L}_{domain}
$$

Five terms, all operating on the same feature space ($\hat{\mathbf{f}} \in \mathbb{S}^{D-1}$ for alignment/uniform/domain, $\mathbf{e} \in \mathbb{R}^D$ for triplet), all using symmetric Euclidean distance. The gradient landscape is consistent: all losses agree on what "close" and "far" mean.

**New fork:**

$$
\mathcal{L}^{\text{new}} = \mathcal{L}_{ce} + \mathcal{L}_{tri} + \lambda(\mathcal{L}_{align} + \mathcal{L}_{drift\text{-}align}) + \mathcal{L}_{uniform} + \mathcal{L}_{domain} + \mathcal{L}_\omega + \mathcal{L}_{dt}
$$

Seven+ terms operating on **three different feature spaces** with **two different distance functions:**


| Loss | Feature space | Distance | Geometry |
| --------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------- |
| $\mathcal{L}_{ce}$ | $\mathbf{f} \in \mathbb{R}^D$ (BN, pre-norm) | N/A (classifier) | Euclidean |
| $\mathcal{L}_{tri}$ | $\mathbf{u} = [\mathbf{e} \mid \boldsymbol{\omega}^\perp] \in \mathbb{R}^{D+D_\omega}$ (pre-BN) | $d_F$ (symmetric trapezoidal forced) | Asymmetric |
| $\mathcal{L}_{align}$ | $\hat{\mathbf{f}} \in \mathbb{S}^{D-1}$ (identity-only, normalized) | $d_E$ | Symmetric |
| $\mathcal{L}_{uniform}$ | $\mathbf{z} = [\hat{\mathbf{f}} \mid \boldsymbol{\omega}^\perp] \in \mathbb{R}^{D+D_\omega}$ (NOT on sphere) | $d_F$ | Asymmetric |
| $\mathcal{L}_{domain}$ | $\mathbf{z}$ vs memory bank | $d_F$ | Asymmetric |
| $\mathcal{L}_\omega$ | $\boldsymbol{\omega}^\perp \in \mathbb{R}^{D_\omega}$ (drift-only) | $\ell_2$ norm | Euclidean |
| $\mathcal{L}_{dt}$ | $\text{sg}[\mathbf{f}] \in \mathbb{R}^D$ (identity, detached) | N/A (classifier) | Euclidean |


**The fundamental tension:** $\mathcal{L}_{align}$ pushes identity features to be augmentation-invariant (and ignores drift). $\mathcal{L}_{uniform}$ pushes ALL pairs apart using asymmetric distance (including the drift component). $\mathcal{L}_\omega$ pushes drift norms toward zero. The net effect of these three losses on drift is: alignment ignores it, uniformity tries to suppress it (by driving asymmetry to zero), and omega regularization explicitly penalizes it. There is **no loss term that provides a positive learning signal for non-zero drift** except the triplet loss, which operates on pre-BN features and uses a forced-symmetric approximation of the asymmetry.

This is the mathematical explanation for why drift collapses to near-zero under the full BAU loss configuration.