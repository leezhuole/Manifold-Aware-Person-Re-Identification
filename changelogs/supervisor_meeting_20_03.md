# Supervisor Meeting Notes — 2026-03-20

## Context

Post-presentation discussion with supervisors. The central theme: **clean signal separation between identity-related and domain-related loss terms** to facilitate feature disentanglement and isolate the contribution of asymmetric (Finsler) geometry to domain modeling.

Current empirical state: under the full BAU loss landscape, the optimizer drives alpha→0, recovering Euclidean. Drift norms remain near zero. The only configuration where Finsler marginally outperformed Euclidean was Base+Align+Domain (no uniformity), alpha=0.027.

---

## Idea 1: Split Triplet Loss into Identity-Euclidean + Domain-Finsler

### Stated proposal

Replace the current single triplet loss (Finsler distance, `RandomMultipleGallerySampler`, pid-based mining) with two independent triplet losses:

**1a. Euclidean Identity Triplet** — identical to old BAU: `euclidean_dist`, `RandomIdentitySampler`, batch-hard mining on pid labels.

$$\mathcal{L}_{tri}^{id} = \frac{1}{B}\sum_{i}\bigl[\max_{j: y_j = y_i} d_E(\mathbf{e}_i, \mathbf{e}_j) - \min_{k: y_k \neq y_i} d_E(\mathbf{e}_i, \mathbf{e}_k) + m_{id}\bigr]_+$$

**1b. Finsler Domain Triplet** — new: `finsler_drift_dist`, a new `RandomDomainSampler`, batch-hard mining on domain/camera labels.

$$\mathcal{L}_{tri}^{dom} = \frac{1}{B}\sum_{i}\bigl[\max_{j: d_j = d_i} d_F(\mathbf{z}_i, \mathbf{z}_j) - \min_{k: d_k \neq d_i} d_F(\mathbf{z}_i, \mathbf{z}_k) + m_{dom}\bigr]_+$$

where $d_i$ is domain/camera label.

Rationale: identity invariance stays Euclidean (same-person features close regardless of domain); domain structure is modeled via Finsler (same-domain features clustered in asymmetric space).

### Critical analysis

**Fundamental gradient conflict.** The domain triplet (1b) pulls same-domain embeddings closer and pushes different-domain embeddings apart. The existing `L_Domain` does the **exact opposite**: it repels nearest same-domain memory bank neighbors. The existing `L_Uniform` also repels all pairs. These objectives are directly contradictory.

The BAU domain loss exists precisely because DG-ReID needs domain-*invariant* features — clustering by domain is the failure mode, not the goal (Zhong et al., "Learning to Generalize Unseen Domains via Memory-based Multi-Source Meta-Learning for Person Re-Identification," CVPR 2019). A domain triplet that attracts same-domain embeddings would re-introduce camera-style shortcuts.

**Counter-argument from the user's framing:** The user said "we want to compare images from different domains, but not necessarily from the same identity" — this suggests the intent is to **learn what differs between domains** (direct drift toward modeling domain artifacts), not to collapse them. If this is the intent, then the mining mask should be inverted: the "positive" should be **cross-domain same-identity** and the "negative" should be **same-domain same-identity**, which is not what standard batch-hard triplet mining does.

**Sampler feasibility.** The proposal calls for `RandomIdentitySampler` for loss 1a and a new `RandomDomainSampler` for loss 1b. But the training loop issues a **single batch** per iteration. Running two separate samplers would require either:
- Two data loaders (doubling memory and I/O), or
- A single batch from which both losses are computed using different masks.

The second option is viable: use one sampler (e.g. `RandomIdentitySampler` to stay close to the BAU baseline, as the user suggested), then apply pid-mask for $\mathcal{L}_{tri}^{id}$ and did-mask for $\mathcal{L}_{tri}^{dom}$ to the **same batch**. No new sampler is needed — only a new mining mask.

**What a `RandomDomainSampler` would look like.** If implemented, it would sample $D$ domains with $M$ images per domain, regardless of identity. This produces batches where identity overlap across samples is sparse and random. The batch-hard mining would find the hardest same-domain positive and hardest cross-domain negative. With only 3 source domains and $M \approx 85$ per domain, the hardest same-domain positive is almost certainly a very different identity — the loss would learn to **cluster arbitrary identities by domain**, exactly the failure mode DG-ReID avoids.

### Verdict

**1a (Euclidean identity triplet): Sound.** Reverting to the old BAU identity triplet is a clean ablation baseline. It removes the Finsler distance from identity-level metric learning, which is consistent with the narrative that asymmetry should be at the domain level.

**1b (Finsler domain triplet with same-domain attraction): Contradicts the DG objective.** Same-domain attraction is the adversary, not the objective. If the intent is to **teach the drift to model domain-specific artifacts**, a different formulation is needed — e.g., a contrastive loss that encourages drift vectors from the same domain to be similar (Idea 3) while keeping identity features domain-invariant.

### Refined alternative for 1b

If the goal is "compare images from different domains, not necessarily same identity," the loss should be formulated as **cross-domain contrastive on identity features via Finsler distance**:

$$\mathcal{L}_{contrast}^{dom} = \frac{1}{|\mathcal{C}|}\sum_{(i,j)\in\mathcal{C}} d_F(\mathbf{z}_i, \mathbf{z}_j)^2$$

where $\mathcal{C} = \{(i,j) : y_i = y_j, d_i \neq d_j\}$ — same identity, different domain. This encourages the Finsler distance to **shrink** for cross-domain same-identity pairs, which means the asymmetric term must compensate for the domain shift. This is consistent with the narrative and does not require a new sampler — it needs cross-camera positive pairs, which `RandomMultipleGallerySampler` already provides.

---

## Idea 2: Extend Domain Loss with Cross-Domain Finsler Repulsion

### Stated proposal

Currently `domain_loss` only considers nearest same-domain memory bank prototypes. Supplement with a term that considers nearest prototypes from **different domains**, using Finsler distance. Fallback: substitute some elements of $\mathcal{N}(i)$ with cross-domain instances.

### Critical analysis

**Redundancy with uniformity.** Cross-domain repulsion on the full embedding space is functionally equivalent to `L_Uniform` applied to full embeddings — both push arbitrary pairs apart. The only difference is that the domain loss uses a memory bank rather than in-batch pairs, and selects nearest neighbors rather than all pairs. If `L_Uniform` is already applied (even on identity-only), adding cross-domain Finsler repulsion creates a superposition of symmetric repulsion (uniform) and asymmetric repulsion (cross-domain Finsler), with no clear separation of roles.

**Harm to same-identity cross-domain matching.** Unconditional cross-domain repulsion pushes apart all cross-domain pairs, including true positives (same identity, different camera). This is catastrophic for the core ReID task.

**Forgotten rationale.** The user acknowledged not remembering why this was proposed. Without a clear justification, adding a term that actively harms the objective should not be pursued.

**Memory bank feasibility.** The memory bank stores features indexed by pid, with associated domain labels. Computing cross-domain distances is straightforward by inverting the domain mask. The computation is $O(B \times N_{mem})$ regardless of the mask, so cost is unchanged. The feasibility concern is unfounded.

### Verdict

**Discard.** The rationale is lost, the formulation overlaps with existing losses, and unconditional cross-domain repulsion damages cross-domain retrieval.

---

## Idea 3: Drift-Domain Alignment Loss

### Stated proposal

Align drift vectors $\omega_i$ with their domain-ids or camera-ids, in a form similar to the alignment loss but without augmentation pairs.

### Critical analysis

**Relationship to Idea 1b.** This is **not** the same as the domain triplet. The domain triplet operates on full embeddings and uses batch-hard mining with a margin. A drift-domain alignment loss operates exclusively on the drift subspace and enforces that drift vectors within a domain are similar (no margin, no hard mining). The domain triplet clusters full embeddings by domain; drift alignment clusters only drift vectors by domain while leaving identity features free.

**Drift collapse concern.** The user raised this: will all drift vectors within a domain become identical? With the current `DomainConditionedDriftHead`, the domain prior is shared within a domain (via the domain embedding), but the residual correction (`residual_scale * residual_block(emb)`) and the feature-dependent gate (`domain_gate(emb)`) introduce per-instance variation. If the alignment loss is strong, it will suppress both the residual and the gate variation, collapsing drift to a pure domain prototype. The residual correction only prevents collapse if the alignment loss weight is small relative to the gradients from other losses that maintain instance-level drift variation.

**Consistency with the narrative.** This is the most narratively aligned idea. The research pivot states: "asymmetry should be modeled primarily at the domain/view level." Encouraging drift vectors to be similar within a domain is exactly this. The key question is whether "similar" means "identical" (collapse) or "clustered with some variance" (desired).

**Proposed formulation.** For each domain $d$, compute the batch-wise drift prototype $\bar{\omega}_d = \frac{1}{|\mathcal{B}_d|}\sum_{i \in \mathcal{B}_d} \omega_i$, then:

$$\mathcal{L}_{drift\text{-}align}^{dom} = \frac{1}{B}\sum_{i} \|\omega_i - \bar{\omega}_{d(i)}\|^2$$

This is a variance-minimization objective within each domain. With a small weight $\lambda_{da}$, it gently encourages domain coherence without hard collapse. The residual block provides the escape hatch for instance-level variation.

### Verdict

**Promising, with caveats.** Implement with a small weight ($\lambda_{da} \leq 0.1$). Monitor intra-domain drift variance during training. If variance drops below a threshold (e.g., 1% of mean drift norm), the weight is too high or the formulation needs a diversity regularizer.

---

## Idea 4: Toy Dataset for Sanity Checking

### Stated proposal

Subsample Market1501 and apply synthetic domain-specific augmentations (darkening, blurring for specific cameras) to create a controlled setting where domain-conditioned drift should help.

### Critical analysis

**This is the highest-priority action.** The fundamental question — "can the Finsler distance learn useful domain asymmetry at all?" — remains unanswered. All empirical evidence so far shows alpha→0 and drift→0. Before adding more loss terms, the **optimization dynamics** of the drift branch must be validated in a regime where the answer is known a priori.

**Design requirements for the toy dataset:**
1. **Deterministic, severe, camera-specific corruption.** E.g., Camera 1: original, Camera 2: strong Gaussian blur ($\sigma=3$), Camera 3: severe brightness reduction ($\times 0.3$). The corruptions must be strong enough that a symmetric Euclidean model demonstrably suffers.
2. **Known ground truth.** The same identities appear across all cameras. The query is always from Camera 1 (clean), the gallery spans Cameras 1-3.
3. **Expected outcome.** The Finsler model should learn drift vectors that compensate for the camera-specific corruption. If it cannot (alpha→0 even here), the Finsler distance function or gradient flow is fundamentally broken.
4. **Subsample size.** Use 50-100 identities, 2-4 images per identity per camera. Small enough for rapid iteration, large enough for meaningful batch-hard mining.

**Literature support.** Hendrycks & Dietterich ("Benchmarking Neural Network Robustness to Common Corruptions and Perturbations," ICLR 2019) provide a taxonomy of image corruptions. Volpi et al. ("Generalizing to Unseen Domains via Adversarial Data Augmentation," NeurIPS 2018) use synthetic domain shifts to study generalization.

### Verdict

**Implement first, before any loss modifications.** This is a diagnostic experiment. If the drift branch cannot learn meaningful asymmetry under a controlled synthetic shift, no amount of loss engineering will fix the underlying problem.

---

## Implementation Plan

### Priority 1: Toy Synthetic Domain Dataset (Idea 4)

1. Create `bau/datasets/market1501_synthetic.py` that wraps Market1501 and applies camera-specific transforms at load time.
2. Define 3 corruption profiles: clean, blur($\sigma=3$), darken($\times 0.3$).
3. Map Market1501 camera IDs 1-6 to corruption profiles (e.g., cameras 1-2: clean, 3-4: blur, 5-6: darken).
4. Register as `market1501_synthetic` in `bau/datasets/__init__.py`.
5. Train `resnet50_finsler` with domain-conditioned drift on this dataset.
6. **Diagnostic metrics:** track per-camera drift prototypes, drift norm evolution, alpha evolution, mAP on clean→corrupted and corrupted→clean retrieval directions.

### Priority 2: Euclidean Identity Triplet (Idea 1a) — restore BAU triplet

1. Add `--identity-triplet-only` flag to `examples/train_bau.py`.
2. When active: triplet loss uses `euclidean_dist` on identity-only features (`emb_w[:, :identity_dim]` or `emb_w` for Euclidean baseline), regardless of model `dist_func`.
3. This is essentially `--force-euclidean` applied only to the triplet loss path, not to domain loss.

### Priority 3: Drift-Domain Alignment (Idea 3)

1. Add `BAUTrainer.drift_domain_alignment_loss(omega, dids)` method.
2. Compute per-domain drift prototype (batch mean, detached), then MSE between each drift and its domain prototype.
3. Add `--use-drift-domain-align` flag (default off) and `--drift-domain-align-weight` (default 0.05).
4. Monitor intra-domain drift variance via W&B.

### Priority 4: Cross-Domain Same-Identity Finsler Contrastive (Refined 1b)

1. After Idea 4 validates that drift can learn domain structure, add a contrastive term on cross-camera same-identity pairs using Finsler distance.
2. This requires cross-camera positive pairs in the batch — `RandomMultipleGallerySampler` already provides these.
3. Formulation: for each cross-camera positive pair $(i, j)$ where $y_i = y_j, d_i \neq d_j$, minimize $d_F(\mathbf{z}_i, \mathbf{z}_j)^2$.
4. This teaches the Finsler distance to compensate for domain shift between same-identity images, without requiring a new sampler.

### Discarded

- **Idea 1b as stated** (same-domain attraction triplet): contradicts DG objective.
- **Idea 2** (cross-domain Finsler repulsion): redundant with uniformity, harms cross-domain retrieval, rationale lost.

---

## Follow-Up Questions

1. **Gradient conflict resolution.** If $\mathcal{L}_{tri}^{dom}$ (Idea 1b) attracts same-domain features and $\mathcal{L}_{Domain}$ repels them, what is the expected equilibrium? Has this been modeled, or is the assumption that one dominates?

2. **Sampler architecture.** The user stated "we should use the same sampler as original BAU." If `RandomIdentitySampler` is used, the batch guarantees P identities × K instances, but domain coverage is random. For the Finsler domain triplet to see meaningful cross-domain contrasts, domain diversity within the batch must be ensured. Is a `DomainBalancedIdentitySampler` (P identities, K instances, forced cross-camera) the intended compromise?

3. **Drift collapse diagnostic.** The user argued the residual correction prevents drift collapse within domains. What is the expected magnitude of the residual relative to the domain prior? If `domain_residual_scale=0.1` and the domain prior norm is 0.5, the residual contributes at most 0.05 — is this sufficient to prevent functional collapse under a drift-domain alignment loss?

4. **Toy dataset scope.** Should the synthetic corruptions be applied at train time only (simulating known source-domain shifts) or also at test time (simulating unknown target corruptions)? The diagnostic value differs: train-only tests whether drift can model known shifts; test-only tests whether the soft domain predictor generalizes to unseen corruptions.

5. **Interaction with existing fixes.** The uniform loss was recently restricted to identity-only Euclidean. The k-NN weights were fixed to use identity-only features. Are these changes assumed to be active for all proposed new experiments, or should some experiments revert to the pre-fix state for comparison?
