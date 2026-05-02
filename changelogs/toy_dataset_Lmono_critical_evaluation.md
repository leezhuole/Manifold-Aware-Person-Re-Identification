# Critical Evaluation: L_mono and the Asymmetric Gap Toy Study

**Date:** 2026-05-02 (updated 2026-05-02 to reflect design decisions)
**Scope:** Rigorous peer-level critique of the proposed L_mono / toy-dataset experiment described in the 30 Apr 2026 handwritten notes, with recommendations on model selection, training protocol, and claimable contributions. This document supersedes informal notes and should be read alongside `toy_dataset_v4.md` and `toy_dataset_asymmetry_diagnostics.md`.

**Design decisions confirmed in discussion:**
- Frozen BAU backbone (ResNet-50 pretrained on Market+MS+CS).
- Last linear layer extended by 1 output dimension (not a separate projection module) to produce θ.
- L_mono only — no other loss terms.
- No comparison against the Finsler BAU (Arm 1b) checkpoint; this study builds intuition from the ground up independently.

---

## 1. The Proposed Idea in Precise Form

The notes propose adding a single scalar dimension θ to the existing d=2048 embedding:

$$
\mathbf{z} \in \mathbb{R}^{d+1} = [\underbrace{f}_{\text{identity}}, \underbrace{\theta}_{\text{noise dim}}]
$$

with a globally fixed drift vector ω = [0, α] (mass entirely on the appended axis, α ∈ (0, 0.95)). The Randers distance on this space is:

$$
d_{FC}(\mathbf{x}, \mathbf{y}) = \|\mathbf{z}_y - \mathbf{z}_x\|_2 + \omega^T(\mathbf{z}_y - \mathbf{z}_x) = \|f_y - f_x\|_2 + \alpha(\theta_y - \theta_x)
$$

**L_mono** enforces θ^{k_a} > θ^{k_b} for k_a < k_b (cleaner images have higher θ):

$$
\mathcal{L}_\text{mono} = \frac{1}{|\mathcal{B}|} \sum_{(p,s)} \sum_{0 \le k_a < k_b \le 4} [\theta^{k_a}_{p,s} - \theta^{k_b}_{p,s} + \lambda]_+
$$

The **claimed asymmetric gap** is:

$$
d_{FC}(z^0, z^k) - d_{FC}(z^k, z^0) = 2\alpha(\theta^k - \theta^0) < 0 \quad \text{(since } \theta^k < \theta^0\text{)}
$$

### 1.1 Architecture: extending the last linear layer

The BAU ResNet-50 BN neck maps GeM-pooled backbone features h ∈ R^{2048} through a linear layer to produce the identity embedding. The confirmed design extends this weight matrix from R^{2048×2048} to R^{2048×2049}, where:

- Columns 1..2048: frozen (pretrained identity weights), producing f = W^T h ∈ R^{2048}.
- Column 2049: the single new trainable parameter w_new ∈ R^{2048}, producing θ = w_new^T h ∈ R^1.

This is not identical to adding a separate `Linear(2048, 1)` module that reads from post-BN features f. That design would compute θ = v^T f = v^T BN(W^T h), which is linear in h only after the fixed affine BN transform. The extended-layer design reads directly from h (the pre-BN input to the linear layer). Both span the same space of learnable functions of h (since BN is an invertible affine map at evaluation time), but they differ in gradient scale and input normalization during training.

**Critical implementation note:** if the BN layer following the extended linear layer normalizes all 2049 outputs jointly, it will zero-center and variance-normalize θ per batch, which directly opposes L_mono (the ordering is preserved only within a batch, not across batches or at evaluation time). BN must be applied only to the first 2048 dimensions. θ must remain as raw pre-BN linear output.

---

## 2. What Is Already Established (Do Not Reproduce)

Before evaluating what is new, note what existing changelogs have already resolved:

| Issue | Resolved in |
|---|---|
| Single-axis collinearity (D1 pre-flight) | `toy_dataset_v4.md §Phase 0` |
| Randers positivity: ‖ω‖ < 1 constraint | `toy_dataset_v4.md §Critical evaluation, point 2` |
| Isotropic-random ω in R^{2049} has ω_{D+1}/‖ω‖ ≈ 2.2% — appended axis is geometrically negligible | `toy_dataset_v4.md §One critical geometric concern` |
| Two ω-init regimes (Regime A isotropic, Regime B mass-concentrated) | `toy_dataset_v4.md §Phase 2` |
| Absorption ratios η_B, η_C and their isotropic null | `toy_dataset_asymmetry_diagnostics.md §5` |
| Drift cosine block structure and H1a/H1b/H1c hypotheses | `toy_dataset_asymmetry_diagnostics.md §2-4` |
| Two source crops per PID (v3.0 schema, why v2.0 is defective) | `toy_dataset_asymmetry_diagnostics.md §1` |

The key value added by the handwritten notes and this study:

1. A cleaner, more explicit formulation of L_mono as a standalone loss, decoupled from all other BAU objectives.
2. The specific design choice ω = [0, α] (ω_{1:D} ≡ 0, all mass on appended axis).
3. The page-3 analysis asking whether the asymmetric gap actually helps mAP — which is the most important unresolved question.

---

## 3. The Core Mathematical Critique: Does the Asymmetric Gap Improve mAP?

**This is the load-bearing question. The notes (page 3) correctly identify it but do not answer it.**

For a fixed query q, mAP and Rank-1 rank gallery items by:

$$
d_{FC}(q, g_i) = \|f_{g_i} - f_q\|_2 + \alpha(\theta_{g_i} - \theta_q)
$$

The term −αθ_q is constant across all gallery items and cancels in pairwise comparisons. Ranking is therefore determined by:

$$
\|f_{g_i} - f_q\|_2 + \alpha \cdot \theta_{g_i}
$$

This is Euclidean distance plus a **query-independent additive bias** proportional to the gallery item's noise dimension. Because θ^k < θ^0 (corrupted has lower θ), corrupted gallery items receive a smaller additive penalty compared to cleaner ones.

### 3.1 When (and whether) the bias helps

For a clean query (θ_q = θ^0) querying a gallery of mixed corruption:
- True positive (same PID, high corruption k=4): small bias α·θ^4 → ranks closer relative to Euclidean.
- Impostors (different PID, high corruption k=4): **same** small bias → ranks equally closer.

**The Randers bias is identity-agnostic.** Within a single severity stratum, the additive bias α·θ^{k(i)} is identical for the true positive and every impostor at that severity. The Randers term does not change the relative ordering within a corruption level — it only reorders items across different corruption levels. Specifically:

- All items at σ=4 get the smallest bias → all move equally closer to the query.
- All items at σ=0 get the largest bias → all move equally farther from the query.

**This is rank-preserving within each severity stratum.** The only scenario where mAP changes is when the true positive is at a different severity than the highest-ranked impostor, causing a cross-stratum reordering. This is a second-order effect dependent on the specific gallery composition, not a first-order retrieval mechanism. In a realistic mixed gallery (both TP and impostors at all severity levels), the net mAP effect of the global bias could be negligible or negative.

**Condition for zero effect:** if the toy evaluation uses a gallery where ALL items are at a single fixed severity, α·θ^{k(i)} is constant across all gallery items and Randers ranking is identical to Euclidean. The standard toy protocol (query=σ=0, gallery=σ=1..4) mixes severities, which is a necessary but not sufficient condition for Randers to have any effect.

### 3.2 The correct asymmetry claim

The notes correctly show that d_{FC}(z^0, z^k) < d_{FC}(z^k, z^0) — the gap is strictly negative and proportional to 2α(θ^k − θ^0). But this is a **geometric** statement about the metric, not a **retrieval** statement about mAP:

- **Geometric asymmetry:** the Randers ball around z^0 reaches farther toward corrupted points than the ball around z^k reaches toward clean points. True by construction once L_mono holds.
- **Retrieval asymmetry:** swapping query/gallery roles changes mAP. True from the gallery pool composition alone, independent of whether Randers is used.

Demonstrating Δ < 0 validates the geometric mechanism. Whether mAP improves is a separate empirical question answered by the bidirectional evaluation protocol.

---

## 4. L_mono: Specific Design Critiques

### 4.1 Batch structure requirement

L_mono sums over pairs (k_a, k_b) for the same person-source pair (p, s). This requires multiple severity levels of the same person to co-occur in the same batch.

With a standard RandomIdentitySampler drawing P=16 identities and K=4 instances per identity, whether those 4 instances span multiple severities is sampler-dependent. Unless the sampler is severity-aware, L_mono will have many zero-gradient steps.

**Requirement:** the batch must include at least 2 severity levels per (p, s) for L_mono to have non-zero gradient. Options:
- A severity-stratified sampler (recommended), or
- Large enough K (≥5, one per severity level) with the full 5-severity toy training split.

### 4.2 Why BAU losses are excluded (motivation for L_mono-only design)

BAU's composite loss (alignment + uniformity + triplet + CE + domain) was originally identified as a conflict with L_mono: all five terms push the backbone to suppress corruption-induced variation in f, which is the same variation L_mono needs to route into θ. The README Research Status Update documents the empirical outcome of this competition: "learned drift magnitude often stays close to zero." The confirmed frozen backbone + L_mono-only design resolves this conflict entirely — the backbone is not updated and its losses do not apply. L_mono is the sole training signal for the one trainable parameter.

### 4.3 The disentanglement question

With ω = [0, α], the entire Randers correction depends on θ = w_new^T h. L_mono enforces that θ is monotone in severity, but it does not guarantee that h contains sufficient corruption signal for w_new to exploit. If the frozen BAU backbone has already suppressed corruption variation in h (via the alignment/uniformity training on Market+MS+CS), then L_mono has weak signal and θ variation will be small regardless of training duration.

This is the central empirical unknown. BAU was trained with standard augmentations (crop, flip, color jitter), not with the toy dataset's specific composed corruptions (JPEG + blur + downsample + gamma + noise). The residual corruption signal in h for these specific operators is an open empirical question that the D1 diagnostic (below) partially answers.

### 4.4 Scale calibration of α

The Randers correction α·(θ_y − θ_x) must be competitive in magnitude with ||f_y − f_x||_2 to influence ranking non-trivially. L_mono enforces ordering but not absolute scale of θ. The hinge margin λ determines the minimum gap enforced; this must be set relative to the typical identity distance scale (~0.5–1.0 for L2-normalized features). Recommend sweeping α ∈ {0.1, 0.3, 0.5, 0.9} and reporting which α produces measurable rank changes.

---

## 5. Experimental Protocol

### 5.1 Confirmed design

| Component | Choice |
|---|---|
| Backbone | Frozen ResNet-50, BAU pretrained on Market+MS+CS |
| New parameter | Column 2049 of the last linear layer (w_new ∈ R^{2048}), initialized near-zero |
| θ computation | θ = w_new^T h (pre-BN, from GeM-pooled features) |
| BN | Applied to first 2048 outputs only; θ bypasses BN |
| Training loss | L_mono only |
| Training data | Toy training split (disjoint from 50 eval PIDs) |

This is a linear probing experiment. The backbone never sees the toy distribution; only the single new weight vector is trained. It is not target-domain adaptation in the DG sense because the backbone is unchanged.

### 5.2 Stopping criterion

Pre-specify before training. Recommended: stop when L_mono < λ (all pairs satisfy the margin with slack), or at a fixed epoch count validated on a held-out 10% severity-balanced subset of the training split by Spearman ρ(θ, σ). Do not stop based on observed mAP.

### 5.3 Models

| Model | Description | Evaluates |
|---|---|---|
| **M1** (Euclidean) | Frozen BAU backbone, Euclidean ranking on f, no θ, no L_mono | Baseline — task asymmetry without Randers |
| **M3** (L_mono) | Frozen BAU backbone + extended last layer + L_mono, Randers ranking | Whether L_mono closes the bidirectional mAP gap |

M1 is the implicit control for M3: same frozen backbone, same f, only the Randers term differs. The comparison is maximally clean.

---

## 6. Relationship to Section 4.4

The paper's Section 4.4 shows that drift magnitude ‖ω‖ increases monotonically with synthetic corruption severity (Figure 1) as an emergent property of the trained Finsler BAU model. This study differs in three ways:

1. **Causal vs. emergent:** Section 4.4 observes monotonicity post-hoc; L_mono enforces it as an explicit training objective.
2. **Retrieval-level:** Section 4.4 reports drift/identity norms only; this study reports bidirectional mAP (clean→corrupted vs. corrupted→clean).
3. **Isolated mechanism:** Section 4.4 is confounded by the full Finsler training stack; this study trains exactly one parameter with exactly one loss.

Since the Finsler BAU checkpoint is deliberately excluded from this study's scope, no direct comparison to Section 4.4 results is made. The toy study stands independently as a ground-up validation of the L_mono mechanism.

---

## 7. Training and Evaluation Protocol

### 7.1 Training

- **Sampler:** severity-stratified; each person must appear at ≥2 distinct severity levels per batch.
- **Loss:** L_mono only, with margin λ swept in {0.05, 0.1, 0.2}.
- **α:** fixed constant swept in {0.1, 0.3, 0.5, 0.9}; not learned.
- **Epochs:** fixed by the pre-specified stopping criterion (§5.2).
- **Seeds:** ≥3 independent runs.

### 7.2 Evaluation

**Table 1: Task-direction mAP/Rank-1**

| Model | clean→corrupted | corrupted→clean | Δ |
|---|---|---|---|
| M1 (Euclidean) | ... | ... | ≈ 0 (metric symmetric; task asymmetry from pool alone) |
| M3 (L_mono + Randers) | ... | ... | expected < 0 (Randers should widen clean-query advantage) |

**Table 2: Geometric asymmetric gap**

| Model | mean [d(z^0, z^k) − d(z^k, z^0)] | Spearman ρ(θ, σ) |
|---|---|---|
| M1 | 0 by construction | N/A |
| M3 | target: < 0 | target: > 0.8 |

Report all metrics with bootstrap 95% CI (N=50 eval PIDs is small; statistical power requires careful reporting).

---

## 8. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| D1 failure: composed corruptions non-collinear in feature space → no monotone direction exists in h | High | Run D1 collinearity diagnostic on frozen backbone features before any training; fall back to single-operator (JPEG-only) ladder if needed |
| Frozen BAU h already corruption-invariant → L_mono has near-zero gradient signal | Medium-High | Monitor θ standard deviation across severities during training; if it stays near zero after 10 epochs, the backbone has suppressed the signal |
| BN applied to θ → L_mono's ordering is batch-dependent and breaks at evaluation | High | Verify BN slice in implementation before training |
| α too small → Randers correction negligible → M3 ≈ M1 at evaluation | Medium | Sweep α; report ranking changes explicitly at each α value |
| Single-severity gallery → Randers provides zero ranking gain | Medium | Ensure mixed-severity gallery in evaluation; verify with 5×5 severity-split matrices |
| L_mono gradient zero if sampler does not provide multi-severity batches | Medium | Severity-stratified sampler required |

---

## 9. What Claim Can Actually Be Made

**Claimable (within the toy setting):**

> Under a controlled synthetic corruption ladder (5 severity levels, ImageNet-C-style operators), a single new weight vector trained by a pairwise hinge loss (L_mono) on top of a frozen identity backbone can enforce that the appended scalar dimension θ decreases monotonically with corruption severity (target: Spearman ρ > 0.8). This produces a Randers metric with the correct asymmetric sign: clean-to-corrupted distances are systematically shorter than corrupted-to-clean distances (target: mean Δ < 0, p < 0.05). In a mixed-severity gallery, Randers ranking may close the bidirectional mAP gap relative to Euclidean.

**Stronger if shown:** M3 closes the bidirectional mAP gap by a measurable margin relative to M1 Euclidean. This upgrades the claim from geometric to retrieval.

**Not claimable:**

> The Randers bias preferentially benefits true positives over impostors. (The bias is identity-agnostic within each severity stratum — it reorders severity strata globally.)

> This generalizes to real multi-camera ReID. (It requires a monotone scalar severity axis, which real camera networks do not provide.)

> This improves DG-ReID mAP. (The toy study is controlled and does not touch the DG evaluation protocol.)

---

## 10. Summary Verdict

The mathematical formulation is internally consistent. The L_mono loss is correct and implementable. The confirmed design (frozen backbone, extended last linear layer, L_mono only) is the cleanest possible isolation of the mechanism: one trainable parameter, one loss, one measurable outcome.

**The one surviving concern** from the original four-problem critique: **the Randers bias is identity-agnostic within each severity stratum.** Even with perfect monotonicity (ρ = 1.0), mAP improves only through cross-stratum reorderings. This limits the expected mAP delta and means Δ < 0 (geometric) and ΔmAP > 0 (retrieval) must be reported separately and not conflated.

**Remaining practical requirements:**
1. D1 collinearity pre-flight on frozen backbone features.
2. BN must not be applied to the θ output dimension.
3. Severity-stratified sampler.
4. α sweep to calibrate the Randers correction magnitude.

---

## 11. Independent Review Summary (cv-research-scientist, 2026-05-02)

A second critical pass converges on all major findings above. Key additions:

**On the mAP mechanism:** "The Randers bias does not change the relative ordering within a corruption level. It only changes the relative ordering across corruption levels. A global bias that shifts all gallery items at the same corruption level by the same amount is rank-preserving within each severity stratum." This is the definitive answer to the open question on page 3 of the notes.

**On the backbone signal:** "BAU's alignment loss, if successful, suppresses corruption information in the backbone features. A linear projection from those features has no corruption signal to extract, and L_mono receives near-zero gradient inputs." Mitigated in the confirmed design because BAU was not trained on the toy dataset's specific corruption operators — residual signal plausibly exists.

**On the correct scope:** "The strongest defensible claim is that the toy experiment validates that the Randers formalism can encode a quality-aware directional bias, but it does not by itself improve identity retrieval because the bias is identity-agnostic."
