# Domain-Conditioned Drift Head — Hyperparameter Sweep Design

**Date:** 2026-03-19
**Branch:** domainFinsler
**Baseline config:** `finsler_single_learnableOmega.sbatch` (instance-conditioned drift, RandomIdentity sampler)

---

## 1. Motivation

The domain-conditioned drift pivot replaces per-instance drift prediction with a structured domain/view prior plus a small residual correction. Four new hyperparameters control this mechanism:

| Parameter | CLI flag | Default | Role |
|-----------|----------|---------|------|
| Domain embedding dimension | `--domain-embed-dim` | 64 | Latent width of learnable domain prototypes |
| Domain temperature | `--domain-temperature` | 1.0 | Softmax temperature for soft domain-token inference at evaluation |
| Domain residual scale | `--domain-residual-scale` | 0.1 | Weight on per-instance residual correction relative to domain prior |
| Domain token loss weight | `--domain-token-loss-weight` | 0.1 | Auxiliary CE loss weight training the domain classifier |

The sweep must determine the operational range for each parameter while yielding clear decision signals about the model's behavior.

---

## 2. Why `domain_embed_dim` Is Not Swept

With $S = 3$ source domains, the domain embedding matrix $E \in \mathbb{R}^{3 \times d}$ is projected through $W \in \mathbb{R}^{d \times 2048}$ to produce the domain drift basis $M = EW \in \mathbb{R}^{3 \times 2048}$. The rank of $M$ is bounded by $\min(3, d)$. For any $d \geq 3$, the domain-prior subspace is at most rank 3.

The sigmoid gate $\sigma(\text{Linear}(x)) \in \mathbb{R}^{2048}$ applies element-wise modulation per instance, but this does not increase the rank of the domain prior itself — it only rescales the fixed basis vectors per-sample. Sweeping $d \in \{32, 64, 128\}$ therefore changes the number of dead parameters and the conditioning of the projection matrix, but not the representational capacity of the domain prior.

**Decision:** Fix `domain_embed_dim = 64` (default). This provides sufficient overparameterization for stable gradient flow while keeping parameter count negligible relative to the backbone (~25M params).

---

## 3. Sweep Architecture: Two-Stage OFAT Design

A full grid over 3 hyperparameters at 3-4 levels each would require 36-64 runs. We instead use a staged one-factor-at-a-time (OFAT) design where each stage isolates a specific question and all other factors are held constant.

### 3.1 Stage 1 — Routing Quality (12 runs)

**SLURM array:** `0-11`
**Script:** `sbatch/sweep_S1_routing.sbatch`

**Question:** How well can the model learn to route domain tokens, and how smooth should the target-domain mixture be at evaluation?

**Design:**

| Array ID | `domain_token_loss_weight` | `domain_temperature` | `domain_residual_scale` |
|----------|---------------------------|---------------------|------------------------|
| 0 | 0.01 | 1.0 | 0.0 |
| 1 | 0.01 | 2.0 | 0.0 |
| 2 | 0.01 | 5.0 | 0.0 |
| 3 | 0.05 | 1.0 | 0.0 |
| 4 | 0.05 | 2.0 | 0.0 |
| 5 | 0.05 | 5.0 | 0.0 |
| 6 | 0.10 | 1.0 | 0.0 |
| 7 | 0.10 | 2.0 | 0.0 |
| 8 | 0.10 | 5.0 | 0.0 |
| 9 | 0.25 | 1.0 | 0.0 |
| 10 | 0.25 | 2.0 | 0.0 |
| 11 | 0.25 | 5.0 | 0.0 |

**Fixed parameters:**
- `domain_residual_scale = 0.0` — eliminates instance-specific drift, isolating the domain routing mechanism
- `domain_embed_dim = 64`
- `drift_conditioning = domain`
- `sampler = RandomMultipleGallery` — enforces cross-camera positive pairs (per README decision)
- `bidirectional_triplet = true`
- `use_omega_reg = true`, `omega_reg_weight = 1.5`
- `eval_drift = false` — evaluate on identity-only features (consistent with current best practice)
- All other hyperparameters match `finsler_single_learnableOmega.sbatch`

**Rationale for residual_scale = 0:** Setting the residual to zero forces the drift to be a pure function of the domain routing. This isolates two questions simultaneously:
1. Does the auxiliary domain-token CE loss produce a useful classifier at different weight levels?
2. At evaluation time, does a peaky or smooth soft domain mixture produce better target-domain retrieval?

Without instance-level variation, any observed mAP change is attributable solely to the domain routing quality.

### 3.2 Stage 2 — Instance Relaxation (4 runs)

**SLURM array:** `0-3`
**Script:** `sbatch/sweep_S2_residual.sbatch`

**Question:** Does adding per-instance drift correction improve target mAP over the pure domain prior, or does BAU's optimization landscape suppress it?

**Design:**

| Array ID | `domain_residual_scale` |
|----------|------------------------|
| 0 | 0.00 |
| 1 | 0.05 |
| 2 | 0.10 |
| 3 | 0.20 |

**Fixed parameters:** Optimal `(domain_token_loss_weight, domain_temperature)` from Stage 1 (must be filled in after Stage 1 completes). All other parameters identical to Stage 1.

**Rationale for including 0.0:** Serves as a replication check against the best Stage 1 configuration. If the 0.0 result does not match, there is a reproducibility issue.

---

## 4. Mathematical Analysis and Expected Behavior

### 4.1 Domain Token Loss Weight ($\lambda_{\text{dom}}$)

The auxiliary domain-token CE loss is:

$$
\mathcal{L}_{dt} = \lambda_{\text{dom}} \cdot \text{CE}\bigl(g(\text{sg}[\mathbf{f}]),\, d\bigr)
$$

where $g: \mathbb{R}^{2048} \to \mathbb{R}^{3}$ is a linear classifier, $\text{sg}[\cdot]$ denotes stop-gradient, and $d$ is the source-domain label.

**Low $\lambda_{\text{dom}}$ (0.01):** The domain classifier receives weak supervision. After 60 epochs, the classifier may not converge beyond chance accuracy (~33% for 3 domains). At evaluation, the soft domain probabilities are near-uniform, reducing the domain-conditioned drift to a fixed weighted average of the three domain embeddings — effectively a learnable global bias vector. The drift head degenerates to a single-mode prior regardless of the input.

**Moderate $\lambda_{\text{dom}}$ (0.05–0.10):** The classifier learns to discriminate source domains from identity features. The stop-gradient prevents identity backbone corruption, but the classifier must be accurate enough to produce meaningful soft routing on target data.

**High $\lambda_{\text{dom}}$ (0.25):** While the stop-gradient protects the backbone from direct domain-classification gradients, the domain classifier may overfit to source-domain-specific identity feature patterns. At evaluation, it may produce overconfident predictions on target data, routing CUHK03 images to a single source domain rather than forming a useful mixture. This is a form of the domain-discrimination vs domain-invariance tension documented by [Ganin et al., JMLR 2016].

**Observable diagnostic:** Track `loss_domain_token` convergence rate. If the loss plateaus above $\ln(3) \approx 1.10$ (maximum entropy for 3 classes), the classifier is not learning.

### 4.2 Domain Temperature ($\tau$)

At evaluation, domain probabilities are computed as:

$$
p_k = \frac{\exp(z_k / \tau)}{\sum_{j=1}^{3} \exp(z_j / \tau)}, \quad k = 1, 2, 3
$$

where $z_k$ are the domain classifier logits.

**Low $\tau$ (1.0):** Produces peaked distributions. If the domain classifier is well-trained, this assigns each target image primarily to one source domain. For CUHK03 (indoor surveillance), this risks assigning all images to whichever source domain has the most similar camera geometry (likely Market-1501, also indoor surveillance). The drift becomes nearly domain-specific rather than a mixture.

**High $\tau$ (5.0):** Flattens the distribution toward uniform. The drift becomes approximately $\frac{1}{3}\sum_{k=1}^{3} \omega_k$, a fixed average regardless of input. This is safer for OOD targets but sacrifices the ability to produce input-adaptive drift.

**Interaction with $\lambda_{\text{dom}}$:** Temperature only matters if the classifier produces non-trivial logits. With very low $\lambda_{\text{dom}}$, the logits are near-random, and temperature has no effect — the distribution is already uniform. Temperature becomes informative only when the classifier is moderately or well-trained.

**Observable diagnostic:** Compare the entropy of $p(z|x)$ on target data across temperature settings. If entropy is already near $\ln(3)$ at $\tau = 1.0$, the classifier is too weak and temperature is irrelevant.

### 4.3 Domain Residual Scale ($\lambda_{\text{res}}$)

The final drift is:

$$
\omega(x) = \text{gate}(x) \cdot W_{\text{proj}}(\text{domain\_context}) + \lambda_{\text{res}} \cdot \text{MLP}(x)
$$

**$\lambda_{\text{res}} = 0$:** The drift is purely domain-conditioned. All instances from the same source domain (during training) or with the same soft assignment (during evaluation) receive the same gated drift vector. Within-domain drift variance comes only from the sigmoid gate, which modulates but does not generate new directions.

**$\lambda_{\text{res}} = 0.05$–$0.10$:** The instance MLP contributes a small perturbation. This allows the model to capture within-domain variation (e.g., pose differences within a camera view) without overwhelming the domain structure.

**$\lambda_{\text{res}} = 0.20$:** The residual term's magnitude approaches that of the domain prior. Prior experiments (documented in the README and analysis) showed that unconstrained instance-level drift is suppressed by BAU's alignment and uniformity objectives. Even though uniform loss now operates on identity-only features, the triplet loss (Finsler on full embeddings) and domain loss still influence drift. At this scale, the residual may either provide useful variation or reintroduce the instance-drift collapse mode.

**Observable diagnostic:** Track the ratio $\mathbb{E}[\|\lambda_{\text{res}} \cdot \text{MLP}(x)\|] / \mathbb{E}[\|\text{gate}(x) \cdot W_{\text{proj}}(\text{domain\_context})\|]$. This can be approximated from `omega_norm_mean` across residual scale settings: if omega_norm_mean does not increase monotonically with $\lambda_{\text{res}}$, the optimizer is suppressing the residual.

---

## 5. Interaction Analysis

### 5.1 Which parameters interact?

**$\lambda_{\text{dom}}$ and $\tau$ interact strongly.** Temperature modulates the sharpness of the classifier's output, so its effect depends entirely on the classifier's quality (controlled by $\lambda_{\text{dom}}$). This is why they are swept jointly in Stage 1.

**$\lambda_{\text{res}}$ is approximately independent of $(\lambda_{\text{dom}}, \tau)$.** The residual MLP operates on raw pooled features and does not depend on domain routing. Its effect on mAP is additive to the domain prior's contribution. This justifies sweeping it separately in Stage 2.

### 5.2 Why not sweep $\lambda_{\text{res}}$ in Stage 1?

Adding $\lambda_{\text{res}}$ to Stage 1 would triple the number of runs (36 total) without adding proportional information. The residual correction is a second-order effect: it refines the domain prior but does not change the routing mechanism. If the routing is broken (bad $\lambda_{\text{dom}}$ or $\tau$), no amount of residual correction will fix it. Conversely, if the routing is good, the residual's marginal contribution can be cleanly measured in Stage 2.

---

## 6. Diagnostic Metrics

### Already tracked (via W&B):
- `omega_norm_mean`, `omega_norm_max` — drift magnitude evolution
- `train/loss_domain_token` — domain classifier training quality
- `eval/mAP`, `final/mAP`, `final/CMC_top_1`, `final/CMC_top_5`, `final/CMC_top_10` — retrieval performance

### Recommended additions (optional, for deeper analysis):
1. **Domain classifier accuracy at end of training** — derivable from final `loss_domain_token` but explicit logging is cleaner
2. **Domain assignment entropy on target** — $H(p_{\text{domain}} | X_{\text{target}})$ at evaluation; collapse to 0 indicates routing failure
3. **Pairwise cosine similarity between projected domain base vectors** in $\mathbb{R}^{2048}$ — if $\cos(\omega_i, \omega_j) \approx 1.0$, the three domain priors have collapsed to a single direction and domain conditioning is vacuous

---

## 7. Hardware Requirements

From Grafana monitoring of prior runs with identical architecture and batch size:
- **Peak GPU VRAM:** ~20 GB (batch_size=256, resnet50_finsler, mixed precision)
- **System RAM:** ~12–16 GB actual usage (3 source datasets + memory bank)
- **GPU power envelope:** 300–340 W (A40/A100 class)

**SLURM resource requests:**
- `VRAM:24G` — 4 GB headroom over observed peak
- `mem=28G` — sufficient for data pipeline + model + memory bank
- `cpus-per-task=5` — 4 data workers + main process
- `time=08:00:00` — 60 epochs * 500 iters completes in ~5–6 hours; 8h provides margin

---

## 8. Interpretation Guide

After Stage 1 completes, select the optimal $(\lambda_{\text{dom}}, \tau)$ by:

1. **Primary criterion:** highest `final/mAP` on CUHK03
2. **Tiebreaker:** prefer moderate $\lambda_{\text{dom}}$ (avoid extremes) and higher $\tau$ (safer for OOD generalization)
3. **Sanity check:** verify that `loss_domain_token` converged (final value well below $\ln(3)$) and that `omega_norm_mean` is non-trivial (drift is not collapsed)

After Stage 2 completes:

1. If `residual_scale = 0.0` matches the best Stage 1 result, reproducibility is confirmed
2. If any `residual_scale > 0` improves over 0.0, instance-level variation adds value on top of the domain prior
3. If all `residual_scale > 0` underperform 0.0, the BAU training landscape still suppresses instance drift even with the corrected loss routing — the domain prior is sufficient
4. If `omega_norm_mean` does not increase with $\lambda_{\text{res}}$, the optimizer is actively suppressing the residual contribution

---

## 9. Comparison Baselines

All sweep runs should be compared against:

1. **Euclidean BAU baseline** — `resnet50` with same source/target and `RandomMultipleGallery` sampler (mAP ~43.3% from README A/B test)
2. **Instance-conditioned Finsler** — `resnet50_finsler` with `drift_conditioning=instance` (from existing `finsler_single_learnableOmega.sbatch`)
3. **Domain pivot validation** — `validate_domain_pivot.sbatch` (domain-conditioned, default hyperparameters)

The sweep runs use the corrected loss routing (uniform on identity-only, k-NN weights on identity-only) documented in the 2026-03-18 changelog entry.
