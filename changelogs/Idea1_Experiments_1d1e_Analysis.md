# Camera-Conditioned Auxiliary Losses on the Drift Subspace: Experiments 1d and 1e

This document formalizes two auxiliary loss functions operating on the drift slice $\mathbf{z}^{\omega}$ of the structured embedding $\mathbf{z} = [\mathbf{z}^{\mathrm{id}};\mathbf{z}^{\omega}]$. The losses condition on **camera labels** rather than identity labels, testing the hypothesis that drift vectors should encode camera-specific domain artifacts. We first analyze the orthogonal decoupling mechanism that justifies operating on the drift slice independently, then motivate the proposed losses from the empirical record of experiments 1a–1c, and finally formalize the mathematical expressions and expected interactions with the existing training stack.

---

## 1. Prerequisite: Orthogonal Decoupling of Identity and Drift

### 1.1 The Gram–Schmidt Projection

The backbone produces a pooled descriptor $\mathbf{h}$ feeding both the identity and drift heads. The identity path applies batch normalization and $L_2$-normalization to yield $\hat{\mathbf{z}}^{\mathrm{id}} = \mathbf{z}^{\mathrm{id}} / \|\mathbf{z}^{\mathrm{id}}\|_2 \in \mathbb{R}^{d_{\mathrm{id}}}$. The drift head $g_\phi$ maps $\mathbf{h}$ to a raw drift vector $\tilde{\mathbf{z}}^{\omega} = g_\phi(\mathbf{h}) \in \mathbb{R}^{d_\omega}$, which is then orthogonalized via a single Gram–Schmidt step:

$$
\mathbf{z}^{\omega} = \tilde{\mathbf{z}}^{\omega} - \bigl(\tilde{\mathbf{z}}^{\omega} \cdot \hat{\mathbf{z}}^{\mathrm{id}}\bigr)\,\hat{\mathbf{z}}^{\mathrm{id}} \;.
$$

This removes the component of the drift parallel to the normalized identity direction, enforcing $\langle \hat{\mathbf{z}}^{\omega}, \hat{\mathbf{z}}^{\mathrm{id}} \rangle = 0$ per sample, per forward pass. Throughout this document, $\omega_i \equiv \hat{\mathbf{z}}_i^{\omega}$ denotes the post-projection, post-gating drift vector (matching the $\hat{\mathbf{z}}^{\omega}$ notation in the methodology). A subsequent sigmoid-gated norm scaling ensures $\|\hat{\mathbf{z}}^{\omega}\|_2 < c$ (the Randers positivity bound), confining drift to the interior of a ball in the orthogonal complement of $\hat{\mathbf{z}}^{\mathrm{id}}$.

### 1.2 What Orthogonality Guarantees

The projection provides two concrete guarantees:

1. **Geometric orthogonality.** The drift vector lies in the hyperplane $\{\mathbf{v} \in \mathbb{R}^{d_{\mathrm{id}}} : \langle \mathbf{v}, \hat{\mathbf{z}}^{\mathrm{id}} \rangle = 0\}$. This prevents the drift from degenerating into a scaled copy of the identity direction, which would collapse the asymmetry term in the Finsler distance $d_F$ (eq. \ref{eq:finsler_distance} in the methodology) to a scalar multiple of the Euclidean base distance, eliminating directional expressiveness.

2. **Subspace confinement.** The drift cannot amplify or attenuate the identity signal along $\hat{\mathbf{z}}^{\mathrm{id}}$. Any loss operating on $\hat{\mathbf{z}}^{\omega}$ alone cannot directly push the identity direction; it can only modify the orthogonal complement. This is the structural basis for treating the drift slice as a separate substrate for auxiliary supervision. Note, however, that this constrains the **representation**; the **optimization** still couples identity and drift through $\partial \hat{\mathbf{z}}^{\omega}/\partial \hat{\mathbf{z}}^{\mathrm{id}}$ and the shared backbone (formalized in Sec. 1.5).

### 1.3 What Orthogonality Does Not Guarantee

The projection is a **necessary but not sufficient** condition for full functional decoupling. Three mechanisms can violate independence despite geometric orthogonality:

**Shared backbone features.** Both $\hat{\mathbf{z}}^{\mathrm{id}}$ and $\tilde{\mathbf{z}}^{\omega}$ are functions of the same backbone output $\mathbf{h}$. The Gram–Schmidt step removes the projection onto a single direction $\hat{\mathbf{z}}^{\mathrm{id}} \in \mathbb{R}^{d_{\mathrm{id}}}$, but the identity-relevant subspace is not one-dimensional—it is a complex, nonlinear manifold in feature space. The drift head can encode identity-correlated information in directions orthogonal to $\hat{\mathbf{z}}^{\mathrm{id}}$ but still correlated with identity-discriminative structure. In the language of domain separation networks (Bousmalis et al., NIPS 2016), the projection enforces a **shared-private** decomposition in the output, but the shared backbone permits information leakage through the input.

**Gradient coupling through the backbone.** During backpropagation, gradients from drift-operating losses ($\mathcal{L}_{1d}$, $\mathcal{L}_{1e}$, $\mathcal{L}_\omega$) flow through $g_\phi$ into $\mathbf{h}$, and from $\mathbf{h}$ into the convolutional backbone. These gradients can modify backbone features in ways that affect the identity path. The stop-gradient operator $\mathrm{sg}[\cdot]$ on the domain classifier input prevents domain-classification gradients from corrupting identity features, but no analogous stop-gradient exists between the drift head and the backbone. Optimizer parameter grouping can attenuate but does not eliminate this coupling.

**Finite-capacity leakage.** At finite model capacity, the drift head may learn to encode identity-correlated information in the orthogonal complement of $\hat{\mathbf{z}}^{\mathrm{id}}$. Consider two samples $(\mathbf{x}_i, \mathbf{x}_j)$ with the same identity but different cameras. Their identity directions $\hat{\mathbf{z}}^{\mathrm{id}}_i$ and $\hat{\mathbf{z}}^{\mathrm{id}}_j$ are close but not identical (due to intra-class variation). The orthogonal complements of these two directions are different $(d_{\mathrm{id}}-1)$-dimensional hyperplanes. Information that is orthogonal to $\hat{\mathbf{z}}^{\mathrm{id}}_i$ need not be orthogonal to $\hat{\mathbf{z}}^{\mathrm{id}}_j$, creating a channel through which identity-correlated structure can persist in the drift slice across samples.

### 1.4 Why the Projection Is Sufficient for Camera-Conditioned Losses

Despite the limitations above, the orthogonal projection provides adequate structural separation for the proposed experiments 1d and 1e. The argument rests on three observations:

1. **Camera labels are weakly correlated with identity in multi-source DG-ReID.** In standard multi-source protocols, each source dataset contains many identities distributed across multiple cameras, yielding a high identity-to-camera ratio. In expectation under many identities per camera, camera-conditioned losses average over diverse identities within each camera group, diluting any identity-specific signal that leaks into the drift slice.

2. **Drift-only losses do not optimize identity discrimination.** Even if the drift slice contains residual identity information, losses 1d and 1e condition on camera labels, not identity labels. They cannot directly reinforce identity-discriminative structure in the drift. The risk is indirect: if a camera label is a strong proxy for identity (e.g., a person appears in exactly one camera), camera-conditioned clustering could inadvertently cluster by identity. In multi-source DG-ReID, this proxy relationship is weak by construction.

3. **Empirical precedent from experiment 1c.** Experiment 1c applied $L_2$ regularization directly to the drift slice (same-PID, different-camera pairs) and behaved favorably in the ablation sweep—unlike experiment 1b, which operated on the full embedding and produced large regressions consistent with objective interference. This supports the claim that direct supervision on the drift slice, mediated by the orthogonal projection, does not destructively interfere with identity learning under the current training recipe.

### 1.5 Formal Characterization of the Decoupling

Let $\Pi_{\hat{\mathbf{z}}^{\mathrm{id}}}^{\perp} = I - \hat{\mathbf{z}}^{\mathrm{id}}(\hat{\mathbf{z}}^{\mathrm{id}})^\top$ denote the orthogonal projector onto the complement of $\hat{\mathbf{z}}^{\mathrm{id}}$. The drift output is $\hat{\mathbf{z}}^{\omega} = \sigma\bigl(\|\Pi_{\hat{\mathbf{z}}^{\mathrm{id}}}^{\perp} \tilde{\mathbf{z}}^{\omega}\|_2\bigr) \cdot c \cdot \widehat{\Pi_{\hat{\mathbf{z}}^{\mathrm{id}}}^{\perp} \tilde{\mathbf{z}}^{\omega}}$, where $\sigma$ is the sigmoid gate and $\hat{(\cdot)}$ denotes unit normalization. The Jacobian $\partial \hat{\mathbf{z}}^{\omega} / \partial \hat{\mathbf{z}}^{\mathrm{id}}$ is nonzero (the projector depends on $\hat{\mathbf{z}}^{\mathrm{id}}$), confirming that identity and drift are **not** gradient-decoupled. This Jacobian acts only through the projection removal term $-(\tilde{\mathbf{z}}^{\omega} \cdot \hat{\mathbf{z}}^{\mathrm{id}})\hat{\mathbf{z}}^{\mathrm{id}}$, which is small when the drift head has learned to produce outputs approximately orthogonal to $\hat{\mathbf{z}}^{\mathrm{id}}$ (i.e., when $|\tilde{\mathbf{z}}^{\omega} \cdot \hat{\mathbf{z}}^{\mathrm{id}}| \approx 0$). In practice, the near-zero initialization of the drift head's final layer and the sigmoid gating encourage this regime from early training.

---

## 2. Motivation from Experiments 1a–1c

The Idea-1 ablation series investigated auxiliary triplet-style objectives on the drift subspace under fixed multi-source DG-ReID protocols. The key findings, stated qualitatively:

**Experiment 1a (Euclidean identity triplet).** A batch-hard triplet loss on the identity slice using Euclidean distance, with BAU alignment, uniformity, and domain repulsion active on the identity features. This serves as the controlled baseline. Under the tested protocols, 1a matched the unified Finsler formulation in retrieval metrics, establishing that the split-head architecture with orthogonal drift does not degrade identity discrimination.

**Experiment 1b (Finsler domain triplet on full embedding).** A batch-hard triplet mining camera labels as positives/negatives, operating on the full embedding $[\mathbf{z}^{\mathrm{id}};\mathbf{z}^{\omega}]$ via the asymmetric Finsler distance $d_F$. This objective pulls same-camera features together and pushes different-camera features apart in the full embedding space. The result is a large regression in generalization metrics, consistent with the identity representation fragmenting into camera-specific clusters that retain domain-specific shortcuts at the expense of cross-domain retrieval. When BAU domain repulsion ($\mathcal{L}_{\mathrm{dom}}$) is simultaneously active, the domain triplet and domain repulsion produce incompatible gradient pressures—$\mathcal{L}_{\mathrm{dom}}$ enforces uniformity across domains while $\mathcal{L}_{\mathrm{tri}}^{\mathrm{dom}}$ enforces camera clustering.

**Experiment 1b-refined (cross-camera Finsler contrastive).** A mean-squared Finsler distance penalty for same-identity, different-camera pairs on the full embedding. This is a softer version of 1b that avoids explicit camera clustering but still operates on the full embedding. Performance trails the baseline, suggesting that even soft cross-camera constraints on the full embedding interfere with the BAU alignment/uniformity balance.

**Experiment 1c (drift-only $L_2$ alignment).** A mean-squared $L_2$ penalty on drift vectors for same-identity, different-camera pairs:

$$
\mathcal{L}_{1c} = \frac{1}{|\mathcal{C}_{1c}|} \sum_{(i,j) \in \mathcal{C}_{1c}} \|\omega_i - \omega_j\|_2^2, \quad \mathcal{C}_{1c} = \{(i,j) : y_i = y_j,\; c_i \neq c_j\}
$$

Under domain-conditioned drift with Finsler domain repulsion, 1c was the only auxiliary in the sweep that did not degrade retrieval metrics relative to the baseline. The loss encourages drift vectors of the same identity from different cameras to align—the **opposite** of the original intuition that drift should encode camera-specific artifacts that differ across cameras for the same person.

---

## 3. Reinterpreting the 1c Result

The success of 1c inverts the naïve hypothesis. If drift vectors modeled camera-specific artifacts, one would expect same-identity, different-camera drift vectors to **differ** (each encoding its respective camera's characteristics). Instead, the best-performing configuration penalizes this difference, encouraging drift coherence across cameras for the same identity.

Two interpretations are consistent with this observation:

**Interpretation A: Drift as identity-conditioned directional bias.** Under this reading, the drift vector does not encode camera artifacts per se, but rather an identity-specific directional preference in the asymmetric retrieval score. Aligning drift across cameras for the same identity ensures that the asymmetric component of $d_F$ is consistent regardless of which camera captured the probe or gallery image. This makes the Finsler distance more stable for cross-camera retrieval of the same person.

**Interpretation B: Regularization effect.** The $L_2$ penalty on same-PID cross-camera drift pairs acts as a soft regularizer that prevents the drift head from overfitting to camera-specific noise. Without this constraint, the drift head may learn spurious camera-specific patterns that do not generalize to unseen target domains. The penalty keeps drift vectors in a low-variance regime where they provide a mild, consistent asymmetric signal rather than noisy camera-specific perturbations.

However, the constraint set of 1c ($y_i = y_j, c_i \neq c_j$) conflates two signals: **identity coherence** (same person) and **cross-camera alignment** (different cameras). The improvement could be driven by either or both. Experiments 1d and 1e are designed to disentangle these factors by testing the complementary hypothesis: what if drift vectors should cluster by **camera** (regardless of identity), encoding shared camera-specific artifacts?

---

## 4. Formalization of Experiments 1d and 1e

### Notation

Throughout, we use the notation established in the methodology:
- $\omega_i \equiv \mathbf{z}_i^{\omega} \in \mathbb{R}^{d_\omega}$: drift vector for sample $i$ (after orthogonal projection and sigmoid gating)
- $\hat{\omega}_i = \omega_i / \|\omega_i\|_2$: $L_2$-normalized drift (used when discussing hyperspherical interpretations; **experiment 1e uses raw $\omega_i$ in code**, see §4.2)
- $c_i \in \{1, \ldots, C\}$: global camera label for sample $i$
- $y_i$: person identity label for sample $i$
- $\mathcal{B}$: mini-batch of size $B$

### 4.1 Experiment 1d: Same-Camera Drift Attraction

**Constraint set.**

$$
\mathcal{C}_{1d} = \{(i,j) : c_i = c_j,\; i \neq j\}
$$

This selects all pairs from the same camera, regardless of identity. Compared to 1c ($\mathcal{C}_{1c} = \{(i,j) : y_i = y_j, c_i \neq c_j\}$), the constraint is relaxed in identity (any PID) and inverted in camera (same camera, not different camera).

**Loss function.**

$$
\mathcal{L}_{1d} = \frac{1}{|\mathcal{C}_{1d}|} \sum_{(i,j) \in \mathcal{C}_{1d}} \|\omega_i - \omega_j\|_2^2
$$

**Intuition.** If the drift vector encodes camera-specific domain artifacts (lighting conditions, viewpoint distribution, resolution, background statistics), then samples from the same camera should produce similar drift vectors regardless of the pedestrian's identity. $\mathcal{L}_{1d}$ provides direct supervision toward this inductive bias. The loss operates exclusively on the drift slice, avoiding the catastrophic identity-manifold shattering observed with 1b (which operated on the full embedding).

**Relationship to 1c.** The constraint sets of 1c and 1d are disjoint when restricted to same-identity pairs: 1c selects same-PID different-camera, while 1d selects same-camera (any PID). Together with 1c, they would cover complementary regions of the pair space. However, 1d also includes different-PID same-camera pairs, which 1c does not address.

**Gradient structure.** The gradient of $\mathcal{L}_{1d}$ with respect to $\omega_i$ is:

$$
\frac{\partial \mathcal{L}_{1d}}{\partial \omega_i} = \frac{2}{|\mathcal{C}_{1d}|} \sum_{j : c_j = c_i,\, j \neq i} (\omega_i - \omega_j)
$$

This pulls $\omega_i$ toward the mean drift vector of its camera group in the mini-batch. The gradient is zero when all same-camera drift vectors are identical, which is the global minimum of $\mathcal{L}_{1d}$ (restricted to same-camera groups).

### 4.2 Experiment 1e: Cross-Camera Drift Uniformity

**Constraint set.**

$$
\mathcal{C}_{1e} = \{(i,j) : c_i \neq c_j,\; i < j\}
$$

This selects all cross-camera pairs (upper triangle to avoid double-counting).

**Loss function.** We adopt the **log-mean-exp / Gaussian-potential form** used by Wang & Isola (ICML 2020) for uniformity, conditioned on cross-camera pairs. The implementation applies it to **raw** drift vectors $\omega_i$ in the Randers ball (same tensor as 1d), not to $\hat{\omega}_i$:

$$
\mathcal{L}_{1e} = \log \frac{1}{|\mathcal{C}_{1e}|} \sum_{(i,j) \in \mathcal{C}_{1e}} \exp\!\bigl(-t\,\|\omega_i - \omega_j\|_2^2\bigr)
$$

where $t > 0$ is a temperature parameter controlling the hardness of the repulsion. This matches `drift_cross_camera_uniformity_loss` in `bau/trainers.py` (upper-triangle mask, no `F.normalize` on drift).

**Intuition.** If drift vectors from different cameras should encode distinct domain artifacts, their **feature vectors** in drift space should be well-separated. Minimizing $\mathcal{L}_{1e}$ drives the average Gaussian potential between cross-camera drifts toward zero, which requires the pairwise squared distances $\|\omega_i - \omega_j\|_2^2$ to be large. Because the loss uses raw $\omega_i$, repulsion acts on **both direction and magnitude** within the feasible ball, consistent with 1d operating in the same coordinates. This is the complementary force to 1d: while 1d clusters same-camera drift, 1e disperses different-camera drift.

**Connection to alignment and uniformity.** Wang & Isola (2020) decompose contrastive learning into alignment (positive pairs close) and uniformity (spread on the hypersphere). In the identity space, BAU implements alignment via $\mathcal{L}_{\mathrm{align}}$ and uniformity via $\mathcal{L}_{\mathrm{uniform}}$ on $L_2$-normalized identity features. Experiments 1d and 1e reuse the **same algebraic template** (attraction + log-mean-exp repulsion) in the **drift space**, conditioned on **camera labels**, but 1e’s potential is evaluated on **unnormalized** drift so the Wang–Isola hyperspherical uniformity guarantee does not apply verbatim:

| | Identity space (BAU) | Drift space (1d + 1e) |
|---|---|---|
| **Alignment** | $\mathcal{L}_{\mathrm{align}}$: same-PID views close | $\mathcal{L}_{1d}$: same-camera drift close |
| **Uniformity** | $\mathcal{L}_{\mathrm{uniform}}$: normalized features spread | $\mathcal{L}_{1e}$: cross-camera raw drift spread (Gaussian kernel) |

This parallel structure is deliberate: if the drift subspace is to encode camera-specific structure, it should exhibit the same alignment-uniformity balance that makes the identity subspace discriminative—but with camera labels replacing identity labels.

**Gradient structure.** Let $S = \sum_{(k,l) \in \mathcal{C}_{1e}} \exp(-t\|\omega_k - \omega_l\|_2^2)$ with $\mathcal{C}_{1e}$ in **upper-triangle** form ($k < l$, $c_k \neq c_l$). Then

$$
\frac{\partial \mathcal{L}_{1e}}{\partial \omega_i} = \frac{-2t}{S} \left( \sum_{\substack{j > i \\ c_j \neq c_i}} e^{-t\|\omega_i - \omega_j\|_2^2}(\omega_i - \omega_j) + \sum_{\substack{k < i \\ c_k \neq c_i}} e^{-t\|\omega_k - \omega_i\|_2^2}(\omega_i - \omega_k) \right).
$$

The softmax weighting concentrates the repulsive gradient on the **closest** cross-camera pairs (smallest $\|\cdot\|_2^2$), analogous to the hyperspherical case. Pairs that are already well-separated contribute negligible gradient.

### 4.3 Justification of Wang–Isola Uniformity for 1e

We considered five candidate formulations for the cross-camera repulsion objective. The Wang–Isola uniformity loss was selected based on the following comparative analysis:

**Minimum Hyperspherical Energy (MHE) / Riesz $s$-energy** (Liu et al., NeurIPS 2018). Formulation: $\sum_{(i,j) \in \mathcal{C}_{1e}} \|\hat{\omega}_i - \hat{\omega}_j\|^{-s}$ with $s > 0$. The power-law singularity at zero distance provides harder repulsion for near-collisions than the Gaussian kernel. However, this singularity requires $\epsilon$-clamping for numerical stability and can produce large gradient spikes for very close pairs, potentially destabilizing training. The Wang–Isola Gaussian kernel provides smoother gradients and shares structural similarity with the Gaussian-kernel repulsion already used for $\mathcal{L}_{\mathrm{dom}}$ (Cho et al., 2024).

**Determinantal Point Process (DPP) log-determinant** (Elfeki et al., ICML 2019). Formulation: $-\log \det(K + \epsilon I)$ where $K_{ij} = \hat{\omega}_i^\top \hat{\omega}_j$. This captures higher-order diversity (not just pairwise repulsion) by penalizing collinearity among multiple drift vectors simultaneously. However, the $O(n^3)$ cost for the determinant and the numerical sensitivity of the gradient (requiring matrix inversion of a potentially near-singular Gram matrix) make it impractical as a mini-batch loss. The added expressiveness over pairwise repulsion is not justified for the expected number of distinct cameras ($C \sim 6\text{--}30$) in standard DG-ReID protocols.

**VICReg variance + covariance** (Bardes et al., ICLR 2022). The variance term prevents dimensional collapse; the covariance term decorrelates drift dimensions. These address a different failure mode (dimensional collapse of the drift representation) rather than the camera-separation objective. VICReg does not enforce pairwise separation between specific camera groups and is better suited as a complementary regularizer than a primary repulsion mechanism.

**Reversed Maximum Mean Discrepancy (MMD)**. Maximizing $\text{MMD}^2(\Omega_a, \Omega_b)$ between camera-specific drift distributions is conceptually correct but operationally problematic: the objective is unbounded, the Gaussian kernel bandwidth requires careful tuning, and the reversed-sign usage lacks empirical validation. The pairwise Gaussian potential in the Wang–Isola loss is a simplified, better-behaved version of the cross-term in MMD.

**Conclusion.** The Wang–Isola uniformity loss provides the best balance of theoretical grounding, computational efficiency ($O(B^2 d_\omega)$), numerical stability, and empirical validation within the existing pipeline.

---

## 5. Expected Interactions and Failure Modes

### 5.1 Interaction with BAU Domain Repulsion ($\mathcal{L}_{\mathrm{dom}}$)

$\mathcal{L}_{\mathrm{dom}}$ repels features from their domain centroid in the memory bank, operating on either the full embedding (Finsler mode) or the identity slice (Euclidean mode). In the multi-source protocol, domain labels correspond to source datasets, each containing multiple cameras.

**1d vs. $\mathcal{L}_{\mathrm{dom}}$:** $\mathcal{L}_{1d}$ attracts same-camera drift vectors; $\mathcal{L}_{\mathrm{dom}}$ repels same-dataset features. Since cameras are nested within datasets ($\text{camera} \subset \text{dataset}$), these objectives operate at different label granularities and exert different geometric pulls on shared coordinates. $\mathcal{L}_{1d}$ provides finer-grained camera-level structure within each dataset cluster on the drift slice, while $\mathcal{L}_{\mathrm{dom}}$ enforces inter-dataset separation on the full structured embedding (in Finsler mode) or the identity slice (in Euclidean mode).

**1e vs. $\mathcal{L}_{\mathrm{dom}}$:** Both are repulsive, but on different feature slices and with different label granularity. $\mathcal{L}_{1e}$ repels cross-camera drift vectors (raw $\mathbf{z}^{\omega}$); $\mathcal{L}_{\mathrm{dom}}$ repels same-dataset full features from centroids. These are complementary: $\mathcal{L}_{\mathrm{dom}}$ ensures inter-dataset separation in the identity space, while $\mathcal{L}_{1e}$ ensures inter-camera separation in the drift space.

### 5.2 Interaction with Omega Regularization ($\mathcal{L}_\omega$)

The log-barrier penalty $\mathcal{L}_\omega = -\frac{1}{B}\sum_i \log(c - \|\omega_i\|_2 + \epsilon)$ penalizes drift norms approaching the Randers bound $c$.

**1d + $\mathcal{L}_\omega$:** $\mathcal{L}_{1d}$ pulls same-camera drift vectors together, which can reduce drift norms (if the camera-group mean has smaller norm than individual vectors) or increase them (if the mean is larger). In either case, $\mathcal{L}_\omega$ provides a soft ceiling. The two losses are compatible: $\mathcal{L}_{1d}$ concentrates drift in a lower-dimensional subspace (camera-specific directions) within the feasible region defined by $\mathcal{L}_\omega$.

**1e + $\mathcal{L}_\omega$:** $\mathcal{L}_{1e}$ uses raw drift vectors, so repulsion couples to **both** direction and magnitude. $\mathcal{L}_\omega$ enforces the Randers ball ceiling on $\|\omega_i\|_2$. The two terms are not “direction-only vs. norm-only”; $\mathcal{L}_\omega$ shapes the feasible norm range within which $\mathcal{L}_{1e}$ spreads cross-camera drifts.

### 5.3 Interaction with Identity Alignment ($\mathcal{L}_{\mathrm{align}}$)

$\mathcal{L}_{\mathrm{align}}$ operates on the identity slice only (by design, to prevent drift collapse from augmentation-invariance pressure). Since 1d and 1e operate on the drift slice only, there is no direct objective conflict. The indirect coupling is through the shared backbone (Sec. 1.3), but the orthogonal projection limits the magnitude of this interaction.

### 5.4 Potential Failure Modes

**Mode 1: Drift collapse under 1d.** If the same-camera attraction is too strong, all drift vectors within a camera group collapse to a single point. This eliminates per-instance variation in the drift, reducing the asymmetric distance to a camera-level constant offset. The Randers distance becomes $d_F(\mathbf{x}, \mathbf{y}) \approx \|\mathbf{x}^{\mathrm{id}} - \mathbf{y}^{\mathrm{id}}\|_2 + \langle \bar{\omega}_c, \mathbf{y}^{\mathrm{id}} - \mathbf{x}^{\mathrm{id}} \rangle$ where $\bar{\omega}_c$ is the camera-level drift. This may still be useful (camera-level asymmetry) but sacrifices instance-level expressiveness. Mitigation: keep the auxiliary weight small relative to the primary objectives.

**Mode 2: Conflict between 1d and 1c.** If both 1c and 1d are active, 1c pulls same-PID different-camera drift together while 1d pulls same-camera drift together. For a pair $(i, j)$ with $y_i = y_j$ and $c_i \neq c_j$, 1c attracts their drift vectors while 1d does not (they are in different cameras). For a pair $(i, k)$ with $y_i \neq y_k$ and $c_i = c_k$, 1d attracts their drift vectors while 1c is silent. These objectives are not contradictory—they operate on disjoint pair sets—but their combined effect may over-constrain the drift space. The sweep includes a 1a+1c+1d+1e arm to test this empirically.

**Mode 3: 1e gradient vanishing.** If cross-camera drift vectors are already well-separated (large $\|\omega_i - \omega_j\|_2^2$), the Gaussian kernel $\exp(-t \cdot \text{large})$ saturates near zero, and the log-mean-exp gradient vanishes. By construction, gradients concentrate on insufficiently separated pairs and become passive once separation is achieved. However, if the initial drift vectors are already well-separated (e.g., due to domain-conditioned initialization), $\mathcal{L}_{1e}$ may provide negligible training signal throughout.

---

## 6. Summary of Constraint Sets

| Experiment | Constraint set $\mathcal{C}$ | Feature | Distance |
|---|---|---|---|
| 1a | $y_i = y_j$ (positive), $y_i \neq y_j$ (negative) | $\mathbf{z}^{\mathrm{id}}$ | $d_E$ |
| 1b | $c_i = c_j$ (positive), $c_i \neq c_j$ (negative) | $[\mathbf{z}^{\mathrm{id}};\mathbf{z}^{\omega}]$ | $d_F$ |
| 1b-refined | $y_i = y_j,\; c_i \neq c_j$ | $[\mathbf{z}^{\mathrm{id}};\mathbf{z}^{\omega}]$ | $d_F^2$ |
| 1c | $y_i = y_j,\; c_i \neq c_j$ | $\mathbf{z}^{\omega}$ | $\|\cdot\|_2^2$ |
| **1d** | $c_i = c_j$ | $\mathbf{z}^{\omega}$ | $\|\cdot\|_2^2$ |
| **1e** | $c_i \neq c_j$, $i<j$ | $\mathbf{z}^{\omega}$ (raw drift in ball) | Wang–Isola-style log-mean-exp |

---

## References

- Wang, T. & Isola, P. (2020). Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. *ICML 2020*.
- Bousmalis, K., Trigeorgis, G., Silberman, N., Krishnan, D., & Erhan, D. (2016). Domain Separation Networks. *NeurIPS 2016*.
- Liu, W., Lin, R., Liu, Z., Liu, L., Yu, Z., Dai, B., & Song, L. (2018). Learning towards Minimum Hyperspherical Energy. *NeurIPS 2018*.
- Elfeki, M., Couprie, C., Riviere, M., & Elhoseiny, M. (2019). GDPP: Learning Diverse Generations using Determinantal Point Processes. *ICML 2019*.
- Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. *ICLR 2022*.
- Cho, Y., Kim, W. J., Hong, S., & Yoon, S.-E. (2024). Generalizable Person Re-identification via Balancing Alignment and Uniformity. *CVPR 2024*.
- Bao, D., Chern, S.-S., & Shen, Z. (2000). *An Introduction to Riemann–Finsler Geometry*. Springer.
