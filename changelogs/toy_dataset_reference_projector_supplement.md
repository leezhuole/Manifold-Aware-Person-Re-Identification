# Toy-dataset reference-point projector — supplementary design document

**Date:** 2026-04-22
**Scope:** Addendum to §5 of [changelogs/toy_dataset_asymmetry_diagnostics.md](toy_dataset_asymmetry_diagnostics.md) introducing a **reference-point drift projector** $w_{\mathrm{ref}} = \omega_0/\|\omega_0\|_2$, its interpretability gauge $\bar c_k = \overline{\cos(\omega_0,\omega_k)}$, and extensions to TABLE 2 in [results/toy_analysis_v3/FINDINGS.md](../results/toy_analysis_v3/FINDINGS.md) §6.
**Depends on:** parent design doc (§5.1 projectors A/B/C), [scripts/toy_dataset_analysis.py](../scripts/toy_dataset_analysis.py) (`compute_drift_orthogonal_projection`), [scripts/toy_paper_figures.py](../scripts/toy_paper_figures.py) (`make_drift_alignment_plot`).
**Status:** diagnostic-only. No retraining, no modification of `finsler_drift_dist` or `resnet50_finsler`.

---

## 0. Notation (additions to parent §0)

| Symbol | Meaning |
|---|---|
| $w_{\mathrm{ref}} = \omega_0 / \|\omega_0\|_2$ | Unit drift direction at the **clean reference** ($\sigma = 0$) crop |
| $\mathbf{P}_{\mathrm{ref}}^\perp = \mathbf{I}_{d_{\mathrm{id}}} - w_{\mathrm{ref}} w_{\mathrm{ref}}^\top$ | Orthogonal complement of $\mathrm{span}\{w_{\mathrm{ref}}\}$ |
| $\eta_{\mathrm{ref}} = (w_{\mathrm{ref}}^\top \boldsymbol{\Delta})^2 / \|\boldsymbol{\Delta}\|_2^2$ | Reference-point absorption ratio ( $=\cos^2\angle(w_{\mathrm{ref}},\boldsymbol{\Delta})$ ) |
| $\bar c_k = \frac{1}{|\mathcal{P}|}\sum_{p \in \mathcal{P}} \cos\!\big(\omega_0^{(p)},\omega_k^{(p)}\big)$ | Mean per-severity drift-alignment cosine — the **validity gauge** |
| $d_F(\mathbf{x}_0,\mathbf{x}_k)$ | Randers midpoint distance on full embeddings (parent §0) |

Notation $\omega_0 := \omega_{(p,s,0)}$ and $\omega_k := \omega_{(p,s,k)}$ for the same $(p,s)$; averaging is over all $(p,s)$ pairs unless stated.

---

## 1. Motivation and paper-coupling

The midpoint projector $w_B = (\omega_0 + \omega_k)/\|\omega_0+\omega_k\|_2$ of parent §5.1 is **paper-native**: its inner product with $\boldsymbol{\Delta}$ is — up to a positive scalar — the exact Randers asymmetry term read by $d_F$ in paper eq. (4). It is therefore the correct primary choice for tying the absorption statistic to the trained scoring function.

Two complementary questions $w_B$ does **not** answer:

1. **Is the corruption-drift axis already identifiable from the clean crop alone?** That is, does the trained drift field satisfy $\omega_0 \parallel \omega_k$ across the severity ladder (drift-axis stability), or does $\omega$ undergo directional mutation with $k$?
2. **How much of the identity-drift component absorbed by $w_B$ is actually due to the clean anchor $\omega_0$ versus the corrupted endpoint $\omega_k$?** The midpoint aggregates both; neither endpoint alone is exposed.

The **reference-point projector**
$$w_{\mathrm{ref}} := \frac{\omega_0}{\|\omega_0\|_2}$$
addresses both. It treats the clean-crop drift direction as the a priori drift axis and measures how well $\boldsymbol{\Delta}$ aligns with it.

**Framing note.** The midpoint-vs-endpoint choice is a discretisation of the continuous Randers path integral $d_F(\gamma) = \int_0^1 F(\gamma(t),\dot\gamma(t))\,\mathrm{d}t$. Bao–Chern–Shen, *An Introduction to Riemann–Finsler Geometry* (GTM 200, Springer, 2000, §11.1–11.2) define the Randers 1-form $\beta = b_i(x) y^i$ as base-point dependent but do **not** contain a theorem comparing midpoint- to endpoint-quadrature: the trapezoidal midpoint in paper eq. (4) is a design choice, and $w_{\mathrm{ref}}$ is a sensitivity probe for that choice, not a new Finsler-geometric object.

---

## 2. Construction

For each $(p, s)$ with both $\sigma=0$ and $\sigma=k \in \{1,2,3,4\}$ present:

$$
\boldsymbol{\Delta} = \mathbf{z}_k^{\mathrm{id}} - \mathbf{z}_0^{\mathrm{id}}, \qquad
w_{\mathrm{ref}} = \frac{\omega_0}{\|\omega_0\|_2}, \qquad
\eta_{\mathrm{ref}} = \frac{(w_{\mathrm{ref}}^\top \boldsymbol{\Delta})^2}{\|\boldsymbol{\Delta}\|_2^2} \in [0,1].
$$

By the Pythagorean decomposition (parent §5.2),
$$
\|\mathbf{P}_{\mathrm{ref}}^\perp \boldsymbol{\Delta}\|_2^2 = \|\boldsymbol{\Delta}\|_2^2 - (w_{\mathrm{ref}}^\top \boldsymbol{\Delta})^2 = (1 - \eta_{\mathrm{ref}})\,\|\boldsymbol{\Delta}\|_2^2.
$$

**What $\eta_{\mathrm{ref}}$ is, named plainly.** $\eta_{\mathrm{ref}} = \cos^2\!\angle(w_{\mathrm{ref}},\boldsymbol{\Delta})$ is the **directional $R^2$** of $\boldsymbol{\Delta}$ on the fixed axis $w_{\mathrm{ref}}$ — equivalently the rank-1 coherence, or the squared cosine of the principal angle between $\mathrm{span}\{w_{\mathrm{ref}}\}$ and $\mathrm{span}\{\boldsymbol{\Delta}\}$ (Björck & Golub, *Math. Comp.* 27, 1973, cited in parent §10). The statistic is not novel; the **application** — using the clean-endpoint drift as a fixed probe for corruption-induced identity displacement — is.

### 2.1 Relationship to projectors A, B, C

- $w_{\mathrm{ref}}$ is **distinct** from projector A ($w_A = \omega_k/\|\omega_k\|_2$) of parent §5.1 — A uses the perturbed endpoint and coincides with the per-sample Gram–Schmidt applied inside [bau/models/model.py:354-356](../bau/models/model.py#L354-L356); $w_{\mathrm{ref}}$ uses the clean endpoint and has no counterpart in the network head.
- As $\bar c_k \to 1$ (i.e. $\omega_0$ and $\omega_k$ nearly collinear),
$$
\big\|\omega_0 + \omega_k\big\|_2 \to \|\omega_0\|_2 + \|\omega_k\|_2, \qquad w_B \to \tfrac{\omega_0}{\|\omega_0\|_2} = w_{\mathrm{ref}},
$$
so $\eta_{\mathrm{ref}} \to \eta_B$ and both are dominated by $\eta_C$ (since the 2-D subspace degenerates to a line). In this regime all three projectors converge.

### 2.2 Conditional ordering (not universal)

It is tempting to claim $\eta_{\mathrm{ref}} \leq \eta_B$ universally; this is **false**. The midpoint direction $w_B$ is the bisector of $\omega_0$ and $\omega_k$. If $\omega_k$ has a negative projection onto $\boldsymbol{\Delta}$ while $\omega_0$ has a positive one, averaging drags $w_B$ away from $\boldsymbol{\Delta}$ and $(w_B^\top\boldsymbol{\Delta})^2$ can fall **below** $(w_{\mathrm{ref}}^\top\boldsymbol{\Delta})^2$. The sufficient condition for $\eta_{\mathrm{ref}} \leq \eta_B$ is
$$
\operatorname{sign}(\omega_0^\top\boldsymbol{\Delta}) = \operatorname{sign}(\omega_k^\top\boldsymbol{\Delta}),
$$
which is empirically likely (both inner products negative for clean-to-corrupted pairs — see FINDINGS.md §6 "Randers term is negative at all severities") but not forced by the model. The conditional statement should therefore read *"when $\bar c_k$ is high, $\eta_{\mathrm{ref}} \lesssim \eta_B$ and both are lower bounds of $\eta_C$; the gap $\eta_B - \eta_{\mathrm{ref}}$ is a measure of directional mutation of $\omega$ across the severity ladder."*

---

## 3. The validity gauge $\bar c_k$

### 3.1 Definition and role

$$
\bar c_k = \frac{1}{|\mathcal{P}_k|} \sum_{(p,s) \in \mathcal{P}_k} \frac{\omega_0^{(p,s)\,\top}\,\omega_k^{(p,s)}}{\|\omega_0^{(p,s)}\|_2\,\|\omega_k^{(p,s)}\|_2}, \qquad \mathcal{P}_k = \{(p,s) : \sigma=0,\,\sigma=k \text{ both present}\}.
$$

$\bar c_k$ is the **interpretability gauge** of $\eta_{\mathrm{ref}}$:

- High $\bar c_k$ ($\to 1$): $w_{\mathrm{ref}}$, $w_B$, and $w_A$ converge; $\eta_{\mathrm{ref}}$ is a tight proxy for the 2-D $\eta_C$.
- Low $\bar c_k$: drift spans a genuinely 2-D plane; the 1-D reference projector systematically under-reports coverage and the gap $\eta_B - \eta_{\mathrm{ref}}$ quantifies directional mutation.

**Stratification versus parent §4 block means.** The parent-doc same-PID/same-source/different-severity block (FINDINGS.md §5, $\bar C = 0.351$) averages over all $(σ_i, σ_j)$ with $σ_i \neq σ_j$. $\bar c_k$ is a **severity-stratified refinement** anchored at $\sigma=0$: it isolates the specific direction $\omega_0 \to \omega_k$ rather than the general "any two different severities" union. Under H4 below, $\bar c_1$ should exceed $\bar C = 0.351$ (the block mean pools $k=1$ with $k=4$, which dilutes).

### 3.2 Geodesic-error framing

Comparing $\omega_0$ and $\omega_k$ as free vectors in $\mathbb{R}^{d_{\mathrm{id}}}$ is only exact when the two identity-sphere base points $\hat{\mathbf{n}}_0$ and $\hat{\mathbf{n}}_k$ coincide. On $\mathbb{S}^{d_{\mathrm{id}}-1}$ with sectional curvature $K=1$, the parallel transport of a tangent vector $v \in T_{\hat{\mathbf{n}}_0}\mathbb{S}^{d_{\mathrm{id}}-1}$ along the length-minimising geodesic to $T_{\hat{\mathbf{n}}_k}\mathbb{S}^{d_{\mathrm{id}}-1}$ is (Pennec, *J. Math. Imaging Vis.* 25, 2006, §3.1 Proposition 1)

$$
\mathcal{P}_{\hat{\mathbf{n}}_0 \to \hat{\mathbf{n}}_k}(v) = v - \frac{\hat{\mathbf{n}}_0^\top v}{1 + \hat{\mathbf{n}}_0^\top \hat{\mathbf{n}}_k}\,(\hat{\mathbf{n}}_0 + \hat{\mathbf{n}}_k), \qquad \hat{\mathbf{n}}_0^\top \hat{\mathbf{n}}_k > -1 + \varepsilon.
$$

Expanding for small geodesic distance $\theta = \arccos(\hat{\mathbf{n}}_0^\top \hat{\mathbf{n}}_k)$ and $v \in T_{\hat{\mathbf{n}}_0}\mathbb{S}^{d_{\mathrm{id}}-1}$ (so $\hat{\mathbf{n}}_0^\top v = 0$), the ambient-Euclidean identification $v \mapsto v$ differs from the transported vector by $\|\mathcal{P}(v) - v\|_2 = O(\sin\theta) = O(\theta)$ — **first-order** in the geodesic distance (do Carmo, *Riemannian Geometry*, Birkhäuser 1992, Ch. 4 Prop. 4.6; Ch. 10 §10.2 on holonomy of $\mathbb{S}^n$). Consequently, flat ambient comparison of $\omega_0$ and $\omega_k$ is asymptotically exact exactly in the regime $\bar c_k \to 1$ (equivalently $\theta \to 0$). This is the differential-geometric justification for reading $\eta_{\mathrm{ref}}$ only when $\bar c_k$ is high.

Note that under the current model, $\omega$ is Gram–Schmidt'd to be tangent to $\hat{\mathbf{n}}$ per sample but **not** parallel-transported across samples with different $\hat{\mathbf{n}}$; the model's training signal implicitly treats $\omega$ as a free vector. The validity gauge quantifies the regime where this treatment is defensible.

---

## 4. Finsler distance column (TABLE 2 extension)

The mean Finsler distance between the full clean- and corrupted-embedding pair is

$$
d_F(\mathbf{x}_0, \mathbf{x}_k) = \|\boldsymbol{\Delta}\|_2 + \tfrac{1}{2}\|\omega_0 + \omega_k\|_2 \cdot (w_B^\top \boldsymbol{\Delta}),
$$

where the second term is the **Randers coupling scalar** already tracked per-pair in [results/toy_analysis_v3/drift_orthogonal_projection.json](../results/toy_analysis_v3/drift_orthogonal_projection.json) (`randers_asymmetry_term`). No new inner product is required; the column is algebraically `delta_norm_mean + randers_term_mean` per severity. Its purpose is to expose the *net* distance readable by the scoring function and contrast it with $\|\boldsymbol{\Delta}\|_2$ (the Euclidean baseline on the identity slice alone).

---

## 5. Hypotheses and falsifiers

### 5.1 H4 — drift-axis stability

$$
\text{(H4)}\quad \bar c_k \in [c_{\min}, 1],\ \ c_{\min} \gg \tfrac{1}{\sqrt{d_{\mathrm{id}}}}, \quad \bar c_1 \geq \bar c_2 \geq \bar c_3 \geq \bar c_4.
$$

*Interpretation:* mild corruption preserves the clean drift axis; heavy corruption mutates it. *Falsifiers:*

- Non-monotone $\bar c_k$ — directional mutation is non-graded in severity.
- $\bar c_1 \lesssim 1/\sqrt{d_{\mathrm{id}}} \approx 0.022$ — the clean drift direction is already not identifiable and the downstream interpretation of $w_{\mathrm{ref}}$ is vacuous. In this case $\eta_{\mathrm{ref}}$ should still be reported but flagged as non-interpretable at all $k$.

### 5.2 H5 — reference-point absorption

$$
\text{(H5)}\quad \bar\eta_{\mathrm{ref}}(k) > \eta_{\mathrm{ref}}^{\mathrm{shuffle}}(k) \text{ at all } k,\qquad \lim_{\bar c_k \to 1}\bar\eta_{\mathrm{ref}}(k) = \bar\eta_B(k).
$$

*Interpretation:* the clean drift direction captures above-chance fraction of the identity drift, and converges to the midpoint-projector result when the drift axis is stable. *Falsifier:* $\bar\eta_{\mathrm{ref}} \lesssim \eta_{\mathrm{ref}}^{\mathrm{shuffle}}$ at high-$\bar c_k$ $k$-values would indicate that even when the drift axis is stable, it is not a useful probe for identity displacement — a stronger negative result than the $\bar\eta_B < \mathrm{shuffle}$ at $k \geq 3$ already reported in FINDINGS.md §6.

### 5.3 Pre-registered prediction from the existing FINDINGS.md numbers

From TABLE 2 in [results/toy_analysis_v3/FINDINGS.md](../results/toy_analysis_v3/FINDINGS.md) §6, $\bar\eta_B$ drops below the shuffle null at $k=3$ and $k=4$. The supplementary-doc prediction is that $\bar c_k$ will also decrease with $k$ and that $\bar\eta_{\mathrm{ref}} \approx \bar\eta_B$ at $k=1$ (paired axis, high-$\bar c_k$ regime) but decouple at $k \geq 3$. If instead $\bar\eta_{\mathrm{ref}} \gg \bar\eta_B$ at $k=3,4$, the drift axis is stable (controlled by $\omega_0$) but $\omega_k$ drifts non-informatively, suggesting that the mutation penalty is in the perturbed endpoint, not the reference.

---

## 6. Validity-gauge pairing — peer-level note

The pairing of a directional-coherence statistic ($\eta_{\mathrm{ref}}$) with a direction-stability gauge ($\bar c_k$) has structural parallels but no named pattern in the literature we could verify:

- **Concept Activation Vectors** (Kim et al., *ICML 2018*, §3.1 Def. 1; §3.2 eq. 1) define a single learned direction at a reference input and probe class predictions by inner product. $\eta_{\mathrm{ref}}$ is the $R^2$ analogue of the TCAV directional sensitivity score. TCAV practitioners informally report cross-seed cosine stability of CAVs as a sanity check; we formalise this role as an interpretability threshold.
- **Representational similarity under perturbation** (Kornblith et al., "Similarity of Neural Network Representations Revisited," *ICML 2019*, §3, §5) compares whole-layer Gram matrices across input transformations via CKA; $\bar c_k$ is a per-sample, drift-direction-specific instance of the same coherence question at a single-axis granularity.
- **Linear representation hypothesis** (Park, Choe, Veitch, arXiv:2311.03658, 2023, §3–4) shows concept directions remain stable across input contexts in large language models — directly parallels $\bar c_k \to 1$ as the "drift concept is stably encoded" condition. *(Preprint; cite as arXiv pending venue publication.)*

The pairing is **well-motivated but not an established named pattern**. It is closest in spirit to reliability–validity pairing in psychometrics (report both the measure and a delineation of the regime where it is interpretable). Describing it as "established" would overclaim; describing it as "novel geometry" would also overclaim — the novelty is in the specific application to Finsler-drift diagnostics in ReID.

---

## 7. Isotropic-null calibration

The parent-doc analytic null of parent §5.3 applies directly:
$$
\mathbb{E}_{\boldsymbol{\Delta}\sim\mathrm{Unif}(\mathbb{S}^{d_{\mathrm{id}}-1})}[\eta_{\mathrm{ref}}] = \frac{1}{d_{\mathrm{id}}} = \frac{1}{2048} \approx 4.88\times 10^{-4}
$$
(Stein, *Proc. 3rd Berkeley Symp.*, 1956; Ledoux, *The Concentration of Measure Phenomenon*, AMS 2001). The shuffle null is as in parent §5.3 with $(\omega_0, \omega_k)$ permuted against $\boldsymbol{\Delta}$: the permuted $w_{\mathrm{ref}}$ at each PID is drawn from an unrelated identity.

The interpretation "$\bar\eta_{\mathrm{ref}} \gg 1/d_{\mathrm{id}}$" remains the meaningful test; but H5 is strictly stronger — it asks for *conditional* coupling (shuffle-null) not mere *marginal* coverage (analytic null).

---

## 8. Implementation touchpoints

| File | Function | Change |
|---|---|---|
| [scripts/toy_dataset_analysis.py](../scripts/toy_dataset_analysis.py) | `compute_drift_orthogonal_projection` | Per-pair keys `eta_ref`, `perp_ref_norm`, `cos_omega_0_k`, `finsler_distance`; per-severity means/stds; `eta_ref` shuffle null |
| [scripts/toy_dataset_analysis.py](../scripts/toy_dataset_analysis.py) | `main()` entrypoint | Call `make_drift_alignment_plot`; print $\bar\eta_{\mathrm{ref}}$, $\bar c_k$, $\bar d_F$ alongside existing H3 summary |
| [scripts/toy_paper_figures.py](../scripts/toy_paper_figures.py) | `make_drift_alignment_plot` | New two-row figure: (top) $\bar c_k$ validity gauge; (bottom) $\bar\eta_{\mathrm{ref}}$ vs $\bar\eta_B$ on log scale with shuffle-null bands |
| [results/toy_analysis_v3/FINDINGS.md](../results/toy_analysis_v3/FINDINGS.md) §6 | TABLE 2 | **Manual** extension with columns $\bar c_k$, $\bar\eta_{\mathrm{ref}}$, $\|\mathbf{P}_{\mathrm{ref}}^\perp\boldsymbol{\Delta}\|_2$, $\bar d_F$ from the new JSON keys |

No CLI flag added — the new columns are always computed and cheap. The existing `--projection-report` / `--projection-shuffle-reps` flags are unchanged.

---

## 9. Outputs

New / extended artefacts under `results/toy_analysis_v3/`:

```
drift_orthogonal_projection.json   # extended: eta_ref_mean/_std, cos_omega_0_k_mean/_std,
                                   #           perp_ref_norm_mean/_std, finsler_distance_mean/_std,
                                   #           shuffle_null.<k>.eta_ref_null_mean/_std
fig_drift_alignment.pdf            # NEW: two-row (cos gauge on top, eta log-scale on bottom)
```

---

## 10. Verification

1. **JSON schema**
   ```bash
   python -c "import json; d = json.load(open('results/toy_analysis_v3/drift_orthogonal_projection.json')); print(sorted(d['per_severity_k']['1'].keys()))"
   ```
   Must contain `cos_omega_0_k_mean`, `eta_ref_mean`, `perp_ref_norm_mean`, `finsler_distance_mean`.

2. **Sanity cross-check.** For every $k$:
$$
\text{finsler\_distance\_mean}(k) \stackrel{?}{=} \text{delta\_norm\_mean}(k) + \text{randers\_term\_mean}(k).
$$
A ≥ $10^{-9}$-level mismatch on any cell is a bug.

3. **Stratification consistency.** $\bar c_1$ should be at least as large as the parent-doc block mean $\bar C^{\text{same pid, same src, diff sev}} = 0.351$ (FINDINGS.md §5). If $\bar c_1 < 0.30$, stratification is suspect (e.g. indexing off-by-one in `severity_k`).

4. **Figure.** `fig_drift_alignment.pdf` renders without error; top panel bounded in $[-0.05, 1.05]$; bottom panel on log scale with shuffle-null bands visible.

5. **End-to-end**
   ```bash
   python scripts/toy_dataset_analysis.py \
     --finsler-checkpoint logs/finsler_primary/job_1521403_primary_unified_1c_w0.1_driftInst/best.pth \
     --euclidean-checkpoint logs/AGReIDv2_sweep/job_1502418_dt_cuhk03_EucBaseline/best.pth \
     --dataset-dir examples/data/ToyCorruption \
     --output-dir results/toy_analysis_v3
   ```

---

## 11. Literature anchors

Only references not already cited in parent §10 are expanded here; the others (Pennec 2006, Bao–Chern–Shen 2000, Stein 1956, Björck–Golub 1973, Salzmann 2010) carry over unchanged.

- **Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viégas, F., Sayres, R.** *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV).* **ICML 2018.** §3.1 Def. 1; §3.2 eq. (1). Canonical "fixed learned direction at reference input as a probe" precedent; $\eta_{\mathrm{ref}}$ is the $R^2$ analogue of the TCAV directional-sensitivity score.
- **Kornblith, S., Norouzi, M., Lee, H., Hinton, G.** *Similarity of Neural Network Representations Revisited.* **ICML 2019.** §3 (CKA definition); §5 (perturbation experiments). Population-level stability of representations under input transformations; $\bar c_k$ is a per-sample directional instance of the same coherence question.
- **Park, K., Choe, Y. J., Veitch, V.** *The Linear Representation Hypothesis and the Geometry of Large Language Models.* **arXiv:2311.03658**, 2023. §3–4. Concept directions are stable across input contexts in large language models — direct analogue of $\bar c_k \to 1$ as the "stable drift-concept" condition. *Preprint; flag in citations.*
- **do Carmo, M. P.** *Riemannian Geometry.* Birkhäuser, 1992. Ch. 4 Prop. 4.6; Ch. 10 §10.2. $O(\theta)$ angular error of flat comparison of tangent vectors at different base points on $\mathbb{S}^{d-1}$; grounds the validity-gauge threshold.

All four are verifiable on Google Scholar / arXiv with title + venue + year. Running `paper-rigorous-citation` before camera-ready is recommended to confirm the Park et al. venue (ICLR 2024 if accepted).

---

## 12. Out of scope

- Retraining — $w_{\mathrm{ref}}$ is a post-hoc diagnostic.
- Modifying $\mathrm{finsler\_drift\_dist}$ or $\mathcal{L}_{\mathrm{dcc}}$ — the existing midpoint parametrisation of paper eq. (4) is unchanged.
- Parallel-transport variant of $\eta_{\mathrm{ref}}$ — the ambient formulation is sufficient given the $\bar c_k$ gauge; the Pennec-transport variant is a future extension parallel to parent §4.1's cosine variant.
- Cross-source $w_{\mathrm{ref}}$ — parent §5.5's construction carries over verbatim (replace $\omega_0$ with $\omega_{(p,1,\sigma_q)}$); the same-source variant is the primary because $\bar c_k$ is cleanest there.
