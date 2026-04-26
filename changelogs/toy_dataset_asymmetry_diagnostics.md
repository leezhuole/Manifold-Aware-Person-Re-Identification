# Toy-dataset asymmetry diagnostics — design document

**Date:** 2026-04-19
**Scope:** Extension of `scripts/toy_dataset_analysis.py` with three diagnostics that substantiate the *direction-aware* and *drift-absorbs-identity-slack* narrative of [paper/GuidedResearch_2026_LeeZhuoLe_Dages_Weber.pdf](../paper/GuidedResearch_2026_LeeZhuoLe_Dages_Weber.pdf) on the ToyCorruption controlled benchmark.
**Depends on:** [changelogs/toy_dataset_synthesis.md](toy_dataset_synthesis.md) (v2.0 pipeline), [scripts/generate_toy_dataset.py](../scripts/generate_toy_dataset.py), [scripts/toy_dataset_analysis.py](../scripts/toy_dataset_analysis.py), [bau/models/model.py](../bau/models/model.py) (`resnet50_finsler`), [bau/loss/triplet.py](../bau/loss/triplet.py) (`finsler_drift_dist`, `euclidean_dist`).

---

## 0. Symbols and conventions

We retain the paper's notation end-to-end.

| Symbol | Meaning |
|---|---|
| $\mathbf{x} \in \mathbb{R}^{3 \times H \times W}$ | Input crop after normalisation |
| $\mathbf{z}(\mathbf{x}) = [\mathbf{z}^{\text{id}}(\mathbf{x}) \,\|\, \boldsymbol{\omega}(\mathbf{x})] \in \mathbb{R}^{d_{\text{id}}+d_\omega}$ | Structured Finsler embedding (paper §3.1) |
| $\mathbf{z}^{\text{id}} \in \mathbb{S}^{d_{\text{id}}-1}$ | $\ell_2$-normalised identity slice, $d_{\text{id}}=2048$ |
| $\boldsymbol{\omega} \in \mathbb{R}^{d_\omega}$ | Drift slice, $d_\omega = 2048$ by default, norm-gated to $\|\boldsymbol{\omega}\|_2 < c_{\max} = 0.95$ and Gram–Schmidt'd so that $\boldsymbol{\omega}^\top \mathbf{z}^{\text{id}} = 0$ per sample |
| $d_E(\mathbf{x},\mathbf{y}) = \|\mathbf{z}^{\text{id}}_{\mathbf{x}} - \mathbf{z}^{\text{id}}_{\mathbf{y}}\|_2$ | Euclidean (identity-only) distance |
| $d_F(\mathbf{x},\mathbf{y}) = \|\mathbf{z}^{\text{id}}_{\mathbf{x}} - \mathbf{z}^{\text{id}}_{\mathbf{y}}\|_2 + \big\langle \tfrac{1}{2}(\boldsymbol{\omega}_{\mathbf{x}} + \boldsymbol{\omega}_{\mathbf{y}}),\, \mathbf{z}^{\text{id}}_{\mathbf{y}} - \mathbf{z}^{\text{id}}_{\mathbf{x}} \big\rangle$ | Midpoint-drift Randers score (paper eq. 4) |
| $p \in \mathcal{P}$ | Person identity (PID) |
| $s \in \{1,2\}$ | source-crop index per PID |
| $\sigma \in \{0,1,2,3,4\}$ | Corruption severity level |
| $\mathbf{x}^{(p,s,\sigma)}$ | Synthetic crop of person $p$, source $s$, severity $\sigma$ |

All inner products are Euclidean on $\mathbb{R}^{d_{\text{id}}}$; there is no non-Euclidean backbone geometry (paper §3.1).

---

## 1. ToyCorruption v3.0: two source crops per identity

### 1.1 Why v3.0 is needed

The canonical ReID setup retrieves *a different photo of the same person*, not *the same photo under different perturbations*. The v2.0 generator ([scripts/generate_toy_dataset.py](../scripts/generate_toy_dataset.py)) selects one lexicographically-first Market-1501 crop per PID and applies the five-level corruption ladder to that single crop. Under this schedule the "correct match" at retrieval time is literally the same underlying image modulated by the ImageNet-C-aligned pipeline, so any retrieval metric conflates corruption drift with *trivially perfect* identity content.

The primary diagnostic requested below (Point 1a) requires **two distinct source crops per PID**, each augmented with the full severity ladder, yielding $2 \times 5 = 10$ crops per identity. Same-source cross-severity retrieval (Point 1b) is kept as an ablation because it isolates drift-only signal from compounded within-PID appearance variance (rationale in §3.2).

### 1.2 Dataset schema

Per PID $p \in \mathcal{P}$, we pick the two lexicographically-smallest crops *from different source cameras* in Market-1501 `bounding_box_test`; the filter "PIDs with at least two Market-1501 cameras" already exists in v2.0 [scripts/generate_toy_dataset.py:242-277](../scripts/generate_toy_dataset.py#L242-L277) and guarantees the pick is well-defined.

**Filename convention (breaking change vs v2.0):**

```text
{new_pid:04d}_c{s}s{σ+1}_000001_01.jpg
```

- `c{s}` encodes the source index $s \in \{1,2\}$ — this becomes the **ReID camera id** for `cmc` / `mean_ap` same-(pid,cam) filtering.
- `s{σ+1}` encodes severity via the Market-1501 *sequence* field, $\sigma+1 \in \{1..5\}$.

Regex update for [scripts/toy_dataset_analysis.py:74](../scripts/toy_dataset_analysis.py#L74):

```python
pattern = re.compile(r"(\d+)_c(\d+)s(\d+)")  # (pid, source_idx, severity+1)
```

`ToyDataset.__getitem__` returns $(\text{img}, \text{fname}, p, s, \sigma)$ so that downstream code no longer overloads cam-id as a severity proxy.

**Splits:**

| Split | Contents | Count per PID |
|---|---|---|
| `bounding_box_test/` | all $s \times \sigma$ crops | 10 |
| `query/` | $s=1$, $\sigma=0$ (clean anchor) | 1 |
| `gallery/` | everything else | 9 |

The new diagnostic functions **ignore** the query/gallery folder split and build their own partitions directly from `bounding_box_test/`; the folder split is retained only for compatibility with stock ReID loaders.

### 1.3 Version / provenance

- `TOY_DATASET_VERSION = "3.0"` (bumped; v2.0 metadata consumers must gate on this).
- `metadata.json` schema extended with `dataset.source_crop_selection = "lex_sort_distinct_camera_top2"` and per-image `source_idx`.
- The ImageNet-C operator mapping and `CORRUPTION_TABLE` are **unchanged** — only the source-selection and filename layout change.

### 1.4 Cost estimate

At $N=50$ identities, $2 \times 5 = 10$ crops per PID, total 500 JPEGs (v2.0: 250). Feature extraction at batch 64 on one GPU: $<$ 1 min per checkpoint. Negligible.

---

## 2. Point 1a — Cross-source, cross-severity retrieval (**primary**)

### 2.1 Setup

For a fixed pair $(s_q, \sigma_q, s_g, \sigma_g)$ with $s_q \neq s_g$:

$$
\mathcal{Q}_{s_q,\sigma_q} = \{\mathbf{x}^{(p, s_q, \sigma_q)} : p \in \mathcal{P}\}, \qquad
\mathcal{G}_{s_g,\sigma_g} = \{\mathbf{x}^{(p, s_g, \sigma_g)} : p \in \mathcal{P}\}.
$$

For each query $\mathbf{q} \in \mathcal{Q}_{s_q,\sigma_q}$, the ranked gallery is produced under two scoring functions evaluated on **independently trained checkpoints** (consistent with the existing v2.0 analyser):

- Euclidean BAU: $\mathbf{z}^{\text{id}}$ from `resnet50`, ranking by $d_E$.
- Finsler BAU: $[\mathbf{z}^{\text{id}} \,\|\, \boldsymbol{\omega}]$ from `resnet50_finsler`, ranking by $d_F$ (`method="symmetric_trapezoidal"`).

The `mean_ap` / `cmc` filter at [bau/evaluation_metrics/ranking.py:18-119](../bau/evaluation_metrics/ranking.py#L18-L119) drops same-(pid, cam) pairs. Because $s_q \neq s_g$, the filter never removes the true match; because we sweep all severities, the diagonal $\sigma_q = \sigma_g$ is well-defined and informative (unlike the v2.0 setup).

Reported tensors (one per scoring function):

$$
\text{mAP}[s_q, \sigma_q, s_g, \sigma_g] \in [0,1], \qquad \text{R1}[s_q, \sigma_q, s_g, \sigma_g] \in [0,1].
$$

With $s_q \neq s_g$ and the bidirectional pair $(s_q, s_g) \in \{(1,2), (2,1)\}$, we report two $5 \times 5$ severity matrices per scoring function. Averaging over source direction gives a single $5 \times 5$ heat map.

### 2.2 Hypotheses and falsifiers

**H1a (monotone degradation).** $\text{mAP}[\sigma_q, \sigma_g]$ is non-increasing in $|\sigma_q - \sigma_g|$ for fixed diagonal $\sigma_q = \sigma_g$. *Falsifier:* non-monotone off-diagonal rows — diagnoses an ill-behaved corruption ladder rather than a model defect.

**H1b (Randers asymmetry signature).** The Finsler mAP matrix is **non-symmetric** across the severity axis even after averaging over $(s_q, s_g)$:

$$
A[\sigma_q, \sigma_g] = \text{mAP}_F[\sigma_q, \sigma_g] - \text{mAP}_F[\sigma_g, \sigma_q] \not\equiv 0,
$$

whereas the Euclidean baseline is symmetric up to sampling noise. This is the direct behavioural signature of the $\langle \tfrac{1}{2}(\boldsymbol{\omega}_{\mathbf{x}}+\boldsymbol{\omega}_{\mathbf{y}}),\,\mathbf{z}^{\text{id}}_{\mathbf{y}}-\mathbf{z}^{\text{id}}_{\mathbf{x}}\rangle$ term and is orthogonal to H1a.

**H1c (drift-shift Spearman).** Over all within-PID image pairs $(\mathbf{x}_i, \mathbf{x}_j)$ regardless of source, the drift-shift magnitude tracks severity difference:

$$
\rho_s\!\left(\,|\sigma_i - \sigma_j|\,,\, \|\boldsymbol{\omega}_i - \boldsymbol{\omega}_j\|_2\,\right) > 0 \ \text{with}\ p < 0.05.
$$

This extends the v2.0 C2 check from per-image $\|\boldsymbol{\omega}_k\|$ to pair-wise $\|\boldsymbol{\omega}_i - \boldsymbol{\omega}_j\|$ and is the falsifiable form of the user's noise-additive reading.

### 2.3 Caveat (to appear verbatim in the figure caption)

At $N=50$ identities, each $(s_q, \sigma_q, s_g, \sigma_g)$ cell has exactly 50 queries and 50 gallery items with a single correct match per query. The mAP estimate per cell is the mean of $1/\text{rank}$ over 50 queries; its standard error is $O(1/\sqrt{50}) \approx 0.14$ under random-baseline assumptions. This suffices to resolve the asymmetry in H1b at $|A| \gtrsim 0.05$ but not finer; camera-ready should rerun at $N \in [150, 300]$ if the observed $|A|$ is smaller.

---

## 3. Point 1b — Same-source, cross-severity retrieval (**ablation, kept**)

### 3.1 Setup

For a fixed pair $(\sigma_q, \sigma_g)$ with $\sigma_q \neq \sigma_g$ and $s_q = s_g = s$:

$$
\mathcal{Q}^{\text{(1b)}}_{\sigma_q} = \bigcup_{s \in \{1,2\}} \{\mathbf{x}^{(p, s, \sigma_q)}\}, \qquad
\mathcal{G}^{\text{(1b)}}_{\sigma_g} = \bigcup_{s \in \{1,2\}} \{\mathbf{x}^{(p, s, \sigma_g)}\}.
$$

The `mean_ap` same-(pid, cam) filter *does* remove same-source matches here — so the retrieval is forced to resolve the correct PID via the *opposite* source, while the query and correct match share corruption severities $\sigma_q, \sigma_g$ respectively. This collapses the compounded variance back onto a single corruption axis.

### 3.2 Rationale for retention (response to user's question)

Point 1a is the paper-facing canonical test, but on its own it conflates two nuisance factors:

1. **Within-PID appearance drift.** Source crops $s=1,2$ differ in pose, illumination, background — content variation that a clean-image ReID model already has to handle and that has nothing to do with camera corruption.
2. **Corruption drift.** The five-level severity ladder is the variable we want to study.

A Finsler gain in Point 1a over Euclidean can arise from either (i) absorbing corruption-induced directional drift into $\boldsymbol{\omega}$, or (ii) better handling of within-PID appearance through the enlarged $[\mathbf{z}^{\text{id}} \,\|\, \boldsymbol{\omega}]$ feature space in general. Point 1b **pins $\mathbf{x}$-content fixed at the crop level** and varies only the corruption channel, so any mAP delta $\text{mAP}_F - \text{mAP}_E$ is attributable to drift behaviour alone. The two experiments jointly decompose the total Finsler gain:

$$
\underbrace{\Delta_{\text{total}}}_{\text{Point 1a}} \;=\; \underbrace{\Delta_{\text{corruption}}}_{\text{Point 1b}} \;+\; \underbrace{\Delta_{\text{appearance}}}_{\text{residual}} \,.
$$

Additionally, Point 1b is the **strongest possible test of directional consistency of $\boldsymbol{\omega}$**: if drift encodes severity in a retrieval-usable way, then under same-content pairs the Randers asymmetry matrix $A[\sigma_q,\sigma_g]$ in 3.1 should be maximal — any signal that washes out in Point 1a but survives in Point 1b localises the Finsler mechanism to the drift channel.

1b is therefore retained as an ablation figure in the supplementary, with the decomposition above stated once in the main body.

---

## 4. Point 2 — Drift cosine-similarity matrix

### 4.1 Definitions

For every pair of extracted embeddings $(i, j)$ from the full `bounding_box_test/` set (ambient variant, **primary**):

$$
C^{\text{amb}}_{ij} \;=\; \frac{\boldsymbol{\omega}_i^\top \boldsymbol{\omega}_j}{\|\boldsymbol{\omega}_i\|_2 \, \|\boldsymbol{\omega}_j\|_2} \;\in\; [-1, 1].
$$

Because the per-sample Gram–Schmidt at [bau/models/model.py:354-356](../bau/models/model.py#L354-L356) enforces $\boldsymbol{\omega}_i \perp \mathbf{z}^{\text{id}}_i$ and $\boldsymbol{\omega}_j \perp \mathbf{z}^{\text{id}}_j$ but *not* $\boldsymbol{\omega}_i \perp \mathbf{z}^{\text{id}}_j$, the ambient cosine mixes two effects: (i) genuine directional agreement of the drift correction, and (ii) residual identity leakage across samples with different $\mathbf{z}^{\text{id}}$.

**Optional parallel-transport variant** (CLI flag `--omega-cos-transport parallel`, reusing the analytical branch of `finsler_drift_dist` at [bau/loss/triplet.py:242-257](../bau/loss/triplet.py#L242-L257)): transport $\boldsymbol{\omega}_j$ from the tangent space of $\mathbb{S}^{d_{\text{id}}-1}$ at $\hat{\mathbf{n}}_j := \mathbf{z}^{\text{id}}_j$ to the tangent space at $\hat{\mathbf{n}}_i$ along the unique length-minimising geodesic, using the closed-form parallel transport of Pennec (2006, §3):

$$
\mathcal{P}_{\hat{\mathbf{n}}_j \to \hat{\mathbf{n}}_i}(\boldsymbol{\omega}_j) \;=\; \boldsymbol{\omega}_j \;-\; \frac{\hat{\mathbf{n}}_j^\top \boldsymbol{\omega}_j}{1 + \hat{\mathbf{n}}_i^\top \hat{\mathbf{n}}_j}\,(\hat{\mathbf{n}}_i + \hat{\mathbf{n}}_j), \qquad \text{if } \hat{\mathbf{n}}_i^\top \hat{\mathbf{n}}_j > -1 + \epsilon.
$$

Then

$$
C^{\text{par}}_{ij} \;=\; \frac{\boldsymbol{\omega}_i^\top \mathcal{P}_{\hat{\mathbf{n}}_j \to \hat{\mathbf{n}}_i}(\boldsymbol{\omega}_j)}{\|\boldsymbol{\omega}_i\|_2 \, \|\mathcal{P}_{\hat{\mathbf{n}}_j \to \hat{\mathbf{n}}_i}(\boldsymbol{\omega}_j)\|_2}.
$$

Note $\|\mathcal{P}_{\hat{\mathbf{n}}_j \to \hat{\mathbf{n}}_i}(\boldsymbol{\omega}_j)\|_2 = \|\boldsymbol{\omega}_j\|_2$ since parallel transport is an isometry; the denominator simplifies accordingly.

### 4.2 Group structure of the matrix

Ambient cosines are aggregated by the three-axis indicator $(P,S,\Sigma)$ where $P \in \{\text{same pid}, \text{diff pid}\}$, $S \in \{\text{same source}, \text{diff source}\}$, $\Sigma \in \{\text{same sev}, \text{diff sev}\}$:

$$
\bar{C}^{(P,S,\Sigma)} \;=\; \frac{1}{|\mathcal{I}_{P,S,\Sigma}|} \sum_{(i,j) \in \mathcal{I}_{P,S,\Sigma}} C^{\text{amb}}_{ij}.
$$

| Block | Paper hypothesis | Expected sign |
|---|---|---|
| (same pid, any $S$, same sev) | Drift is identity-invariant across source camera, consistent with $\mathcal{L}_{\text{dcc}}$ (paper eq. 10) | large positive |
| (diff pid, any $S$, same sev) | Drift is camera/severity-aligned — paper's Randers-nuisance reading | positive, not negligible |
| (same pid, any $S$, diff sev) | Drift direction *shifts* with severity | smaller positive or near-zero |
| (diff pid, any $S$, diff sev) | Background null | near zero |

The relative ordering of the first two block means is **the direct test** of whether the trained $\boldsymbol{\omega}$ obeys identity-coherence ($\mathcal{L}_{\text{dcc}}$) or camera-alignment, and this is the finding the paper's §3.5 Cross-camera drift coherence argument rests on.

### 4.3 Geometric-assumption audit (peer-level critical note)

Three assumptions are implicit in the ambient cosine:

1. $\boldsymbol{\omega}_i, \boldsymbol{\omega}_j$ are comparable as free vectors in $\mathbb{R}^{d_{\text{id}}}$. This is only approximately true because Gram–Schmidt makes each orthogonal to its own $\mathbf{z}^{\text{id}}$; circular reliance on identity invariance is the caveat.
2. Cosine is a valid similarity on drift vectors of different $\ell_2$ norms. Because norm-gating confines $\|\boldsymbol{\omega}\|_2 \in (0, c_{\max})$ and $c_{\max} = 0.95$, the dynamic range is modest and cosine is defensible.
3. Magnitude information is discarded. Reporting $C_{ij}$ alongside $\|\boldsymbol{\omega}_i - \boldsymbol{\omega}_j\|_2$ (already part of Point 1a H1c) restores it.

The parallel-transport variant addresses (1) directly. Both variants ship in the JSON output; the paper-facing figure uses ambient with the caveat in the caption.

---

## 5. Point 3 — Drift-orthogonal projection of identity drift

### 5.1 Construction

Fix a PID $p$ and a pair of same-source samples at severities $0$ and $k$:

$$
(\mathbf{z}^{\text{id}}_0, \boldsymbol{\omega}_0),\ (\mathbf{z}^{\text{id}}_k, \boldsymbol{\omega}_k), \qquad \boldsymbol{\Delta} := \mathbf{z}^{\text{id}}_k - \mathbf{z}^{\text{id}}_0 \in \mathbb{R}^{d_{\text{id}}}.
$$

Three projectors are relevant.

**(A) Per-sample single-axis (legacy reference).**

$$
\mathbf{w}_A = \frac{\boldsymbol{\omega}_k}{\|\boldsymbol{\omega}_k\|_2}, \qquad \mathbf{P}_A^\perp = \mathbf{I}_{d_{\text{id}}} - \mathbf{w}_A \mathbf{w}_A^\top.
$$

This coincides with the per-sample Gram–Schmidt step already applied inside `resnet50_finsler.forward` ([bau/models/model.py:354-356](../bau/models/model.py#L354-L356)) and is kept only in the JSON dump for completeness.

**(B) Midpoint single-axis — primary.**

$$
\mathbf{w}_B = \frac{\boldsymbol{\omega}_0 + \boldsymbol{\omega}_k}{\|\boldsymbol{\omega}_0 + \boldsymbol{\omega}_k\|_2}, \qquad \mathbf{P}_B^\perp = \mathbf{I}_{d_{\text{id}}} - \mathbf{w}_B \mathbf{w}_B^\top.
$$

The Randers asymmetry term in paper eq. (4) is

$$
\big\langle \tfrac{1}{2}(\boldsymbol{\omega}_0 + \boldsymbol{\omega}_k),\,\boldsymbol{\Delta}\big\rangle \;=\; \tfrac{1}{2}\|\boldsymbol{\omega}_0 + \boldsymbol{\omega}_k\|_2 \cdot (\mathbf{w}_B^\top \boldsymbol{\Delta}),
$$

so $\mathbf{w}_B^\top \boldsymbol{\Delta}$ is — up to the positive scalar $\tfrac{1}{2}\|\boldsymbol{\omega}_0 + \boldsymbol{\omega}_k\|_2$ — exactly the one-dimensional projection of $\boldsymbol{\Delta}$ that the Finsler score reads. Therefore

$$
\|\mathbf{P}_B^\perp \boldsymbol{\Delta}\|_2^2 \;=\; \|\boldsymbol{\Delta}\|_2^2 - (\mathbf{w}_B^\top \boldsymbol{\Delta})^2
$$

is **the component of the identity drift that the Randers asymmetry term cannot absorb**. This is the most paper-coupled choice and is the primary scalar reported.

*Derivation:* Because $\mathbf{P}_B^\perp$ is an orthogonal projector, it is symmetric ($(\mathbf{P}_B^\perp)^\top = \mathbf{P}_B^\perp$) and idempotent ($\mathbf{P}_B^\perp \mathbf{P}_B^\perp = \mathbf{P}_B^\perp$). Evaluating the squared $\ell_2$-norm:
$$
\begin{align*}
\|\mathbf{P}_B^\perp \boldsymbol{\Delta}\|_2^2 &= (\mathbf{P}_B^\perp \boldsymbol{\Delta})^\top (\mathbf{P}_B^\perp \boldsymbol{\Delta}) \\
&= \boldsymbol{\Delta}^\top (\mathbf{P}_B^\perp)^\top \mathbf{P}_B^\perp \boldsymbol{\Delta} \\
&= \boldsymbol{\Delta}^\top (\mathbf{P}_B^\perp \mathbf{P}_B^\perp) \boldsymbol{\Delta} \\
&= \boldsymbol{\Delta}^\top \mathbf{P}_B^\perp \boldsymbol{\Delta} \\
&= \boldsymbol{\Delta}^\top (\mathbf{I}_{d_{\text{id}}} - \mathbf{w}_B \mathbf{w}_B^\top) \boldsymbol{\Delta} \\
&= \boldsymbol{\Delta}^\top \boldsymbol{\Delta} - (\boldsymbol{\Delta}^\top \mathbf{w}_B)(\mathbf{w}_B^\top \boldsymbol{\Delta}) \\
&= \|\boldsymbol{\Delta}\|_2^2 - (\mathbf{w}_B^\top \boldsymbol{\Delta})^2.
\end{align*}
$$

**(C) Two-dimensional Gram–Schmidt projector — supporting.**

$$
\mathbf{u}_1 = \tilde{\boldsymbol{\omega}}_0, \qquad \tilde{\boldsymbol{\omega}}_i := \boldsymbol{\omega}_i / \|\boldsymbol{\omega}_i\|_2,
$$

$$
\mathbf{u}_2 = \frac{\tilde{\boldsymbol{\omega}}_k - (\mathbf{u}_1^\top \tilde{\boldsymbol{\omega}}_k)\,\mathbf{u}_1}{\big\|\tilde{\boldsymbol{\omega}}_k - (\mathbf{u}_1^\top \tilde{\boldsymbol{\omega}}_k)\,\mathbf{u}_1\big\|_2}, \qquad \mathbf{U} = [\mathbf{u}_1 \mid \mathbf{u}_2] \in \mathbb{R}^{d_{\text{id}} \times 2},
$$

$$
\mathbf{P}_C^\perp = \mathbf{I}_{d_{\text{id}}} - \mathbf{U}\mathbf{U}^\top, \qquad \|\mathbf{P}_C^\perp \boldsymbol{\Delta}\|_2 = \|\boldsymbol{\Delta}\|_2\,\sin\theta_1,
$$

where $\theta_1$ is the principal angle between $\text{span}\{\boldsymbol{\Delta}\}$ and $\text{span}\{\boldsymbol{\omega}_0, \boldsymbol{\omega}_k\}$ in the sense of Björck & Golub (1973). $\mathbf{P}_C^\perp$ is the principled generalisation of the per-sample orthogonalisation to *pairs*: it removes every direction expressible as a linear combination of the two drift vectors.

### 5.2 Absorption statistics

**Geometric variance interpretation:** In this context, the term "variance" is utilized in the geometric sense of total signal energy or sum of squared deviations, rather than as a statistical population variance. Because $\boldsymbol{\Delta} := \mathbf{z}^{\text{id}}_k - \mathbf{z}^{\text{id}}_0$ represents the embedding shift from the clean state to the corrupted state, $\mathbf{z}^{\text{id}}_0$ acts as the local expected value or origin for that specific sample. The squared $\ell_2$ norm evaluates to:

$$
\|\boldsymbol{\Delta}\|_2^2 = \sum_{m=1}^{d_{\text{id}}} (\Delta_m)^2
$$

This formulation is exactly the sum of squared coordinate deviations across all $d_{\text{id}}$ dimensions. Therefore, $\|\boldsymbol{\Delta}\|_2^2$ algebraically defines the total spatial variation or displacement energy introduced by the corruption. This allows orthogonal projection operators to partition this total deviation into additive, uncorrelated subspace components.

Define the drift-subspace *absorption ratios* as the variance fraction captured by the corresponding projector:

$$
\eta_B \;=\; \frac{(\mathbf{w}_B^\top \boldsymbol{\Delta})^2}{\|\boldsymbol{\Delta}\|_2^2} \;=\; 1 - \frac{\|\mathbf{P}_B^\perp \boldsymbol{\Delta}\|_2^2}{\|\boldsymbol{\Delta}\|_2^2} \;\in [0,1],
$$

$$
\eta_C \;=\; \frac{\|\mathbf{U}^\top \boldsymbol{\Delta}\|_2^2}{\|\boldsymbol{\Delta}\|_2^2} \;=\; \cos^2\theta_1 \;\in [0,1].
$$

Reported per $(p, k)$ pair and aggregated to $(\bar{\eta}_B(k), \bar{\eta}_C(k))$ as functions of severity.

**Derivation of $\eta_B$ (Midpoint 1D Subspace Absorption):**
Let $\mathbf{P}_B = \mathbf{w}_B \mathbf{w}_B^\top$ be the orthogonal projection matrix onto the 1D subspace spanned by the unit vector $\mathbf{w}_B$. By the Pythagorean theorem for inner product spaces, the total squared norm of $\boldsymbol{\Delta}$ decomposes orthogonally:
$$ \|\boldsymbol{\Delta}\|_2^2 = \|\mathbf{P}_B \boldsymbol{\Delta}\|_2^2 + \|\mathbf{P}_B^\perp \boldsymbol{\Delta}\|_2^2 $$
Because $\mathbf{w}_B$ is a unit vector, the squared norm of the projection onto it is strictly the squared dot product:
$$ \|\mathbf{P}_B \boldsymbol{\Delta}\|_2^2 = \|\mathbf{w}_B (\mathbf{w}_B^\top \boldsymbol{\Delta})\|_2^2 = (\mathbf{w}_B^\top \boldsymbol{\Delta})^2 $$
Dividing the decomposition by the total variance $\|\boldsymbol{\Delta}\|_2^2$ yields the absorption ratio $\eta_B$:
$$ \frac{(\mathbf{w}_B^\top \boldsymbol{\Delta})^2}{\|\boldsymbol{\Delta}\|_2^2} + \frac{\|\mathbf{P}_B^\perp \boldsymbol{\Delta}\|_2^2}{\|\boldsymbol{\Delta}\|_2^2} = 1 \implies \eta_B = \frac{(\mathbf{w}_B^\top \boldsymbol{\Delta})^2}{\|\boldsymbol{\Delta}\|_2^2} = 1 - \frac{\|\mathbf{P}_B^\perp \boldsymbol{\Delta}\|_2^2}{\|\boldsymbol{\Delta}\|_2^2} $$
*Literature Grounding:* The decomposition of variance via orthogonal projection aligns with dimensional subspace constraints used in latent representation bottlenecks, such as Salzmann et al., "Factorized Orthogonal Latent Spaces" (AISTATS 2010).

**Derivation of $\eta_C$ (Endpoint 2D Subspace Absorption):**
$\mathbf{U} = [\mathbf{u}_1 \mid \mathbf{u}_2]$ is an orthonormal basis for $\text{span}\{\boldsymbol{\omega}_0, \boldsymbol{\omega}_k\}$ constructed via Gram-Schmidt. The projection of $\boldsymbol{\Delta}$ onto this 2D subspace is $\mathbf{U}\mathbf{U}^\top \boldsymbol{\Delta}$.
The squared norm of this projection evaluates to:
$$ \|\mathbf{U}\mathbf{U}^\top \boldsymbol{\Delta}\|_2^2 = (\mathbf{U}^\top \boldsymbol{\Delta})^\top \mathbf{U}^\top \mathbf{U} (\mathbf{U}^\top \boldsymbol{\Delta}) = \|\mathbf{U}^\top \boldsymbol{\Delta}\|_2^2 $$
Dividing by the total squared norm gives the fraction of variance located within the 2D plane:
$$ \eta_C = \frac{\|\mathbf{U}^\top \boldsymbol{\Delta}\|_2^2}{\|\boldsymbol{\Delta}\|_2^2} $$
Geometrically, if we define $\theta_1$ as the minimum angle between the 1D line defined by $\boldsymbol{\Delta}$ and the 2D plane defined by $\mathbf{U}$, the cosine of this angle is exactly the ratio of the projected length to the original length:
$$ \cos\theta_1 = \frac{\|\mathbf{U}\mathbf{U}^\top \boldsymbol{\Delta}\|_2}{\|\boldsymbol{\Delta}\|_2} \implies \cos^2\theta_1 = \frac{\|\mathbf{U}^\top \boldsymbol{\Delta}\|_2^2}{\|\boldsymbol{\Delta}\|_2^2} $$
*Literature Grounding:* This identity leverages the first principal angle between vector subspaces defined by Björck & Golub, "Numerical Methods for Computing Angles Between Linear Subspaces" (Math. Comp., 1973). Using a sub-space projector to enforce/measure hard orthogonality mimics metrics used in Eom & Ham, "Learning Disentangled Representation for Robust Person Re-identification" (NeurIPS 2019).

### 5.3 Isotropic-null calibration (peer-level critical note)

In $d_{\text{id}} = 2048$, any fixed $m$-dimensional subspace $\mathcal{S}$ captures only $m/d_{\text{id}}$ of the variance of a direction drawn uniformly from $\mathbb{S}^{d_{\text{id}}-1}$:

$$
\mathbb{E}_{\boldsymbol{\Delta} \sim \text{Unif}(\mathbb{S}^{d_{\text{id}}-1})}\!\left[\,\frac{\|\Pi_{\mathcal{S}}\boldsymbol{\Delta}\|_2^2}{\|\boldsymbol{\Delta}\|_2^2}\,\right] \;=\; \frac{m}{d_{\text{id}}} \;=\; \begin{cases} 1/2048 & m=1,\ \text{(B)} \\ 2/2048 & m=2,\ \text{(C)} \end{cases}
$$

by Stein (1956). The Finsler-absorbs-identity-slack claim therefore reduces to

$$
\text{(H3)}\quad \bar{\eta}_B \;\gg\; \tfrac{1}{d_{\text{id}}}, \qquad \bar{\eta}_C \;\gg\; \tfrac{2}{d_{\text{id}}}
$$

with an effect size that is orders of magnitude above the isotropic null, not merely statistically non-zero. Two null calibrations ship:

- *Analytic null* $m/d_{\text{id}}$.
- *Shuffle null* — recompute $\eta_B, \eta_C$ after randomly permuting the $(\boldsymbol{\omega}_0, \boldsymbol{\omega}_k)$ pairing against $\boldsymbol{\Delta}$; this preserves empirical marginals and tests only the *conditional* coupling between drift subspace and identity drift.

Both appear as horizontal reference bands on the paper figure.

### 5.4 Connection to shared–private latent decomposition

The projector $\mathbf{P}_C^\perp$ instantiates the orthogonality regulariser used in factorised-latent-space learning: Salzmann, Ek, Urtasun, Darrell (AISTATS 2010); Tsai et al. (ICLR 2019) in multimodal VAEs; and more recently enforced hard orthogonality in disentangled ReID (Eom & Ham, NeurIPS 2019; Jin et al., CVPR 2020). The novelty of the diagnostic is not the projector itself but the **pair-wise, post-hoc evaluation of how much of the empirical identity drift lives in the 2-D drift subspace spanned by the endpoint drifts** — a quantity that is *not* optimised during training and therefore provides a genuine test of the trained model's geometry. The Randers-midpoint coupling of $\mathbf{P}_B^\perp$ to paper eq. (4) makes this the first diagnostic that is tied end-to-end to the scoring function in use.

### 5.5 Cross-source extension

The derivation of 5.1–5.2 carries over to the cross-source pair $(p, s_q=1, \sigma_q)$ vs $(p, s_g=2, \sigma_g)$ without modification: redefine

$$
\boldsymbol{\Delta} \;:=\; \mathbf{z}^{\text{id}}_{(p, 2, \sigma_g)} - \mathbf{z}^{\text{id}}_{(p, 1, \sigma_q)}, \qquad
\mathbf{w}_B \;=\; \frac{\boldsymbol{\omega}_{(p,1,\sigma_q)} + \boldsymbol{\omega}_{(p,2,\sigma_g)}}{\big\|\boldsymbol{\omega}_{(p,1,\sigma_q)} + \boldsymbol{\omega}_{(p,2,\sigma_g)}\big\|_2},
$$

and the construction of $\mathbf{U}$ is analogous. For the paper-facing figure, the **same-source** variant is plotted against severity difference $|\sigma_q - \sigma_g|$ (diagonal is degenerate at $\boldsymbol{\Delta}=0$ and skipped); the cross-source variant is reported in a supplementary table.

---

## 6. Files to modify

| File | Change | New LoC (approx.) |
|---|---|---|
| [scripts/generate_toy_dataset.py](../scripts/generate_toy_dataset.py) | Select 2 source crops per PID (from distinct Market-1501 cameras); new filename convention; `TOY_DATASET_VERSION = "3.0"`; extended metadata | +40 |
| [scripts/toy_dataset_analysis.py](../scripts/toy_dataset_analysis.py) | Regex update; `ToyDataset` returns $(p, s, \sigma)$; three new analysis functions (`compute_cross_source_severity_retrieval`, `compute_same_source_severity_retrieval`, `compute_drift_cosine_matrix`, `compute_drift_orthogonal_projection`); four new plotting helpers; new CLI flags (`--omega-cos-transport {ambient,parallel}`, `--projection-report {B_primary,all}`); new JSON outputs | +300 |
| [scripts/toy_paper_figures.py](../scripts/toy_paper_figures.py) | New helpers for $5\times 5$ heat-map and projection-vs-severity twin-axis panel, reusing existing `FONT_*`, `LINE_W`, `viridis` conventions | +80 |
| [changelogs/toy_dataset_synthesis.md](toy_dataset_synthesis.md) | Append "v3.0 — two source crops" section with schema diff | +30 |

**No change** to [bau/models/model.py](../bau/models/model.py), [bau/loss/triplet.py](../bau/loss/triplet.py), or the training pipeline. All diagnostics are post-hoc on existing checkpoints.

---

## 7. Functions to reuse (verified locations)

- Feature extraction — `extract_features` [scripts/toy_dataset_analysis.py:160-176](../scripts/toy_dataset_analysis.py#L160-L176).
- Identity/drift split conventions — `compute_retrieval_metrics` [scripts/toy_dataset_analysis.py:236-297](../scripts/toy_dataset_analysis.py#L236-L297); mirror `q_id = F.normalize(q_feats[:, :identity_dim])` exactly.
- Distance functions — `euclidean_dist` [bau/loss/triplet.py:67-95](../bau/loss/triplet.py#L67-L95); `finsler_drift_dist(method="symmetric_trapezoidal")` [bau/loss/triplet.py:132-273](../bau/loss/triplet.py#L132-L273); analytical parallel-transport branch [bau/loss/triplet.py:242-257](../bau/loss/triplet.py#L242-L257) for Point 2 transport variant.
- Ranking — `cmc`, `mean_ap` [bau/evaluation_metrics/ranking.py:18-119](../bau/evaluation_metrics/ranking.py#L18-L119).
- Per-sample Gram–Schmidt reference — `resnet50_finsler.forward` [bau/models/model.py:354-356](../bau/models/model.py#L354-L356).
- Plot style conventions — `plot_quantitative` [scripts/toy_dataset_analysis.py:545-597](../scripts/toy_dataset_analysis.py#L545-L597) and the `FONT_*` / `LINE_W` constants in [scripts/toy_paper_figures.py](../scripts/toy_paper_figures.py).

---

## 8. Outputs

Under `results/toy_analysis_v3/`:

```
drift_identity_metrics.json                # superset of v2.0 keys; adds H1c Spearman
retrieval_metrics.json                     # superset; adds cross_source_severity_mAP[5,5,2 directions]
omega_cosine_blocks.json                   # block means + full NxN matrix (for sup.)
drift_orthogonal_projection.json           # per-(p,k) {delta_norm, eta_B, eta_C, parallel_B_term} + shuffle null
fig_cross_severity_mAP_dE.pdf              # 5x5 heat map, viridis
fig_cross_severity_mAP_dF.pdf              # 5x5 heat map, viridis
fig_cross_severity_asymmetry_A.pdf         # diverging colormap centred at 0
fig_omega_cosine_matrix.pdf                # full matrix, ordered (pid, source, sev)
fig_omega_cosine_blocks.pdf                # 8 block means as a bar chart
fig_drift_orthogonal_absorption.pdf        # eta_B and eta_C vs severity, with null band
```

---

## 9. Verification protocol (end-to-end)

1. **Regenerate ToyCorruption v3.0:**
   ```bash
   rm -rf examples/data/ToyCorruption/bounding_box_test/* examples/data/ToyCorruption/query/* examples/data/ToyCorruption/gallery/*
   python scripts/generate_toy_dataset.py \
     --source-dir /home/stud/leez/storage/user/reid/data/Market-1501-v15.09.15/bounding_box_test \
     --output-dir examples/data/ToyCorruption \
     --num-identities 50 --seed 42
   ```
   Check `metadata.json` has `toy_dataset_version == "3.0"` and 10 crops per PID.

2. **Run extended analyser:**
   ```bash
   python scripts/toy_dataset_analysis.py \
     --finsler-checkpoint logs/finsler_primary/job_1521403_primary_unified_1c_w0.1_driftInst/best.pth \
     --euclidean-checkpoint logs/AGReIDv2_sweep/job_1502418_dt_cuhk03_EucBaseline/best.pth \
     --dataset-dir examples/data/ToyCorruption \
     --output-dir results/toy_analysis_v3
   ```

3. **Assertions:**
   - $\text{mAP}_F[\sigma,\sigma]$ on the cross-source diagonal is within 5 pp of $\text{mAP}_E[\sigma,\sigma]$ at $\sigma=0$ (sanity).
   - $\max_{\sigma_q,\sigma_g} |A[\sigma_q,\sigma_g]| > 0.03$ (asymmetry signature of $d_F$, H1b).
   - Spearman in H1c $> 0.3$ with $p < 0.05$ over within-PID drift-pair distances.
   - Block-mean ordering in Point 2: $\bar{C}^{\text{(same pid, same sev)}} > \bar{C}^{\text{(diff pid, diff sev)}}$ by at least 0.1.
   - $\bar{\eta}_B, \bar{\eta}_C$ at $k=4$ exceed their analytic isotropic nulls ($1/2048, 2/2048$) by at least one order of magnitude *and* the shuffle null by at least a factor of 2.
   - If H3 fails, the paper's Finsler-absorbs-slack claim is not supported on ToyCorruption; report the negative result and escalate rather than patching the diagnostic.

---

## 10. Literature anchors

- **Randers / Finsler geometry.** Randers, G. *On an Asymmetrical Metric in the Four-Space of General Relativity*, Phys. Rev. 59, 195 (1941); Bao, Chern, Shen, *An Introduction to Riemann–Finsler Geometry*, Springer 2000, §11.1. Midpoint-drift is the flat-Randers template underlying paper eq. (1)–(4).
- **Principal angles between subspaces.** Björck, Å., Golub, G. H. *Numerical Methods for Computing Angles Between Linear Subspaces*, Math. Comp. 27, 579–594 (1973). Justifies $\|\mathbf{P}_C^\perp \boldsymbol{\Delta}\|_2 = \|\boldsymbol{\Delta}\|_2 \sin\theta_1$.
- **Concentration of measure on the sphere.** Stein, C. *Inadmissibility of the Usual Estimator for the Mean of a Multivariate Normal Distribution*, Proc. 3rd Berkeley Symp. (1956); Ledoux, M. *The Concentration of Measure Phenomenon*, AMS 2001. Justifies the $m/d_{\text{id}}$ isotropic null.
- **Shared-private latent decomposition.** Salzmann, Ek, Urtasun, Darrell, *Factorized Orthogonal Latent Spaces*, AISTATS 2010; Tsai, Liang, Zadeh, Morency, Salakhutdinov, *Learning Factorized Multimodal Representations*, ICLR 2019. Motivates the orthogonality interpretation of $\mathbf{P}_C^\perp$.
- **Disentangled ReID.** Eom, C., Ham, B. *Learning Disentangled Representation for Robust Person Re-identification*, NeurIPS 2019; Jin, X. et al. *Style Normalization and Restitution for Generalizable Person Re-identification*, CVPR 2020. Prior art for identity/nuisance factorisation in ReID.
- **Parallel transport on spheres.** Pennec, X. *Intrinsic Statistics on Riemannian Manifolds: Basic Tools for Geometric Measurements*, J. Math. Imaging Vis. 25 (2006). Closed-form formula underpinning the Point 2 parallel-transport variant.
- **ImageNet-C taxonomy (dataset lineage).** Hendrycks, D., Dietterich, T. *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*, ICLR 2019. Corruption pipeline anchor, unchanged from v2.0.

All six refs are canonical and verifiable on first pass; a `paper-rigorous-citation` or `cv-research-scientist` agent run is recommended before camera-ready.

---

## 11. Out-of-scope

- Retraining the Finsler checkpoint.
- Modifying `finsler_drift_dist` semantics — $\mathbf{P}^\perp$ is a *diagnostic*, not a new scoring function.
- Scaling to $N > 50$ in this pass (flagged as a recommendation, deferred).
- Multi-target extension beyond Market-1501 source crops — v3.0 remains Market-only.
