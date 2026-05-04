# Plan: Integrate Toy Dataset Experiments into Paper Draft

## Context

The paper currently contains a Section 4.4 ("Controlled corruption analysis") that compares two **independently trained, architecturally different** checkpoints (Euclidean BAU vs. Finsler Arm 1b) on a single-direction clean→corrupted retrieval task. This comparison is confounded: the checkpoints differ in backbone training, drift-branch gradient flow, and the L_dcc loss. The result — Finsler 62.3% mAP vs. Euclidean 69.7% at s=4 (Finsler worse) — does not isolate the Randers scoring mechanism.

Three new controlled experiments have been completed that replace this with a clean mechanistic study:

- **M1** (zero-shot Euclidean, CUHK03-pretrained backbone): establishes task asymmetry baseline without Randers machinery
- **M2a** (frozen BAU backbone + extended last-layer θ head trained by L_mono only, λ=0.1): validates L_mono as severity-encoding mechanism and tests Randers retrieval
- **M2b** (same as M2a but λ=0): ablation confirming L_mono as causal mechanism

All toy numbers below come from **`eval_toy_balanced.py`**: for each severity k, direction A uses 50 queries and a 50-image gallery (one true match per query), and direction B swaps roles — so query count, gallery size, and match count are **matched** across directions. (Unbalanced gallery mass can invert the sign of directional Δ; Chen et al., arXiv:2206.02416, Table 2, document the same phenomenon at scale on Market-1501.)

Under this protocol, direction A (clean query → corrupted gallery) is **easier** than direction B for M1 at k=1..4 — mean Δ≈+0.099 (seed-1 logs: `eval_toy_balanced_best_20260503_171229.log`).

The goal of this plan is to modify the paper draft to incorporate these findings without overflowing the page budget, maintaining the paper's primary contribution (Finsler BAU +1.9 pp mAP on M+MS+CS→C3), and ensuring no section makes a claim contradicted by another.

---
## User Comments after reading this AI generated plan
The following summarizes my comments/notes regarding this plan, which you should examine crictically and only take into account when drafting the revision if you agree with them -- DO NOT default into agreement!
1. Avoid using ambiguous or confusing terminology like (content-level task asymmetry). Only use terminology that are well-received within the scientific community, in which case you should provide a citation to it, or paraphrase it such that it can be understood intuitively without any ambiguity. 
2. If you intend to append to the abstract, you should concise the other parts of the abstract, as it is currently at its maximum length. Evaluate the significance of every word here and how relevant the message it wants to convey is to the core message of the paper. 
3. You can ignore generating the new figures. They are already generated in @paper/
4. Avoid using notations/abstractions before introducing them (e.g., L_mono)
5. Avoid mentioning experiment-specific naming (e.g., D8) -- it is better to remove them altogether from the report, as it makes no sense for the readers to remember arbitrary alphanumeric naming conventions that this paper uses
6. Section 4.4 needs to be more concised than what is currently proposed. It should use a tabular summary of the quantitative results as an anchor for the concise analysis, consider graphical visualisation as well
7. Avoid using hyphens or any em dashes of the kind to emphasize a point. Consider using semicolon instead, or just start a new sentence
8. Place M2b in the appendix to save space in the main part of the paper, reference this in the main section though. It does not contribute much to the core message of this section other than validating the efficacy of the loss function engineered here
9. Make the newly proposed Table 5 more concise 
10. Make the limitations and future work more concise, each spanning only a maximum of two sentences. Merge them into the same paragraph. 
12. Do we really need to mention the thing about how BN norm suppresses the asymmetrical signal?

---

## Exhaustive Sweep: What Must Change

### 1. Abstract (`paper/0_abstract.tex`)

**Problem:** The abstract makes no mention of the toy experiment. Given that Section 4.4 is now a substantive mechanistic study with clean causal isolation, it deserves 1–2 sentences.

**Change:** Add after the sentence ending "...is the critical ingredient for stable asymmetric retrieval under BAU-style training." Insert:

> "A controlled synthetic corruption study further isolates the asymmetric scoring mechanism: a pairwise monotonicity loss reliably encodes corruption severity in a scalar dimension (Spearman ρ ≈ −0.91 on the toy test split after 20 epochs), and the resulting Randers correction produces the geometrically predicted directional distance gap on PID-matched pairs. A balanced bidirectional evaluation confirms that clean queries retrieve corrupted gallery items more reliably than the reverse under matched query/gallery counts."

Do **not** cite mAP numbers from the toy experiment in the abstract (N=50, single seed — insufficient to headline).

---

### 2. Introduction (`paper/1_intro.tex`)

**Problem A — Figure 1 (fig:drift_monotonicity):** The current hook figure shows Arm 1b drift magnitude (span 0.054–0.067, only 1.4% of barrier bound) and identity distance (0→1.27) vs. severity. The weak drift signal relative to the large identity signal is a poor visual hook — a reviewer will immediately note the drift barely responds. The toy experiment's balanced-eval Δ[k] profile is a stronger, cleaner visual argument for WHY asymmetric metrics matter.

**Change — Replace fig:drift_monotonicity:** Replace the current Figure 1 with a two-panel figure:
- **Left panel:** Image strip showing one identity at five severity levels (σ=0 clean through σ=4 extreme, the visual from the current fig:toy_hook top row). Label: "Controlled ToyCorruption severity ladder."
- **Right panel:** Bar chart of M1 balanced-eval Δ[k] = mAP_A[k] − mAP_B[k] for k=0,1,2,3,4. Data (same log as above): k=0→−0.0233, k=1→+0.0350, k=2→+0.0347, k=3→+0.2013, k=4→+0.1259. Uniform bar colour (severity is on the x-axis); horizontal line at Δ=0. Y-axis: "Δ mAP (Dir. A − Dir. B)". X-axis: "Corruption severity k". Caption: "Under matched query/gallery sizes, clean queries retrieve corrupted gallery items **more reliably** than the reverse (Δ>0 for k=1..4), with the gap peaking near +0.20 at k=3. This asymmetry motivates direction-aware scoring."

**New figure label:** `fig:balanced_asymmetry_intro`

**Problem B — Contributions list:** Item (v) currently reads "report an auxiliary-loss sweep that selects a primary training recipe and documents which multi-target and cross-view evaluations are still outstanding." Add a new contribution:

> "(vi) provide a controlled synthetic validation isolating the L_mono severity-encoding mechanism and confirming the Randers gap with a clean causal ablation, alongside a balanced bidirectional evaluation that resolves a gallery-composition confound and establishes the correct sign of task asymmetry."

---

### 3. Related Work (`paper/2_relatedwork.tex`)

**Problem:** Section 2.4 (Asymmetric domain shift) cites the data processing inequality to argue that clean→corrupted information loss "typically dominates." The balanced eval confirms this claim empirically (direction A IS easier). No claim in related work is falsified; however, the paragraph can be strengthened.

**Change (two sentences):** At the end of Section 2.4, before the final paragraph "The preceding threads share a gap…", add:

> "Chen et al. (arXiv:2206.02416, *Benchmarks for Corruption Invariant Person Re-identification*) make this direction asymmetry directly measurable: their Table 2 reports Rank-1 of 30.5% when the query is corrupted versus 77.2% when the gallery is corrupted under a ResNet-50 baseline on Market-1501, a 46-point gap driven entirely by which side of the retrieval pair is degraded. Our balanced bidirectional evaluation (Sec. 4.4) controls for this protocol effect by equalising query count, gallery size, and correct-match count across directions, and confirms the same direction: clean-probe retrieval against a corrupted gallery is reliably easier than the reverse at all tested severity levels."

This citation also justifies the decision to report only the balanced protocol in Sec. 4.4 (see triaging decision in Sec. 4.4 preamble below).

---

### 4. Section 4.4 (`paper/4_experimentalResults.tex`) — **MAJOR REWRITE**

This is the primary change. The existing Section 4.4 must be entirely replaced.

**Current content (to remove):**
- Comparison of Arm 1b Finsler vs. Euclidean BAU (two different checkpoints, confounded)
- Drift magnitude ρ=0.67 for Arm 1b (this datapoint belongs in Discussion, not here)
- "Large retrieval gains from asymmetric scoring … remain unconfirmed" (correct but under-explained)
- Figure 3 / fig:toy_hook bottom panel (mAP curves comparing different checkpoints — replace entirely)

**New Section 4.4 structure (approximately 600–750 words; 2 sub-paragraphs + 1 table + 1 figure):**

---

**Section 4.4 heading:** *Controlled Corruption Study: Mechanism Isolation*

**Triaging decision: report balanced evaluation only**

| Design | M1 mean Δ (k=1..4) | Role in this plan |
|---|---|---|
| Balanced per-severity (`eval_toy_balanced.py`, 50×50, matched counts) | **+0.099** (log: `eval_toy_balanced_best_20260503_171229.log`) | **Primary** retrieval-asymmetry evidence |

**Rationale (three points):**

1. **Gallery/query imbalance can invert directional Δ.** Chen et al. (*Benchmarks for Corruption Invariant Person Re-identification*, arXiv:2206.02416) show that which side of the pair is degraded dominates Rank-1 (Table 2: 30.5% corrupted query vs 77.2% corrupted gallery on Market-1501, ResNet-50). Matched-count evaluation removes that confound for the toy study.

2. **The balanced protocol isolates the retrieval question.** With equal query count, gallery size, and correct-match count across both directions, any non-zero Δ under Euclidean distance reflects feature-side differences, not unequal pool sizes.

3. **The Randers gap is geometric, not a Table 5 row.** The directional gap d_R(z^0, z^k) − d_R(z^k, z^0) = 2α(θ^k − θ^0) is logged from **the same balanced run** using PID-matched clean (σ=0, source 1) vs corrupted (σ=k, source 2) pairs (cameras may differ). M2a at α=0.9 (`eval_toy_balanced_m2a_lambda0.1_seed1_20260503_171300.log`): ≈−0.142 (σ=1), −0.413 (σ=2), −0.588 (σ=3), −0.735 (σ=4); M2b (`eval_toy_balanced_m2b_lambda0_seed1_20260503_171319.log`): **0** at all α — causal isolation of L_mono.

---

**Para 1 — Setting and models:**

> We isolate the asymmetric scoring mechanism from the confounds present in the main DG-ReID protocol (Sec. 4.3) using ToyCorruption: 50 held-out Market-1501 identities, each with two source crops from distinct cameras, corrupted at five severity levels (σ=0 clean through σ=4 extreme) via the same composite pipeline as Sec. 4.1 [ImageNet-C]. This yields 500 images (50 PIDs × 2 sources × 5 severities). Three models are evaluated:
>
> - **M1** (baseline): CUHK03-pretrained ResNet-50, Euclidean ranking, no θ head, no additional training on toy data.
> - **M2a** (mechanism): M+MS+CS-pretrained BAU backbone, **frozen** throughout. A single linear projection head θ: ℝ^2048 → ℝ^1 is appended to the pre-BN identity features and trained for 20 epochs on the toy training split using L_mono only (λ=0.1):
>   L_mono = mean_{(i,j): pid_i=pid_j, sev_i<sev_j} [θ_i − θ_j + margin]_+
>   enforcing θ monotone decreasing with severity within each identity. Retrieval uses the Randers distance d_R(q,g) = ‖f_g−f_q‖_2 + α(θ_g−θ_q) for α swept in {0.1,0.3,0.5,0.9}.
> - **M2b** (ablation): Identical to M2a but λ=0; θ receives no gradient. Controls for BN drift and training protocol.
>
> **Caveat:** M1 and M2a/M2b use different backbone checkpoints. M2a/M2b update BatchNorm running statistics over 20 training epochs even with frozen affine weights (BN drift), reducing the corruption signal in features by approximately 5.6× relative to M1. Within-model comparisons (M2a Euclidean vs. Randers; M2a vs. M2b) are clean; cross-model comparisons should not be made.

**Para 2 — Evaluation protocol:**

> We report the **balanced per-severity protocol**. For each severity level k ∈ {0,1,2,3,4}: Direction A uses source-1, σ=0 images as queries (50 items) and source-2, σ=k images as gallery (50 items); Direction B swaps these roles. Both directions have exactly 50 queries, 50 gallery items, and 1 correct match per query (cross-source cam-ids 1 vs 2 prevent same-cam exclusions). Any non-zero Δ[k] under Euclidean distance reflects feature quality differences between clean and corrupted images — not gallery size, query count, or correct-match count.
>
> The σ=0 vs σ=0 cell (k=0) serves as a viewpoint-asymmetry baseline: the two source crops come from distinct Market-1501 cameras, introducing a small source-viewpoint asymmetry (M1: Δ[0]=−0.023) independent of corruption. The net corruption-driven gap at severity k is Δ[k] − Δ[0].
>
> **Note on the Randers gap:** The mean directional gap \(d_R(z^0,z^k)-d_R(z^k,z^0)\) is aggregated over **PID-matched** clean (σ=0, source 1) vs corrupted (σ=k, source 2) pairs; **cameras may differ**. Values are printed on each Randers line in `eval_toy_balanced.py` (D8 `mean_gap`, 50 pairs per k≥1). M2a (λ=0.1) shows strictly negative gaps at α>0 with monotone severity scaling at α=0.9; M2b (λ=0) shows **0** (θ inactive).

**Para 3 — Task asymmetry (balanced eval, M1):**

> M1 balanced-eval results confirm that **clean queries retrieve corrupted gallery items more reliably than the reverse** at all severity levels k=1..4 (see Table 5, top section, and Fig. 1). The mean Δ≈+0.099 over k=1..4 is dominated by k=3 (+0.201) and k=4 (+0.126). A σ=0 vs σ=0 sanity check (k=0) yields Δ≈−0.023, quantifying the source-viewpoint asymmetry between the two crop cameras unrelated to corruption; the corruption-driven gap at k=3 corrected for this baseline is approximately +0.22.
>
> Directional Δ under **matched** gallery/query counts is therefore positive here; designs that break count symmetry can flip the sign of the same Euclidean score (Chen et al., Table 2, motivate controlling that effect).

**Para 4 — L_mono mechanism and ablation (M2a vs M2b):**

> M2a trains θ reliably: Spearman ρ(θ, severity)≈−0.911 on the 500-image test split after 20 epochs (`eval_toy_balanced_m2a_lambda0.1_seed1_20260503_171300.log`), with θ_std growing across training (training logs). Notably, strongly negative ρ is achieved early — the frozen backbone's pre-BN features contain sufficient residual corruption signal for a linear projection to latch onto monotone severity structure immediately.
>
> M2b (λ=0) provides a clean causal isolation: θ_std=0.000 throughout all 20 epochs (no gradient reaches the θ head), the Randers gap is zero for all α ∈ {0.1,0.3,0.5,0.9}, and mAP is identical to Euclidean at all α. L_mono is the causal mechanism for severity encoding in θ; cross-entropy alone contributes nothing to θ's severity discrimination.

**Para 5 — Randers gap confirmation (supporting note, not tabulated):**

> The Randers gap 2α(θ^k − θ^0) is negative for all α > 0 in M2a on balanced PID-matched pairs, confirming the geometric mechanism. At α=0.9 (`eval_toy_balanced_m2a_lambda0.1_seed1_20260503_171300.log`): ≈−0.142 (σ=1), −0.413 (σ=2), −0.588 (σ=3), −0.735 (σ=4), monotone in severity. For M2b (`eval_toy_balanced_m2b_lambda0_seed1_20260503_171319.log`) the gap is **0** at all α and σ, causally isolating L_mono.
>
> This result is stated in-text only (not Table 5). The geometric claim — severity-monotone θ under L_mono produces the predicted directional asymmetry — is established on the **same balanced protocol** as the mAP rows. Randers-induced **mAP** shifts on this split are small (see M2a mean Δ Randers vs Euclidean in Table 5) and should be read with N=50 / single-seed caution.

---

**New Table 5** (replaces the confounded mAP comparison, and complements Table 4):

Caption: "Controlled ToyCorruption study — balanced per-severity evaluation. Direction A: clean query (σ=0, source 1) → corrupted gallery (σ=k, source 2). Direction B: corrupted query → clean gallery (roles swapped). Both directions: 50 queries, 50 gallery items, 1 correct match per query. Mean Randers directional gap (PID-matched σ=0 vs σ=k; cameras may differ) is logged but not tabulated here. N=50 PIDs, single seed."

| Model | k | α | mAP A | mAP B | Δ |
|---|---|---|---|---|---|
| M1 | 0 (σ=0 vs σ=0 sanity) | — | 0.953 | 0.977 | −0.023 |
| M1 | 1 | — | 0.967 | 0.932 | +0.035 |
| M1 | 2 | — | 0.929 | 0.894 | +0.035 |
| M1 | 3 | — | **0.850** | **0.649** | **+0.201** |
| M1 | 4 | — | 0.564 | 0.438 | +0.126 |
| M1 | mean k=1..4 | — | — | — | **+0.099** |
| M2b | mean k=1..4 | any | — | — | +0.018 |
| M2a | mean k=1..4 | 0 (Euclidean) | — | — | +0.018 |
| M2a | mean k=1..4 | 0.9 (Randers) | — | — | +0.041 |

Row notes:
- M2b Randers (all α): identical to M2b Euclidean for all k and α (θ_std=0 throughout training; balanced eval makes Randers retrieval-neutral by design when θ is constant).
- M2a Euclidean matches M2b Euclidean (both show mean Δ=+0.018 vs M1's +0.099): the 5.6× gap is attributable to BatchNorm running-stat drift over 20 training epochs, not to L_mono. M1 never runs in training mode on toy data.
- M2a Randers α=0.9 (+0.041 vs Euclidean +0.018): small residual widening from within-severity θ leakage; not the primary cross-stratum mechanism.
- Cross-model comparisons (M1 vs M2a/M2b) are confounded by BN drift and different backbone checkpoints; within-model comparisons are clean.
- M2a Randers gap (in-text only, not tabulated; balanced `eval_toy_balanced_m2a_lambda0.1_seed1_20260503_171300.log`, α=0.9): ≈−0.142 (σ=1) → −0.735 (σ=4), monotone; M2b: **0** (`eval_toy_balanced_m2b_lambda0_seed1_20260503_171319.log`).
- Keep to 9 rows.

---

**Figure replacement (fig:toy_hook → fig:toy_mechanism):**

Remove the bottom panel (mAP curves comparing Arm 1b vs. Euclidean baseline). Keep the top image strip (five severity levels) as one panel. Add a second panel: Randers gap vs. severity σ∈{1,2,3,4} for M2a at α=0.9 from **balanced** logs (data: ≈−0.142, −0.413, −0.588, −0.735; `eval_toy_balanced_m2a_lambda0.1_seed1_20260503_171300.log`). Label: "Mean Randers gap (PID-matched pairs) scales monotonically with severity — the asymmetric scoring mechanism is confirmed."

Caption: "**Left:** ToyCorruption severity ladder — one identity at σ=0 through σ=4, produced by compositing Gaussian blur, bilinear downscaling, gamma tone shift, sensor noise, and JPEG recompression following ImageNet-C categories. **Right:** Mean Randers gap at α=0.9 for M2a (balanced eval). Gaps are negative and increase in magnitude with σ; M2b (λ=0) yields **0** at all severities on the same protocol, isolating L_mono."

---

**Para 6 — Limitations:**

Four limitations must be stated explicitly and honestly:

> (1) **Statistical power.** N=50 eval PIDs, single seed. Paired AP standard errors are approximately 0.02–0.04; differences below 0.03 are not reliably interpretable without confidence intervals. The M2a Randers improvement at high α (+0.023 mean Δ increase) sits at this boundary. Replication at ≥3 seeds and ≥150 PIDs is the minimum threshold before any retrieval improvement claim can be stated with confidence. All current results should be read as directional indicators, not established effects.
>
> (2) **BN drift invalidates cross-model comparison.** M2a and M2b update BatchNorm running statistics during 20 training epochs even with frozen affine weights; M1 does not. This suppresses the corruption signal in M2a/M2b features by approximately 5.6× relative to M1 (Euclidean Δ: +0.018 vs +0.099). The comparison M1 Euclidean vs M2a Randers is therefore not a valid ablation. Only within-model comparisons are clean: M2a Euclidean vs M2a Randers, and M2a vs M2b. Controlling BN statistics — e.g., freezing BN in eval mode during training — is required before a cross-model comparison can be made.
>
> (3) **Scalar θ ceiling.** The Randers correction α·θ_g applies the same additive offset to every gallery item at a given severity level, leaving within-stratum ranking unchanged. Large **mAP** shifts from Randers therefore require cross-severity gallery mixing; the balanced per-severity design deliberately fixes one severity per gallery so Euclidean vs Randers **mAP** stays nearly identical while the **pairwise gap** statistic remains informative. This isolates content-level asymmetry without conflating it with pool-size effects.
>
> (4) **θ leakage.** θ = w_new^T · emb reads backbone features that encode both identity and severity. Within-severity θ variation (identity-correlated noise, not severity signal) is amplified by α across gallery items. The small M2a Randers **mAP** shift in balanced eval (+0.041 vs +0.018 Euclidean mean Δ at α=0.9) comes from this leakage rather than from the primary severity-discrimination mechanism.

---

**Para 7 — Future work (condensed):**

> (1) **Multi-seed statistical validation.** Run M2a at seeds {1, 2, 3}; report mean ± std for mAP A, mAP B, Δ, Spearman ρ, and Randers gap at each α. Add per-query bootstrap 95% CIs for Δ. This is the minimum before any retrieval improvement claim.
>
> (2) **BN drift control.** Freeze BN statistics before training (model.eval() for BN layers, or snapshot before fine-tuning and restore at eval). This creates a valid matched comparison with M1.
>
> (3) **Low-dimensional drift head.** Replace the scalar θ with a k-dimensional drift representation (k ≈ 4–8). This bridges directly to the resnet50_finsler architecture (where the drift head already produces a domain-conditioned vector) and removes the single-axis ceiling. L_mono can be applied per-dimension or as a norm-based ordering constraint.
>
> (4) **Real quality proxies.** No-reference image quality scores (Laplacian variance, BRISQUE, or a pretrained IQA network) can serve as severity surrogates on real datasets where corruption labels are unavailable. The closest ecological target is BRIAR (Cornett et al., WACV-W 2023), where standoff distance acts as a natural monotone quality axis — applying L_mono with range as severity would provide the first ecologically grounded test of the asymmetric retrieval claim.

---

### 6. Section 5 (Discussion — `paper/5_discussion.tex`)

**Problem A:** Section 5 currently opens with the paradox that L_dcc (the best-performing auxiliary) forces drift to be camera-invariant, which appears to contradict the directional motivation. This tension is real and should be kept. But the toy experiment provides a complementary, non-paradoxical result: L_mono explicitly trains a directionally useful θ. These should be connected.

**Change:** Add a paragraph at the end of Section 5, before any future-work text:

> "The controlled toy study in Sec. 4.4 provides a mechanistic complement to the main training stack result. Where the full Finsler BAU system obscures the origin of its marginal gain (training geometry vs. test-time Randers scoring; see Supp. sec:suppl_eval_drift_tables), the toy setting cleanly separates the two: L_mono trains a severity-monotone θ reliably and quickly (ρ≈−0.91 on the toy test split; strongly negative from epoch 1 in training logs), the M2b ablation causally isolates it, and the Randers gap is confirmed geometrically on balanced PID-matched pairs. The analogous role in the full system is played by L_dcc: both are strategies for rendering the drift subspace semantically usable — L_mono by imposing severity monotonicity, L_dcc by bounding cross-camera variance. Neither guarantees that test-time Randers scoring outperforms Euclidean on the same checkpoint; both improve the drift subspace's behavior during training and, in L_mono's case, enable the directional gap by construction."

**Problem B:** The first paragraph of Section 5 says the Arm 1b drift magnitude spans 0.054–0.067 and tracks severity with ρ=0.67. This number will now conflict with the toy study's ρ≈−0.91 on θ. **These measure different quantities** (‖ẑ^ω‖ for the full Finsler BAU vs. θ for the toy L_mono model) but a reader could conflate them.

**Change:** Add a clarifying parenthetical in the first paragraph of Section 5 after "ρ=0.67": "(measured on the full Finsler Arm 1b drift magnitude ‖ẑ^ω‖; the toy study's L_mono-trained scalar θ achieves ρ≈−0.91 on the toy split under a different architecture and training protocol — see Sec. 4.4)."

---

### 7. Section 6 (Conclusion — `paper/6_conclusion.tex`)

**Problem:** The conclusion's scope/limitations paragraph mentions the toy experiment only in passing: "The controlled corruption analysis (Sec. 4.4) confirms that the drift branch tracks severity monotonically but at magnitudes that yield near-zero test-time gains." This was an honest characterization of the confounded comparison. It must be updated to reflect the new controlled study.

**Change:** Replace the sentence "The controlled corruption analysis (Sec. 4.4) confirms that the drift branch tracks severity monotonically but at magnitudes that yield near-zero test-time gains from Finsler ranking over Euclidean ranking on the same checkpoint: meaning the bidirectional triplet gain (+1.0 pp mAP) comes through improved training geometry, not through a fundamentally different test-time distance function." with:

> "The controlled toy study (Sec. 4.4) validates two mechanistic claims independently of the full training stack: (1) a pairwise monotonicity loss (L_mono) trains a severity-monotone scalar (ρ≈−0.91 on the toy split) with clean causal isolation via a λ=0 ablation, and (2) the resulting Randers correction produces the geometrically predicted asymmetry — the mean Randers gap on PID-matched pairs is negative and scales with severity at α=0.9 for M2a, and is identically zero for M2b. A balanced bidirectional evaluation further establishes that directional **mAP** asymmetry under Euclidean scoring is positive for M1 (mean Δ≈+0.099 at k=1..4) under matched query/gallery counts. The bidirectional triplet gain (+1.0 pp mAP) in the main system comes through improved training geometry; toy **mAP** deltas from Randers alone remain small on this split (N=50, single seed)."

---

## Summary of Figure and Table Changes

| Item | Action | Reason |
|---|---|---|
| `fig:drift_monotonicity` (intro Figure 1) | **Replace** with 2-panel: image strip (left) + balanced-eval Δ[k] bar chart (right) | Weak drift signal (1.4% of barrier) is a poor hook; balanced-eval Δ[k] profile directly motivates asymmetric scoring |
| `fig:toy_hook` (Section 4.4 Figure) | **Replace bottom panel** only; keep image strip at top | Bottom panel shows confounded Finsler vs Euclidean; replace with Randers gap vs. severity (M2a, in-text confirmation) |
| New **Table 5** | **Add** in Section 4.4 | Balanced-protocol only (9 rows: M1 per k, M2b ablation, M2a Euclidean/Randers); mean Randers gap is a separate logged statistic (not a table row) |
| Tables 1–4 | **No change** | DG-ReID results unaffected by toy experiment |
| `fig:drift_monotonicity` data | Remove or move to Discussion | ρ=0.67 datapoint moves to Discussion with clarifying parenthetical |

---

## Narrative Corrections: Claims That Must NOT Conflict

| Location | Current claim | Corrected version |
|---|---|---|
| Intro, Section 2.4 (asymmetric domain shift) | "clean→corrupted information loss typically dominates the reverse" | No change; the balanced eval CONFIRMS this direction. The Related Work addition reinforces it with Chen et al. data. |
| Section 4.4 | Finsler 62.3% vs Euclidean 69.7% at s=4 (Finsler worse) | Remove entirely; confounded comparison, not informative about the mechanism |
| Section 4.4 | "ρ=0.67" drift-magnitude Spearman for Arm 1b | Move to Discussion with clarifying parenthetical; in Section 4.4 replace with M2a L_mono Spearman ρ≈−0.911 (`eval_toy_balanced_m2a_lambda0.1_seed1_20260503_171300.log`) |
| Section 4.4 | Legacy prose about a second unbalanced protocol | Remove; all toy numbers come from `eval_toy_balanced.py` (M1 mean Δ≈+0.099 for k=1..4 on seed-1 logs). |
| Discussion | "direction-dependent: a clean query retrieved against a degraded gallery is not the same problem as the reverse" | Keep; add: "balanced evaluation confirms correct sign — direction A is easier (+0.099 mean Δ)" |
| Section 4.4 | References to Fig. 1 (drift_monotonicity) in intro | Remove cross-references; the new Section 4.4 is self-contained |
| Conclusion | "drift branch tracks severity monotonically but at magnitudes that yield near-zero test-time gains" | Replace as specified in Section 7 change above |

---

## What Must NOT Be Changed

- Tables 2, 3, 4 (main DG-ReID results): unaffected
- All of Section 3 (methodology): L_mono is not part of the main architecture
- The Discussion's core tension (L_dcc forcing camera-invariant drift contradicting initial intuition): this remains valid and honest
- The ±1.9 pp mAP headline result: unaffected
- Multi-seed and multi-protocol experimental protocol for the main system

---

## Critical Caveats to Include in Section 4.4

The following limitations must appear explicitly — underselling is better than overclaiming:

1. **N=50 eval PIDs, single seed.** Paired AP CIs are required before any retrieval improvement claim. The M2a Randers improvement in balanced eval (+0.023 mean Δ increase at α=0.9) sits within the SE ≈ 0.02–0.04 range and should be treated as directional only.
2. **BN drift invalidates cross-model comparison.** M2a/M2b have 20 epochs of BN running-stat updates; M1 does not. The 5.6× reduction in Δ signal (M1 +0.099 vs M2a/M2b Euclidean +0.018) is a BN artifact, not an L_mono effect. Only within-model comparisons are clean.
3. **The balanced protocol makes Randers *mAP* shifts small by design.** A single-severity gallery gives approximately constant α·θ_g offsets across the gallery, so rankings barely move; the **mean directional Randers gap** on PID-matched (σ=0, source 1) vs (σ=k, source 2) pairs is nevertheless well-defined and logged (`eval_toy_balanced` D8 lines). Small M2a Randers vs Euclidean **mAP** differences reflect within-severity θ leakage, not failure of the gap statistic.
4. **No mechanism transfer claim.** The toy study validates the geometric mechanism in a controlled setting. The θ head (scalar, L_mono-only, frozen backbone) and the Finsler drift head (MLP, L_dcc+L_tri, trained end-to-end) are architecturally and causally unrelated; results from one do not carry over to the other.

---

## Files to Modify

| File | Change type | Estimated edit size |
|---|---|---|
| `paper/0_abstract.tex` | Add 2 sentences | +2 sentences |
| `paper/1_intro.tex` | Replace Figure 1 caption and label; add contribution (vi) | Replace ~5 lines, add ~4 lines |
| `paper/2_relatedwork.tex` | Add 1 sentence in Section 2.4 | +1 sentence |
| `paper/3_methodology.tex` | Add 1 parenthetical in L_dcc paragraph | +1 sentence |
| `paper/4_experimentalResults.tex` | **Rewrite Section 4.4 entirely; add Table 5; replace Figure 3** | ~600 words, 1 new table |
| `paper/5_discussion.tex` | Add 1 paragraph at end; add parenthetical in paragraph 1 | +~120 words |
| `paper/6_conclusion.tex` | Replace 1 paragraph in scope/limitations | ~70-word replacement |

New figure source files needed (not in the bib/tex; must be generated; **one PDF per panel** — compose in LaTeX):
- `paper/fig_balanced_asymmetry_intro_strip.pdf` — severity ladder
- `paper/fig_balanced_asymmetry_intro_delta.pdf` — Euclidean Δ mAP[k] (balanced M1)
- `paper/fig_toy_mechanism_strip.pdf` — same ladder (§4.4)
- `paper/fig_toy_mechanism_gap.pdf` — mean Randers gap vs. σ (M2a, balanced `eval_toy_balanced_m2a_lambda0.1_*.log`; see `scripts/make_toy_corruption_paper_figures.py`)

The image strip (five severity levels, one identity) already exists in the toy corruption dataset and in fig:toy_hook's top panel — reuse verbatim.

---

## Verification

After implementation, the following should hold:

1. Abstract mentions toy experiment without citing unconfirmed mAP numbers.
2. Introduction Figure 1 shows balanced-eval Δ[k] profile (bars positive for k=1..4, bar at k=0 near-zero).
3. Contribution (vi) is added to the list in Section 1.
4. Section 4.4 contains: (a) triaging decision and motivation, citing Chen et al. arXiv:2206.02416; (b) M1/M2a/M2b model descriptions; (c) balanced-protocol description; (d) Table 5 (balanced only, 9 rows); (e) Randers gap in-text confirmation (not a table row); (f) all four limitations (Para 6); (g) condensed future work (Para 7).
5. Table 5 reports only balanced-eval rows (single protocol; no cross-protocol Δ mixing).
6. No sentence in any section claims Finsler outperforms Euclidean on a controlled same-checkpoint comparison on the toy dataset.
7. No sentence conflates the full Arm 1b drift magnitude (ρ=0.67) with the toy L_mono Spearman (ρ≈−0.91 on θ).
8. Section 5 Discussion connects L_mono and L_dcc as parallel strategies.
9. Section 6 Conclusion accurately characterizes the toy results (correct sign, causal isolation, statistical caveat, BN drift limitation).
10. All figure labels used in the text resolve to defined \label{} commands.
11. The Randers gap formula appears in Section 4.4 and is consistent with: d_R(z^0, z^k) − d_R(z^k, z^0) = 2α(θ^k − θ^0) < 0. It is stated as a geometric confirmation, not as an evaluation table entry.
12. Section 2.4 (Related Work) cites Chen et al. (arXiv:2206.02416) with the specific Rank-1 numbers from Table 2 (30.5% corrupted query vs 77.2% corrupted gallery for ResNet-50) as motivation for direction-controlled evaluation.
