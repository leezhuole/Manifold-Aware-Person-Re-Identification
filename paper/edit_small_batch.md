# Edit-small batch — single-line wording / notation fixes

**Date:** 2026-05-04
**Scope:** Items classified as Edit-small in `paper/comments_triage_assessment.md`. These are unambiguous, single-line fixes (notation, hyphenation, typo, jargon swap) ready to apply once approved.
**Status:** Awaiting batch approval. NO `paper/*.tex` edits applied yet.

Each row gives: comment id, file:section, current text → proposed text, justification.

---

## Section 0 — Abstract

| ID | File | Current | Proposed | Justification |
|----|------|---------|----------|---------------|
| A1 | `paper/0_abstract.tex` L2 | "formulate a unified asymmetric scoring layer with norm-constrained drift" | "formulate a Randers/Finsler-style asymmetric distance with norm-constrained drift" | "unified asymmetric scoring layer" reads as marketing; the next sentence already says "Randers/Finsler-style". |
| A2 | `paper/0_abstract.tex` L2 | "On M+MS+CS$\to$CUHK03, five-seed mean mAP is $43.8\pm0.3\%$" | "On Market-1501 + MSMT17 + CUHK-SYSU $\to$ CUHK03, the five-seed mAP is $43.8\pm0.3\%$" | Abbreviations not yet introduced; per `publication_facts.md` Reported_metrics_with_provenance convention to spell out source/target. |
| A4 | `paper/0_abstract.tex` L2 | "A controlled corruption experiment isolates the mechanism" | "A controlled corruption experiment isolates the asymmetric scoring mechanism" | "the mechanism" is vague — concrete referent supplied. |
| A5 | `paper/0_abstract.tex` L2 | "The Randers correction produces" | "The asymmetric distance term produces" | "Randers correction" undefined in the abstract; main paper introduces it in Sec. 4.5. |

---

## Section 1 — Introduction

| ID | File | Current | Proposed | Justification |
|----|------|---------|----------|---------------|
| I2 | `paper/1_intro.tex` L15 | "It treats probe→gallery and gallery→probe as interchangeable" | "It treats probe-to-gallery and gallery-to-probe scoring as interchangeable" | Arrow notation in prose is unprofessional (per `review_style_directives.md` general style). |
| I3 | `paper/1_intro.tex` L17 | "a failure mode analyzed in Sec.~\ref{subsec:interference_analysis} and a cautionary datapoint for ``hard'' disentanglement" | "a failure mode for hard disentanglement under BAU-style multi-source training, analyzed in Sec.~\ref{subsec:interference_analysis}" | "cautionary datapoint" is filler; reorder clauses for clarity. |
| I4 | `paper/1_intro.tex` L17 | "retaining a single continuous feature map while letting asymmetry enter through the scoring function" | "retaining a single shared feature extractor while letting asymmetry enter through the scoring function" | "continuous feature map" is ambiguous — the intent is "single shared backbone-derived embedding". |
| I-extra | `paper/1_intro.tex` L19 | (commented-out 6-bullet contributions list) | DELETE | Duplicate of L21 active list; dead code. |

---

## Section 2 — Related Work

| ID | File | Current | Proposed | Justification |
|----|------|---------|----------|---------------|
| R1 | `paper/2_relatedwork.tex` L7 | `\subsection{Retrieval geometry}` | `\subsection{Distance functions for ReID retrieval}` | "Retrieval geometry" reads as a coined term; the subsection actually catalogs distance-function design choices (symmetric Euclidean, modality-specific, intrinsic non-Euclidean, extrinsic asymmetric scoring). |
| R4 | `paper/2_relatedwork.tex` L25 | "A Randers metric combines a Riemannian quadratic piece and a linear drift" | "A Randers metric combines a Riemannian quadratic term and a linear drift term" | "piece" is informal; standard Finsler texts (Bao–Chern–Shen 2000) use "term". |
| R5 | `paper/2_relatedwork.tex` L37 | "the additive form, Euclidean chord plus a linear functional of the displacement, yields a closed-form asymmetric cost" | "the additive form, Euclidean chord plus a term linear in the displacement, yields a closed-form asymmetric cost" | Fixes "linear functional of the displacement" awkwardness flagged in R5. |
| R6 | `paper/2_relatedwork.tex` L37 | "must satisfy integrand-level regularity conditions to define genuine Finsler metrics \cite{bao2000introduction}" | "must satisfy the pointwise positivity condition $\|\omega(x)\|_{M(x)^{-1}} < 1$ to define a genuine Finsler metric \cite{bao2000introduction}" | Replaces vague "integrand-level regularity conditions" with the actual condition (Bao–Chern–Shen Theorem 1.1.5). |
| R7 | `paper/2_relatedwork.tex` L40 | "clean$\to$corrupted information loss typically dominates the reverse" | "clean-to-corrupted information loss typically dominates the reverse" | Same arrow-in-prose fix as I2; bundle. |
| R-extra | `paper/2_relatedwork.tex` L4–5 | (commented-out paragraph "property-preserving model") | DELETE | Forbidden_claims phrasing ("structurally encoding"). Dead text. |
| R-extra2 | `paper/2_relatedwork.tex` L14–15 | (commented-out paragraph "Finsler manifold") | DELETE | Forbidden_claims phrasing ("mathematically guaranteed separation"). Dead text. |

---

## Section 3 — Methodology

| ID | File | Current | Proposed | Justification |
|----|------|---------|----------|---------------|
| M4 | `paper/3_methodology.tex` L7 (existing "Geometric properties" paragraph — only if Item 3a in `restructuring_plan.md` is NOT applied; else N/A) | "satisfying the Randers positivity condition $\|\omega\| < 1$" | "satisfying the Randers positivity condition $\|\omega\|_2 < 1$ (Sec.~\ref{subsec:asymmetric_finsler_distance}, Randers positivity constraint paragraph)" | Add the explicit norm subscript and the back-reference to where the condition is justified. |
| M5 | `paper/3_methodology.tex` L7 | "since the projector $\Pi_{\hat{\mathbf{z}}^{\mathrm{id}}}^{\perp}$ depends on $\hat{\mathbf{z}}^{\mathrm{id}}$" | "since the projection $\Pi_{\hat{\mathbf{z}}^{\mathrm{id}}}^{\perp}(\mathbf{v}) = \mathbf{v} - (\mathbf{v}\cdot\hat{\mathbf{z}}^{\mathrm{id}})\hat{\mathbf{z}}^{\mathrm{id}}$ depends on $\hat{\mathbf{z}}^{\mathrm{id}}$" | Defines the projector inline; alternative is to drop the projector reference if Item 3a moves the paragraph to supplement. |
| M6 | `paper/3_methodology.tex` L17 | (sentence "Setting $\hat{\mathbf{z}}^{\omega}{=}\mathbf{0}$ in the distance below recovers $d_E$ on the identity slice." in Euclidean baseline paragraph) | MOVE to right after eq.~\ref{eq:finsler_distance} | Forward dangling reference; sentence belongs with the distance it describes. |
| M7 | `paper/3_methodology.tex` L20 | "We replace the global drift in the canonical Randers form (eq.~\ref{eq:finsler_dist_canonical}) with the midpoint of the two endpoint drifts and restrict displacement to the identity subspace" | "We replace the global drift in the canonical Randers form (eq.~\ref{eq:finsler_dist_canonical}) with the midpoint of the two endpoint drifts, paired with the identity displacement $(\mathbf{y}^{\mathrm{id}}-\mathbf{x}^{\mathrm{id}})$" | Removes ambiguous "restrict displacement to the identity subspace"; the equation already shows the inner-product structure. |
| M14 | `paper/3_methodology.tex` L52 | "The sigmoid scaling asymptotically bounds the pre-orthogonalization drift norm below $c_{\max}{=}0.95$; the Gram-Schmidt step reduces the norm further but the post-orthogonalization bound is not explicitly re-enforced." | DELETE | Tautological / counterproductive: the math (Pythagoras for GS) does establish the post-GS bound from the pre-GS bound; the disclaimer is wrong. |
| M21 | `paper/3_methodology.tex` L127 (Sec. 3.6) | "$\mathcal{L}_{\mathrm{dcc}}$ avoids camera-vs-identity hard-pair conflict by regularizing drift variance instead: the regime in which we prefer to operate when stability is paramount" | "$\mathcal{L}_{\mathrm{dcc}}$ avoids the camera-vs-identity hard-pair conflict by bounding drift variance directly, which we adopt as the primary recipe" | Removes the awkward "regime in which we prefer to operate" colon-clause. |
| M22 | `paper/3_methodology.tex` L127 (last sentence of Sec. 3.6) | "Multi-term training precludes unique attribution; the empirical pattern is nonetheless consistent with this account." | DELETE | Defensive hedge with no information content (per `review_style_directives.md` directive 1). |

---

## Section 4 — Experimental Results

| ID | File | Current | Proposed | Justification |
|----|------|---------|----------|---------------|
| E1 | `paper/4_experimentalResults.tex` L3 | "aligning with BAU~\cite{cho2024generalizable}" | "aligning with BAU~\cite{cho2024generalizable} (statistics in Table~\ref{tab:dataset_stats})" | Adds the missing forward-reference to the dataset stats table. |
| E4 | `paper/4_experimentalResults.tex` L122 | "Retrieval uses the simplified Randers score $d_R(q,g)=\|\mathbf{f}_g-\mathbf{f}_q\|_2+\alpha(\theta_g-\theta_q)$." | "Retrieval uses the simplified Randers score $d_R(q,g) = \|\mathbf{f}_g - \mathbf{f}_q\|_2 + \alpha\,(\theta_g - \theta_q)$, where $\mathbf{f}_q, \mathbf{f}_g \in \mathbb{R}^{2048}$ are pre-BN identity features and $\theta_q, \theta_g \in \mathbb{R}$ are the learned severity scalars from $\mathcal{L}_{\mathrm{mono}}$ (eq.~\ref{eq:l_mono})." | Defines $\mathbf{f}$ and $\theta$ at first use. |
| E5 | `paper/4_experimentalResults.tex` L167 | "The Randers correction produces the geometrically predicted directional gap" | "The Randers correction $\alpha\,(\theta_g - \theta_q)$ produces the geometrically predicted directional gap" | Defines "Randers correction" inline with the explicit term. |
| E6 | `paper/4_experimentalResults.tex` L177 | "Two caveats apply. First, statistical power is limited ..." | "Two caveats temper the toy result. Statistical power is limited ($N{=}50$, single seed); differences below $0.03$ in mAP should be treated as directional, and the trained variant's Randers improvement of $+0.023$ mean $\Delta$ at $\alpha{=}0.9$ over its Euclidean counterpart sits at this threshold; multi-seed replication is required before any retrieval-improvement claim can be made. Second, the balanced single-severity protocol ..." | Removes the "Two caveats apply. First," staccato; integrates E7 disambiguation in same pass. |
| E7 | `paper/4_experimentalResults.tex` L177 | "the Randers improvement of the trained variant ($+0.023$ mean $\Delta$ at $\alpha\!=\!0.9$ over its Euclidean counterpart)" | "the trained variant's Randers improvement of $+0.023$ mean $\Delta$ at $\alpha=0.9$ over its Euclidean counterpart (i.e., $+0.041$ absolute mean $\Delta$ minus the trained-Euclidean control of $+0.018$, see Table~\ref{tab:toy_balanced_trained})" | DISAGREE with original comment 59 that the number should be 0.041. Defends the 0.023 by spelling out the arithmetic. Bundled with E6 in one rewrite. |

---

## Application order (when approved)

1. Apply DELETE rows first (R-extra, R-extra2, I-extra, M14, M22) — these are pure removals.
2. Apply notation-only swaps (R4, R5, R6, R7, I2) — atomic substitutions.
3. Apply rewording rows (A1, A2, A4, A5, R1, I3, I4, M5, M7, M21, E1, E4, E5) — single-sentence replacements.
4. Apply position moves (M6) — verify the destination is unchanged after Items 1–3.
5. Apply integrated multi-line edits (E6+E7 bundled).

After application, run `pdflatex` once to confirm no broken references.

---

## QA pass summary (bundled, applies to all small edits)

- **scientific-writing-coach:** All edits reduce filler / increase precision; no edit introduces a new claim.
- **citation-substantiation:** No new external citations; one internal cross-reference added (R6 still cites Bao–Chern–Shen which is already in `main.bib`).
- **repetition-reducer:** Deletions in Items 1 (R-extra, R-extra2, I-extra, M14, M22) reduce duplication. No new prose added that duplicates existing text.
