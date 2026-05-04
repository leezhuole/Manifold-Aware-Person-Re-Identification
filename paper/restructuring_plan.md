# Restructuring plan — Edit-large items

**Date:** 2026-05-04
**Scope:** Items classified as Edit-large in `paper/comments_triage_assessment.md` requiring section moves, paragraph rewrites, or new prose. Edit-small items are bundled separately in `paper/edit_small_batch.md`.
**Status:** Awaiting user approval. NO `paper/*.tex` edits applied yet.

QA checklist (per `paper-comment-triage` schema): every proposed prose chunk in this plan has been vetted against (1) `paper-scientific-writing-coach` (Carlini/Karpathy/Nature checklist), (2) `paper-citation-substantiation` two-path (cite vs log in `unsourced_or_vague_claims.md`), (3) `paper-repetition-reducer` (cross-section duplication). Verdicts attached at the end of each item.

---

## 1. Abstract — coherent number pairing + scoped toy summary (A3, A6, E-extra)

### Comments addressed
- A3 (line 4 — "protocol-matched run" jargon)
- A6 (line 7 — "include the toy conclusion that we could prove the intuition")
- E-extra (cross — abstract `+1.9%` vs Table P2 `+2.5%` mismatch)

### Assessment
- A3: AGREE on rewording.
- A6: PARTIAL — mention what the toy *actually* proved (mechanism + sign of geometric gap), not the broader retrieval intuition (Forbidden under `publication_facts.md` Forbidden_claims).
- E-extra: AGREE — pick one number-pairing (Option A or B in `publication_facts.md` Headline_number_internal_consistency_flag).

### Proposed structural change
Decision required from author: **Option A (recommended)** uses 1521465 (44.3 / +1.9) consistently, drops 44.9 from main; **Option B** uses 1521403 (44.9 / +2.5) and adds a snapshot footnote. Below is Option A.

### Proposed abstract prose (Option A)
> Domain-generalizable person re-identification (DG-ReID) typically optimizes symmetric distances on Euclidean embeddings, yet probe-to-gallery retrieval in heterogeneous camera networks is direction-dependent: viewpoint, resolution, and sensing pipelines induce ordered mismatch that a symmetric metric discards. We study Randers/Finsler-style asymmetric retrieval scores on top of alignment–uniformity training~\cite{cho2024generalizable}, using a structured identity/drift representation while keeping the backbone Euclidean. Nuisance-aligned drift objectives interfere with domain-invariant training pressures; we therefore adopt a unified asymmetric distance with norm-constrained drift that recovers the Euclidean regime at initialization. On Market-1501 + MSMT17 + CUHK-SYSU $\to$ CUHK03, Finsler BAU achieves 44.3\% mAP, $+1.9\%$ over a configuration-matched Euclidean baseline; a five-seed multi-seed estimate is $43.8\pm0.3\%$ mAP. A controlled corruption study isolates the asymmetric scoring mechanism: a pairwise monotonicity loss reliably encodes corruption severity in a learned scalar (Spearman $\rho \approx -0.91$), and the resulting Randers correction produces the geometrically predicted directional distance gap. The results suggest that drift-subspace variance regularization, not explicit domain-mining auxiliaries, is the critical ingredient for stable asymmetric retrieval. Code is available at \url{https://github.com/leezhuole/Manifold-Aware-Person-Re-Identification}.

### Provenance lines for each new claim
- "Finsler BAU achieves 44.3\% mAP" → `publication_facts.md` Reported_metrics_with_provenance §"Primary Finsler runs" row 1521465 (P2 M+MS+CS→C3, 44.3%).
- "$+1.9\%$ over a configuration-matched Euclidean baseline" → `publication_facts.md` Headline_number_reconciliation row 1, paired with 1521474 (42.4%).
- "five-seed multi-seed estimate is $43.8\pm0.3\%$" → `publication_facts.md` §"Multi-seed Finsler primary" jobs 1521477/1521774/1521825/1521826/1521831 (note: iters=500 caveat applies; abstract intentionally rounds to one decimal).
- "Spearman $\rho \approx -0.91$" → `publication_facts.md` §"Controlled corruption study" (added 2026-05-04) row 1; `changelogs/toy_lmono_diagnostic_analysis.md` §3.2.
- "Randers correction produces the geometrically predicted directional distance gap" → `publication_facts.md` §"Controlled corruption study" row 3.

### QA verdicts
- **scientific-writing-coach:** PASS. One idea per sentence. Hook (asymmetric retrieval) → method (split id/drift + Randers) → result (44.3% / +1.9%) → mechanism validation (toy) → punch line (variance regularization). 6 sentences, ~1900 chars (under CVPR abstract limit).
- **citation-substantiation:** PASS. Every numeric claim has a `publication_facts.md` row. The phrase "Nuisance-aligned drift objectives interfere..." is supported by Sec. 3.6 + Table~\ref{tab:aux_loss_sweep} Arm 5. The claim "drift-subspace variance regularization is the critical ingredient" is the Discussion thesis (Sec. 5).
- **repetition-reducer:** PASS. The toy summary (last 2 sentences) does not duplicate Sec. 4.5 verbatim; it summarizes mechanism + sign, leaving the full table to Sec. 4.5.

---

## 2. Introduction — contributions list refactor (I5, I-extra)

### Comments addressed
- I5 (line 16 — contributions list duplicates abstract + previous paragraph; (vi) word-for-word identical to abstract)
- I-extra (lines 19 vs 21 — duplicate active and commented-out contribution lists)

### Assessment
- I5: AGREE — the current 6-bullet list is a laundry-list per `paper-cvpr-tight-prose` (anti-laundry-list). Abstract sentence about balanced bidirectional eval is duplicated verbatim.
- I-extra: AGREE — delete the commented-out duplicate at line 19.

### Proposed structural change
Replace the 6-bullet contribution list with 4 contributions framed by *what the reader gains*:

### Proposed contributions paragraph
> Our main contributions are: (i) we treat asymmetric probe-to-gallery scoring as a complementary axis to domain-invariance for DG-ReID, motivated by direction-dependent retrieval in cross-view and cross-modality deployments; (ii) we instantiate this as a Randers/Finsler-style score on a split identity/drift representation with norm-constrained drift that recovers Euclidean geometry at initialization, on top of BAU-style alignment and uniformity training; (iii) we identify and quantify objective interference between identity-mined and camera-mined hard-pair losses, motivating a drift-coherence regularizer in place of camera-mined nuisance triplets; (iv) we corroborate the asymmetric scoring mechanism in a controlled synthetic study where a pairwise monotonicity loss encodes corruption severity in a learned scalar (Spearman $\rho \approx -0.91$) and the Randers correction produces the predicted directional distance gap.

### Provenance lines
- (i) qualitative axis claim — supported by `paper/2_relatedwork.tex` paragraph on cross-modal/aerial-ground; `\cite{wu2017rgb,nguyen2024agreidv2}`.
- (ii) → Sec. 3 (split embedding + asymmetric distance + barrier).
- (iii) "objective interference" → `publication_facts.md` Reported_metrics_with_provenance §"Idea-1" Arm 5 (mAP 36.4 vs 44.1, regression $-7.7$pp); paper Sec. 3.6 + Table~\ref{tab:aux_loss_sweep} Arm 5.
- (iv) Spearman + Randers gap → `publication_facts.md` §"Controlled corruption study" rows 1 and 3.

### QA verdicts
- **scientific-writing-coach:** PASS. Each contribution is a complete clause with a verb and a falsifiable claim. No "we propose"; uses "we treat", "we instantiate", "we identify", "we corroborate" — concrete actions.
- **citation-substantiation:** PASS. (i) cites two prior works; (ii)–(iv) cite internal sections / tables.
- **repetition-reducer:** PASS — bullet (iv) summarizes mechanism, abstract summarizes outcome+number; no verbatim overlap.

---

## 3. Methodology — drift-property paragraph and Sec. 3.3 relationship paragraph move to supplement (M2, M13)

### Comments addressed
- M2 (line 29 — "Geometric properties of the drift factor" paragraph deals with implementation more than effect; move to supplement)
- M13 (line 40 — "Relationship between the two designs" paragraph belongs in supplement)

### Assessment
- M2: AGREE — the paragraph is implementation detail (sigmoid scaling, Jacobian, channels for residual identity leakage). Supp.~\ref{sec:suppl_orthogonal_rationale} already covers the Gram-Schmidt rationale.
- M13: PARTIAL — the conceptual point (residual = domain-prototype with $\lambda=1$, $\omega_{\mathrm{dom}}\equiv 0$) is useful as a 1-liner; the rest is supplement material.

### Proposed structural change
**3a (M2):** Delete the "Geometric properties of the drift factor" paragraph from Sec. 3.1. Replace with one sentence at the end of the first Sec. 3.1 paragraph: "The sigmoid-gated scaling of \S\ref{subsec:drift_head_design} together with Gram-Schmidt places $\mathbf{z}^{\omega,\perp}$ inside the open ball $B(\mathbf{0}, c_{\max})$, so the Randers positivity condition (Sec.~\ref{subsec:asymmetric_finsler_distance}) holds; the residual identity-leakage analysis is in Supp.~\ref{sec:suppl_orthogonal_rationale}."

Append the deleted paragraph to Supp.~\ref{sec:suppl_orthogonal_rationale} after the existing content, with a sub-heading: "Identity leakage channels."

**3b (M13):** Delete the 6-line "Relationship between the two designs" paragraph from Sec. 3.3 (last paragraph). Replace with one sentence at the end of the residual-drift paragraph: "Setting $\lambda=1$ and $\omega_{\mathrm{dom}}\equiv\mathbf{0}$ in eq.~\ref{eq:domain_drift_decomposition} recovers this residual variant from the domain-prototype head."

Append the deleted prose to Supp.~\ref{sec:suppl_domain_proto_head} as a final paragraph "Relationship to the residual variant."

### Provenance
Both edits are pure structural moves; no new claims, no new numbers. The `c_{\max}=0.95$ value, the Jacobian non-zero claim, and the GS-norm-non-increase claim are unchanged and live in the supplement.

### QA verdicts
- **scientific-writing-coach:** PASS. Reduces methodology length by ~15 lines, freeing space for the higher-priority objective-interference rewrite (item 6 below).
- **citation-substantiation:** PASS. No new claims.
- **repetition-reducer:** PASS — eliminates duplicate description of GS rationale (currently appears in both Sec. 3.1 paragraph 1 and paragraph 2).

---

## 4. Methodology — memory-bank semantics correction (M19)

### Comment addressed
- M19 (line 46 — paper text says $\mathbf{m}_k$ is "centroid of domain $k$"; code shows it is per-PID memory with a domain mask)

### Assessment
**AGREE — paper text is wrong.** Verified against `bau/models/memory.py` (lines 16–22) and `bau/trainers.py` (lines 78, 130–139). The memory bank is per-sample (effectively per-PID), and `domain_loss` masks bank entries by `did` to find same-domain candidates for repulsion — it is **not** a per-domain centroid. See `publication_facts.md` Memory_bank_semantics_correction (added 2026-05-04).

### Proposed structural change — option (b) in publication_facts.md (shorter)
Replace the current text:
> Let $\mathbf{m}_k$ denote the centroid of domain $k$ in the memory bank:
> $\mathcal{L}_{\mathrm{dom}} = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \phi(d_F^2(\mathbf{z}_i, \mathbf{m}_{c(i)}))$,
> where $\phi(\cdot)$ is the BAU repulsion kernel \cite{cho2024generalizable} and $c(i)$ is the domain label of sample $i$.

with:
> The BAU domain repulsion~\cite{cho2024generalizable} maintains a per-PID memory bank $\{\mathbf{m}_j\}_{j=1}^{N}$ updated by exponential moving average and labeled by training domain $\mathrm{dom}(j)$. For each batch sample $i$ with domain $c(i)$, repulsion is mined over the same-domain mask $\{\mathbf{m}_j : \mathrm{dom}(j) = c(i),\ j \neq i\}$:
> $\mathcal{L}_{\mathrm{dom}} = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \phi\!\left(d_F^2(\mathbf{z}_i,\, \mathbf{m}_{j^\star(i)})\right), \qquad j^\star(i) = \arg\min_{j:\mathrm{dom}(j)=c(i),\ j\neq i} d_F^2(\mathbf{z}_i, \mathbf{m}_j),$
> where $\phi(\cdot)$ is the BAU repulsion kernel and $d_F$ replaces the Euclidean distance to inject directional asymmetry into the per-domain repulsion.

### Provenance
- `bau/models/memory.py` lines 16–22 (`MemoryBank` storage and `momentum_update` keyed on the second arg, called with `pids` in `bau/trainers.py` line 78).
- `bau/trainers.py` lines 130–139 (`domain_loss`: `domain_mask = torch.eq(dids.unsqueeze(-1), self.memory_bank.labels)`; `sorted_dist[:, 1:m+1]` selects the hardest non-self positives within the same domain).
- `publication_facts.md` Memory_bank_semantics_correction (added 2026-05-04).

### QA verdicts
- **scientific-writing-coach:** PASS. Equation now matches code; no overstatement.
- **citation-substantiation:** PASS. BAU citation retained; new explicit code references via the `publication_facts.md` row.
- **repetition-reducer:** PASS. No duplication introduced.

**NOTE:** This is a load-bearing correction; the original equation describes a *different* loss than what the code computes. The author should confirm before applying.

---

## 5. Methodology — Sec. 3.4 Total-loss paragraph relocation + Sec. 4.2 deduplication (M16, E2, E3)

### Comments addressed
- M16 (line 43 — "Total loss" paragraph appears before $\mathcal{L}_{\mathrm{dcc}}$ is defined; move to end of 3.4)
- E2 (line 54 — Sec. 4.2 "Training" paragraph restates the entire loss stack verbatim)
- E3 (line 55 — Sec. 4.3 "Primary model" paragraph restates the recipe)

### Assessment
- M16: AGREE — read-order issue: $\mathcal{L}_{\mathrm{dcc}}$ is defined in Sec. 3.5 eq.~\ref{eq:loss_dcc}; the total-loss equation in Sec. 3.4 references it forward.
- E2: AGREE — the "Training" paragraph in 4.2 lists every loss term again, including weights, which is also stated in Sec. 3.4 "Total loss". Per `review_style_directives.md` directive 5 ("Forward-references to sections should include `\ref{}`"), use a single canonical home.
- E3: AGREE — "Primary model" paragraph in 4.3 also restates the recipe.

### Proposed structural change

**5a (M16):** Move the "Total loss" paragraph (Sec. 3.4 last paragraph, eq.~\ref{eq:total_loss}) to **end of Sec. 3.5** (after `Same-camera drift attraction`). All summands ($\mathcal{L}_{\mathrm{ce}}, \mathcal{L}_{\mathrm{tri}}^{\mathrm{F}}, \mathcal{L}_{\mathrm{align}}, \mathcal{L}_{\mathrm{uniform}}, \mathcal{L}_{\mathrm{dom}}, \mathcal{L}_\omega, \mathcal{L}_{\mathrm{dcc}}$) are then defined before the sum.

**5b (E2):** Replace the "Training" paragraph in Sec. 4.2 with: "We train with the composite loss eq.~\ref{eq:total_loss}; primary-recipe weights are $\lambda_\omega = 1.5$ and $w_{\mathrm{dcc}} = 0.1$. The drift head uses a reduced learning rate ($\times 0.05$ of backbone LR) and gradient clipping (max norm 1.0)."

**5c (E3):** Replace the first sentence of "Primary model" in Sec. 4.3 with: "Our strongest configuration on M+MS+CS$\rightarrow$C3 is **Arm 1b** (Sec.~\ref{subsec:training_objective}, Table~\ref{tab:aux_loss_sweep}): bidirectional unified Finsler triplet with $\mathcal{L}_{\mathrm{dcc}}$ and residual drift."

### Provenance
- Eq.~\ref{eq:total_loss} weights ($\lambda_\omega=1.5$, $w_{\mathrm{dcc}}=0.1$) → `publication_facts.md` §"Auxiliary loss configuration sweep" job 1521019; §"Primary Finsler runs" jobs 1521463–1521897.
- Drift LR multiplier 0.05, gradient clip 1.0 → README §"Useful fork-specific flags".

### QA verdicts
- **scientific-writing-coach:** PASS. Eliminates the laundry-list restatement in 4.2.
- **citation-substantiation:** PASS. All weights have provenance.
- **repetition-reducer:** PASS. Three near-verbatim restatements collapsed to one canonical home + two short pointers.

---

## 6. Methodology — objective-interference paragraph rewrite + relocation (M15, M20, M21, M22)

### Comments addressed
- M15 (line 42 — "without $\mathcal{L}_{\mathrm{dcc}}$, bidirectional mining does not consistently improve...")
- M20 (line 47 — Sec. 3.6 cites Table 4 numerics in methodology)
- M21 (line 48 — "the regime in which we prefer to operate when stability is paramount" awkward)
- M22 (line 49 — "Multi-term training precludes unique attribution" defensive hedge)

### Assessment
All AGREE on rewriting. The current Sec. 3.6 (and the paragraph in Sec. 3.4 that M15 covers) is *empirical*-flavored prose in a methodology section: it cites mAP regression numbers and table arms, which `review_style_directives.md` directive 9 forbids in methodology.

### Proposed structural change

**6a (M15):** Delete the sentence "In isolation, without $\mathcal{L}_{\mathrm{dcc}}$, bidirectional mining does not consistently improve results (Table~\ref{tab:aux_loss_sweep}, Arms~0, 3, 4); combined with $\mathcal{L}_{\mathrm{dcc}}$, it yields the strongest performance in the auxiliary-loss sweep (Sec.~\ref{subsec:quantitative_analysis}, Table~\ref{tab:aux_loss_sweep})."

Replace with: "We pair bidirectional mining with the cross-camera drift coherence regularizer $\mathcal{L}_{\mathrm{dcc}}$ (Sec.~\ref{subsec:drift_regularization}) in the primary recipe; the empirical comparison is in Sec.~\ref{subsec:quantitative_analysis}."

**6b (M20, M21, M22):** Compress Sec. 3.6 ("Objective interference analysis") to a 3-sentence definitional paragraph (no numbers, no table cells):

> **Objective interference.** Identity batch-hard mining rewards cross-camera same-PID proximity; camera batch-hard mining rewards same-camera proximity, which usually means *different* identities. Jointly optimizing both pulls in the same hinge therefore induces a hard-pair conflict on the structured embedding, and the unconstrained drift subspace absorbs most of the slack. The cross-camera drift coherence regularizer $\mathcal{L}_{\mathrm{dcc}}$ (eq.~\ref{eq:loss_dcc}) avoids this conflict by bounding drift variance directly rather than mining camera-paired drift; we adopt it as the primary recipe and report the numeric comparison in Sec.~\ref{subsec:quantitative_analysis}.

Move the empirical numbers (`-7.5\%$ mAP, $\max\|\hat\omega\|$ 0.09→0.11) to Sec. 4.3 "Auxiliary-loss sweep" paragraph.

### Provenance
- Definitional content (no numbers) requires no provenance row.
- Numbers moved to 4.3 → `publication_facts.md` §"Auxiliary loss configuration sweep" Arm 5 (mAP 36.4, R1 35.5).

### QA verdicts
- **scientific-writing-coach:** PASS. Sec. 3.6 becomes 3 sentences, no table refs in methodology.
- **citation-substantiation:** PASS. Empirical numbers retained but moved to results.
- **repetition-reducer:** PASS. Eliminates the redundancy where Sec. 3.6 and Sec. 4.3 both describe Arm 5's mAP regression.

---

## 7. Methodology — drift coherence interpretation paragraph trim (M18)

### Comment addressed
- M18 (line 45 — "Two interpretations are consistent with this..." is unproven speculation)

### Assessment
AGREE. The two interpretations ((i) drift as identity-conditioned directional bias, (ii) soft regularizer reducing effective drift dimensionality) are speculative; neither was tested. Per `review_style_directives.md` directive 1 (defensive interpretive prose belongs in Discussion, not Methodology), trim.

### Proposed structural change
Replace the current trailing sentences of Sec. 3.5 "Cross-camera drift coherence":

> The effect is not that drift semantically encodes person identity; it is that drift variance is bounded so that the asymmetric scoring term is reliable. Two interpretations are consistent with this: (i) drift as an identity-conditioned directional bias that should be view-invariant for a given person, and (ii) a soft regularizer that reduces the effective drift dimensionality required for stable Randers scoring.

with:
> $\mathcal{L}_{\mathrm{dcc}}$ does not require drift to encode identity; it bounds the drift subspace variance to keep the asymmetric scoring term well-conditioned. We discuss the empirical signature and competing interpretations in Sec.~\ref{sec:discussion}.

The deleted interpretive content is already covered by Sec. 5 ("Discussion") paragraphs (i) and (ii). The current Sec. 5 already articulates the same two interpretations with empirical anchors (post-GS norms 0.054–0.067, $\rho=0.67$ with severity); methodology can defer cleanly.

### Provenance
- Empirical anchor for interpretation (i) — `paper/5_discussion.tex` paragraph (i).

### QA verdicts
- **scientific-writing-coach:** PASS. Methodology becomes concrete; Discussion retains the interpretive richness.
- **citation-substantiation:** PASS.
- **repetition-reducer:** PASS — eliminates duplicate interpretation between Sec. 3.5 and Sec. 5.

---

## 8. Methodology — alignment / uniform identity-only justification rewrite (M9, M10, M11)

### Comments addressed
- M9 (line 36 — $f_s, f_w$ notation collision with $\omega$ and missing definition)
- M10 (line 37 — "destroying the asymmetric signal that augmentations are meant to induce" too strong)
- M11 (line 38 — "preventing drift magnitude from biasing the repulsion kernel" lacks derivation)

### Assessment
All AGREE. The "Identity-only alignment" paragraph in Sec. 3.2 needs three local fixes that together flow as a single rewrite.

### Proposed structural change
Replace the current "Identity-only alignment" paragraph with:

> **Identity-only alignment.** The BAU alignment loss~\cite{cho2024generalizable} matches the identity features of weak (subscript $w$) and strong (subscript $s$) augmentations of the same image. Aligning the full $[\mathbf{z}^{\mathrm{id}}; \mathbf{z}^{\omega,\perp}]$ embedding would force $\mathbf{z}^{\omega,\perp}_w \approx \mathbf{z}^{\omega,\perp}_s$, removing the per-view directional signal that the asymmetric distance relies on. We restrict alignment to the identity slice:
> [eq.~\ref{eq:identity_only_alignment} unchanged with $\mathbf{z}^{\mathrm{id}}_{w,i}, \mathbf{z}^{\mathrm{id}}_{s,i}$ in place of $f_w, f_s$]
> where $w_i$ are reciprocal $k$-NN Jaccard weights~\cite{zhong2017re,cho2024generalizable} computed on identity features only. The uniform loss is similarly restricted to $\mathbf{z}^{\mathrm{id}}$: on the full embedding, drift dispersion would inflate the kernel argument $\|f_i - f_j\|^2$ and trivially satisfy the uniformity objective without improving identity-space coverage (derivation in Supp.~\ref{sec:suppl_knn_uniform}).

### Provenance
- Subscripts $w$ (weak augmentation) and $s$ (strong augmentation) — BAU \cite{cho2024generalizable} convention, also used in `bau/trainers.py` `inputs_w, inputs_s`.
- Uniform-loss derivation — Supp.~\ref{sec:suppl_knn_uniform} already documents the math.

### QA verdicts
- **scientific-writing-coach:** PASS. Notation now consistent across Sec. 3.1–3.2; "destroying" softened to "removing"; uniform-loss justification now points to the existing derivation.
- **citation-substantiation:** PASS. Two BAU citations retained.
- **repetition-reducer:** PASS — uses Supp.~\ref{sec:suppl_knn_uniform} reference instead of restating the math inline.

---

## 9. Related work — clean up commented-out paragraphs and Finsler-MDS lead-in (R-extra, R-extra2, R2)

### Comments addressed
- R-extra (lines 4–5 — commented-out paragraph contradicts narrative)
- R-extra2 (lines 14–15 — commented-out paragraph violates Forbidden_claims)
- R2 (line 20 — Finsler-MDS sentence is a non-sequitur)

### Assessment
AGREE on all three.

### Proposed structural change
**9a (R-extra, R-extra2):** Delete both commented-out paragraphs from `paper/2_relatedwork.tex`. They contain Forbidden_claims phrasing ("unified continuous Finsler manifold", "mathematically guaranteed separation") and dead text drives reviewer confusion.

**9b (R2):** Rewrite the closing sentence of Sec. 2.2 "Retrieval geometry":

Current:
> A fourth design axis keeps a standard Euclidean feature map and injects direction dependence in the pairwise score rather than in the geodesics of the embedding space; Finsler-MDS \cite{dages2025finsler} is a recent example of this template (see Sec.~\ref{subsec:asymmetric_finsler_distance}).

Proposed:
> A complementary axis to the three above keeps the feature map Euclidean and injects directional dependence into the *pairwise score* rather than into the geometry of the embedding space. Finsler-MDS \cite{dages2025finsler} instantiates this template for asymmetric dissimilarity embedding; we adopt the same scoring template inside a DG-ReID training stack (Sec.~\ref{subsec:asymmetric_finsler_distance}).

### Provenance
- Finsler-MDS as embedding-side example — \cite{dages2025finsler} (already cited).
- Logged as resolved in `unsourced_or_vague_claims.md` UV-RW-014 ("closest formal precedent" superlative is removed; the rewrite stays inside the cite scope).

### QA verdicts
- **scientific-writing-coach:** PASS.
- **citation-substantiation:** PASS — the rewrite drops the "closest" superlative flagged in UV-RW-014 and uses scoped language.
- **repetition-reducer:** PASS. Deletes 28 lines of commented-out duplicate prose.

---

## Approval gating

The following items in this plan **must be confirmed by the user before any `paper/*.tex` edit**:

1. **Item 1 — Abstract Option A vs B.** Affects abstract + Table~\ref{tab:template_multiprotocol_bau} P2 row + Sec. 4.3 (Δ wording). Default: **Option A** (44.3% / +1.9% throughout).
2. **Item 4 — Memory-bank equation rewrite.** This is a load-bearing correction; the original equation is wrong. Author confirms before apply.
3. **Item 6 — Sec. 3.6 → 3-sentence rewrite + numbers moved to 4.3.** Affects two sections.

Other items (2, 3, 5, 7, 8, 9) are mechanical; once Items 1, 4, 6 are settled, the rest can be applied in one batch.

---

## Files NOT touched in this plan

- `paper/X_suppl.tex` — only receives appended content from items 3a, 3b (existing supplementary sections gain trailing paragraphs); no restructuring.
- `paper/main.bib` — no new citations beyond what is already cited; the Bao–Chern–Shen 2000 citation already exists for items 8 / R6.
- `paper/comments_triage.md` — not modified; the assessment ledger is the new artifact.

Edit-small items (notation/wording fixes, hyphen substitutions, etc.) are bundled in `paper/edit_small_batch.md` for separate batch approval.
