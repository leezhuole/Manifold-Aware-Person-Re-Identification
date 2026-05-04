# Comments triage assessment ledger

**Date:** 2026-05-04
**Source comments file:** `paper/comments_triage.md`
**Drafting_phase:** `results_implementation_lock`
**Reviewer posture:** Peer-level critical engagement. Comments are challenged where the draft is correct; the draft is challenged where the comments are correct. No default agreement.

Classification schema (from `paper-comment-triage`):
- **Ask** — clarification needed from the author / requires a research decision before any action.
- **Edit-small** — single-line wording, notation, or punctuation fix; unambiguously correct.
- **Edit-large** — restructure, move, rewrite, or delete a paragraph / subsection / table block.

Verdicts: **AGREE** / **PARTIAL** / **DISAGREE**. Each row carries a one-line justification grounded in code, changelog, or `publication_facts.md`.

Key codebase corroboration: at the current branch (`toyDataset_v4`, head `0ebdbce`) the `bau/` tree was reverted to a near-vanilla BAU + a `theta_head` (commit `1c0e181`); `FinslerDriftHead`, `finsler_drift_dist`, and `AlphaParameter` no longer live in `bau/models/model.py` or `bau/loss/triplet.py`. The paper still reasons about that earlier implementation, which is captured in `publication_facts.md` rows derived from job logs (1521465, 1521403, 1521774–1521831, 1521019, etc.). All implementation-grounding below uses those logs, not the currently-checked-out `bau/`.

---

## Section 0 — Abstract (6 comments)

| # | Line | Class | Verdict | Justification |
|---|------|-------|---------|---------------|
| A1 | 2 | Edit-small | AGREE | "unified asymmetric scoring layer" reads as marketing; "Randers/Finsler-style asymmetric retrieval score" already appears one sentence earlier (paper L2). Drop or substitute "asymmetric scoring layer". |
| A2 | 3 | Edit-small | AGREE | Abstract is L2-introduced for `M+MS+CS$\to$CUHK03`; expand to "Market-1501 + MSMT17 + CUHK-SYSU $\to$ CUHK03" (consistent with `publication_facts.md` Reported_metrics_with_provenance which always names full source/target). |
| A3 | 4 | Edit-small | AGREE | "protocol-matched run" is unintroduced jargon. The 44.9% number is **job 1521403 from an earlier code snapshot** (per `publication_facts.md` Headline_number_reconciliation, line "44.9 / 45.8 single-run is from 1521403; do not pair with +1.9 vs 42.4 — that delta is +2.5"). The current abstract already has a number-pairing inconsistency: `+1.9%` aligns with 1521465 (44.3 vs 42.4), not with the 44.9 protocol-matched run (which gives +2.5). Replace with "a single-seed run on the same protocol" and either (a) report the 44.3% / +1.9 pairing aligned with the configuration-matched Eucl. baseline, or (b) report the 44.9% stand-alone with no Δ in the abstract. |
| A4 | 5 | Edit-small | AGREE | "mechanism" is vague. Replace with the concrete object: "the asymmetric scoring layer" or "the Randers correction term" (whichever survives A1). |
| A5 | 6 | Edit-small | PARTIAL | "Randers correction" is introduced in Sec. 4.5 as a defined term, but in the abstract it is unintroduced and reads as a coined phrase. Comment 57 makes the same point in the experiments section — bundle. Replace with "the asymmetric distance term" in the abstract; keep "Randers correction" only inside Sec. 4.5 with its defining equation. |
| A6 | 7 | Edit-large | PARTIAL | Author asks to add "the conclusion from the toy dataset, that we could prove the intuition". `changelogs/toy_lmono_diagnostic_analysis.md` §5 explicitly distinguishes **Established** (severity encoding via L_mono with Spearman ≈ −0.91, M2b causal isolation, geometric D8 sign correct) from **Not Established** (Randers genuinely improves retrieval beyond noise, transfer to real ReID, beating a matched Euclidean baseline). The abstract already cites the Spearman number and "predicted directional distance gap" — that is what the toy proves. Adding "we proved the intuition" would overclaim because the *retrieval* intuition (asymmetric ranking improves mAP) was **not** proven by the toy (mean Δ +0.023 sits below the noise floor; see paper L177 limitation). Action: leave the toy summary as is, or sharpen one sentence to "validates the asymmetric scoring **mechanism** in a controlled setting" (already present at L2 abstract). DO NOT add "we proved the intuition". Forbidden by `publication_facts.md` Forbidden_claims (no overclaim from toy to real ReID). |
| A* | 9 | Edit-large | AGREE | The closing instruction "do not reference notations, terminology or results that have not been introduced" is correctly applied across A1–A5. Bundle into one abstract revision pass. |

---

## Section 1 — Introduction (5 comments + structural)

| # | Line | Class | Verdict | Justification |
|---|------|-------|---------|---------------|
| I1 | 12 | Ask | DISAGREE | The non-monotonic Δ profile (k=1: +0.035, k=2: +0.035, k=3: +0.20, k=4: +0.13) is exactly the empirical signal reported in `changelogs/toy_dataset_v4_balanced_eval.md` and the M1 log. Explaining the dip from k=3 to k=4 in the **Introduction** would push toy-dataset mechanics into the hook; CVPR-style narrative reserves the explanation for Sec. 4.5. The peak-then-fall is a standard saturation effect at heavy corruption (both directions degrade). The current caption need not explain it; pointing to Sec. 4.5 is sufficient. Recommendation: leave caption as is or trim "with the gap peaking near +0.20 at k=3" — the dip from k=3 to k=4 needs no inline explanation. |
| I2 | 13 | Edit-small | AGREE | Replace "probe→gallery and gallery→probe" with "probe-to-gallery and gallery-to-probe scoring" (also flagged at comment 26 for related work). Bundle. |
| I3 | 14 | Edit-small | AGREE | "cautionary datapoint" is filler. The sentence structure is "X is analyzed in §Y and a cautionary datapoint for Z" which reads awkwardly. Replace with "a documented failure mode for hard disentanglement under BAU-style multi-source training (Sec.~\ref{subsec:interference_analysis})". |
| I4 | 15 | Edit-small | AGREE | "continuous feature map" is ambiguous — does it mean a single-headed feature extractor without a separate disentanglement branch, or a topologically continuous map? The intended meaning is the former. Replace with "a single shared feature extractor" or "a single backbone-derived embedding". |
| I5 | 16 | Edit-large | AGREE | Last paragraph (contributions list) duplicates the previous paragraph and the abstract verbatim; item (vi) is word-for-word identical to abstract sentence about "balanced bidirectional evaluation". Refactor: collapse to 3–4 contributions framed by *what the reader gains*: (i) treat asymmetric probe-to-gallery scoring as a complementary axis to invariance; (ii) instantiate it as a Randers/Finsler score on a split identity/drift representation; (iii) document the objective-interference failure mode for nuisance-aligned drift losses; (iv) controlled toy validation of the asymmetric scoring mechanism. Drop the abstract-duplicate sentence about the bidirectional eval. |
| I-extra | 19 vs 21 | Edit-small | AGREE | Lines 19 and 21 contain near-duplicate contribution lists (one commented out, one active). Delete the commented-out line 19 to remove confusion. |

---

## Section 2 — Related Work (8 comments)

| # | Line | Class | Verdict | Justification |
|---|------|-------|---------|---------------|
| R1 | 19 | Edit-small | AGREE | "Retrieval geometry" as a heading is unusual. The subsection mixes (a) symmetric distances in standard ReID, (b) cross-modal directional heads, and (c) intrinsic vs extrinsic non-Euclidean embeddings. Better heading: "Distance functions in person re-identification" or "Symmetric and asymmetric retrieval distances". |
| R2 | 20 | Edit-small | AGREE | The Finsler-MDS sentence is a non-sequitur in a "retrieval geometry" subsection that has just discussed Euclidean and intrinsic non-Euclidean spaces. The lead-in needs a transition sentence: "A separate template keeps a Euclidean feature map but injects direction dependence into the pairwise score; Finsler-MDS \cite{dages2025finsler} is a recent embedding-side instance, and motivates the scoring layer in Sec.~\ref{subsec:asymmetric_finsler_distance}." |
| R3 | 21 | Ask | PARTIAL | "Nuisance" is an established statistics/ML term (nuisance variable, nuisance parameter — see Bishop 2006 §3.5). It is also used in DG-ReID literature (e.g., disentanglement papers). However, it is not the dominant term in re-id papers; "domain-specific factors" or "view/style factors" is more common. Recommendation: keep "nuisance" but on first use add a parenthetical "(e.g., camera, viewpoint, illumination)". Currently the term is used multiple times without ever being defined operationally. |
| R4 | 22 | Edit-small | AGREE | "piece" (in "Riemannian quadratic piece and a linear drift") is informal. Standard Finsler texts (Bao–Chern–Shen 2000) use "term" or "component". Replace "piece" with "term". |
| R5 | 23 | Edit-small | AGREE | "linear functional of the displacement" is technically correct (a linear map $u \mapsto \omega^\top u$) but reads awkwardly because "of the displacement" is bolted on. Replace with "linear in the displacement" (the standard phrasing for one-form drift in Finsler geometry). |
| R6 | 24 | Edit-small | AGREE | "integrand-level regularity conditions" is jargon without a referent. The intended technical content: for a position-dependent drift field $\omega(x)$, the Randers metric $F_x(u) = \sqrt{u^\top M(x) u} + \omega(x)^\top u$ defines a genuine Finsler metric only if $\|\omega(x)\|_{M(x)^{-1}} < 1$ holds **pointwise** (Bao–Chern–Shen 2000, Theorem 1.1.5). Replace "integrand-level regularity conditions" with "the pointwise positivity condition $\|\omega(x)\|_{M(x)^{-1}}<1$". |
| R7 | 26 | Edit-small | AGREE | Same fix as I2: replace "clean$\to$corrupted" with "clean-to-corrupted". |
| R-extra | 4–5 | Edit-large | AGREE | Lines 4–5 contain a commented-out paragraph that contradicts the running narrative ("property-preserving model, structurally encoding domain shifts as asymmetrical traversal costs"). Either delete it or migrate it to changelog notes. |
| R-extra2 | 14–15 | Edit-large | AGREE | Lines 14–15 contain another commented-out paragraph about "Finsler manifold" that conflicts with `publication_facts.md` Forbidden_claims ("Unified continuous Finsler manifold as replacement for implemented = asymmetric distance on Euclidean feature blocks"). Delete the commented block. |

---

## Section 3 — Methodology (18 comments)

| # | Line | Class | Verdict | Justification |
|---|------|-------|---------|---------------|
| M1 | 28 | Edit-large | AGREE | The hat in $\hat{\mathbf{z}}^\omega$ collides with the hat used for augmented views in Fig.~\ref{fig:architecture} ($\mathbf{x}$ vs $\hat{\mathbf{x}}$). Two ways out: (a) rename Gram-Schmidt output to $\mathbf{z}^{\omega,\perp}$ (or $\bar{\mathbf{z}}^\omega$); (b) rename augmented view to $\mathbf{x}'$ or $\mathbf{x}_{\mathrm{aug}}$. Option (a) is preferable because hat is canonically reserved for *normalized* identity ($\hat{\mathbf{z}}^{\mathrm{id}}$), and the post-projection drift is **not** unit-normalized. Bundles with M11 (omega-vs-superscript notation). |
| M2 | 29 | Edit-large | AGREE | "Geometric properties of the drift factor." paragraph (Sec. 3.1, second paragraph) is implementation detail (sigmoid scaling threshold, Jacobian non-zero, projector dependence). Move to Supp.~\ref{sec:suppl_orthogonal_rationale} which already discusses the rationale, and keep one sentence in the main: "The sigmoid scaling and Gram-Schmidt step together place $\mathbf{z}^{\omega,\perp}$ inside the open ball $B(0, c_{\max})$, satisfying the Randers positivity condition (proof in Supp.~\ref{sec:suppl_orthogonal_rationale})." |
| M3 | 30 | Ask | DISAGREE | Author asks to "Change this to alpha to match the notation from section 2.4". This conflicts with the **legacy** $\alpha$ usage in Supp.~\ref{sec:suppl_alpha} (the scalar gate $\alpha \in [0, \alpha_{\max}]$ with $\alpha_{\max}=1.0$, see eq.~\ref{eq:finsler_distance_alpha}). $c_{\max}=0.95$ in the main is a **drift-norm bound**, conceptually distinct from the legacy interpolation gate $\alpha_{\max}$. Conflating them would re-introduce the confusion the supplementary explicitly disclaims. Action: keep $c_{\max}$ in the main; add one footnote "$c_{\max}$ here is the drift-norm bound; the legacy scalar interpolation gate $\alpha$ in Supp.~\ref{sec:suppl_alpha} is a separate symbol." |
| M4 | 31 | Edit-small | AGREE | The math of the open-ball claim is correct: pre-orthogonalization sigmoid scaling forces $\|\mathbf{z}^\omega\|_2 < c_{\max}$, and Gram-Schmidt cannot increase the norm (Pythagoras: $\|\hat{\mathbf{z}}^{\omega,\perp}\|^2 = \|\mathbf{z}^\omega\|^2 - \langle \mathbf{z}^\omega, \hat{\mathbf{z}}^{\mathrm{id}}\rangle^2 \leq \|\mathbf{z}^\omega\|^2$), so the post-projection drift remains in $B(0, c_{\max})$. The justification "*should* lie in an open ball" is the **Randers positivity condition** (Bao–Chern–Shen 2000): $F(u) = \|u\|_2 + \omega^\top u > 0$ for all $u \neq 0$ requires $\|\omega\|_2 < 1$. Add one sentence: "The open-ball constraint is required for Randers positivity (see Sec.~\ref{subsec:asymmetric_finsler_distance}, *Randers positivity constraint* paragraph)." |
| M5 | 32 | Edit-small | AGREE | Projector $\Pi^\perp_{\hat{\mathbf{z}}^{\mathrm{id}}}$ is invoked without an explicit formula. Either define inline once: $\Pi^\perp_{\hat{\mathbf{z}}^{\mathrm{id}}}(\mathbf{v}) = \mathbf{v} - (\mathbf{v}\cdot\hat{\mathbf{z}}^{\mathrm{id}})\hat{\mathbf{z}}^{\mathrm{id}}$, or remove the projector reference and write $\partial \mathbf{z}^{\omega,\perp}/\partial \hat{\mathbf{z}}^{\mathrm{id}} \neq \mathbf{0}$ in elementwise form. |
| M6 | 33 | Edit-small | AGREE | "Setting $\hat{\mathbf{z}}^{\omega}{=}\mathbf{0}$ in the distance below recovers $d_E$ on the identity slice." sits inside the Euclidean-baseline paragraph but talks about the Randers distance "below" — a forward dangling reference. Move this sentence to immediately after eq.~\ref{eq:finsler_distance} where it belongs. |
| M7 | 34 | Edit-small | AGREE | "restrict displacement to the identity subspace" is ambiguous (does the displacement live in id-space, or do we project the displacement?). The intended meaning per eq.~\ref{eq:finsler_distance} is that the inner product is taken between the midpoint drift and the *identity-displacement* $(\mathbf{y}^{\mathrm{id}} - \mathbf{x}^{\mathrm{id}})$, not the full displacement. Rephrase: "and pair it with the identity-only displacement $(\mathbf{y}^{\mathrm{id}}-\mathbf{x}^{\mathrm{id}})$" — or remove the subphrase entirely (the equation makes the choice unambiguous). |
| M8 | 35 | Ask | AGREE (Randers positivity is correct) | Verified: positive-definiteness of the Randers fundamental tensor — equivalently, $F(u) > 0$ on $T \setminus \{0\}$ — requires $\|\omega\|_{M^{-1}} < 1$ (Bao–Chern–Shen 2000, Theorem 1.1.5; in the flat case $M = I$, this collapses to $\|\omega\|_2 < 1$). The paper's claim is correct. The author's "double-check" is satisfied; no math change needed, but the prose at L28 ("the flat-space specialization of $\|\omega\|_{M^{-1}}<1$") could add the parenthetical "(positivity of the Randers norm $F$)". |
| M9 | 36 | Edit-large | AGREE | Two issues bundled: (a) the BAU alignment loss is written in terms of $f_s, f_w$ but Sec. 3.1 / 3.2 use $\mathbf{z}^{\mathrm{id}}, \mathbf{z}^\omega$ — notation drift; (b) subscripts $s$ (strong augmentation) and $w$ (weak augmentation) are unintroduced; $w$ also visually collides with $\omega$. Fix: (a) replace $f_s, f_w$ with $\mathbf{z}^{\mathrm{id}}_s, \mathbf{z}^{\mathrm{id}}_w$ to match the rest of Sec. 3; (b) introduce subscripts at first use ("strong $(s)$ and weak $(w)$ augmentations following BAU \cite{cho2024generalizable}"). |
| M10 | 37 | Edit-small | AGREE | "destroying the asymmetric signal that augmentations are meant to induce" is too strong on two counts: (i) "destroying" overstates a regularization effect; (ii) "augmentations are meant to induce [asymmetry]" is unsupported — augmentations in BAU are designed for invariance, not asymmetry, and our use is a re-purposing. Soften: "would collapse drift to augmentation-invariant values, removing the per-view directional signal that the asymmetric distance relies on." |
| M11 | 38 | Edit-small | AGREE | "preventing drift magnitude from biasing the repulsion kernel" is asserted without derivation. The actual reason: the BAU uniform loss is $\log \mathbb{E}_{i\neq j}[\exp(-2\|f_i - f_j\|^2)]$. On the full $[\mathbf{z}^{\mathrm{id}};\mathbf{z}^\omega]$ embedding, $\|f_i-f_j\|^2 = \|\Delta \mathbf{z}^{\mathrm{id}}\|^2 + \|\Delta \mathbf{z}^\omega\|^2$, so any drift dispersion inflates the kernel argument and trivially satisfies uniformity without identity-space coverage. This is exactly what Supp.~\ref{sec:suppl_knn_uniform} already says. Replace the in-line justification with: "Restricting to $\mathbf{z}^{\mathrm{id}}$ prevents drift dispersion from artificially satisfying uniformity (Supp.~\ref{sec:suppl_knn_uniform})." |
| M12 | 39 | Ask | DISAGREE | The "convex combination of $S$" in Sec. 3.3 (domain-prototype drift) is mathematically a convex combination: $p_s$ are softmax outputs (non-negative, sum to one), so $\sum_{s=1}^S p_s \mathbf{e}_s$ **is** a convex combination of $\{\mathbf{e}_s\}$. The reviewer's "Double check if it really is convex" can be answered: yes. No change needed. |
| M13 | 40 | Edit-large | PARTIAL | "Relationship between the two designs" paragraph is largely architectural plumbing (residual = domain-prototype with $\lambda=1$, $\omega_{\mathrm{dom}}\equiv 0$). Move to Supp.~\ref{sec:suppl_domain_proto_head} after the full architecture; replace with one sentence in main: "Setting $\lambda=1$ and $\omega_{\mathrm{dom}}\equiv 0$ in eq.~\ref{eq:domain_drift_decomposition} recovers the residual variant." This compresses ~6 lines into 1 in main text. |
| M14 | 41 | Edit-small | AGREE | "The sigmoid scaling asymptotically bounds ... but the post-orthogonalization bound is not explicitly re-enforced." adds little — the math at M4 already proves the post-GS bound holds. Delete this sentence. |
| M15 | 42 | Edit-large | AGREE | The claim "without $\mathcal{L}_{\mathrm{dcc}}$, bidirectional mining does not consistently improve results (Arms~0, 3, 4); combined with $\mathcal{L}_{\mathrm{dcc}}$, it yields the strongest performance" is partially supported by `results/metric sweeps/sweep_auxiliary_loss_configs_metrics.md` (job 1521019): EXP 0/3/4 (bidir, no DCC) reach 43.6/43.7/43.9 mAP; EXP 1 (DCC, no bidir) reaches 43.9. The combined bidir+DCC arm 1b at 44.9 is from a **different** job (1521403, an earlier code snapshot — see `publication_facts.md` Headline_number_reconciliation), not from this sweep. So the claim mixes provenance. The author's instruction to remove the sentence on stylistic grounds (no result references in methodology) is also correct per `review_style_directives.md` directive 9. Action: delete the sentence. |
| M16 | 43 | Edit-large | AGREE | "Total loss." paragraph currently sits before $\mathcal{L}_{\mathrm{dcc}}$ is defined (eq.~\ref{eq:loss_dcc} is in Sec. 3.4, but the total-loss sum cites $\mathcal{L}_{\mathrm{dcc}}$). Move "Total loss" paragraph to the end of Sec. 3.5 (after `Same-camera drift attraction`) so all summands are defined first. |
| M17 | 44 | Edit-large | PARTIAL | The notation issue (subsection title uses `\omega` while Sec. 3.1–3.2 use $\mathbf{z}^\omega$) is real; the "highlight that $\omega$ is not some new output, but a slice of the output feature" requirement also applies to Sec. 3.3 ("Drift-head design variants"). Recommendation: add one sentence at the start of Sec. 3.3: "Throughout, $\omega \equiv \mathbf{z}^\omega \in \mathbb{R}^{d_\omega}$ denotes the drift slice of the structured embedding (Sec.~\ref{subsec:structured_embedding}); the two notations are interchangeable." Then standardize on $\omega$ inside Sec. 3.4–3.5 / equations. |
| M18 | 45 | Edit-large | AGREE | "Two interpretations are consistent with this..." paragraph speculates about what drift "is" without empirical anchor. Per `review_style_directives.md` directive 1, defensive interpretive prose belongs in a Discussion footnote, not the methodology. Trim to one sentence: "$\mathcal{L}_{\mathrm{dcc}}$ does not require drift to encode identity; it bounds drift-subspace variance to keep the asymmetric scoring term well-conditioned. We discuss interpretations in Sec.~\ref{sec:discussion}." |
| M19 | 46 | Edit-large | AGREE — paper text is wrong | Verified against `bau/models/memory.py` (line 16–22): the memory bank stores **per-sample** features `self.features[y]` indexed by `y` (the dataset index, equal to PID after ID encoding), and the buffer `self.labels` stores the `did` (domain ID) of the sample at that slot. Confirmed in `bau/trainers.py` line 78 (`self.memory_bank.momentum_update(f_w, pids)` — the second argument is **pids**, not dids) and line 136 (`domain_mask = torch.eq(dids.unsqueeze(-1), self.memory_bank.labels)` — the bank is filtered by `did` to find candidates *within the same domain*). So $\mathbf{m}_k$ in eq.~\ref{eq:domain_loss} is **not the centroid of domain $k$**; it is a **per-PID feature** that is then masked by domain identity. The paper text and the equation are misaligned. Two fixes possible: (a) re-write eq.~\ref{eq:domain_loss} as a sum over per-PID memory entries with a domain mask (faithful to code); (b) declare that the implementation uses a per-PID memory and a domain mask, while the equation summarizes the effective per-domain repulsion semantics. (a) is more honest; (b) is shorter. Need to flag for `publication_facts.md` correction. |
| M20 | 47 | Edit-large | AGREE | Sec. 3.6 ("Objective interference analysis") cites Table 4 numerically ($-7.5\%$ mAP, $\max\|\hat\omega\|$ 0.09→0.11). Per `review_style_directives.md` directive 9, methodology should not cite specific table cells. Move the empirical content to Sec. 4.3 (auxiliary-loss sweep paragraph) or merge with the Sec. 4.3 paragraph about Arm 5. Keep one methodology paragraph that defines *the failure mode* (objective interference) without citing numbers. |
| M21 | 48 | Edit-small | AGREE | "Cross-camera drift coherence $\mathcal{L}_{\mathrm{dcc}}$ avoids camera-vs-identity hard-pair conflict by regularizing drift variance instead: the regime in which we prefer to operate when stability is paramount" — the colon-clause is awkward. Replace with: "$\mathcal{L}_{\mathrm{dcc}}$ avoids the camera-vs-identity hard-pair conflict by regularizing drift variance directly, which we adopt as the primary recipe." |
| M22 | 49 | Edit-small | AGREE | "Multi-term training precludes unique attribution; the empirical pattern is nonetheless consistent with this account" — defensive hedge with no information content. Per `review_style_directives.md` directive 1, drop. |

---

## Section 4 — Experimental Results (8 comments)

| # | Line | Class | Verdict | Justification |
|---|------|-------|---------|---------------|
| E1 | 53 | Edit-small | AGREE | "aligning with BAU~\cite{cho2024generalizable}" lacks a forward-reference to Table~\ref{tab:dataset_stats}. Add: "...aligning with BAU~\cite{cho2024generalizable} (statistics in Table~\ref{tab:dataset_stats})." |
| E2 | 54 | Edit-large | AGREE | The "Training" paragraph in 4.2 (lines 51–53) restates the entire `\mathcal{L}_{\mathrm{ce}} + \mathcal{L}_{\mathrm{tri}}^{\mathrm{bi}} + \mathcal{L}_{\mathrm{align}} + \mathcal{L}_{\mathrm{uniform}} + \mathcal{L}_{\mathrm{dom}} + \lambda_\omega \mathcal{L}_\omega + w_{\mathrm{dcc}}\mathcal{L}_{\mathrm{dcc}}$ stack with weights — verbatim from Sec. 3.4 "Total loss." Pick one home (preferably Sec. 3.4 with hyperparameter values inlined) and replace the 4.2 restatement with a one-line "We train with the composite loss of eq.~\ref{eq:total_loss}, with $\lambda_\omega=1.5$ and $w_{\mathrm{dcc}}=0.1$ (drift-head LR ×0.05; gradient clip 1.0)." |
| E3 | 55 | Edit-large | AGREE | Sec. 4.3 "Primary model" paragraph also restates the recipe ("bidirectional unified Finsler triplet + $\mathcal{L}_{\mathrm{dcc}}$ (weight 0.1), residual drift, no $\mathcal{L}_{\mathrm{tri}}^{\mathrm{dom}}$"). With E2's deduplication, this can read: "Our strongest configuration is **Arm 1b** (Sec.~\ref{subsec:training_objective})". |
| E4 | 56 | Edit-small | AGREE | $d_R(q,g) = \|\mathbf{f}_g - \mathbf{f}_q\|_2 + \alpha(\theta_g - \theta_q)$ in Sec. 4.5 introduces $\mathbf{f}$ and $\theta$ without definition. Add inline: "where $\mathbf{f}_q, \mathbf{f}_g \in \mathbb{R}^{2048}$ are pre-BN identity features and $\theta_q, \theta_g \in \mathbb{R}$ are the learned severity scalars from $\mathcal{L}_{\mathrm{mono}}$ (eq.~\ref{eq:l_mono})." |
| E5 | 57 | Edit-small | AGREE | "The Randers correction" coined inline at L122. Bundle with A5: introduce as defined term once at the first use of the asymmetric scoring layer, then refer back. In Sec. 4.5, "Randers correction" can be defined: "the Randers correction $\alpha(\theta_g - \theta_q)$" makes it concrete. |
| E6 | 58 | Edit-small | AGREE | "Two caveats apply. First, ..." reads list-y. Convert to a single connecting clause: "Two caveats temper the toy result. Statistical power is limited ($N=50$, single seed) ... Second, the balanced single-severity protocol ...". |
| E7 | 59 | Ask | DISAGREE — number 0.023 is correct | The paper text "the Randers improvement of the trained variant ($+0.023$ mean $\Delta$ at $\alpha=0.9$ over its Euclidean counterpart)" computes $\Delta_{\mathrm{Randers}} - \Delta_{\mathrm{Euclidean}} = 0.041 - 0.018 = 0.023$ (Table~\ref{tab:toy_balanced_trained} rows: $\alpha=0$ Euclidean = +0.018, $\alpha=0.9$ Randers = +0.041). The 0.041 is the absolute mean Δ at $\alpha=0.9$; the +0.023 is the *gain over the trained Euclidean baseline*. Both numbers are correct under different definitions. Action: keep 0.023 but disambiguate: "...sits at this threshold ($+0.023$ above the trained-variant Euclidean control of $+0.018$, vs.\ the absolute mean $\Delta=+0.041$ at $\alpha=0.9$)." |
| E-extra | 4 (numbers) | Edit-large | AGREE | The abstract `+1.9% mAP` and the 4.3 paragraph `+2.5% mAP` are inconsistent: per `publication_facts.md` Headline_number_reconciliation, `+1.9` is grounded in jobs 1521465 (44.3) vs 1521474 (42.4), and `+2.5` is grounded in the older 1521403 (44.9) vs 1521474 (42.4). The Table~\ref{tab:template_multiprotocol_bau} uses 44.9 (1521403), and the abstract uses +1.9. This requires a coherent editorial decision (covered in `restructuring_plan.md`). |

---

## Cross-section issues (not enumerated in comments, surfaced by triage)

| Issue | Location | Action |
|-------|----------|--------|
| Headline number mismatch (abstract +1.9 vs Table P2 row 44.9 implying +2.5) | abstract L2 + 4_experimentalResults.tex L37, L59 | Decide one pairing; document in publication_facts. Already flagged in `publication_facts.md` Headline_number_reconciliation. |
| Memory-bank semantics (m_k label) | 3_methodology.tex L110 + eq:domain_loss | Update both paper text and `publication_facts.md` (new INTERNAL row). |
| Code base reverted to vanilla BAU + theta_head (commit 1c0e181) | bau/* | All paper claims about `FinslerDriftHead`, `finsler_drift_dist`, `AlphaParameter` now reference *historical code snapshots* (jobs 1521403 etc.). This is fine for `results_implementation_lock` provenance — but a reproducibility statement should clarify "code at commit X" rather than "current bau/ tree". Add a "Code provenance" footnote in Sec. 4.2 or in Supp. |

---

## Summary counts

| Section | Total | Edit-small | Edit-large | Ask | AGREE | PARTIAL | DISAGREE |
|---------|-------|-----------|-----------|-----|-------|---------|----------|
| 0 Abstract | 7 (incl. closing) | 5 | 2 | 0 | 5 | 2 | 0 |
| 1 Introduction | 6 (incl. extra) | 4 | 1 | 1 | 5 | 0 | 1 |
| 2 Related Work | 9 (incl. 2 extras) | 6 | 2 | 1 | 8 | 1 | 0 |
| 3 Methodology | 22 | 8 | 12 | 2 | 18 | 2 | 2 |
| 4 Experimental Results | 8 (incl. extra) | 5 | 3 | 1 | 6 | 1 | 1 |
| **Totals** | **52** | **28** | **20** | **5** | **42** | **6** | **4** |

(Original `paper/comments_triage.md` lists ~60 lines of bullet comments; many are sub-bullets or reference markers, which collapse to 52 actionable items above.)

---

## Disagree-with-justification (4 items)

- **I1 (intro line 12):** Do NOT explain non-monotonic Δ profile in caption — explanation belongs in Sec. 4.5.
- **M3 (methodology line 30):** Do NOT replace $c_{\max}$ with $\alpha$ — it would conflict with the legacy scalar gate $\alpha$ in Supp.~\ref{sec:suppl_alpha}.
- **M12 (methodology line 39):** "Convex combination of $S$" is mathematically correct (softmax outputs are convex weights); no change.
- **E7 (experimental results line 59):** Number $+0.023$ is correct (delta-of-deltas); add disambiguating clause rather than replacing with $0.041$.

---

## Partial-agreement items requiring author input (6 items)

- **A6 (abstract toy conclusion):** Add what was *actually* proven by the toy (mechanistic severity encoding, sign of geometric gap) — but DO NOT claim "we proved the intuition" of asymmetric ranking helping retrieval (Forbidden_claim).
- **A5/E5/A1 (Randers correction / unified scoring layer phrasing):** Bundled — needs one consistent term across abstract / 4.5 / 4.5.
- **R3 (nuisance terminology):** Keep + clarify on first use; or replace globally with "domain-/view-specific factors". Author chooses.
- **M13 (Sec. 3.3 relationship paragraph):** Move to supplement, keep one-line bridge in main.
- **M17 (Sec. 3.4 omega notation):** Need a single clarifying sentence to bridge $\mathbf{z}^\omega$ ↔ $\omega$.
- **E-extra (headline number coherence):** Choose one pairing (44.3/+1.9 OR 44.9/+2.5); update abstract or table accordingly. See `restructuring_plan.md` for two options.

---

## Provenance corroboration index (for the high-stakes comments)

| Comment | Evidence consulted | Verdict source |
|---------|-------------------|----------------|
| 31 (open-ball reasoning) | Math derivation (Pythagoras for GS); Bao–Chern–Shen Theorem 1.1.5 | AGREE; add one sentence stating positivity = the reason for the bound |
| 35 (Randers positive-definiteness) | Bao–Chern–Shen 2000 §1.1; eq.~\ref{eq:finsler_norm} positivity | AGREE; current paper text is correct, no change required (M8) |
| 39 (convex combination of S) | softmax outputs are non-negative and sum to 1 | DISAGREE; math is correct |
| 42 (bidir vs L_dcc direction) | `results/metric sweeps/sweep_auxiliary_loss_configs_metrics.md` (job 1521019); `publication_facts.md` Headline_number_reconciliation | AGREE on removal (mixed-provenance + style violation) |
| 46 (memory bank centroid) | `bau/models/memory.py` lines 16–22; `bau/trainers.py` lines 78, 136 | AGREE — paper text is wrong; needs `publication_facts.md` correction |
| 59 (0.023 vs 0.041) | `paper/4_experimentalResults.tex` Table~\ref{tab:toy_balanced_trained}; arithmetic 0.041−0.018=0.023 | DISAGREE on replacement; AGREE on disambiguation |
| Abstract 7 (toy proves intuition) | `changelogs/toy_lmono_diagnostic_analysis.md` §5 Established vs Not Established | PARTIAL — toy proved the *mechanism*, not the *retrieval intuition*; do not overclaim |

---

End of ledger.
