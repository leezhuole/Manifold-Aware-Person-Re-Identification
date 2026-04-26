# Changelog

**Timestamp:** 2026-04-22

## [2026-04-22] - Added mathematical derivation of residual variance in `toy_dataset_asymmetry_diagnostics.md`

### Files modified
- `changelogs/toy_dataset_asymmetry_diagnostics.md` — Added formal Pythagorean derivation of the component of the identity drift unmatched by the 1D Finsler subspace projector $\mathbf{P}_B^\perp$.

### Problem this addresses
The mathematical steps linking the Randers asymmetry term to the closed-form orthogonal projection residual variance expression were unstated, hindering reproducibility and immediate clarity. 

### Expected behavior
Readers will find a clear, step-by-step algebraic breakdown of the squared $\ell_2$-norm evaluation directly below the equation text in the design document.

---

**Timestamp:** 2026-04-14

## [2026-04-14] - paper: Related Work asymmetric shift + `nguyen2024agreidv2` bib

### Files modified
- `paper/draft_5/sec/2_relatedwork.tex` — Condensed asymmetric / non-commutative domain-shift subsection; cite AG-ReID.v2~\cite{nguyen2024agreidv2} (replaces TODO).
- `paper/draft_5/main.bib` — Added `nguyen2024agreidv2` (IEEE TIFS 2024, DOI 10.1109/TIFS.2024.3353078; metadata from dblp).

### Problem this addresses
Verbose subsection and missing aerial--ground citation.

### Expected behavior
`\cite{nguyen2024agreidv2}` resolves; prose fits repetition-reducer + tighter scientific-register goals.

---

## [2026-04-14] - paper: tighten objective interference in `paper/draft_5/sec/3_methodology.tex`

### Files modified
- `paper/draft_5/sec/3_methodology.tex` — `\subsection{Objective interference analysis}` condensed (repetition-reducer + scientific-writing pass): single block for empirical claim, definition, mechanism, $\mathcal{L}_{\mathrm{dom}}$ / $\mathcal{L}_{\mathrm{dcc}}$ roles, and attribution caveat; preserves $-7.5\%$, Arm~5, and $\max\|\hat{\omega}\|$ numbers.

### Problem this addresses
Redundant sentences and parallel “pull” enumerations inflated the subsection without adding testable precision.

### Expected behavior
Faster read; §4.2 remains the numeric anchor for Arm~5.

---

## [2026-04-14] - paper: abstract avoids unintroduced dataset shorthand

### Files modified
- `paper/draft_5/sec/0_abstract.tex` — Replaced `M+MS+CS$\rightarrow$C3` with prose: multi-source DG-ReID with CUHK03 as the target domain (same experimental referent; symbols deferred to the main text).

### Problem this addresses
Abstract used source abbreviations and arrow notation before notation is defined.

### Expected behavior
Abstract remains self-contained; full protocol naming stays in Sec.~experimental setup.

---

## [2026-04-14] - paper: refresh `tab:multiseed` from multiseed rerun (Slurm 1522341, 1522344--1522347)

### Files modified
- `paper/draft_5/sec/4_experimentalResults.tex` — Table~\ref{tab:multiseed}: per-seed mAP/R1/R5/R10 and mean$\pm$std from final evaluation in `logs/multiseed_primary_c3/job_152234{1,4,5,6,7}_primary_unified_1c_w0.1_driftInst_seed{5,1,2,3,4}/log.txt`; reproducibility paragraph: Rank-1 $43.6\pm0.4\%$, R5/R10 aggregate stds; removed outdated Rank-1 $1.0\%$ std narrative.

### Problem this addresses
Multiseed primary recipe was re-run; paper still listed the previous sweep’s per-seed cells and $44.2\pm1.0\%$ Rank-1.

### Expected behavior
Multiseed statistics match the new logs (best-checkpoint final eval). Abstract/conclusion lines that only cite $43.8\pm0.3\%$ mAP remain numerically consistent (mean $43.82\%$ rounded to one decimal).

---

## [2026-04-14] - paper: apply draft\_5 triage plan to `paper/draft_5/**/*.tex`

### Files modified
- `paper/draft_5/sec/3_methodology.tex` — Added `eq:domain_token_loss` after `eq:total_loss`; composite-loss / primary-recipe note; `eq:domain_triplet` in domain-loss subsection; rewrote objective interference (identity vs.\ camera triplet, secondary $\mathcal{L}_{\mathrm{dom}}$, $\mathcal{L}_{\mathrm{dcc}}$ contrast); fixed `subsec:ablations` → `subsec:quantitative_analysis` forward ref.
- `paper/draft_5/sec/4_experimentalResults.tex` — Datasets labels; evaluation-metrics paragraph; multi-protocol table + 3-split averages + Protocol~3 wording; training list without $\mathcal{L}_{\mathrm{dom\text{-}tok}}$; primary-model and multi-target text aligned with provenance; multiseed paragraph; merged `tab:ablation_main` into `tab:aux_loss_sweep` (H1--H5); Arm~1b cells $44.3/44.8$; uniformity pointer + new supp ref; controlled-corruption rewrite (EL-10/11/12/16), optional `fig/fig_drift_monotonicity.pdf` figure with placeholder.
- `paper/draft_5/sec/5_discussion.tex` — Removed duplicate interference/future-work blocks (absorbed elsewhere); updated bidir gain wording.
- `paper/draft_5/sec/6_conclusion.tex` — Positive-first conclusion (EL-14 Option II numbers); future work retains Randers-as-supervision line with cites.
- `paper/draft_5/sec/0_abstract.tex` — Abstract aligned with headline reconciliation (EL-17).
- `paper/draft_5/sec/1_intro.tex` — Toy figure caption: `pp` → `\%` mAP.
- `paper/draft_5/sec/X_suppl.tex` — `tab:ablation_main` references → H-rows / `tab:aux_loss_sweep`; domain-triplet section points to main `eq:domain_triplet`; new `sec:suppl_uniformity_alpha`; bidirectional supp sentence updated.

### Problem this addresses
Executes the approved triage plan in-source: provenance-consistent numbers, merged tables, methodology equations, and reduced repetition.

### Expected behavior
PDF builds under a full CVPR toolchain; `pdflatex` in this environment may still fail if `cvpr.sty` is not installed locally. Place `fig_drift_monotonicity.pdf` under `paper/draft_5/fig/` to replace the corruption figure placeholder (or leave absent to use the framed placeholder).

---

## [2026-04-14] - docs: draft 5 triage plan — post-review integration (EL-07, EL-14, EL-17, publication_facts)

### Files modified
- `changelogs/draft_5_triage_plan.md` — Merged outcomes from `draft_5_plan_review_citations.md` (complete), `draft_5_plan_review_writing_coach.md`, and `draft_5_plan_review_repetition.md`: **EL-14** rewritten for provenance-clean **Option II** (44.3/44.8 vs 42.4, +1.9%; multiseed 43.8 ± 0.3%); **EL-07** prose (no `itemize`); **EL-02/04/05/08** tightened per reviews; new **EL-17** (abstract alignment); orchestrator handoff section; author decision **#7** for UV-CITE-004.
- `.cursor/paper/publication_facts.md` — New subsection **Headline_number_reconciliation** pairing headline mAP/Δ claims with jobs **1521465**, **1521474**, multiseed stats, and legacy **1521403** (no mixing with +1.9%).
- `.cursor/paper/unsourced_or_vague_claims.md` — **UV-CITE-004** marked **resolved** with pointer to the above.

### Problem this addresses
Prior agent left citation-substantiation finished but writing-coach/repetition flags on **EL-14** and **EL-07** not yet folded into the executable plan; **44.9/+1.9%** pairing was internally inconsistent.

### Expected behavior
Authors implement draft\_5 from a single reconciled plan; headline numbers and Δ claims trace to **Headline_number_reconciliation**; abstract edits deferred to **EL-17** after conclusion text is approved.

---

## [2026-04-14] - paper: drop SOTA comparison table from `paper/draft_4/4_experimentalResults.tex`

### Files modified
- `paper/draft_4/4_experimentalResults.tex` — Removed the DG-ReID comparison table (`tab:sota_comparison`, BAU Table~4 placeholder / TBD rows). No `\ref{tab:sota_comparison}` existed elsewhere in `paper/draft_4/`.

### Problem this addresses
Author requested removing that table and any references; prior-work numbers remain citable in prose via BAU~\cite{cho2024generalizable} where needed.

### Expected behavior
Table numbering in the compiled PDF shifts; no dangling labels for `tab:sota_comparison`.

---

## [2026-04-14] - paper: tabularx replaces resizebox in `paper/draft_4/*.tex`

### Files modified
- `paper/draft_4/4_experimentalResults.tex` — All tables: removed `\resizebox{...}{!}{...}`; use `\begin{tabularx}{\columnwidth}` for single-column floats and `\begin{tabularx}{\textwidth}` for `table*`; flexible text columns use `>{\raggedright\arraybackslash}X` (or `l` + `X` where noted); numeric columns stay `r`/`c` as before. Comment at top of section lists `\usepackage{tabularx}` for the main preamble.
- `paper/draft_4/X_suppl.tex` — Same for eval-drift Tables~A/B (`\textwidth`).

### Problem this addresses
Scaled tables (`\resizebox`) shrink body font relative to running text; `tabularx` fixes total width and lets columns absorb space so font size matches the document.

### Expected behavior
Tables span `\columnwidth` or `\textwidth` without scaling; long cells wrap inside `X` columns. Main file must load `tabularx` (and already needs `booktabs`, `multirow`).

---

## [2026-04-14] - paper: dataset table inline cites in `paper/draft_4/4_experimentalResults.tex`

### Files modified
- `paper/draft_4/4_experimentalResults.tex` — `tab:dataset_stats`: dropped the Reference column; moved each dataset’s citation next to its name in the first column (`lrrr` tabular).

### Problem this addresses
Redundant column when references can sit on the dataset label.

### Expected behavior
Same bibliographic pointers as before, narrower table.

---

## [2026-04-14] - paper: side-by-side dataset + protocol tables in `paper/draft_4/4_experimentalResults.tex`

### Files modified
- `paper/draft_4/4_experimentalResults.tex` — Combined `tab:dataset_stats` and `tab:protocol_splits` into one `table` float with two `minipage`s (56\%/42\% of `\linewidth`), each `\resizebox{\linewidth}{!}{...}`; single merged caption (Left/Right) with both `\label`s on the same float.

### Problem this addresses
Back-to-back full-width tables wasted vertical space; a unified caption reduces repetition.

### Expected behavior
Both table references still resolve to one table number; panels scale proportionally within the column (`\linewidth` matches `\columnwidth` in twocolumn column floats and `\textwidth` in single-column).

---

## [2026-04-14] - paper: condense datasets/protocols opening in `paper/draft_4/4_experimentalResults.tex`

### Files modified
- `paper/draft_4/4_experimentalResults.tex` — Merged the former opening paragraph plus `\paragraph{Multi-target evaluation.}` and `\paragraph{Metrics.}` into a single subsection paragraph; retained “subset of Protocol~2” scope and `\label{subsubsec:metrics}` on the metrics/ranking sentence.

### Problem this addresses
Redundant “following BAU / Protocol~2” framing and three micro-headings broke CVPR-style pacing at the start of Experimental Results.

### Expected behavior
One dense setup block states Protocol~2 alignment with BAU, the three leave-one-out splits (with primary split rationale), the six-split table reference, and CMC/mAP plus Finsler vs Euclidean ranking conventions.

---

**Timestamp:** 2026-04-13

## [2026-04-13] - docs: Idea‑1 1d/1e results analysis markdown in `bau/docs/`

### Files modified
- `bau/docs/Idea1_1d1e_results_analysis.md` — New document: slide talking points, for/against a Euclidean \(\mathcal{L}_{\mathrm{dom}}\) sweep for 1d/1e, paper‑substitute narrative, and pointers to `results/metric sweeps/` provenance.

### Problem this addresses
The compiled table / discussion from the Idea‑1 1d/1e ablation lived only in chat; the repo needed a single reference copy aligned with `changelogs/Idea1_Experiments_1d1e_Analysis.md` and sweep ledgers.

### Expected behavior
Authors can paste or adapt sections into `paper/draft_4/` and slides; empty Euclidean‑\(\mathcal{L}_{\mathrm{dom}}\) cells for 1d/1e are explicitly labeled unrun, not null results.

---

## [2026-04-13] - changelogs: align Idea1 1d/1e analysis with `drift_cross_camera_uniformity_loss` (raw drift)

### Files modified
- `changelogs/Idea1_Experiments_1d1e_Analysis.md` — §4.2 loss and notation: **unnormalized** $\omega_i$ (matches `bau/trainers.py` after 2026-04-10 no-norm change); corrected gradient sketch for upper-triangle pair sum; updated BAU parallel table, §5.1/§5.2/§5.4, and §6 table; clarified that Wang–Isola hyperspherical guarantee does not apply verbatim to 1e on the Randers ball.
- `bau/trainers.py` — Docstring/comment on `drift_cross_camera_uniformity_loss` updated to describe raw-drift operation (no `F.normalize`).

### Problem this addresses
Section 4.2 still described $L_2$-normalized drift and a gradient sum over all $j$ with $c_j \neq c_i$, which disagreed with the current trainer (Gaussian kernel on raw drift; upper-triangle mask). §5.2 incorrectly claimed 1e was norm-invariant.

### Expected behavior
The analysis document matches the implemented constraint set $\mathcal{C}_{1e}=\{(i,j): c_i\neq c_j,\, i<j\}$ and $\mathcal{L}_{1e}=\log\,\mathrm{mean}_{(i,j)\in\mathcal{C}_{1e}}\exp(-t\|\omega_i-\omega_j\|_2^2)$.

---

## [2026-04-13] - toy_paper_figures: nudge `100.0` down, `62.3` up on mAP plot

### Files modified
- `scripts/toy_paper_figures.py` — `VALUE_TEXT_NUDGE_POINTS`: `100.0` extra offset now includes **−3 pt** vertical (slightly lower); added `62.3` with **+5 pt** vertical (and kept `62.6` alias) so the low-Finsler label sits higher while preserving horizontal left nudge.

### Problem this addresses
Intro figure `fig_retrieval_mAP.pdf`: merged `100.0` sat slightly high; Finsler ~62.3 label needed a bit more clearance from the bottom.

### Expected behavior
Re-run `scripts/regenerate_toy_figures.py` (or toy retrieval stage) to refresh PDFs.

---

## [2026-04-13] - toy paper figures: shared `toy_paper_figures.py`, Eucl-above / Fins-below mAP labels, draft\_4 sync

### Files modified
- `scripts/toy_paper_figures.py` — New module: `make_corruption_strip`, `make_retrieval_mAP_plot` (Euclidean annotations above markers, Finsler below; merged gray label when curves coincide; one-decimal value nudges for `100.0` / `68.8` / `62.6` to avoid clipping past the axes), `copy_paper_outputs` (hook PDFs + optional full output mirror).
- `scripts/regenerate_toy_figures.py` — Delegates to `toy_paper_figures`; fixes `sys.path` so imports work when run as `python scripts/regenerate_toy_figures.py`; optional `--also-mirror-to`.
- `scripts/toy_dataset_analysis.py` — Uses shared paper hooks (`fig_corruption_strip.pdf`, `fig_retrieval_mAP.pdf`); keeps optional `fig_paper_corruption_strip.pdf` (ω norms per panel, **no suptitle**) for draft\\_3; adds `--paper-fig-dir`, `--paper-mirror-dir`; paper-style cleanup on other figures (no suptitles / plot titles where they duplicated captions; Spearman stats as small axes text on drift monotonicity; higher DPI/pad for exports).

### Problem this addresses
Paper `draft_4` hooks must match regeneration styling; mAP value labels needed Euclidean-above / Finsler-below placement with manual nudges for edge labels; full toy run should refresh `paper/draft_4/fig` and optionally mirror the whole results folder.

### Expected behavior
`python scripts/regenerate_toy_figures.py` and the retrieval/strip stages of `toy_dataset_analysis.py` produce the same hook PDF styling; `fig_retrieval_mAP.pdf` places Eucl values on top of markers and Fins values underneath, with targeted shifts for 100.0 / 68.8 / 62.6.

---

## [2026-04-13] - sbatch: MSMT17-target redo for protocol23_primary multitarget (OOM)

### Files modified
- `sbatch/protocol23_primary_multitarget_msmt17_redo.sbatch` — New 2-task array (`0` = P2 `p2_m-cs-c3_dt-msmt17`, `1` = P3 `p3_mdg-cs-c3dg_dt-msmt17`) mirroring `protocol23_primary_multitarget.sbatch` tasks 1 and 4; SLURM `--mem` raised from 32G to **64G**; distinct job/log names and W&B run names/tags (`protocol23-msmt17-redo`, `oomfix-mem64G`) so results do not collide with the failed parent job.

### Problem this addresses
Array tasks targeting MSMT17 in job `1521463` were **cgroup OOM-killed** during MSMT17 evaluation (SLURM: `oom_kill event`, `srun: ... Out Of Memory` in `logs/slurm_logs/finsler-p23-primary-1521463_{1,4}.err`), after training had started; MSMT17’s large test split makes eval memory-heavy under the original 32G RAM limit.

### Expected behavior
Submit with `sbatch sbatch/protocol23_primary_multitarget_msmt17_redo.sbatch`; both runs should complete full training and eval without OOM if the cluster grants 64G RAM. If OOM persists, increase `--mem` further or reduce eval batch/workers in `train_bau` (not changed here).

---

**Timestamp:** 2026-04-12

## [2026-04-12] - toy_dataset_analysis: `fig_retrieval_mAP` matches combined-paper (a) style (no table)

### Files modified
- `scripts/toy_dataset_analysis.py` — `plot_retrieval_mAP`: removed under-plot table; restored **short** legend strings (`Finsler BAU ($d_F$)`, `Euclidean BAU ($d_E$)`), `fontsize=9` legend with `loc="upper right"`, `framealpha=0.95`, **one-decimal** annotations at each marker (`xytext` offsets $(8,10)$ and $(-8,-12)$, `fontsize=8`), axis label `fontsize=11`, padded `ylim`, `clip_on=False` on lines—same as the former `fig_combined_paper.pdf` panel (a).

- `paper/draft_3/4_experimentalResults.tex` — Fig.~\ref{fig:corruption_analysis} caption: values **next to markers**, not tabulated below.

- `docs/toy_dataset_synthesis.md` — Output blurb updated.

### Problem this addresses
Authors preferred marker-adjacent mAP labels and combined-paper legend formatting without the bottom table.

### Expected behavior
`fig_retrieval_mAP.pdf` is a single axes with curve annotations only; re-run `toy_dataset_analysis.py` to regenerate.

---

## [2026-04-12] - toy_dataset_analysis: split paper figures; mAP table under line plot

### Files modified
- `scripts/toy_dataset_analysis.py` — Removed `plot_combined_paper_figure` and `fig_combined_paper.pdf`. Added `plot_paper_corruption_strip` → `fig_paper_corruption_strip.pdf` (single row of five crops). `plot_retrieval_mAP` uses **gridspec**: line plot + **tabular** mAP values (no on-curve text). Main writes the strip after retrieval metrics.

- `paper/draft_3/4_experimentalResults.tex` — Fig.~\ref{fig:corruption_analysis} stacks `fig_retrieval_mAP.pdf` and `fig_paper_corruption_strip.pdf`.

- `docs/toy_dataset_synthesis.md` — Output list updated.

### Problem this addresses
Combined two-panel PDF still had cramped mAP labels; authors asked for **two separate** generated PDFs for the paper panels.

### Expected behavior
Regenerating the toy analysis produces `fig_retrieval_mAP.pdf` (plot + table) and `fig_paper_corruption_strip.pdf` independently; both are required for the controlled-corruption figure in draft\_3.

---

## [2026-04-12] - paper/draft_3/X_suppl.tex: eval-drift tables use `c` columns (no siunitx `S`)

### Files modified
- `paper/draft_3/X_suppl.tex` — Replaced `S[table-format=...]` column specs with `@{ }lc*{6}{c}@{ }` for Tables A and B (eval-drift supplementary). Header row uses plain `c` cells (removed brace-wrapped literals intended for `S` columns).

### Problem this addresses
`S` columns require `\usepackage{siunitx}` and treat cells as parsed numerics; `\textbf{...}` and signed $\Delta$ cells conflict with that model, causing build errors or fragile markup when the main document does not load `siunitx`.

### Expected behavior
Supplementary eval-drift tables compile with only `booktabs`/`multirow`-class dependencies; alignment remains centered for all numeric and $\Delta$ cells.

---

## [2026-04-12] - toy_dataset_analysis: fix retrieval mAP plot annotations and y-limits

### Files modified
- `scripts/toy_dataset_analysis.py` — `plot_retrieval_mAP` and panel (a) of `plot_combined_paper_figure`: numeric labels use horizontal offsets (Finsler left-above marker, Euclidean right-below) so coincident curves no longer overlap; `set_ylim` adds bottom/top padding; `clip_on=False` on lines; `savefig(..., pad_inches=0.2)`; legend `loc="upper right"` with `framealpha=0.95` to stay clear of the downward trend.

### Problem this addresses
mAP value annotations stacked on the same vertical offsets and clipped past the axes for high mAP values.

### Expected behavior
`fig_retrieval_mAP.pdf` / combined paper panel (a) show readable, non-overlapping labels inside padded y-range after re-running the script.

---

## [2026-04-12] - Toy analysis: subsampled ω/identity PCA + optional t-SNE; paper figure stub

### Files modified
- `scripts/toy_dataset_analysis.py` — Added `subsample_pids_with_full_severity`, `gather_identity_drift_arrays`, and `plot_omega_identity_pca_panels` (PCA on $\hat{\mathbf{z}}^{\omega}$ with per-identity polylines $s{=}0{\rightarrow}4$, separate PCA on $\mathbf{z}^{\mathrm{id}}$; optional `sklearn` $t$-SNE scatter on $\omega$ only, non-metric). New CLI: `--viz-num-identities`, `--viz-seed`, `--fig-omega-identity-pca`, `--viz-tsne`, `--fig-omega-tsne`, `--tsne-perplexity`, `--tsne-seed`. Writes `drift_projection_provenance.json` under `--output-dir`.

- `paper/draft_3/4_experimentalResults.tex` — `\input{fig_toy_omega_projection}` after the controlled-corruption figure; qualitative subsection references Fig.~\ref{fig:toy_omega_identity_pca} and PCA vs.\ $t$-SNE caveats.

### Files added
- `paper/draft_3/fig_toy_omega_projection.tex` — Figure environment + caption for `fig_omega_identity_pca.pdf` (copy from `results/toy_analysis/` into the LaTeX build directory or set `\graphicspath`).

### Problem this addresses
Provide a **reviewer-defensible**, linear (PCA) qualitative view of the drift slice vs.\ the identity slice on the toy protocol, plus optional exploratory $t$-SNE, aligned with the paper’s asymmetric-drift narrative.

### Expected behavior
Running `toy_dataset_analysis.py` emits `fig_omega_identity_pca.pdf` and provenance JSON; with `--viz-tsne`, also `fig_omega_tsne.pdf`. PDFs are not committed here; authors sync them into the paper tree for compilation.

---

## [2026-04-12] - ToyCorruption v2.0: ImageNet-C–aligned pipeline, metadata schema, synthesis doc

### Files modified
- `scripts/generate_toy_dataset.py` — Replaced v1 pipeline (blur → JPEG → resize → additive brightness) with **defocus blur → horizontal motion blur → bilinear down/up → gamma tone → Poisson + Gaussian noise → JPEG**, anchored to Hendrycks & Dietterich (ICLR 2019) / ImageNet-C operator families. Added `TOY_DATASET_VERSION` `"2.0"`, deterministic noise RNG per `(seed, pid, level)`, and **`metadata.json` schema v2** (`dataset` + `images`). Motion blur uses **NumPy-only** averaging (no SciPy runtime dependency).

### Files added
- `docs/toy_dataset_synthesis.md` — Full design synthesis: C1–C3 claims, pipeline order, literature mapping, exact severity table, limitations, regeneration (including stale-file warning).

### Files regenerated (local run)
- `examples/data/ToyCorruption/` — 50 identities × 5 severities; `results/toy_analysis/` figures and JSON metrics refreshed via `toy_dataset_analysis.py` (BAU conda env).

### Problem this addresses
Toy degradations needed explicit **community benchmark** justification, more **camera-relevant** diversity (motion, exposure-like gamma, sensor noise), and **documented provenance** for paper/reviewer use.

### Expected behavior
Drift validation remains defined on **severity = cam_id − 1**. Correlation of drift magnitude with severity should stay **positive** on sufficiently large identity samples; exact Spearman values depend on the Finsler checkpoint. Regenerating into a non-empty output tree without clearing may **mix old and new** image counts—clear `bounding_box_test/`, `query/`, and `gallery/` first when changing `--num-identities`.

---

## [2026-04-12] - Protocol-2 sample images: three identities, two cameras each

### Files added
- `protocol2_identity_pairs/` — Six crops copied from local DG-ReID roots: Market-1501 (pid 2, cameras 1 vs 6), MSMT17 (pid 953, cameras 1 vs 15), CUHK03-NP detected (pid 1056, cameras 1 vs 2). Renamed for clarity.

### Problem this addresses
Need a small, reproducible set of same-ID cross-camera pairs from Protocol-2–relevant datasets for qualitative figures or inspection.

### Expected behavior
CUHK-SYSU is omitted here: `bau/datasets/cuhksysu.py` assigns a dummy camera id to all SYSU crops, so distinct `cid` pairs are not defined in this codebase.

---

## [2026-04-12] - Paper draft\_3: integrate draft tables via \texttt{\\input}

### Files modified
- `paper/draft_3/4_experimentalResults.tex` — `\paragraph{Historical loss-structure ablations}` now introduces Table~\ref{tab:ablation_main} and `\input`s `draft_table_main` (with `\IfFileExists` fallback for different main-file locations).
- `paper/draft_3/X_suppl.tex` — New Sec.~\ref{sec:suppl_eval_drift_tables} with bridging prose and `\input` of `draft_table_supplementary`.

### Problem this addresses
Table bodies lived in standalone files and were not part of the compiled sections.

### Expected behavior
Building the paper from `paper/draft_3/` or `paper/` picks up the correct table fragments; supplementary eval-drift tables appear under the new supplement section.

---

## [2026-04-12] - Paper draft\_3: replace internal experiment codes with formal loss names

### Files modified
- `paper/draft_3/3_methodology.tex` — Renamed: identity-sliced triplet; $\mathcal{L}_{\mathrm{dcc}}$ (cross-camera drift coherence), $\mathcal{L}_{\mathrm{sca}}$ (same-camera drift attraction); labels `subsec:unified_vs_identity_sliced`, `eq:loss_dcc`; weights $w_{\mathrm{dcc}}$, $w_{\mathrm{sca}}$.
- `paper/draft_3/4_experimentalResults.tex`, `0_abstract.tex`, `1_intro.tex` — Propagated notation; auxiliary table uses DCC/SCA abbreviations with caption definitions; `EXP` $\rightarrow$ `Arm`.
- `paper/draft_3/draft_table_main.tex`, `paper/draft_3/draft_table_supplementary.tex` — Method column uses descriptive names (identity-sliced triplet, domain triplet only, $\mathcal{L}_{\mathrm{dcc}}$, etc.).
- `.cursor/paper/publication_facts.md` — Recommended primary row relabeled with $\mathcal{L}_{\mathrm{dcc}}$ (INTERNAL ledger still uses historical sweep labels elsewhere).

### Problem this addresses
Internal labels (1a, 1c, 1d) are not appropriate for formal submission prose.

### Expected behavior
Main text and tables read in standard terminology; equation references use `\eqref{eq:loss_dcc}`.

---

## [2026-04-12] - Paper draft\_3: results lock, methodology objective, experiments section

### Files modified
- `.cursor/paper/publication_facts.md` — Set `Drafting_phase` to `results_implementation_lock`; added **Auxiliary loss configuration sweep (job 1521019)** under `Reported_metrics_with_provenance`.
- `paper/draft_3/3_methodology.tex` — New subsections: unified vs.\ 1a triplet; drift auxiliaries $\mathcal{L}_{1c}$/$\mathcal{L}_{1d}$; total objective eq.~\eqref{eq:total_loss}; interference subsection closing sentence updated.
- `paper/draft_3/4_experimentalResults.tex` — Quantitative analysis (primary model, experiment inventory, outstanding runs), template tables for multi-protocol and extended ranks, auxiliary sweep Table~\ref{tab:aux_loss_sweep}, ablation narrative (BAU-style).
- `paper/draft_3/0_abstract.tex`, `paper/draft_3/1_intro.tex` — Aligned with results lock and primary configuration wording.
- `paper/draft_3/draft_table_main.tex` — Header comment updated for `results_implementation_lock`.

### Problem this addresses
Theory-first constraints no longer match the author’s shift to locking results and documenting the best auxiliary configuration (unified + 1c, instance drift); methodology lacked the full composite loss and experiments section was largely empty.

### Expected behavior
Agents reading `publication_facts.md` treat benchmark numbers and job IDs as admissible in main text where cited; draft\_3 states the provenanced primary model, full $\mathcal{L}$ decomposition, remaining benchmark placeholders, and auxiliary sweep table.

### Follow-up (same day)
- Post `paper-scientific-writing-coach` / `paper-repetition-reducer` pass: tightened abstract closing, split intro contributions (iv)/(v), replaced “formalize” interference wording, pointed total-objective primary recipe to `tab:aux_loss_sweep`, added corruption-figure caption takeaway.

---

## [2026-04-12] - Collate sweep_auxiliary_loss_configs metrics from job logs

### Files modified
- `results/metric sweeps/sweep_auxiliary_loss_configs_metrics.md` (new) — Final evaluation (best checkpoint) mAP and CMC (R1, R5, R10) for all 12 runs under `logs/sweep_auxiliary_loss_configs/job_1521019_*`, parsed from each `log.txt` after `Loaded best model for final evaluation`.

### Problem this addresses
No single table summarized the completed auxiliary-loss sweep (job 1521019).

### Expected behavior
Provenance table matches numbers in per-job `log.txt`; notes call out best mAP/R1 cells and EXP 5 regression.

---

**Timestamp:** 2026-04-10

## [2026-04-10] - Paper Draft 3: condense controlled corruption subsection

### Files modified
- `paper/draft_3/4_experimentalResults.tex` — Tightened `\subsection{Controlled corruption analysis}`: single setup paragraph; C2/C1/C3 without repeating suppression-vs-retrieval and benchmark-vs-toy claims; shortened Fig.~\ref{fig:corruption_analysis} panel (b) caption to avoid duplicating Spearman and drift magnitude from the body; one interpretation paragraph (diagnostic + Sec.~\ref{subsec:domain_conditioned_drift} link + open direction).

### Problem this addresses
The subsection restated the same conclusions (monotonic drift, small norm, training suppression) across C2, the figure caption, and Interpretation.

### Expected behavior
Readers get one clear arc: controlled toy protocol $\rightarrow$ drift tracks severity but stays small; identity absorbs corruption; two-checkpoint retrieval comparison; why this matters and what is left open.

---

**Timestamp:** 2026-04-10

## [2026-04-10] - toy_dataset_analysis: independent Euclidean vs Finsler checkpoints

### Files modified
- `scripts/toy_dataset_analysis.py` — Retrieval mAP and qualitative rank-list compare **two checkpoints**: Euclidean BAU `resnet50` (default `logs/AGReIDv2_sweep/job_1502418_dt_cuhk03_EucBaseline/best.pth`) with Euclidean distance on 2048-d embeddings vs. Finsler `resnet50_finsler` (`--finsler-checkpoint`) with `finsler_drift_dist` on full 4096-d embeddings. Replaces the prior same-checkpoint identity-only vs `d_F` comparison. Added `load_bau_euclidean_resnet50` and `baseline_euclidean_resnet50` on retrieval helpers. `retrieval_metrics.json` uses keys `euclidean_bau_resnet50_dE`, `finsler_bau_dF`, plus `euclidean_checkpoint` / `finsler_checkpoint` paths.

### Expected behavior
- `--euclidean-checkpoint` defaults to the path above; use another BAU `resnet50` `best.pth` with matching `--num-classes` if needed.

---

**Timestamp:** 2026-04-10

## [2026-04-10] - Paper Draft 3: Architecture figure — encoder parallelogram + input arrows

### Files modified
- `paper/draft_3/fig_architecture.tex` — BAU encoder drawn as a parallelogram via `path picture` (vertical left edge, top-right vertex shifted for a slanted right edge; avoids `shape=parallelogram`, which is missing on some TeX Live installs). Weak/strong view arrows now terminate on the **left face** using projection onto the `north west`–`south west` segment (`calc`).

### Expected behavior
The figure compiles with `shapes.geometric` + `calc` as documented in the file header; inputs meet the encoder’s left margin at the correct heights.

---

**Timestamp:** 2026-04-10

## [2026-04-10] - Paper Draft 3: BAU-style architecture figure polish

### Files modified
- `paper/draft_3/fig_architecture.tex` — Removed unused stop-gradient arrow style; folded default-eval note into the structured-feature node (avoids a misleading extra “flow” to a side box); minor encoder node typography (`\textbf{BAU encoder}` + fixed width). Figure remains BAU-level abstract: dual views, shared BAU encoder, split identity/drift, structured embedding, caption points to methodology and experimental-results sections for objectives.

### Problem this addresses
Post-review polish after the BAU-style abstraction: fewer visual elements competing for attention and no arrow implying eval is a forward pass from $\mathbf{z}$.

### Expected behavior
`pdflatex` on a document that inputs `fig_architecture.tex` (with TikZ libraries as in the figure header comment) compiles; cross-references resolve when the full paper defines the cited labels and bibliography.

---

**Timestamp:** 2026-04-10

## [2026-04-10] - Controlled corruption toy dataset and drift vector validation

### Files added
- `scripts/generate_toy_dataset.py` — Generates a synthetic dataset from Market-1501 test identities with 5 severity levels of composite camera-realistic corruptions (blur, JPEG, downscale, brightness). Each severity level is assigned a distinct synthetic camera ID. Output follows Market-1501 naming convention.
- `scripts/toy_dataset_analysis.py` — Standalone analysis script: loads Finsler and Euclidean checkpoints, extracts full [identity|drift] embeddings on the toy dataset, computes per-severity drift magnitude and identity distance metrics, runs clean→corrupted retrieval experiments, and generates paper-ready figures (quantitative, retrieval mAP with under-plot table, corruption strip, qualitative rank-list, optional ω PCA).

### Files modified
- `paper/draft_3/4_experimentalResults.tex` — Added new §4.3 "Controlled corruption analysis" with three diagnostic claims (C1: identity sensitivity, C2: drift monotonicity, C3: asymmetric retrieval), figure reference, and interpretation paragraph. Honest reporting: drift magnitude is statistically monotonic (Spearman ρ>0.5, p<1e-17) but too small (~0.05) to alter retrieval rankings.

### Generated outputs
- `results/toy_analysis/drift_identity_metrics.json` — Per-severity drift norm and identity distance statistics with Spearman correlation.
- `results/toy_analysis/retrieval_metrics.json` — mAP/Rank-1 for Euclidean vs Finsler ranking at each gallery severity.
- `results/toy_analysis/fig_drift_monotonicity.pdf` — Dual-axis plot: drift magnitude (monotonic increase) and identity distance (substantial increase) vs. severity.
- `results/toy_analysis/fig_retrieval_mAP.pdf` — mAP vs. gallery severity for Finsler $d_F$ vs Euclidean $d_E$, with tabulated values under the plot.
- `results/toy_analysis/fig_retrieval_qualitative.pdf` — Rank-list comparison panels.
- `results/toy_analysis/fig_paper_corruption_strip.pdf` — Single-row corruption strip with $\|\omega\|$ per severity (paper panel).
- `/home/stud/leez/storage/user/reid/data/ToyCorruption/` — 250 images (50 identities × 5 severity levels).

### Key findings
1. **Drift monotonicity confirmed:** Drift magnitude increases monotonically with corruption severity (ρ=0.51, p=2.5e-18), proving the drift branch responds to controlled camera-like degradation.
2. **Drift magnitude suppressed:** Absolute drift norms remain in [0.053, 0.062], far too small to materially alter pairwise distances. Domain-invariant training (alignment + uniformity) actively suppresses drift.
3. **Identity NOT invariant:** Identity distance from clean grows 0→1.26 across severity levels. The identity branch absorbs corruption variation rather than delegating it to drift.
4. **Retrieval delta zero:** Finsler d_F and Euclidean d_E produce identical mAP at all severity levels. The drift vector does not contribute to ranking under the current recipe.
5. **Interpretation:** The experiment confirms the eval-drift-true sweep finding (near-zero ΔmAP) with a controlled, interpretable diagnostic. The drift branch has the correct directional response but insufficient magnitude to be useful.

### Problem this solves
The paper claims asymmetric retrieval geometry is motivated by non-commutative camera degradation, but no prior experiment isolated whether the drift vector actually encodes this. The toy dataset provides controlled evidence that the drift branch responds to severity but is suppressed by the training stack.

### Expected behavior after this patch
Running `python scripts/generate_toy_dataset.py` generates the toy dataset. Running `python scripts/toy_dataset_analysis.py --finsler-checkpoint <path>` produces all figures and metrics. The paper draft now includes a §4.3 with honest reporting of the diagnostic results.

---

**Timestamp:** 2026-04-08

## [2026-04-08] - Paper Draft 3: Conciseness tightening pass (Sec 3, 4.2, supplementary)

### Files modified
- `paper/draft_3/3_methodology.tex` — Full rewrite for CVPR-level conciseness
- `paper/draft_3/4_experimentalResults.tex` — Sec 4.2 implementation details tightened
- `paper/draft_3/X_suppl.tex` — Two new supplementary sections added
- `paper/draft_3/main.bib` — Two citations added

### Problem
Sections 3 (Methodology) and 4.2 (Implementation Details) were verbose and contained documentation-style language, repeated arguments across paragraphs, and tangential derivations that interrupted the main narrative flow. Six claim-calibration warnings were identified (metric axiom contradiction, overstated guarantees, hypothesized mechanisms stated as fact, loaded language, borderline single-factor attribution). Two citations were missing.

### Changes applied

**Section 3 (Methodology):**
- Cut the 8-line roadmap paragraph; replaced with one-sentence preamble
- Merged "Structured embedding" and "Orthogonal decomposition rationale" into one tighter paragraph with Supp. forward-ref
- Tightened "Randers-type distance": removed redundant recovery-limit restatement; inlined drift-dim-scaling as one sentence with Supp. forward-ref
- Tightened "Randers positivity": replaced "breaking the metric axiom" (contradicts disclaimer of metric status) with "producing non-physical negative scores that destabilize triplet mining"
- Tightened "Identity-only alignment": merged two sentences making the same point
- Tightened "Bidirectional triplet": cut obvious symmetric-case sentence; compressed from 8 lines to 5
- Tightened "Drift norm regularization": removed redundant divergence explanation; removed duplicate hard-clamping contrast (already in Sec 3.2); added Boyd & Vandenberghe citation
- Tightened "Domain-conditioned drift": replaced "simplicity bias" causal assertion with hedged observation; compressed where-clause; replaced "corrupting" with neutral "modifying"; compressed test-time paragraph to one sentence
- Tightened "Interference analysis": replaced "documented to produce" with "observed to coincide with"; added single-factor attribution caveat per publication_facts.md

**Section 4.2 (Implementation Details):**
- Compressed backbone paragraph to one sentence
- Compressed instance drift head: removed documentation-style parentheticals, ornamental phrases
- Compressed domain-conditioned drift head: removed redundant norm-scaling restatement
- Compressed training recipe: removed internal method names (SLERP, analytical); removed "when enabled" config-toggle language
- Compressed batch construction: removed class name; replaced absolute gradient claim with hedged version
- Compressed memory bank: removed "depending on configuration"; replaced repeated alignment rationale with cross-ref
- Compressed evaluation: removed CLI flag name; removed "without retraining"

**Supplementary (X_suppl.tex):**
- Added Sec "Orthogonal Decomposition Rationale" with full degenerate-case argument and Bousmalis analogy
- Added Sec "Drift Dimension Scaling Derivation" with Cauchy-Schwarz bound and gamma correction

**Citations (main.bib):**
- Added `zhong2017re` (Zhong et al. 2017, k-reciprocal encoding) — cited for reciprocal k-NN Jaccard weights
- Added `boyd2004convex` (Boyd & Vandenberghe 2004, Convex Optimization) — cited for log-barrier penalty

### Expected behavior
- Sec 3 is ~1 column shorter while retaining all formal definitions and equations
- Sec 4.2 is ~0.5 column shorter with no documentation-style language
- All 6 WARN-level claim calibration issues resolved
- Two missing citations substantiated
- Derivations and rationale offloaded to supplementary with forward-refs from main text

---

## [2026-04-08] - Paper Draft 3: TikZ architecture figure rewrite (dense reference style)
**Files Modified:** `paper/draft_3/fig_architecture.tex`
**Files Created (earlier):** `.cursor/skills/tikz-academic-figure/SKILL.md`, `.cursor/agents/tikz-figure-reference-crawler.md`, `.cursor/agents/tikz-codebase-architecture-concretizer.md`, `.cursor/agents/tikz-qa-reviewer.md`

### Problem Addressed
The initial architecture figure had excessive whitespace, included legacy/inactive modules, and did not match the dense reference style requested. Losses were floating above the main flow rather than stacked on the right. Several arrow paths crossed through node interiors.

### Changes Made
- **`fig_architecture.tex`** completely rewritten:
  - Dense layout modelled after the user's reference diagram: minimal whitespace, explicit sub-modules with math formulas inside blocks, tensor shape annotations on arrows.
  - Legacy/inactive modules (alpha gate, domain triplet, cross-domain contrastive) **removed** from the diagram entirely.
  - Input shows weak/strong augmentation split feeding the shared backbone.
  - Drift head expanded to show all sub-modules: Domain Classifier, Softmax/τ, Embed+Proj, Gate g, ⊙, Residual MLP, +, σ-Gate, Gram–Schmidt.
  - Losses stacked vertically in a dedicated right column: CE, Triplet (id-only, d_E), Alignment, Uniform, Domain (d_F, full), L_ω (log-barrier), Domain-tok CE.
  - Grouping boxes for Identity Branch (blue) and Domain-Conditioned Drift Head (orange).
  - Stop-gradient path from identity to domain classifier shown with dashed arrow and sg[z^id] label.
  - Notation aligned with 3_methodology.tex after QA review: sg[z^id] (not hat), ‖ω‖ (not hat-z^ω) on barrier.
  - Caption includes abstraction note about pool/BN ordering vs implementation.
- **Agent/skill infrastructure** created in earlier pass (unchanged).

### Expected Behavior
`fig_architecture.tex` compiles cleanly with `pdflatex` (tested). Insert via `\input{fig_architecture}` in a document loading `tikz`, `xcolor`, `amsmath`, and TikZ libraries `positioning, fit, arrows.meta, calc, backgrounds, shapes.geometric`. Uses `figure*` with `\resizebox{\textwidth}{!}`.

### Timestamp: 2026-04-08

---

## [2026-04-06] - Paper Draft 2: CVPR-style LaTeX table drafts produced (pre-draft artifacts)
**Files Created:** `paper/draft_2/draft_table_main.tex`, `paper/draft_2/draft_table_supplementary.tex`

### Problem Addressed
Experiments section needs provenanced table drafts ready for insertion once `results_implementation_lock` is set. The `paper-table-formatter` skill was applied; user explicitly authorized including real metric values in these pre-draft artifacts despite `theory_first` being active.

### Changes Made
- `draft_table_main.tex`: Single `\resizebox{\columnwidth}{!}` booktabs table, 7 arms (Arm 1 baseline, 5 Arm 2 variants, Arm 3 Unified Finsler), columns = Method / Eval / mAP / Rank-1. All rows carry `% Provenance: job XXXXXX` comments. `\textbf{}` applied to Baseline (1a), 1a+1c, and Unified Finsler as candidate method rows (author must confirm final selection before submission).
- `draft_table_supplementary.tex`: Two `table*` environments — Table A (FD training, eval-drift δ, 8 columns: Arm / Best cond. / Eucl. mAP / Finsler mAP / ΔmAP / Eucl. R1 / Finsler R1 / ΔR1) and Table B (ED training, same columns). Table B isolates the 1b arm ΔmAP = −6.1 diagnostic under Euclidean-domain training.

### Expected Behavior
Both files compile as standalone table fragments; import with `\input{draft_table_main}` / `\input{draft_table_supplementary}`. `siunitx` `S` column alignment is used for numeric columns; `\usepackage{siunitx,booktabs,multirow}` required in preamble.

### Timestamp: 2026-04-06

---

## [2026-04-06] - Paper Draft 2: Methodology structural edits (EL-001–EL-004, CF-001–CF-007, citation)
**Files Modified:** `paper/draft_2/markdowns/3_methodology.tex`, `paper/draft_2/markdowns/2_relatedwork.tex`, `paper/draft_2/markdowns/1_intro.tex`, `paper/draft_2/markdowns/main.bib`, `paper/draft_2/markdowns/X_suppl.tex` (new, replaces `X_suppl.md`)

### Problem Addressed
Multiple structural, notational, and citation issues identified in the triage round for `3_methodology.tex` required author-approved edits:
- Prior narrative assumed a sequential development from Euclidean → domain Finsler → unified Finsler; code inspection confirmed both were parallel tests over the same `finsler_drift_dist`.
- `alpha` parameter accepted by `finsler_drift_dist` in both BAU and ReNorm2 codebases is not used in the computation body; omega drift vector is the current learnable component.
- Notation mixed superscript (`\mathbf{z}^{\mathrm{id}}`) and subscript (`\mathbf{x}_{\mathrm{id}}`) for split-embedding slices across equations.
- `X_suppl.md` incorrectly stated alpha and omega "coexist in the final implementation".
- Citation for LOO DG-ReID template attributed to "Luo et al." — correct first author is Zhao (zhao2021m3l).
- Label collision `eq:finsler_distance` used in both related work and methodology.

### Changes Made
- **EL-001/EL-003 (Option B restructure + notation):** Reordered §3 sections — §3.2 now defines split embedding and $d_F$ first; §3.3 applies $d_F$ to domain objectives; §3.4 is interference analysis. All split-embedding slice notation standardized to superscript throughout.
- **EL-001 bridging paragraph:** Integrated bridging derivation (canonical Randers → sample-adaptive midpoint form) into the `\paragraph{Randers-type distance.}` in §3.2. Finsler metric disclaimer moved here from related work (CF-006).
- **EL-002:** Replaced `\paragraph{Asymmetric domain loss.}` placeholder with full $\mathcal{L}_{\mathrm{dom}}$ formulation using $d_F$ directly (no alpha).
- **EL-004:** Replaced `\paragraph{Learnable Euclidean–Asymmetric Interpolation.}` with a one-sentence supplementary pointer. `X_suppl.md` converted to `X_suppl.tex` (CF-003) with A.2 rewritten to remove "coexist" claim, A.3 updated to state alpha pegs to $\alpha_{\max}$ in all runs (shortcut behavior), A.4 reframed as research stepping stone superseded by omega architecture.
- **CF-001:** Fixed `hermans2017defense` from `@inproceedings` (arXiv booktitle) to `@misc` with proper eprint fields.
- **CF-002:** Renamed `\label{eq:finsler_distance}` → `\label{eq:finsler_dist_canonical}` in `2_relatedwork.tex`; methodology `eq:finsler_distance` label now free for $d_F$.
- **CF-004:** Updated `1_intro.tex` line 8 to forward-reference `\ref{subsec:interference_analysis}` instead of stating quantitative claim inline.
- **CF-005:** Shortened "inductive bias / finite capacity" phrase in `2_relatedwork.tex` §2.3.
- **CF-006:** Moved Finsler manifold disclaimer from `2_relatedwork.tex` §2.4 last paragraph into methodology §3.2 bridging content; replaced with forward reference in related work.
- **CF-007:** Trimmed redundant §2.2 "Euclidean features / asymmetry via score" explanation to a forward reference.
- **Citation:** Verified and added `zhao2021m3l` (Zhao et al., CVPR 2021 — first author is Zhao, not Luo) to `main.bib`; inserted `\cite{zhao2021m3l}` alongside `\cite{cho2024generalizable}` in §3.4.
- **Writing coach pass:** Applied CVPR-style concision edits (claim-first topic sentences, removed narration, compressed proxy-chain clause, hedged "produces" → "is documented to produce" in theory-first section).

### Expected Behavior
`3_methodology.tex` compiles with no forward-reference tension: $d_F$ is fully defined before use. All citations are well-formed. Supplementary cross-references to `\ref{sec:suppl_alpha}` resolve via the new `X_suppl.tex`.

---

**Timestamp:** 2026-04-05

## [2026-04-05] - Sbatch: `sweep_eval_drift_true_consecutive` repo root + checkpoint discovery
**Files Modified:** `sbatch/sweep_eval_drift_true_consecutive.sbatch`

### Problem Addressed
A consecutive eval job reported **Found 0 checkpoints** because discovery used a hardcoded `ROOT` and/or relative `find` paths that did not match the directory Slurm used as submit cwd, or because a missing sweep subtree caused brittle `find` behavior.

### Modification
- Resolve **`ROOT`** as `REPO_ROOT` (env) → `SLURM_SUBMIT_DIR` → parent of `sbatch/` via `BASH_SOURCE`.
- Search each sweep tree with **`find "${ROOT}/${d}"`** separately; skip missing dirs with a warning.
- If zero `model_best.pth.tar` files: print **diagnostics** (subdirs + sample `*.pth.tar`) and **exit 1** with fix hints; optional commented **`#SBATCH --chdir=...`** in the header.
- Document **`REPO_ROOT=... sbatch ...`** override.

### Expected Behavior
Checkpoints are found when the user submits from the repo root or sets `REPO_ROOT` / `--chdir`; empty runs fail fast with actionable output instead of a silent zero-length sweep.

**Timestamp:** 2026-04-05

### Follow-up (same day)
**Problem:** Discovery still found 0 files because **`examples/train_bau.py` saves `best.pth`**, not `model_best.pth.tar` (see `torch.save(..., 'best.pth')` around the best-mAP hook).

**Modification:** Both `sweep_eval_drift_true_consecutive.sbatch` and `sweep_eval_drift_true.sbatch` now search for **`best.pth` or `model_best.pth.tar`**, excluding `code_snapshot/`. Array script `ROOT` resolution aligned with the consecutive script (`REPO_ROOT` / `SLURM_SUBMIT_DIR` / script parent).

**Timestamp:** 2026-04-05

## [2026-03-30] - Metrics markdown: `sweep_unified_finsler_idea1_gap` (job 1516604)
**Files Created:** `results/sweep_unified_finsler_idea1_gap_metrics.md`

### Problem Addressed
Final **mAP** / **Rank-1** from `logs/sweep_unified_finsler_idea1_gap/` needed collation for the paper table (1b row + Unified Finsler Euc. Dom. columns).

### Modification
Parsed each `log.txt` at **best-model final evaluation**; documented copy-paste markdown tables and log path index.

### Expected Behavior
Single place to pull numbers for job 1516604 gap sweep runs.

**Timestamp:** 2026-03-30

## [2026-03-30] - Sbatch: Idea-1 gap — W&B name, `--no-triplet` for domain-triplet arm
**Files Modified:** `sbatch/sweep_unified_finsler_idea1_gap.sbatch`

### Problem Addressed
1. **`WANDB_NAME=gapUnifiedFin_*`** implied every array task used a unified Finsler PID triplet; **ARM=0** is domain-triplet-only.  
2. **`--use-domain-triplet` without `--no-triplet`** stacks `loss_tri_dom` on top of the main PID triplet (`bau/trainers.py` sums both), so Idea **1a**-style or unified PID triplet was never “replaced” by domain triplet alone.

### Modification
- **`WANDB_NAME` → `idea1Gap_${RUN_NAME}`**; first W&B tag **`sweep-idea1-gap`**; Slurm **`#SBATCH --job-name=idea1-gap`**, stdout/stderr **`idea1-gap-%A_%a.*`**.
- **ARM=0:** prepend **`--no-triplet`** to the existing `--use-domain-triplet` block so **`loss_tri` is disabled** and the only `TripletLoss`-derived term in the total loss is **`loss_tri_dom`**.
- Header comment documents the stack vs replace behavior.

### Expected Behavior
Domain-triplet-only tasks train without PID batch-hard triplet; unified-Finsler tasks unchanged. W&B run names no longer overstate “unified Finsler” for the domain-triplet arm.

**Timestamp:** 2026-03-30

## [2026-03-30] - Sbatch: unified Finsler Idea-1 gap sweep (8 tasks)
**Files Created:** `sbatch/sweep_unified_finsler_idea1_gap.sbatch`

### Problem Addressed
Metric table had empty cells for **Idea 1b-only** and **unified Finsler** rows across instance/domain drift conditioning and Finsler vs Euclidean `L_domain`, which are not covered by the existing Idea-1 sweeps (those fix `--identity-triplet-only` and extensions use `--bidirectional-triplet`).

### Modification
- **`#SBATCH --array=0-7`** with `COND = TASK_ID % 2`, `L_DOM_KIND = (TASK_ID / 2) % 2`, `ARM = TASK_ID / 4`.
- **`ARM == 0`**: `--use-cross-domain-contrastive` (camera nuisance, `XDOM_W=0.1`) — refined 1b **without** separate Euclidean identity triplet.
- **`ARM == 1`**: no cross-domain contrastive — pure unified Finsler main triplet + BAU stack.
- **`L_DOM_KIND == 0`**: `--memory-bank-mode full`, `--use-omega-reg --omega-reg-weight 1.5` (same as Finsler-domain Idea-1 sweep).
- **`L_DOM_KIND == 1`**: `--use-euclidean-domain-loss` + identity memory bank (same as EuclideanDom sweep).
- **Explicitly omitted** from all tasks: `--identity-triplet-only`, `--bidirectional-triplet`.
- Logs under `logs/sweep_unified_finsler_idea1_gap/`; Slurm stdout/stderr under `logs/slurm_logs/unified-finsler-idea1-gap-*.out/.err`.

### Expected Behavior
`sbatch sbatch/sweep_unified_finsler_idea1_gap.sbatch` runs the eight configurations needed to fill the gap table; main triplet uses Finsler distance on the full `[identity|drift]` embedding.

**Timestamp:** 2026-03-30

## [2026-03-30] - EuclideanDom sweep metrics table (job 1514816)
**Files Created:** `results/sweep_loss_ablation_Idea1_EuclideanDom_metrics.md`

### Problem Addressed
Final **mAP** and **Rank-1** from `logs/sweep_loss_ablation_Idea1_EuclideanDom/` needed collation for paper/slide tables.

### Modification
Parsed each `log.txt` at the **post–best-checkpoint** evaluation block; documented paths and a copy-paste markdown table (five arms × instance/domain).

### Expected Behavior
Single source of truth for Euclidean-domain-loss sweep numbers alongside Finsler-domain columns.

**Timestamp:** 2026-03-30

## [2026-03-24] - finsler_single_best_24_03: Protocol-2 array sweep (-ds/-dt)
**Files Modified:** `finsler_single_best_24_03.sbatch`

### Problem Addressed
The single-job script only ran one BAU Protocol-2 split (`market1501 msmt17 cuhksysu` → `cuhk03`). Reproducing the full original BAU DG protocol requires the three source/target permutations from `old_bau/BAU/train.sh`.

### Modification
- **`#SBATCH --array=0-2`**: task `0`/`1`/`2` set `SOURCE_DATASET` and `TARGET_DATASET` to match the three `train.sh` lines (order preserved).
- **Logs**: `logs/slurm_logs/finsler-p2-%A_%a.out` / `.err`; per-task `logs/Protocol-2/job_${SLURM_JOB_ID}_${RUN_NAME}` with `RUN_NAME` / W&B tags including `p2-split-${IDX}` and a short `PROTOCOL_TAG`.
- **Hyperparameters**: unchanged from the previous single-run recipe (`ITERS=500` for all splits; original BAU used `--iters 200` only for the `msmt17` target — noted in script header).

### Expected Behavior
`sbatch finsler_single_best_24_03.sbatch` submits three independent training jobs with identical Finsler flags but different `-ds`/`-dt`.

**Timestamp:** 2026-03-24

## [2026-03-24] - Euclidean BAU-style domain loss toggle + Idea-1 EuclideanDom sweep
**Files Modified:** `bau/trainers.py`, `examples/train_bau.py`  
**Files Created:** `sbatch/sweep_loss_ablation_Idea1_EuclideanDom.sbatch`

### Problem Addressed
Ablations needed to turn **Finsler-style `L_domain`** (full embedding + model `dist_func`) **off** and restore **original BAU** Euclidean repulsion on **identity features** with an **identity-only memory bank**, via a single CLI flag.

### Modification
- **`--use-euclidean-domain-loss`**: requires `--arch resnet50_finsler`; forces `--memory-bank-mode identity`; centroid init slices extracted features to `memory_bank_dim` before mean/normalize.
- **`BAUTrainer`**: `use_euclidean_domain_loss` uses `f_w_align`/`f_s_align` in `domain_loss`, sets `effective_dist_func = euclidean_dist` and `alpha = None`; `momentum_update` uses the identity slice when `memory_bank.num_features == identity_dim`.
- **Sweep** `sweep_loss_ablation_Idea1_EuclideanDom.sbatch`: same 10-job factorial as `sweep_loss_ablation_Idea1_extensions.sbatch`, with `--use-euclidean-domain-loss` and **without** `--use-omega-reg` (avoids drift→identity gradient via orthogonalization in the drift head).

### Expected Behavior
Toggling the flag reproduces BAU’s `L_domain` geometry on the identity arm while keeping the Finsler backbone; new logs under `logs/sweep_loss_ablation_Idea1_EuclideanDom/`.

**Timestamp:** 2026-03-24

## [2026-03-23] - CSV: Idea-1 loss ablation metrics (job 1512947)
**Files Created:** `results/sweep_loss_ablation_Idea1_metrics.csv`

### Problem Addressed
Sweep results needed a machine-readable, self-describing table for papers, slides, and spreadsheets.

### Modification
Added one row per configuration (5 arms × 2 drift conditionings) with `arm_id`, short name, `arm_detail`, `drift_conditioning`, `map_pct`, `rank1_pct`, shared protocol columns (sources, target, arch, sampler, key booleans, seed, epochs), SLURM `job_id`, and `notes`.

### Expected Behavior
Import into Excel/LaTeX/pandas without parsing log files.

**Timestamp:** 2026-03-23

## [2026-03-23] - Publication-style figure for Idea-1 loss ablation sweep
**Files Created:** `scripts/plot_sweep_loss_ablation_idea1.py`; outputs `results/plots/sweep_loss_ablation_Idea1.pdf` and `.png` (300 DPI)

### Problem Addressed
Sweep metrics (job 1512947) needed a single, presentation-ready graphic with publication-style typography and a colorblind-safe palette.

### Modification
Added a matplotlib script: dual-panel grouped bar chart (mAP and Rank-1) for five loss arms × two drift conditionings (Okabe–Ito blue/orange), panel labels (a)/(b), methods footnote (protocol, single-seed caveat).

### Expected Behavior
`python scripts/plot_sweep_loss_ablation_idea1.py --output-dir results/plots` regenerates vector PDF and high-resolution PNG.

**Timestamp:** 2026-03-23

## [2026-03-23] - Plot script: portable fonts, SVG, PDF Type 42
**Files Modified:** `scripts/plot_sweep_loss_ablation_idea1.py`

### Problem Addressed
Some environments could not open the first PDF/PNG reliably (missing Times fonts, PDF font embedding, transparency).

### Modification
Use **Agg** backend; **DejaVu Serif** + `mathtext.fontset=dejavuserif`; `pdf.fonttype=42`; opaque **white** `facecolor` on save; default export adds **SVG**; `--formats` flag; slightly larger figure.

### Expected Behavior
PNG/PDF/SVG regenerate consistently; SVG opens in browsers; PDF embeds TrueType for PowerPoint/Acrobat.

**Timestamp:** 2026-03-23

## [2026-03-23] - Analysis report: sweep_loss_ablation_Idea1 (job 1512947)
**Files Created:** `changelogs/sweep_loss_ablation_Idea1_analysis_2026-03-23.md`

### Problem Addressed
The Idea‑1 loss ablation sweep (`sbatch/sweep_loss_ablation_Idea1_extensions.sbatch`, job `1512947`) completed; results needed consolidation against the meeting premise and README research narrative.

### Modification
Added a structured report: sweep aims, factorial design (5 arms × 2 drift conditionings), final mAP/Rank‑1 table from `logs/sweep_loss_ablation_Idea1/job_1512947_*/log.txt`, interpretation (baseline best; domain triplet harmful; refined contrastive slightly negative; drift‑only neutral), mermaid figures, literature references, and prioritized next experiments aligned with README (multi‑seed, dual eval with/without drift, toy corruptions, AG‑ReIDv2).

### Expected Behavior
Readers can cite one document for sweep conclusions and follow‑up work without re‑parsing raw logs.

**Timestamp:** 2026-03-23

## [2026-03-20] - ReNorm2 FinslerEmbeddingHead: identity_norm after BN neck (BAU parity)
**Files Modified (ReNorm2):** `fastreid/modeling/heads/finsler_head.py`
**Problem Addressed:** `identity_norm` was `F.normalize(global_feat)` instead of `F.normalize(bn_neck(global_feat))`, diverging from `resnet50_finsler` in `bau/models/model.py` (lines 336–337).
**Expected Behavior:** Finsler identity slice and orthogonalization match BAU; drift head input remains pre-BN pooled features.

**Timestamp:** 2026-03-20

## [2026-03-21] - Extend loss-ablation sbatch: drift conditioning × 5 arms + dom-triplet with L_domain
**Files Modified:** `sbatch/sweep_loss_ablation_Idea1_extensions.sbatch`

### Problem Addressed
Ablation needed **instance vs domain** drift conditioning for every loss arm, plus a **domain-triplet run with BAU `L_domain` enabled** (no `--no-domain`) to observe explicit conflict with same-nuisance attraction.

### Modification
Array expanded to **0–9**: `EXP = TASK_ID / 2` in {0..4}, `COND = TASK_ID % 2` → **instance** vs **domain** `--drift-conditioning`. **EXP=4** duplicates domain triplet but **omits `--no-domain`**. Run names include `driftInst` / `driftDom`; **EXP=4** uses `withLdom` in the name.

### Expected Behavior
Ten jobs; compare within same `EXP` across `driftInst`/`driftDom`, and **EXP=2 vs EXP=4** for domain triplet with vs without memory domain loss.

**Timestamp:** 2026-03-21

## [2026-03-21] - SLURM array: loss ablation for Idea-1 extensions (intuition sweep)
**Files Created:** `sbatch/sweep_loss_ablation_Idea1_extensions.sbatch`

### Problem Addressed
Multiple optional losses (camera-nuisance refined contrastive, Finsler domain triplet, drift-only cross-camera L2) needed isolated A/B/C comparison on a fixed Finsler+1a recipe without per-method hyperparameter sweeps.

### Modification
Added SLURM array **0–3**: (0) **1a-only baseline**, (1) **camera `cross-domain-contrastive-nuisance` + `use-cross-domain-contrastive`**, (2) **`use-domain-triplet` with camera mining + `--no-domain`** to limit gradient conflict with BAU domain loss, (3) **`use-drift-only-cross-contrastive`**. Shared settings mirror `finsler_single_learnableOmega.sbatch` (RMGS, omega reg, instance drift, bidirectional triplet). Resources set to **32G VRAM / 32G RAM / 6 CPUs** for headroom over the prior 28G recipe; single forward per step (no domain-balanced second loader).

### Expected Behavior
`sbatch sbatch/sweep_loss_ablation_Idea1_extensions.sbatch` submits four comparable runs; W&B tags `sweep-loss-ablate` plus arm-specific tags. Interpret mAP with the caveat that arm 2 disables `L_domain` by design.

**Timestamp:** 2026-03-21

## [2026-03-21] - Camera nuisance labels, Finsler domain triplet, drift-only contrastive, domain-balanced loader
**Files Modified:** `bau/utils/data/preprocessor.py`, `bau/utils/data/sampler.py`, `bau/trainers.py`, `examples/train_bau.py`, `README.md`

### Problem Addressed
Refined 1b contrastive used merged **dataset** id (`did`), so no same-PID cross-dataset pairs exist in multi-source DG. Original meeting **domain triplet (1b)** was not implemented. Optional **domain-balanced** batches and **drift-only** cross-camera supervision were needed for experiments.

### Modification
1. **`TwoViewPreprocessor`:** Returns **`(img_w, img_s, pid, did, cid)`** for two-view training; supports **3-tuple** train rows with **`did=0`**. Memory-init single-view path unchanged **`(img, fname, pid, cid)`**.
2. **`BAUTrainer`:** Parses **`cids`**; **`cross_domain_finsler_contrastive_loss`** takes generic **nuisance** labels; **`contrastive_nuisance`** `dataset|camera` selects **`dids` vs `cids`**. **`--use-domain-triplet`** adds second **`TripletLoss`** on **`f_w`** with batch-hard on **`--domain-triplet-mining-label`** (`camera`|`dataset`). Optional **`domain_triplet_loader`**: second forward on domain-balanced batch only for **`loss_tri_dom`**. **`drift_only_cross_camera_contrastive_loss`**: mean **L2²** on drift for **same PID, different `cid`**. W&B: **`train/loss_tri_dom`**, **`train/loss_drift_xcontrast`**; console **L_TriDom**, **L_DriftX**.
3. **`RandomDomainBalancedSampler`:** Groups by **camera** or **dataset**; **`get_domain_balanced_train_loader`** mirrors train augmentations. **`--use-domain-balanced-second-loader`** requires **`--use-domain-triplet`**.
4. **CLI:** **`--cross-domain-contrastive-nuisance`**, **`--use-domain-triplet`**, **`--domain-triplet-weight`**, **`--domain-triplet-margin`**, **`--domain-triplet-mining-label`**, **`--use-domain-balanced-second-loader`**, **`--domain-balanced-batch-size`**, **`--domain-balanced-instances-per-group`**, **`--use-drift-only-cross-contrastive`**, **`--drift-only-cross-contrastive-weight`**.

### Expected Behavior
- **`--cross-domain-contrastive-nuisance camera`** + **`--use-cross-domain-contrastive`**: signal on **same person, different camera** (merged global **`cid`**).
- **Domain triplet** intentionally **conflicts** with memory-based **`L_domain`** (same-nuisance attraction vs repulsion); use **`--no-domain`** or accept ablation.
- **`model(..., domain_ids=...)`** still uses **dataset** `did` only; domain-conditioned **`num_domains`** unchanged.

**Timestamp:** 2026-03-21

## [2026-03-20] - SBATCH: Idea 1a+1b learnable-omega Finsler job recipe
**Files Modified:** `finsler_single_learnableOmega.sbatch`

### Problem Addressed
Cluster launch script did not enable the new Idea 1 training flags or the sampler layout that yields cross-domain same-identity pairs for the contrastive term.

### Modification
Set default `SAMPLER` to `RandomMultipleGallery`; added `CROSS_DOMAIN_CONTRAST_WEIGHT=0.1`; passed `--identity-triplet-only`, `--use-cross-domain-contrastive`, and `--cross-domain-contrastive-weight` into `train_bau.py`; updated `WANDB_TAGS` and `RUN_NAME` to mark Idea 1a/1b runs.

### Expected Behavior
`sbatch finsler_single_learnableOmega.sbatch` runs the previous Omega-reg + bidirectional Finsler setup with Euclidean identity-only triplet and weighted cross-domain same-ID contrastive loss; W&B tags distinguish these experiments.

**Timestamp:** 2026-03-20

## [2026-03-20] - Idea 1: Euclidean identity triplet + cross-domain Finsler contrastive
**Files Modified:** `bau/trainers.py`, `examples/train_bau.py`

### Problem Addressed
Supervisor meeting Idea 1 called for separating identity-level metric learning (Euclidean on identity features) from asymmetric domain-aware geometry, and for a refined supplementary loss that pulls same-identity cross-domain pairs together under the model distance \(d_F\) without a new sampler.

### Modification
1. **`BAUTrainer` / triplet (1a):** Added `identity_triplet_only`. When True, `TripletLoss` uses `euclidean_dist` with no alpha, and the forward pass slices `emb_w[:, :identity_dim]` for Finsler models so `euclidean_dist` does not apply the legacy last-dimension alpha hack.
2. **`BAUTrainer` / contrastive (refined 1b):** Added `use_cross_domain_contrastive` and `cross_domain_contrastive_weight`. New method `cross_domain_finsler_contrastive_loss(f_w, pids, dids, alpha)` minimizes mean \(d(\mathbf{z}_i,\mathbf{z}_j)^2\) over pairs with the same PID and different source-domain IDs; uses `self.dist_func` (Finsler or Euclidean). Returns 0 if no such pairs exist in the batch.
3. **Logging:** AverageMeter, total loss, W&B key `train/loss_xdom_contrast`, and console `L_XDom`. `train/loss_total` now matches the backward sum including `lam * (align + drift_align)` and the new term.
4. **CLI:** `--identity-triplet-only`, `--use-cross-domain-contrastive`, `--cross-domain-contrastive-weight` (default `0.1`).

### Expected Behavior
- With `--identity-triplet-only` on `resnet50_finsler`, triplet mining is Euclidean on the identity slice only; domain/triplet geometry conflict for ID is reduced.
- With `--use-cross-domain-contrastive`, training adds weighted contrastive pressure on same-ID cross-source pairs in the weak view; signal is strongest when batches contain such pairs (e.g. `--sampler RandomMultipleGallery`).
- W&B step logs include `train/loss_xdom_contrast` when the flag is on (zero line when off).

**Timestamp:** 2026-03-20 (implementation per approved plan)

## [2026-03-20] - Supervisor meeting: loss signal separation and toy dataset proposal
**Files Created:** `changelogs/supervisor_meeting_20_03.md`

### Problem Addressed
Post-presentation discussion identified the need for clean separation between identity-related and domain-related loss signals. Four ideas were proposed: (1) split triplet loss into identity-Euclidean + domain-Finsler, (2) extend domain loss with cross-domain Finsler repulsion, (3) drift-domain alignment loss, (4) toy synthetic dataset for sanity checking the Finsler hypothesis.

### Modification
Created `changelogs/supervisor_meeting_20_03.md` documenting structured critical analysis of all four proposals, including mathematical formulations, gradient conflict identification, feasibility assessment, and a prioritized implementation plan. Key findings: Idea 1b (same-domain Finsler triplet attraction) directly contradicts the DG objective and existing `L_Domain`; Idea 2 is redundant with uniformity and harmful to cross-domain retrieval; Idea 3 (drift-domain alignment) is narratively consistent but risks drift collapse; Idea 4 (toy dataset) is highest priority as a diagnostic before any loss modifications.

### Expected Behavior
The meeting notes provide a clear decision record: implement toy dataset first (Priority 1), then Euclidean identity triplet (Priority 2), then drift-domain alignment (Priority 3). Idea 1b (as stated) and Idea 2 are discarded with documented rationale.

## [2026-03-19] - Comprehensive analysis of Finsler port to ReNorm2
**Files Created:** `changelogs/finsler_port_renorm2_analysis.md`
**Problem Addressed:** Document all modifications made to the ReNorm2 codebase with the same rigour as `changelogs/analysis_18_03.md`. Identifies two HIGH-severity confounds: (1) Finsler triplet uses L2-normalized identity features while the Euclidean baseline uses unnormalized features, and (2) evaluation-time RN+EN fusion produces non-unit-normed identity features that break the Finsler distance geometry. Both must be addressed before experiment results can be interpreted cleanly.
**Expected Behavior:** The analysis document provides a complete reference for interpreting upcoming experimental results and a checklist of pre-experiment fixes.

## [2026-03-19] - Port Finsler asymmetric distance to ReNorm2 for cross-framework sanity check
**Files Modified (in /home/stud/leez/reid/src/ReNorm2/):** `fastreid/modeling/heads/finsler_head.py` (new), `fastreid/modeling/heads/__init__.py`, `fastreid/modeling/losses/triplet_loss.py`, `fastreid/modeling/meta_arch/ReNorm.py`, `fastreid/config/defaults.py`, `fastreid/utils/compute_dist.py`, `fastreid/evaluation/reid_evaluation.py`, `tools/train_net.py`, `configs/ReNorm_Finsler.yml` (new), `finsler_single.sbatch` (new)
**Functions Altered:** `ARCH_RENORM.__init__`, `ARCH_RENORM.losses`, `build_dist`, `ReidEvaluator.evaluate`

### Problem Addressed
The Finsler asymmetric distance hypothesis was tested only within BAU. To determine whether the signal generalizes beyond BAU's loss landscape, the core Finsler components (split identity/drift head, `finsler_drift_dist` with symmetric_trapezoidal, batch-hard mining on asymmetric distance) were ported into the clean ReNorm2 codebase. ReNorm's contribution is in normalization layers (RN + EN), which is orthogonal to both BAU's losses and Finsler's distance geometry, making it the ideal cross-framework sanity check. Literature survey confirmed no other DG-ReID method offers a comparably clean separation of concerns.

### Modification
Minimal-invasion port: added `FinslerEmbeddingHead` (drift branch after GeM pool, orthogonalized against L2-normalized identity), `FinslerTripletLoss` (batch-hard on Finsler distance), and `"finsler"` eval metric. Wired conditionally into `ARCH_RENORM` via `cfg.MODEL.FINSLER.ENABLED`. ReNorm backbone, normalization modules, data pipeline, CE loss, and optimizer are untouched. Euclidean baseline is preserved when `FINSLER.ENABLED = False`. Smoke test passed: all losses compute, gradients flow through drift head, eval produces 4096-d features with measurable asymmetry.

### Expected Behavior
Running `finsler_single.sbatch` produces Protocol-2 results for the 2x2 comparison matrix {BAU, ReNorm} x {Euclidean, Finsler}. If Finsler helps both frameworks, the asymmetric hypothesis is strengthened. If it helps only BAU or neither, the signal is framework-specific or absent.

## [2026-03-19] - Add configurable backbone InstanceNorm and Stage 3 ablation sweep
**Files Modified:** `bau/models/model.py`, `examples/train_bau.py`, `sbatch/sweep_S3_instancenorm.sbatch` (new)
**Functions Altered:** `_build_resnet50_base` (new), `resnet50.__init__`, `resnet50_finsler.__init__`, `_parse_in_stages` (new), `create_model` in `examples/train_bau.py`

### Problem Addressed
The InstanceNorm2d layers after ResNet stages 1-3 (inherited from BAU, originating in IBN-Net / Jia et al. BMVC 2019) are hardcoded. This prevents ablating whether shallow-layer IN is complementary or redundant with the domain-conditioned drift head. The fork's narrative is that domain-specific variance should be explicitly modeled in the drift vector, which is in tension with IN stripping that variance before it reaches the drift head.

### Modification
1. Extracted backbone construction into `_build_resnet50_base(pretrained, in_stages)`, shared by both `resnet50` and `resnet50_finsler`. The `in_stages` parameter controls which stages (1-3) receive `InstanceNorm2d` after them.
2. Added `--backbone-in-stages` CLI argument accepting comma-separated stage indices or `"none"` (default: `1,2,3`).
3. Created `sbatch/sweep_S3_instancenorm.sbatch` — SLURM array job (4 configs) sweeping `backbone_in_stages` in {1,2,3 | 1,2 | 1 | none} with placeholders for optimal routing/residual parameters from Stages 1+2.

### Expected Behavior
- Default behavior (`--backbone-in-stages 1,2,3`) is identical to the old hardcoded IN placement.
- `--backbone-in-stages none` removes all IN layers, allowing domain/style statistics to flow unmodified into the drift head.
- Stage 3 ablation should reveal whether the drift head can compensate for the loss of IN-based domain normalization, or whether the two mechanisms serve complementary roles.

## [2026-03-19] - Create two-stage domain-conditioned drift hyperparameter sweep; optimize SLURM resource requests
**Files Modified:** `sbatch/sweep_S1_routing.sbatch` (new), `sbatch/sweep_S2_residual.sbatch` (new), `changelogs/sweep_domain_drift.md` (new), `finsler_single_learnableOmega.sbatch`
**Functions Altered:** N/A (infrastructure and documentation only)

### Problem Addressed
The domain-conditioned drift head introduces four new hyperparameters (`domain_embed_dim`, `domain_temperature`, `domain_residual_scale`, `domain_token_loss_weight`) with no empirical guidance on their operational range. A systematic sweep is needed before the first domain-conditioned experiments can be interpreted. Additionally, SLURM resource requests (VRAM: 28 GB, RAM: 36 GB) were over-provisioned relative to observed peak usage (~20 GB VRAM, ~16 GB RAM) from Grafana monitoring of prior runs.

### Modification
1. Created `changelogs/sweep_domain_drift.md` documenting the full sweep rationale, mathematical analysis of each hyperparameter, expected behavior per configuration, interaction analysis, diagnostic metrics, and an interpretation guide.
2. Created `sbatch/sweep_S1_routing.sbatch` — a SLURM array job (12 configs) sweeping `domain_token_loss_weight` in {0.01, 0.05, 0.10, 0.25} x `domain_temperature` in {1.0, 2.0, 5.0} with `domain_residual_scale = 0.0` to isolate the routing mechanism.
3. Created `sbatch/sweep_S2_residual.sbatch` — a SLURM array job (4 configs) sweeping `domain_residual_scale` in {0.00, 0.05, 0.10, 0.20} with placeholder variables for the optimal routing parameters from Stage 1.
4. `domain_embed_dim` is not swept: with 3 source domains, the rank of the domain prior subspace is bounded by min(3, d), making sweeps above d=3 mathematically vacuous.
5. Reduced VRAM request from 28 GB to 24 GB and system RAM from 36 GB to 28 GB in `finsler_single_learnableOmega.sbatch` and both sweep scripts.

### Expected Behavior
- Stage 1 determines whether the domain routing mechanism produces useful soft tokens and how smooth the target-domain mixture should be. Runs with very low token loss weight (0.01) should show near-random domain classification; runs with very high weight (0.25) may show domain-entangled identity features.
- Stage 2 determines whether per-instance drift correction adds value over the pure domain prior, or whether BAU's optimization landscape suppresses it.
- Reduced memory requests should schedule without OOM; 4 GB VRAM headroom is provided over observed peak.

## [2026-03-18] - Fix k-NN weight contamination and uniform loss incompatibility; design ablation experiments
**Files Modified:** `bau/trainers.py`, `examples/train_bau.py`, `README.md`, `sbatch/ablation_A_capacity.sbatch`, `sbatch/ablation_B_uniform.sbatch`, `sbatch/ablation_C_drift_conditioning.sbatch`, `sbatch/ablation_F_crossview.sbatch`, `sbatch/validate_domain_pivot.sbatch`
**Functions Altered:** `BAUTrainer.train` (k-NN sims, uniform loss call), `BAUTrainer.uniform_loss`, `main_worker` (--force-euclidean flag)

### Problem Addressed
Two bugs identified during the critical loss function analysis (plan Part 5):

1. **k-NN weight contamination (Discrepancy A3):** The reciprocal k-NN Jaccard weight matrix was computed from `sims = matmul(f, f.t())` where `f = [identity_norm | drift]`. The drift component is not L2-normalized, so the inner product was dominated by drift magnitude rather than identity similarity. This biased alignment weights away from true identity-level neighborhoods.

2. **Uniform loss incompatibility (Discrepancies U1, U2):** Uniform loss was applied to `[identity | drift]` features using asymmetric Finsler distance. This is theoretically inconsistent — the Wang & Isola uniformity objective assumes a symmetric metric on the unit hypersphere. The asymmetric distance created contradictory gradients that suppressed drift (alpha → 0), as confirmed by `ablation_summary.csv`.

Additionally, no `--force-euclidean` flag existed to override the model's distance function, preventing capacity-controlled ablation experiments.

### Modification
- `bau/trainers.py`: k-NN sims now computed on identity-only features (`f_for_sims = cat([f_w_align, f_s_align])`). `uniform_loss()` signature changed to `uniform_loss(self, f)` — unconditionally uses `euclidean_dist` on identity-only features, removing alpha parameter and Finsler/manifold distance paths.
- `examples/train_bau.py`: Added `--force-euclidean` flag that overrides `model.dist_func` with `euclidean_dist` in the trainer and evaluator. Enables capacity-controlled ablation (resnet50_finsler architecture with Euclidean-only distance).
- `README.md`: Added architecture diagram errata section documenting 4 discrepancies (domain gate dims, classifier output, drift head input, L_p label).
- Created 5 ablation sbatch scripts: Ablation A (capacity control, 3 configs), Ablation B (uniform loss × geometry, 4 configs), Ablation C (drift conditioning, 4 configs), Ablation F (cross-view directionality on AG-ReIDv2, 4 configs), and domain-pivot validation.

### Expected Behavior
- Alignment weights now reflect identity-level neighborhood structure, not drift magnitude.
- Uniform loss produces symmetric repulsion gradients that do not conflict with drift learning.
- The combination of fixes removes the two identified mechanisms for drift suppression/alpha collapse, allowing the asymmetric component to persist if it provides genuine benefit.
- Ablation scripts are ready for submission via `sbatch sbatch/ablation_*.sbatch` and `sbatch sbatch/validate_domain_pivot.sbatch`.

## [2026-03-18] - Document training sampler decision (RandomMultipleGallerySampler) in README
**Files Modified:** `README.md`
**Functions Altered:** N/A (documentation only)

### Problem Addressed
The decision to use `RandomMultipleGallerySampler` for the Finsler pivot (cross-camera positive pairs per identity) was supported by an A/B test against `RandomIdentitySampler` on the Euclidean baseline, but the rationale and results were not yet recorded in the repository.

### Modification
Added a new subsection **Training batch sampler: RandomMultipleGallerySampler** under Research Status Update in `README.md`, including: (1) the distinction between the two samplers and why cross-camera positives matter for view-invariant alignment, (2) a results table from the A/B test (Market1501+MSMT17+CUHK-SYSU → CUHK03-NP) showing +1.1% mAP and gains on Rank-1/5/10 with `RandomMultipleGallerySampler`, (3) the formal decision to use `--sampler RandomMultipleGallery` for forthcoming Finsler/domain-conditioned experiments, with a short justification tied to the per-domain/per-camera asymmetry pivot, and (4) caveats (single seed, single target). Also added `--sampler` to the expanded experimental surface list and to the Useful fork-specific flags table.

### Expected Behavior
Readers and launch scripts can treat `RandomMultipleGallerySampler` as the documented choice for the Finsler pivot; the README explains why and cites the Euclidean baseline comparison. Backward compatibility is preserved (default remains `RandomIdentity` in code).

## [2026-03-18] - Add peer-critical-tone Cursor rule
**Files Modified:** `.cursor/rules/peer-critical-tone.mdc`, `changelog.md`
**Functions Altered:** N/A (new rule, documentation)

### Problem Addressed
No persistent rule instructed the agent to treat the user as a peer, critically analyse claims instead of defaulting to agreement, flag README/changelog conflicts, stay concise, use a post-doctorate research tone in Asking mode for theory-heavy topics, and avoid analogies or wording that downplay semantics.

### Modification
Created `.cursor/rules/peer-critical-tone.mdc` with `alwaysApply: true`. Rule enforces peer-level engagement, explicit signalling of conflicts with README narrative or changelog-documented failures, concision, research-theory tone in Asking mode, and semantic precision without diluting language.

### Expected Behavior
The agent will apply this rule in every session: respond as a peer with critical analysis, surface README/changelog misalignments early, keep replies concise, adopt a post-doctorate researcher register for research/theory questions in Asking mode, and avoid analogies or softening language that weakens semantic content.

## [2026-03-12 08:59:37 UTC] - Document domain-conditioned drift head architecture in README
**Files Modified:** `README.md`
**Functions Altered:** N/A (documentation update)

### Problem Addressed
The new `DomainConditionedDriftHead` pivot introduced a more complex conditioning path than the original per-instance `FinslerDriftHead`, and the current README did not yet explain clearly how the new drift prior is formed, how it connects to the identity branch, or how evaluation works when ground-truth target-domain labels are unavailable.

### Modification
Appended a new README section that explains the domain-conditioned drift architecture in plain language, describes how the auxiliary domain-token predictor is used during training and evaluation, and adds a Mermaid flowchart showing the identity branch, domain-token classifier, domain embedding mixture, residual drift correction, final concatenation, and the new auxiliary domain-token loss.

### Expected Behavior
Readers should now be able to understand how the new drift head differs from the older instance-conditioned head, why the evaluation-time prior is a soft mixture over source domains, and how the added auxiliary loss fits into the existing BAU/Finsler training pipeline.

## [2026-03-11 22:18:12 UTC] - Add W&B tags CLI support for training launches
**Files Modified:** `examples/train_bau.py`, `finsler_single_learnableOmega.sbatch`
**Functions Altered:** `main_worker` in `examples/train_bau.py`

### Problem Addressed
Training runs could already set a W&B run name from the CLI, but there was no matching way to attach W&B tags directly from sbatch launch scripts. That made it harder to organize experiment groups, especially while comparing the new domain-conditioned Finsler pivot against older baselines.

### Modification
Added a new `--wandb-tags` CLI argument to `examples/train_bau.py` and passed it through to `wandb.init(..., tags=...)`. Updated `finsler_single_learnableOmega.sbatch` to define a `WANDB_TAGS` Bash array and forward it into the training command.

### Expected Behavior
Sbatch launch scripts can now attach arbitrary W&B tags directly at launch time without editing Python code. Runs should appear in W&B with both the configured run name and the provided tag list, making sweeps and ablations easier to filter and compare.

## [2026-03-11 21:55:18 UTC] - Introduce domain-conditioned Finsler drift pivot scaffolding
**Files Modified:** `bau/models/model.py`, `bau/trainers.py`, `examples/train_bau.py`, `examples/test.py`, `README.md`, `changelogs/perDomainPerViewAsymmetry.md`
**Functions Altered:** `FinslerDriftHead.forward`, `resnet50_finsler.__init__`, `resnet50_finsler.forward`, `BAUTrainer.__init__`, `BAUTrainer.train`, `create_model` in `examples/train_bau.py`, `create_model` in `examples/test.py`

### Problem Addressed
The previous Finsler implementation modeled asymmetry purely at the per-instance level. Current experiments suggest that this is too weak and too easily suppressed by BAU's domain-generalization objectives to support a strong scientific claim. The repository therefore needed a low-risk path to test whether asymmetry is better represented as a structured domain/view effect while preserving the existing BAU and Finsler baselines for comparison.

### Modification
Added an opt-in domain-conditioned drift pathway to `resnet50_finsler` via a new `DomainConditionedDriftHead`, along with a lightweight soft domain-token predictor for inference-time conditioning. Updated the trainer to pass source-domain IDs already present in the DG pipeline and to optionally supervise the token predictor through a small auxiliary loss. Extended the training and standalone evaluation CLIs with domain-conditioning controls, improved standalone checkpoint unwrapping for wrapped training checkpoints, corrected the Finsler memory-bank split normalization to use the model identity slice width, appended a dated README status update describing the research pivot, and created `changelogs/perDomainPerViewAsymmetry.md` to document the intended staged implementation strategy.

### Expected Behavior
The codebase now supports direct A/B testing between the original instance-conditioned Finsler drift and a new domain-conditioned variant without disrupting the baseline BAU path. Training can reuse existing source-domain labels as drift-conditioning metadata, and evaluation can optionally infer soft domain tokens from images so the asymmetric branch remains usable on unseen target data. The new documentation should also make the current research direction and safest validation protocol explicit.

## [2026-03-09 00:00:00 UTC] - Rewrote README to document the fork-specific contributions over BAU
**Files Modified:** `README.md`
**Functions Altered:** N/A (documentation update)

### Problem Addressed
The previous README described the project at a high level, but it did not meticulously distinguish which improvements over the original BAU repository were actually implemented in the current codebase. It also contained several documentation mismatches with active code, including outdated architecture naming, an incorrect evaluation flag, and an overstated description of gradient-isolated Finsler training.

### Modification
Rewrote `README.md` to explicitly compare this repository against the original BAU codebase and document the concrete implementation deltas that plausibly affect metrics: the `resnet50_finsler` backbone, `FinslerDriftHead`, `finsler_drift_dist`, `AlphaParameter`, AG-ReIDv2 view-filtered and cross-view protocol support, geometry-aware trainer/evaluator behavior, Finsler-aware memory bank settings, and the expanded CLI surface for ablations. The rewrite also adds conservative attribution language, implementation caveats, and corrected command/default examples that match the current code.

### Expected Behavior
The README now functions as an accurate contribution audit for the asymmetric BAU fork. Users should be able to identify which code-level changes distinguish this repository from old BAU, understand why those changes may influence mAP/Rank-based metrics, and avoid being misled by stale commands or unsupported claims when reproducing experiments.

## [2026-03-04 15:00:00 UTC] - Implement SLERP integration for Finsler drift distance
**Files Modified:** `bau/loss/triplet.py`
**Functions Altered:** `finsler_drift_dist`

### Problem Addressed
The previous `Option 3: Spherical Trapezoidal Approximation` did not properly capture the continuous integration path along the hypersphere. The user requested implementing a true Spherical Linear Interpolation (SLERP) approach to calculate the ambient drift vector overlap along the geodesic path, prioritizing computational efficiency.

### Expected Behavior
The `finsler_drift_dist` function now computes the drift penalty by taking a Riemann sum over discretised steps along the SLERP trajectory between points `x` and `y`. To maximize computational efficiency, the expensive dot products (`term_xx`, `term_xy`, `term_yy`, `term_yx`) are computed identically as before across the batch. The Riemann integration simply interpolates these scalars over time using velocity coefficients $A_t$ and $B_t$, avoiding repeated higher-dimensional tensor allocations within the integration loop.

## [2026-03-04 15:28:16] - Introduced computation methods for Finsler Drift
- **Files modified**: `bau/loss/triplet.py`
- **Problem**: The explicit computation path for the metric tensor geometry was previously hardcoded or commented out (Options 1, 2, 3.1, 3.2).
- **Modification**: Parameterized the distance computation in the `finsler_drift_dist` function to support a `method` configuration corresponding to 'constant_drift', 'symmetric_trapezoidal', 'slerp', and 'analytical'. Included boundary/edge-case handling for values of theta approaching 0.
- **Expected behavior**: Depending on the exact method passed via `functools.partial` or as a standard argument (e.g., `method="analytical"`), the drift distance operates reliably and robustly. Without explicit configuration, it falls back gracefully to the original un-commented `symmetric_trapezoidal` approximation.

## [2026-03-04 16:22:30] - Added CLI hook for Finsler Drift computation method
- **Files modified**: `examples/train_bau.py`, `bau/models/model.py`
- **Problem**: The explicit computation method was hardcoded or restricted to Python-level overrides, making it inaccessible via hyperparameter tuners or shell scripts.
- **Modification**: Attached a `--drift-method` command-line argument directly to the main parsing logic in `train_bau.py` and propagated this attribute through to `resnet50_finsler`'s initialization.
- **Expected behavior**: Running `python examples/train_bau.py --drift-method analytical` automatically utilizes Exact Analytical Integration of the Parallel-Transported Field with no extra code modification.

## [2026-03-04 16:45:00] - Triplet Loss Distance Function Override
- **Files modified**: `bau/loss/triplet.py`
- **Functions altered**: `TripletLoss.__init__`
- **Problem**: Triplet Loss required a hardcoded Finsler evaluation method (`symmetric_trapezoidal`) to ensure stability, but the model globally passed a user-defined method intended only for uniform and domain losses.
- **Modification**: Intercepted the `dist_func` assignment inside `TripletLoss.__init__`. If the passed function (or `functools.partial` wrapper) evaluates to `finsler_drift_dist`, we dynamically overwrite the keyword arguments to strictly force `method="symmetric_trapezoidal"`.
- **Expected behavior**: Triplet Loss distance calculations will exclusively and safely use the symmetric trapezoidal approximation, while the evaluator and other margin losses properly respect the CLI-provided `--drift-method` flag.
## [2026-03-11 00:00:00 UTC] - Fix Float16 underflow in arccos gradients during SLERP/Analytical drift computation
**Files Modified:** `bau/loss/triplet.py`
**Functions Altered:** `finsler_drift_dist`

### Problem Addressed
When training with mixed precision (`float16`), the `slerp` and `analytical` drift computation methods in `finsler_drift_dist` caused gradient explosion and model divergence. This occurred because the `torch.clamp` bounds (`-1.0 + 1e-6`, `1.0 - 1e-6`) were ineffective in `float16` since the machine epsilon is roughly `9.77e-4`. As a result, the input to `torch.acos()` could reach exactly `1.0` or `-1.0`, leading to a division by zero in the backward pass and producing `NaN` gradients.

### Modification
Updated the `finsler_drift_dist` function to dynamically set the numerical epsilon (`eps`) for clamping based on the tensor's datatype. If the input is `float16`, the clamping bound is set to a safe `1e-4`; otherwise, it defaults to the original `1e-6`.

### Expected Behavior
The `slerp` and `analytical` methods will now correctly clamp `arccos` inputs away from the critical `-1.0` and `1.0` boundaries even when trained with mixed precision, completely averting the `NaN` gradient cascades and allowing the model to train stably.

## [2026-03-11 14:47:16 UTC] - Stabilize angular Finsler drift backpropagation for SLERP and analytical modes
**Files Modified:** `bau/loss/triplet.py`
**Functions Altered:** `finsler_drift_dist`, `_angular_extrapolation_bound`, `_safe_acos_linear_extrapolation`, `_stable_theta_sin_ratio`

### Problem Addressed
The previous `slerp` and `analytical` branches still backpropagated through a raw `torch.acos()` evaluation near cosine values of $\pm 1$, which remained poorly conditioned even after dtype-aware clamping. Under mixed precision this caused unstable angular gradients, and the small-angle ratio $\theta / \sin(\theta)$ also relied on an abrupt fallback that did not preserve the local series behavior around $\theta = 0$.

### Modification
Reworked the angular computation path in `finsler_drift_dist` so that the `slerp` and `analytical` methods: (1) run their angle-sensitive matrix products in `float32`, (2) replace the raw `torch.acos()` call with a linearly extrapolated safe variant that keeps gradients finite beyond a configurable interior bound, and (3) use an explicit Taylor expansion for the small-angle limit of $\theta / \sin(\theta)$ instead of a hard constant branch.

### Expected Behavior
The `slerp` and `analytical` drift penalties should now remain numerically stable for nearly aligned identity embeddings, avoid the classical arccosine-gradient blow-up in mixed precision training, and preserve a smooth small-angle limit during backpropagation so these drift modes can optimize instead of stalling or diverging.

### Auto-update: Handle `domain_ids` passing dynamically in BAUTrainer
- **Files Altered**: `bau/trainers.py`
- **Problem**: Baseline (Euclidean/non-Finsler) models do not expect `domain_ids` in their forward signature, causing training loops with Euclidean baselines to crash when explicitly passing it.
- **Expected Behavior**: `BAUTrainer` dynamically inspects the inner module's forward method. It only passes `domain_ids=domain_ids` when the backbone actively declares it as a valid kwarg, preventing unexpected keyword crashes for legacy model variations.
- **Timestamp**: 2026-03-16 17:17:45

## [2026-04-01 00:00:00 UTC] - Write comprehensive analysis of Idea 1 loss ablations
**Files Modified:** `results/Idea1_Ablation_Comprehensive_Analysis.md` (created)
**Functions Altered:** N/A (analysis markdown created)

### Problem Addressed
The user needed an academically rigorous, impartial analysis of the consolidated sweep results for the "Idea 1" loss ablations in the asymmetric Finsler Domain Generalizable Re-ID framework. This analysis needed to cover the performance of the baseline vs. unified Finsler, the catastrophic failure of explicit domain clustering (domain triplet 1b), the superiority of Finsler over Euclidean domain memory banks, and the impact of auxiliary constraints.

### Modification
Created `results/Idea1_Ablation_Comprehensive_Analysis.md` containing a detailed discussion of the ablation results, backed by relevant CV literature (Jia et al., Zhou et al., Wang & Isola, Chen & He) and framed from the perspective of a senior CV research scientist.

### Expected Behavior
The new markdown file provides a clear, theory-grounded explanation of why explicit domain clustering harms DG-ReID, why the unified asymmetric Finsler manifold performs optimally without explicit disentanglement, and why over-constraining drift vectors degrades performance. This document will serve as a foundational reference for the theoretical narrative of the paper/research direction.

### 2026-03-24 10:00:00
*   **Modifications:** Created a new markdown document at `bau/docs/finsler_disentanglement_analysis.md`.
*   **Problem:** The theoretical viability, utility of the drift vector during retrieval, and experimental roadmap for Finsler asymmetry in DG-ReID needed to be clearly structured, consolidated, and documented following an in-depth analytical discussion. 
*   **Expected Behavior:** Researchers and collaborators can now reference `bau/docs/finsler_disentanglement_analysis.md` for a complete, exhaustive summary of the "representation disentanglement" paradigm, the resolution of the `--eval-drift` misconception, and the concrete 3-phase experimental roadmap (Toy Dataset -> Disentanglement Validation -> Asymmetric Eval Activation) moving forward.

### 2026-04-06 10:44:24
*   **Modifications:** Reformatted `results/eval_drift_true_finsler_ranking_table.md`.
*   **Problem:** The original single 8-column table (2 training regimes × 2 eval modes × 2 metrics × 2 drift conditionings) was too dense for presentation. All information was present but not digestible.
*   **Expected Behavior:** The file now follows a two-primary-table + two-supplementary-table structure: **Table A** (Finsler-domain training, best-of-conditioning, mAP/R1 + Δ), **Table B** (Euclidean-domain training, same format — highlights the catastrophic −6.1 mAP for 1b), **Supp. S1** (full per-conditioning breakdown for reproducibility), **Supp. S2** (Finsler-eval R5/R10). Best conditioning is selected per arm by Finsler-eval mAP.

## [2026-04-06] — Curate experiment provenance into publication_facts.md and create narrative_relevance_map.md
**Files Modified:** `.cursor/paper/publication_facts.md` (appended), `paper/draft_2/narrative_relevance_map.md` (created)
**Functions Altered:** N/A (documentation)

### Problem Addressed
`publication_facts.md` contained only 3 rows from the early job-1512947 sweep; 8 subsequent experiment batches (jobs 1516935, 1517230, 1516604, 1518384, 1481835, 1482168, 1483098, 1485934, 1512315) were not reflected. No mapping between experiments and the ablation arms defined in `4_experiments.md` existed.

### Modification
Appended 8 new subsections to `Reported_metrics_with_provenance` in `publication_facts.md`:
- Full Idea-1 re-run under Finsler-domain training (job 1516935 + gap rows 1516604)
- Full Idea-1 re-run under Euclidean-domain training (job 1517230 + gap rows 1516604)
- eval-drift-true sweep tables A/B (job 1518384)
- Learnable alpha sweep (jobs 1481835, 1482168): alpha converges to 0.098–0.150 regardless of init
- Finsler duet 5-run RNG sweep (jobs 1483098, 1485934): no statistically significant Finsler gain
- S1 domain-conditioned routing sweep (job 1512315): all 12 runs below 1a baseline (40.3–41.8% mAP)
- Early Euclidean ablation baselines (ablationEuclidean-P3.csv, ablationEuclideanPaper.csv, ablationHyperbolic.md) with provenance caveat
Created `paper/draft_2/narrative_relevance_map.md` mapping every experiment to ablation arms 1–4 with primary/secondary/irrelevant/figure-candidate tables.

### Expected Behavior
Paper agents can now directly look up which logged experiment corresponds to which ablation arm, with log paths and job IDs. `publication_facts.md` is current. `narrative_relevance_map.md` serves as the reference for table population once `results_implementation_lock` is set.

## [2026-04-06 20:39 UTC] - Experiments section rewrite, agent architecture expansion, and experiment curation pipeline

**Files Modified:**
- `paper/draft_2/markdowns/4_experiments.md`
- `.cursor/paper/review_style_directives.md`
- `.cursor/agents/paper-experiments-curator.md`
- `.cursor/paper/GUIDE.md`

**Files Created:**
- `.cursor/skills/paper-table-formatter/SKILL.md`
- `paper/draft_2/narrative_relevance_map.md`
- `paper/draft_2/draft_table_main.tex`
- `paper/draft_2/draft_table_supplementary.tex`
- `paper/draft_2/figure_audit.md`

### Problem Addressed

**Track A — Small edits to `4_experiments.md`:**
1. Protocol description was too vague ("classic DG pattern") and lacked an explicit note that the paper uses a subset of Protocol 2. Cross-view AG-ReIDv2 sentence was unjustified at draft stage.
2. "exact split names and filters are listed in the supplement when numerics are locked" was placeholder prose in the body, violating the draft standard.
3. `## Implementation sketch` used bold label-colon fragment headers (`**Backbone.**`, `**Optimization.**`, `**Asymmetry.**`) — not academic prose; same issue in `## Qualitative analysis`.

**Track B — Agent architecture gap:**
No existing agent mapped experiments to the paper narrative, and no skill produced CVPR-ready LaTeX tables from provenanced metrics. The default pipeline in `GUIDE.md` had no quantitative-section workflow.

### Modifications

**`4_experiments.md`:** Protocol note now explicitly names Protocol 2, M+MS+CS→C3, exploratory scope, and computational constraint. AG-ReIDv2 moved to `%TODO`. Supplement placeholder replaced with `%TODO`. Implementation sketch rewritten as single academic prose paragraph (backbone → drift head → loss stack → α scheduling → eval toggle). Qualitative analysis rewritten as declarative prose paragraph with `%TODO`.

**`review_style_directives.md`:** Added three new global directives (11–13): no placeholder prose in body, no bold label-colon fragment headers, no bullet-enumerated qualitative-analysis plans.

**`paper-experiments-curator.md`:** Extended with Output 2 — produces `paper/draft_2/narrative_relevance_map.md` mapping each experiment to ablation arms in `4_experiments.md` (primary / secondary / irrelevant / figure candidates).

**`paper-table-formatter` skill:** New skill encoding CVPR `booktabs` table production from `narrative_relevance_map.md` + `results/*.md|csv`; phase-dependent behavior (structure-only under `theory_first`; provenanced numbers under `results_implementation_lock`); anti-pattern rules.

**`GUIDE.md`:** Added Stage 0 (experiment → narrative mapping, table formatter) before the existing Stage 1 pipeline; updated mermaid diagram; added `paper-table-formatter` to skills list.

**Artifact outputs:**
- `narrative_relevance_map.md`: 12 primary candidates, 8 secondary, irrelevant/superseded table, 10 figure candidates with readiness flags
- `draft_table_main.tex`: 7-arm ablation table (Arm 1–3) with provenanced metrics and `%TODO` gates
- `draft_table_supplementary.tex`: eval-drift δ tables A and B (FD and ED training) for supplementary
- `figure_audit.md`: per-figure assessment for 9 expected plots; all rated NEEDS_POSTPROCESSING; rcParams template; 8-item priority action list

### Expected Behavior

Paper agents following the GUIDE.md pipeline will now: (1) run Stage 0 before drafting quantitative sections, producing a narrative map and LaTeX table skeleton; (2) apply the 13 review_style_directives globally; (3) have access to provenanced table drafts for the experiments section once `results_implementation_lock` is set.

---

## 2026-04-08T19:06:26Z — Expand Methodology (Sec 3) and Implementation Details (Sec 4.2); Create Supplementary Material

### Problem

Section 3 (Methodology) covered only four subsections (Euclidean baseline, asymmetric Finsler distance, domain objectives, interference analysis) and omitted several implemented methodological contributions: orthogonal decomposition rationale, Randers positivity constraint analysis, drift dimension scaling, identity-only alignment restriction, bidirectional triplet loss, drift norm regularization (log barrier), and domain-conditioned drift prior. Section 4.2 (Implementation details) was a single dense paragraph lacking structured coverage of the drift head architecture, training recipe, batch construction, memory bank, and evaluation protocol. No supplementary material existed for the learnable alpha, alternative drift integration methods, norm constraint alternatives, k-NN/uniform loss corrections, IN ablation design, or ReNorm2 port.

### Modifications

**`paper/draft_3/3_methodology.tex`** — Expanded from ~60 to ~105 lines (theory only):
- Sec 3.2 (Asymmetric Finsler Distance): Added four new `\paragraph{}` blocks: orthogonal decomposition rationale (Gram-Schmidt, domain separation analogy), Randers positivity constraint (||ω|| < 1, metric axiom), drift dimension scaling (Cauchy-Schwarz bound, γ correction), identity-only alignment (eq. for restricted alignment loss, k-NN on identity only).
- New Sec 3.3 (Bidirectional asymmetric triplet): Formalized D^T mining under asymmetric d_F; eq. for L_tri^bi.
- New Sec 3.4 (Drift norm regularization): Logarithmic barrier penalty; gradient analysis at boundary.
- New Sec 3.5 (Domain-conditioned drift prior): Decomposition ω = ω_dom + λ·ω_res; domain prior with soft assignment, sigmoid gate, stop-gradient; auxiliary CE loss; evaluation-time generalization.
- Existing Sec 3.3 (Domain objectives) renumbered to 3.6; Sec 3.4 (Interference analysis) to 3.7. Section preamble updated with all new subsection references.

**`paper/draft_3/4_experimentalResults.tex`** — Sec 4.2 expanded from 1 paragraph to 7 structured `\paragraph{}` blocks:
- Backbone and identity branch, instance-conditioned drift head (2-layer MLP, sigmoid gating, zero init, pre-BN input, Gram-Schmidt), domain-conditioned drift head, training recipe (loss stack, drift LR ×0.05, gradient clipping max_norm 1.0), batch construction (RandomMultipleGallerySampler cross-camera), memory bank, evaluation (eval-drift toggle).

**`paper/draft_3/5_supplementary.tex`** — New file with 6 appendix sections:
- A: Learnable alpha (centered sigmoid parameterization, gradient starvation, convergence)
- B: Alternative drift integration (spherical trapezoidal, SLERP Riemann sum, analytical parallel transport, numerical stability, empirical outcome)
- C: Drift norm constraint alternatives (hard clamping, tanh, sigmoid, LogSumExp)
- D: k-NN weight contamination and uniform loss corrections
- E: InstanceNorm ablation design
- F: Cross-framework validation (ReNorm2 port, negative result)

**`paper/draft_3/main.bib`** — Added `bousmalis2016domain` (Domain Separation Networks, NeurIPS 2016).

All mathematical expressions verified against codebase: `bau/loss/triplet.py` (finsler_drift_dist symmetric_trapezoidal, AlphaParameter, TripletLoss bidirectional, scaling_factor), `bau/models/model.py` (FinslerDriftHead, DomainConditionedDriftHead, scale_drift_vector sigmoid gating, orthogonalization), `bau/trainers.py` (identity-only alignment, uniform loss restriction, k-NN on identity features, omega_loss log barrier, gradient clipping).

### Expected Behavior

The paper draft now documents all implemented methodological contributions with a strict theory (Sec 3) vs. practice (Sec 4.2) boundary. Cross-references between sections are consistent. The supplementary material covers negative results and design alternatives that inform the ablation narrative.

---

## 2026-04-08T19:14:23Z — Consolidate Supplementary; Purge Legacy Alpha References from Main Text

### Problem

1. Two supplementary files existed: the original `X_suppl.tex` (with a well-framed alpha section documenting it as a superseded design step) and a newly generated `5_supplementary.tex` (with sections B--F on drift integration, norm constraints, k-NN/uniform fixes, IN ablation, ReNorm2 port). These needed to be merged.
2. The main text (abstract, intro, methodology) still contained language implying a learnable scalar alpha parameter as an active component of the distance function (e.g., "bounded learnable weight that interpolates toward the Euclidean triplet regime", "learnable Euclidean-asymmetry interpolation", "A scalar variant $d_F^\alpha$..."). Since alpha is a legacy/superseded design documented only in the supplementary, these references were misleading.

### Modifications

**`paper/draft_3/X_suppl.tex`** — Appended five new sections after the existing alpha documentation:
- Sec 2: Alternative Drift Integration Methods (spherical trapezoidal, SLERP, analytical, numerical stability, empirical outcome)
- Sec 3: Drift Norm Constraint Alternatives (hard clamping, tanh, sigmoid, LogSumExp)
- Sec 4: k-NN Weight Contamination and Uniform Loss Corrections
- Sec 5: InstanceNorm Ablation Design
- Sec 6: Cross-Framework Validation (ReNorm2 Port)

**`paper/draft_3/5_supplementary.tex`** — Deleted (redundant after merge into X_suppl.tex).

**`paper/draft_3/3_methodology.tex`** — Removed the sentence "A scalar variant $d_F^\alpha = d_E + \alpha \cdot \langle \mathbf{z}^{\omega}, \cdot \rangle$, which recovers Euclidean at $\alpha{=}0$, is developed and analyzed in Sec.~\ref{sec:suppl_alpha}." from the Randers-type distance paragraph.

**`paper/draft_3/1_intro.tex`** — Reworded two passages:
- "with a bounded learnable weight that interpolates toward the Euclidean triplet regime" → "that recovers Euclidean geometry when the drift vanishes, with norm-constrained drift initialization ensuring a smooth departure from the symmetric baseline"
- "learnable Euclidean-asymmetry interpolation" → "norm-constrained drift that recovers Euclidean geometry at initialization"

**`paper/draft_3/0_abstract.tex`** — Reworded "bounded learnable interpolation toward the Euclidean triplet regime" → "norm-constrained drift initialization that recovers the Euclidean triplet regime at startup".

### Expected Behavior

The main text (abstract, intro, methodology, experiments) no longer references the scalar alpha parameter. The alpha design is documented exclusively in the supplementary (X_suppl.tex Sec 1) as a superseded stepping stone. All cross-references between main text and supplementary resolve correctly.

## 2026-04-08T21:40:24Z — Implement Experiments 1d and 1e: Camera-Conditioned Drift Losses

### Problem

The Idea-1 ablation series (1a–1c) tested auxiliary losses on the drift subspace but did not test the complementary hypothesis: whether drift vectors should cluster by **camera** (encoding shared camera-specific artifacts) rather than by identity. The best-performing auxiliary (1a+1c) encourages same-PID cross-camera drift alignment—the opposite of the original intuition. Two new experiments extend the investigation:

- **1d**: Same-camera drift attraction (L2² on drift for same-camera pairs, any PID)
- **1e**: Cross-camera drift uniformity (Wang–Isola log-mean-exp Gaussian potential on L2-normalized drift for cross-camera pairs)

Together, 1d+1e form an alignment-uniformity decomposition on the drift space conditioned on camera labels, structurally parallel to BAU's alignment+uniformity on the identity space conditioned on PID labels.

### Modifications

**`bau/trainers.py`**:
- Added `drift_same_camera_attraction_loss(self, drift_w, cids)`: mean L2² over same-camera pairs (diagonal excluded). Constraint set C_1d = {(i,j) : c_i = c_j, i ≠ j}.
- Added `drift_cross_camera_uniformity_loss(self, drift_w, cids, t=2.0)`: Wang–Isola uniformity on L2-normalized drift for cross-camera upper-triangle pairs. Returns log(mean(exp(-t * dist²))). Constraint set C_1e = {(i,j) : c_i ≠ c_j, i < j}.
- Extended `BAUTrainer.__init__` with parameters: `use_drift_same_cam_attract`, `drift_same_cam_attract_weight`, `use_drift_cross_cam_uniform`, `drift_cross_cam_uniform_weight`, `drift_cross_cam_uniform_t`.
- Extended `train()` loop: compute both losses on `f_w_drift` and `cids`, add to total loss sum, add AverageMeter tracking, W&B step logging, and print-line entries.

**`examples/train_bau.py`**:
- Added 5 CLI arguments: `--use-drift-same-cam-attract`, `--drift-same-cam-attract-weight` (default 0.1), `--use-drift-cross-cam-uniform`, `--drift-cross-cam-uniform-weight` (default 0.1), `--drift-cross-cam-uniform-t` (default 2.0).
- Passed all through to `BAUTrainer` constructor.

**`sbatch/sweep_loss_ablation_Idea1_1d1e.sbatch`** (new):
- 10-job array (5 arms × 2 drift conditioning).
- EXP 0: baseline 1a only (reference). EXP 1: 1a+1d. EXP 2: 1a+1e. EXP 3: 1a+1d+1e. EXP 4: 1a+1c+1d+1e.
- Each crossed with instance and domain drift conditioning.
- Base recipe matches `sweep_loss_ablation_Idea1_extensions.sbatch`: resnet50_finsler, M+MS+CS→C3, batch 256, 60 epochs, Finsler domain loss (full memory bank), omega reg (w=1.5), identity triplet only, eval-drift false.

**`results/Idea1_Experiments_1d1e_Analysis.md`** (new):
- Paper-ready analysis document (theory_first phase) with: orthogonal decoupling investigation (Gram–Schmidt guarantees and limitations), motivation from 1a–1c, reinterpretation of the 1c result, full mathematical formalization of 1d and 1e, justification of Wang–Isola over alternatives (MHE, DPP, VICReg, reversed MMD), expected interactions and failure modes.

### Expected Behavior

New CLI flags are inactive by default (no behavioral change to existing runs). When activated, 1d pulls same-camera drift vectors together and 1e pushes cross-camera drift directions apart on the unit hypersphere. The sweep produces results directly comparable to slide 9 of the 03.04.2026 meeting deck (same base recipe, same eval protocol). Potential failure modes: drift collapse under strong 1d weight, gradient vanishing in 1e if drift vectors are already well-separated.

---

## 2026-04-10 — Remove L2 normalization from 1e (cross-camera drift uniformity)

### Problem

The Wang–Isola uniformity loss in `drift_cross_camera_uniformity_loss` projected drift vectors onto the unit hypersphere via `F.normalize` before computing the Gaussian potential. This discards drift magnitude, which carries semantic content in the Randers distance (the asymmetric contribution is ‖ω‖·⟨ω̂, v̂⟩). The normalization also creates an asymmetry between 1d (operates on raw drift in the ball) and 1e (operates on directions only), leaving the cross-camera norm distribution unsupervised. The hyperspherical uniformity guarantee from Wang & Isola (2020) does not transfer to the open Randers ball where drift vectors actually live.

### Modifications

**`bau/trainers.py`**:
- Commented out `drift_norm = F.normalize(drift_w, p=2, dim=1)` in `drift_cross_camera_uniformity_loss`. The Gaussian potential now operates on raw drift vectors: exp(-t·‖ω_i - ω_j‖²) instead of exp(-t·‖ω̂_i - ω̂_j‖²).

**`sbatch/sweep_1e_no_norm.sbatch`** (new):
- 8-job array (4 arms × 2 drift conditioning): baseline, 1a+1e, 1a+1d+1e, 1a+1c+1d+1e.
- Identical recipe to `sweep_loss_ablation_Idea1_1d1e.sbatch`. The only code difference is the removed normalization. Baseline arms included for direct comparison.
- W&B tag `sweep-1e-no-norm` distinguishes from the original `sweep-loss-ablate-1d1e` runs.

### Expected Behavior

The 1e loss now repels cross-camera drift vectors in both direction and magnitude within the Randers ball, rather than only repelling directions on the hypersphere. This makes 1e geometrically consistent with 1d (both operate in the native drift space). The temperature parameter t=2.0 may need retuning since the scale of ‖ω_i - ω_j‖² in the ball (max ~3.6 for max_norm=0.95) differs from the hyperspherical scale (max 4.0 for unit vectors).

---

## 2026-04-10 — Auxiliary loss configuration sweep (Unified Finsler focus)

### Problem

Prior sweeps tested drift auxiliaries (1c, 1d, 1e) exclusively under the 1a regime (`--identity-triplet-only`, Euclidean triplet on identity slice). The Unified Finsler baseline (full d_F triplet, no `--identity-triplet-only`) matches 1a on mAP (44.1) and beats it on R1 (44.0–44.3), but has never been combined with the drift auxiliaries that boosted R1 under 1a (1c → 44.9 R1, 1d → 44.5 R1 under domain conditioning). Additionally, `--eval-drift true` and `--bidirectional-triplet` were not tested on the strongest 1a+1d arm. The paper's interference thesis (Section 3.7) lacks an ablation row for L_tri^dom under Unified Finsler.

### Modifications

**`sbatch/sweep_auxiliary_loss_configs.sbatch`** (new):
- 12-job array (6 configs × 2 drift conditioning: instance/domain).
- All configs use `--eval-drift true`, `--use-omega-reg` (w=1.5), `--memory-bank-mode full`.
- EXP 0: Unified Finsler + 1d (w=0.1) + bidirectional. Tests whether Unified base + strongest R1 auxiliary exceeds both individually.
- EXP 1: Unified Finsler + 1c (w=0.1). Tests Unified base + highest-R1 auxiliary (44.9 under 1a).
- EXP 2: Unified Finsler + 1c + 1d (w=0.1 each). Complementary drift pair without 1e (which caused stacking degradation). Tests whether camera-clustered drift subspace helps under Unified.
- EXP 3: 1a + 1d (w=0.1) + bidirectional + eval-drift true. Adds two untested flags to the proven 1a+1d arm (previously run with eval-drift false, no bidirectional).
- EXP 4: Unified Finsler + 1d (w=0.3) + bidirectional. Weight sensitivity test — 3× the default 1d weight. Comparison with EXP 0 isolates weight effect.
- EXP 5: Unified Finsler + L_tri^dom (camera mining, w=1.0) + bidirectional. Paper Section 3.7 interference ablation — tests whether the asymmetric domain triplet degrades Unified Finsler as it did under 1a.
- Base recipe: resnet50_finsler, M+MS+CS→C3, batch 256, 60 epochs, symmetric trapezoidal drift, omega reg w=1.5.

### Expected Behavior

EXP 0/1 are the highest-priority runs: if Unified Finsler + 1d or 1c under domain conditioning exceeds 44.1 mAP or 44.9 R1, it establishes a new best. EXP 2 tests whether 1c+1d avoids the degradation seen in 1c+1d+1e (the uniformity objective 1e was the likely conflict source). EXP 3 should show marginal gains from eval-drift and bidirectional on the already-strong 1a+1d arm. EXP 4 probes whether 0.1 was suboptimal for 1d. EXP 5 is expected to degrade (validating the interference thesis), but if it doesn't, the interference is 1a-regime-specific and the paper narrative needs revision.

## [2026-04-22 22:20:29] Created missing runs for Idea 1 ablation with Euclidean domain loss
* **Files altered:** Added `sbatch/sweep_loss_ablation_Idea1_1d1e_EuclideanDom.sbatch`
* **Problem solved:** The ablation table had missing runs under the Euclidean Domain Loss configuration for runs `1a+1d`, `1a+1e`, `1a+1d+1e`, and `Unified Finsler + 1c`, focused entirely on instance conditioning.
* **Expected behavior:** Executing the new sbatch script will perform a 4-run array job filling the gap for Euclidean L_domain instance-conditioning experiments. The `--identity-triplet-only` flag correctly encapsulates all `1a` variations but is actively dropped for the `Unified Finsler + 1c` run.

## [2026-04-23 13:39:13] Parsed logs for Euclidean domain loss Idea 1 ablation
* **Problem solved:** Requested to populate the missing cells for Euclidean domain loss (instance conditioning) on runs 1a+1d, 1a+1e, 1a+1d+1e, and Unified + 1c.
* **Files altered:** Extracted logs into `results/metric sweeps/sweep_loss_ablation_Idea1_1d1e_EuclideanDom_metrics_missing.md`
