# Changelog

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
