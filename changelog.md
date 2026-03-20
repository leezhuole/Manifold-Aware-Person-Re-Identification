# Changelog

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
