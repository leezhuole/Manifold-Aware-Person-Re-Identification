# Changelog

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
