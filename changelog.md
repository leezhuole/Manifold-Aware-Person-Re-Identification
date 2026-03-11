# Changelog

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
