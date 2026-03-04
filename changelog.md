# Changelog

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