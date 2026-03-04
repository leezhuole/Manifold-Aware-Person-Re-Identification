
## Learnable Randers Alpha (New)
18.01.2026

This branch introduces a learnable global Randers $\alpha$ that modulates the asymmetric term in the canonical Randers distance. The parameter is stored separately from the backbone to keep model structure clean and to enable incremental extensions (e.g., per-domain or per-layer alpha).

**New CLI Flags (train):**

- `--alpha`: initializes learnable $\alpha$ if set (legacy behavior: fixed scalar).
- `--alpha-init`: overrides `--alpha` for initialization (use `0.0` to start from Euclidean baseline).
- `--alpha-max`: max value for scaled sigmoid (default `1.0`).
- `--alpha-temp`: temperature for scaled sigmoid (default `1.0`).

**Example:**

```bash
python examples/train_bau.py \
    --alpha 0.0 \
    --alpha-max 1.0 \
    --alpha-temp 1.0
```

**W&B Logging:**

- `alpha/value` is logged once per epoch when W&B is enabled.

## Implementation Notes (Learnable Alpha)
- **Learnable parameter module:** `AlphaParameter` with scaled sigmoid in `bau/loss/triplet.py`.
- **Training losses:** `BAUTrainer` now uses a global `alpha_module` and passes `alpha_value` to triplet/align/uniform/domain losses in `bau/trainers.py`.

- **Evaluation distances:** pairwise distance now accepts `alpha` and uses it in `bau/evaluators.py`.
- **Checkpointing:** `examples/train_bau.py` now saves dictionaries with `state_dict` and optional `alpha_state`.
- **Loading:** `examples/train_bau.py` loads `alpha_state` if present (backward compatible with old checkpoints).
