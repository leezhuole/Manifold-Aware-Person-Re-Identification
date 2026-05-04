# Plan: Balanced Per-Severity Evaluation (Section 8, toy_lmono_diagnostic_analysis.md)

## Context

The existing toy eval protocol (`eval_toy_checkpoint.py`) uses unbalanced query/gallery sizes: 50 clean queries vs. 400-item corrupted gallery (Direction A) against 400 corrupted queries vs. 50-item clean gallery (Direction B). This gallery-composition confound makes the reported Δ uninterpretable as a content-level asymmetry signal. Section 8 specifies a balanced per-severity eval that holds query count, gallery size, and correct-match count equal across both directions, isolating the genuine content-level asymmetry.

Two code changes are required. No model retraining, no changes to existing eval scripts or evaluators.

---

## Verified preconditions

- `examples/data/ToyCorruption/bounding_box_test/` exists with **500 files** (50 PIDs × 2 sources × 5 severities).  
- Filename pattern: `{pid:04d}_c{source_idx}s{severity+1}_000001_01.jpg` — source and severity are unambiguously encoded.  
- `examples/eval_toy_balanced.py` does **not** yet exist.  
- `bidirectional_evaluate` in `bau/evaluators.py:319` accepts arbitrary `(query_clean, gallery_corrupted, query_corrupted, gallery_clean)` lists — no changes needed there.  
- D8 inside `bidirectional_evaluate` is computed by `_collect_d8_pairs` (line 159), which matches by `(pid, cam)`. In the balanced eval, query is source 1 (cam=1) and gallery is source 2 (cam=2), so no same-cam pairs exist → D8 will be NaN throughout. This is expected and must be documented in the script output.

---

## Change 1 — `bau/datasets/toy_corruption.py`

**Goal:** Expose `self.by_source_severity: dict[(source_idx, severity), list[tuple]]` containing `(bounding_box_test/ path, pid0, cam_id)` triples for all eval images.

### What to modify

**`_build_lists`** (line 89): add population of `by_source_severity` inside the `elif split == "eval"` branch. No changes to the train branch or existing eval-split assignments.

```python
def _build_lists(self, images_map):
    train, query_s1, query_s2, gallery = [], [], [], []
    by_source_severity = {}                          # NEW

    for key in sorted(images_map.keys()):
        entry = images_map[key]
        split = entry.get("split", key.split("/", 1)[0])
        basename = self._basename_from_key(key)

        if split == "train":
            # ... unchanged ...

        elif split == "eval":
            subdir, triple = self._eval_subdir_and_tuple(basename, entry)
            img_path = osp.join(self.dataset_dir, subdir, triple[0])
            row = (img_path, triple[1], triple[2])
            if subdir == "query_s1":
                query_s1.append(row)
            elif subdir == "query_s2":
                query_s2.append(row)
            else:
                gallery.append(row)

            # NEW: unified bounding_box_test/ path for balanced eval
            severity   = int(entry["severity"])
            source_idx = int(entry["source_idx"])
            bbt_path   = osp.join(self.dataset_dir, "bounding_box_test", basename)
            ss_key     = (source_idx, severity)
            by_source_severity.setdefault(ss_key, []).append(
                (bbt_path, triple[1], triple[2])
            )

    return train, query_s1, query_s2, gallery, by_source_severity  # 5-tuple now
```

**`__init__`** (line 43): update unpack and assignment.

```python
train, query_s1, query_s2, gallery, by_source_severity = self._build_lists(payload["images"])
# ... existing verbose/print block ...
self.by_source_severity = by_source_severity
```

**Expected state after change:** `dataset.by_source_severity[(src, sev)]` returns a list of 50 triples for each `(src ∈ {1,2}, sev ∈ {0,1,2,3,4})` — 10 cells × 50 = 500 images total. All paths point to `bounding_box_test/`.

---

## Change 2 — `examples/eval_toy_balanced.py` (new file)

**Goal:** Per-severity balanced bidirectional eval (50 queries × 50 gallery in each direction for each k).

### Structure (modelled on `eval_toy_checkpoint.py`)

**Arguments** (match `eval_toy_checkpoint.py` exactly plus two new):
- `--resume` (required)
- `--data-dir` (required)
- `--alpha` (default `"0.0,0.1,0.3,0.5,0.9"`)
- `--log-dir`, `--log-file`
- `--eval-batch-size`, `--workers`, `--height`, `--width`, `--eval-print-freq`
- `--no-theta` (new flag): use when evaluating an M1-style checkpoint (no `theta_head`); sets `return_theta=False` and loads `resnet50` instead of `toy_resnet50`

**Checkpoint loading** (handle both formats):
```python
ckpt = load_checkpoint(args.resume)
if isinstance(ckpt, dict) and "model" in ckpt:
    # M2a/M2b format from train_toy_lmono.py
    model = models.create("toy_resnet50", num_classes=0, pretrained=False)
    model.load_state_dict(ckpt["model"], strict=False)
    return_theta = not args.no_theta
else:
    # M1 state-dict format
    model = models.create("resnet50", num_classes=0, pretrained=False,
                          with_theta_head=False)
    copy_state_dict(ckpt, model)
    return_theta = False
```

**Main eval loop:**

```python
severities = [0, 1, 2, 3, 4]  # k=0 is σ=0 vs σ=0 sanity check
results_per_k = {}

for k in severities:
    query_A  = dataset.by_source_severity[(1, 0)]   # source 1, sev 0 (50 items)
    gallery_A = dataset.by_source_severity[(2, k)]  # source 2, sev k (50 items)
    all_rows = sorted(set(query_A) | set(gallery_A))
    loader   = build_eval_loader(all_rows, ...)

    metrics = bidirectional_evaluate(
        model, loader,
        query_clean=query_A,
        gallery_corrupted=gallery_A,
        query_corrupted=gallery_A,   # swap: Direction B
        gallery_clean=query_A,
        alpha_values=alpha_values,
        return_theta=return_theta,
        print_freq=args.eval_print_freq,
        verbose=False,
    )
    metrics.pop("theta_by_fname", None)
    results_per_k[k] = metrics
```

**Spearman ρ (only if return_theta):** After the loop, build a single DataLoader over all 500 `bounding_box_test` items (`flatten(dataset.by_source_severity.values())`), run `extract_features(..., return_theta=True)`, call `spearman_rho_theta_severity`.

**Output format:**
```
========== eval_toy_balanced ==========
NOTE: D8 is NaN for all k — cross-source balanced eval has no same-cam pairs.
      D8 is a per-pair geometric gap; balanced eval uses source-1 vs source-2 images.

severity k=0 (σ=0 vs σ=0 sanity check — expect Δ≈0):
  Euclidean mAP A=X.XXXX  B=X.XXXX  Δ=X.XXXX
  Randers α=0.1 mAP A=... B=... Δ=...
  ...

severity k=1:
  ...

severity k=4:
  ...

--- Summary: Balanced per-severity Δ (mean over k=1..4) ---
  Euclidean: mean_Δ=X.XXXX  [per-k: k1=..., k2=..., k3=..., k4=...]
  Randers α=0.5: mean_Δ=X.XXXX ...

Spearman rho(theta, severity) = X.XXXX
========== eval_toy_balanced finished ==========
```

---

## Critical implementation notes

1. **D8 NaN is expected and must be documented in-script** (print explanation before the per-severity table). Do not suppress or treat as an error.

2. **Same-query reuse across k:** `query_A = by_source_severity[(1, 0)]` is identical for every k. The DataLoader for k=0 (σ=0 vs σ=0) may have duplicate images if source-1-sev-0 and source-2-sev-0 have the same filenames — they don't (different `c{src}` field), so `set()` dedup is safe.

3. **Cam-id filter in `mean_ap`:** query cam=1, gallery cam=2 for all k — no same-cam exclusions will fire. Verify that `mean_ap` and `cmc` don't silently treat "no same-cam matches to exclude" as an error condition. From the code, `mean_ap` excludes gallery items where `query_cam == gallery_cam`, so with cam 1 vs cam 2, all gallery items are retained — correct.

4. **Within-severity θ variance:** At high α, identity-correlated θ noise may cause small Randers ≠ Euclidean mAP even in the balanced eval. If this occurs, print a note — it indicates θ leakage, not a bug.

5. **`_build_lists` return arity change:** The internal change from 4-tuple to 5-tuple return is backwards-compatible since `_build_lists` is a private method called only from `__init__`.

---

## Files modified / created

| File | Change type |
|---|---|
| `bau/datasets/toy_corruption.py` | Modify `_build_lists` and `__init__` (~12 lines) |
| `examples/eval_toy_balanced.py` | New file (~150 lines) |

No changes to: `bau/evaluators.py`, `examples/eval_toy_checkpoint.py`, `examples/eval_toy_m1.py`, `bau/utils/randers.py`.

---

## Verification

1. `python -c "from bau.datasets.toy_corruption import ToyCorruption; d = ToyCorruption('examples/data/ToyCorruption'); assert len(d.by_source_severity) == 10; assert all(len(v)==50 for v in d.by_source_severity.values()); print('OK')`
2. Run `eval_toy_balanced.py --resume <m1_ckpt> --data-dir examples/data/ToyCorruption --no-theta` — k=0 should give Δ ≈ 0; k=1..4 should give Δ ≠ 0.
3. Run with M2a checkpoint (no `--no-theta`) — confirm Spearman ρ matches the value from `eval_toy_checkpoint.py` (same model, same eval images).
4. Confirm D8 is NaN for all k in the balanced output.
