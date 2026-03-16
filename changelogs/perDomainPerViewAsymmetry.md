# Per-Domain / Per-View Asymmetry Pivot Plan

_Date: 2026-03-11_

## Goal

Pivot the current `resnet50_finsler` pipeline away from a purely per-instance drift vector toward a **domain-conditioned or view-conditioned asymmetric component** that is more consistent with domain-generalizable person ReID.

The central hypothesis is:

> The main asymmetric bias in AG-ReIDv2-style retrieval is induced by **view/domain acquisition geometry**, not by person identity itself.

This document records the intended implementation direction so the code changes remain modular, understandable, and easy to ablate.

---

## Why pivot away from per-instance drift?

The existing codebase already shows the main symptoms:

- the learned `omega` vectors often remain small,
- gains over Euclidean BAU are marginal,
- BAU-style alignment/uniformity objectives favor identity invariance,
- an unconstrained per-instance drift branch can be interpreted by DG training as domain nuisance and therefore suppressed.

That makes the following interpretation plausible:

1. the retrieval problem may still contain directional structure,
2. but the current **instance-conditioned** drift parameterization is not the correct level of abstraction,
3. so the drift should be reinterpreted as **shared domain/view bias** plus, at most, a small residual correction.

---

## Design principles

The implementation should follow these rules:

1. **Preserve the identity pathway.**
   - The current identity slice remains the primary carrier of person discrimination.
   - Classification and triplet behavior should stay as close to the current BAU/Finsler recipe as possible.

2. **Keep the code modular.**
   - Add new components rather than rewriting large existing blocks.
   - Preserve the current `instance`-conditioned drift path for clean A/B testing.

3. **Exploit metadata already present in the pipeline.**
   - The first pass should use existing multi-source `did` labels.
   - AG-ReIDv2 view-filtered source datasets already map naturally onto domain IDs.

4. **Avoid hard target-domain lookup at inference.**
   - Evaluation should infer a soft domain/view token from the image instead of assuming target labels are known.

5. **Minimize risk in the first pass.**
   - Do not redesign `finsler_drift_dist()` initially.
   - Do not change the memory-bank semantics more than necessary.
   - Do not entangle the new drift path too aggressively with the identity representation.

---

## Planned code changes by file

### 1. [bau/models/model.py](../bau/models/model.py)

#### New modules

- `DomainConditionedDriftHead`
  - learns a small bank of domain/view prototypes,
  - projects them into the drift space,
  - optionally applies a feature-dependent gate,
  - optionally adds a small residual per-instance correction,
  - reuses the same drift norm-scaling rule as the original drift head.

#### `resnet50_finsler` updates

Add a configuration switch such as:

- `drift_conditioning="instance" | "domain"`

For `instance` mode:

- keep the original `FinslerDriftHead` behavior.

For `domain` mode:

- create the new `DomainConditionedDriftHead`,
- add a lightweight domain-token predictor,
- allow `forward()` to accept:
  - `domain_ids` during training,
  - or soft `domain_probs` / inferred domain probabilities during evaluation.

#### Intended behavior

The output embedding remains:

```text
[identity | drift]
```

but the drift term now reflects:

```text
shared domain/view bias + small residual correction
```

rather than purely per-instance drift.

---

### 2. [bau/trainers.py](../bau/trainers.py)

#### Training metadata reuse

The trainer already receives `dids` from the multi-source training loader. The first-pass implementation should reuse these directly.

#### New auxiliary loss

Add an optional domain-token supervision loss:

- objective: train the soft domain-token predictor used at test time,
- supervision source: existing `did` labels,
- default weight: small but non-zero,
- gradient scope: prefer keeping this branch lightweight so it does not dominate identity learning.

#### Keep current BAU behavior stable

- keep ID CE on the identity branch,
- keep triplet centered on the current embedding path,
- keep identity alignment as the default,
- avoid large changes to uniform and domain losses in the first pass.

---

### 3. [bau/evaluators.py](../bau/evaluators.py)

The initial goal is to avoid large evaluator rewrites.

Preferred behavior:

- the model itself should infer soft domain tokens during evaluation,
- the evaluator should continue extracting embeddings in the same way,
- the only observable difference should be whether the model produces instance- or domain-conditioned drift.

#### Future diagnostics

Once the first implementation is stable, the evaluator should also report:

- mean drift norm by predicted domain,
- entropy of predicted domain tokens,
- fraction of queries whose top-k changes when drift is enabled,
- forward/reverse asymmetry gap where protocols permit.

---

### 4. [examples/train_bau.py](../examples/train_bau.py)

Expose the pivot through CLI flags while keeping defaults understandable.

Planned or desired flags:

- `--drift-conditioning`
- `--domain-embed-dim`
- `--infer-domain-conditioning`
- `--domain-temperature`
- `--domain-residual-scale`
- `--domain-token-loss-weight`

The training entrypoint should also:

- pass the number of source domains into the model,
- preserve old behavior when `--drift-conditioning instance` is selected,
- keep experiment scripts easy to compare against prior runs.

---

### 5. [examples/test.py](../examples/test.py)

Standalone evaluation should remain usable for the new mode.

At minimum it should:

- construct `resnet50_finsler` with the new drift-conditioning options,
- unwrap `state_dict` checkpoints correctly,
- allow evaluation-time soft domain token inference.

---

## Implementation stages

### Stage 1 — Minimal viable pivot

Objective: test the hypothesis with the lowest possible risk.

- Add `domain` drift mode.
- Reuse existing source `did` labels.
- Train a lightweight domain-token predictor.
- Use inferred soft domain tokens at evaluation.
- Keep `finsler_drift_dist()` unchanged.
- Keep memory-bank semantics unchanged.

### Stage 2 — Better inductive bias

If Stage 1 is stable and promising:

- add stronger factorization between identity and drift,
- optionally add a domain-only classification head on drift,
- optionally reduce identity leakage into the drift branch,
- inspect domain-conditioned drift statistics per protocol.

### Stage 3 — Finer metadata

Only if Stage 1/2 justify it:

- move from source-domain IDs to raw `camid` or explicit view labels,
- allow pair-conditioned asymmetry,
- study query/gallery-direction tokens.

---

## Non-goals for the first pass

To keep the experiment low-risk, the first implementation should **not**:

- redesign the Finsler distance formula,
- replace BAU’s identity objectives,
- overhaul the memory bank into domain prototypes,
- force domain-conditioned drift onto every dataset immediately,
- claim accuracy gains before checking whether the new mechanism actually changes retrieval behavior.

---

## Recommended low-risk experiment matrix

### Baselines

1. `resnet50` Euclidean BAU
2. `resnet50_finsler` with `drift_conditioning=instance`
3. `resnet50_finsler` with `drift_conditioning=domain`

### Dataset focus

Start with AG-ReIDv2 source-view DG:

- `agreidv2_aerial`
- `agreidv2_cctv`
- `agreidv2_wearable`

### Evaluation modes

For each run, report:

1. identity-only ranking,
2. full drift-enabled ranking,
3. drift statistics and ranking-change diagnostics.

### Acceptance criteria

The pivot is worth continuing if at least one of the following holds:

- improves mAP / Rank-1 consistently over the instance-drift version,
- yields a larger, more interpretable drift signal,
- produces meaningful ranking changes aligned with cross-view directionality,
- or clarifies why DG suppresses asymmetry in a way that strengthens the research narrative.

---

## Working conclusion

The project should no longer be framed narrowly as “does Finsler asymmetry beat BAU?”

A stronger framing is:

> DG person ReID appears to suppress instance-level asymmetry, but structured domain/view-conditioned asymmetry may still be a useful nuisance model and diagnostic tool.

That framing is compatible with the current repository, preserves prior work, and gives the next experiments a clearer scientific target.
