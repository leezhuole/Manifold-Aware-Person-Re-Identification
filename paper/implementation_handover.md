# Implementation handover — paper/*.tex edits

**Date:** 2026-05-04
**Author decisions recorded:** 2026-05-04 (see §Author decisions below)
**Drafting_phase:** `results_implementation_lock`
**Assignee:** incoming agent implementing edits into `paper/*.tex` files

---

## Source documents (read these first)

| Document | Path | Purpose |
|----------|------|---------|
| Comment triage ledger | [paper/comments_triage_assessment.md](comments_triage_assessment.md) | 52-row peer-critical verdict table; justifications, provenance index, 4 DISAGREE items |
| Edit-large restructuring plan | [paper/restructuring_plan.md](restructuring_plan.md) | 9 items with proposed prose, QA verdicts, provenance per claim |
| Edit-small batch | [paper/edit_small_batch.md](edit_small_batch.md) | 24 atomic single-line fixes with application order |
| Publication facts | [.cursor/paper/publication_facts.md](../.cursor/paper/publication_facts.md) | Authoritative metrics, provenance, Forbidden_claims, code-snapshot caveat |
| Original comments | [paper/comments_triage.md](comments_triage.md) | Source comment file (read-only reference) |

---

## Author decisions (binding, do not re-litigate)

### Decision 1 — Headline number pairing: Option A (conservative)

Use **44.3% / +1.9%** consistently throughout the paper (grounded in job 1521465 vs. 1521474).
Do **not** use 44.9% (job 1521403, earlier code snapshot) or +2.5% in the main text.

Affects:
- `paper/0_abstract.tex` — see Item 1 proposed prose in [restructuring_plan.md](restructuring_plan.md)
- `paper/4_experimentalResults.tex` Table `\ref{tab:template_multiprotocol_bau}` P2 row — change 44.9 → 44.3; change any Δ claim from +2.5 → +1.9
- `paper/4_experimentalResults.tex` Sec. 4.3 paragraph — update Δ wording to +1.9%

Provenance: `publication_facts.md` §Headline_number_reconciliation row 1; §Primary Finsler runs job 1521465; §Euclidean baseline runs job 1521474.

### Decision 2 — Memory-bank equation rewrite: CONFIRMED

Apply **Item 4** of [restructuring_plan.md](restructuring_plan.md) in full.

**Critical scope extension:** In addition to replacing the equation and surrounding prose in Sec. 3.4 (`paper/3_methodology.tex`), the implementing agent must locate and update **every other reference in `paper/*.tex` to the old formulation** — specifically:
- Any text referring to "$\mathbf{m}_k$" as "centroid of domain $k$" or "domain centroid"
- Any prose saying the memory bank stores or computes per-domain means
- Any equation or notation implying the loss argument is a per-domain aggregated feature rather than a per-PID bank entry

The correct semantics (verified against `bau/models/memory.py:16–22` and `bau/trainers.py:78,130–139`):
- The bank stores **per-PID features** (`self.features[y]`, keyed by PID `y`), labeled by domain ID (`self.labels` = `dids`)
- `momentum_update` is called with `pids` as the second argument (line 78 of `bau/trainers.py`)
- Same-domain filtering is done via `domain_mask = torch.eq(dids.unsqueeze(-1), self.memory_bank.labels)` (line 136)
- The loss mines the hardest non-self same-domain PID memory entry, **not** a domain centroid

Proposed replacement equation and prose are in [restructuring_plan.md](restructuring_plan.md) Item 4.
Provenance is recorded in `publication_facts.md` §Memory_bank_semantics_correction (added 2026-05-04).

### Decision 3 — Sec. 3.6 compression: PROCEED

Apply **Item 6** of [restructuring_plan.md](restructuring_plan.md) in full:
- Compress Sec. 3.6 ("Objective interference analysis") in `paper/3_methodology.tex` to the 3-sentence definitional paragraph (no table numbers, no table cell references)
- Move the empirical numbers (`-7.5% mAP`, `$\max\|\hat\omega\|$ 0.09→0.11`) to the Sec. 4.3 "Auxiliary-loss sweep" paragraph in `paper/4_experimentalResults.tex`

---

## Full application scope — what to implement

### Step 0 — Pre-check (before any edit)

Verify no `paper/*.tex` file has been modified since 2026-05-04. If files have changed, re-read them and reconcile with the plans below before applying.

### Step 1 — Edit-large items (from restructuring_plan.md)

Apply all 9 items in the order below. Items 1, 4, 6 are the three approval-gated items (now approved per §Author decisions). Items 2, 3, 5, 7, 8, 9 were always mechanical.

| Item | File(s) affected | Status |
|------|-----------------|--------|
| **1. Abstract rewrite** (Option A prose) | `paper/0_abstract.tex` | APPROVED — use Option A proposed prose from `restructuring_plan.md` Item 1 verbatim |
| **2. Intro contributions list refactor** | `paper/1_intro.tex` | Mechanical — replace 6-bullet list with 4-contribution paragraph; delete commented-out line 19 |
| **3. Supplement moves** (drift-property para + Sec. 3.3 relationship para) | `paper/3_methodology.tex`, `paper/X_suppl.tex` | Mechanical — delete from main, append to respective supplement sections |
| **4. Memory-bank equation rewrite** | `paper/3_methodology.tex` (primary), all other `paper/*.tex` for stale references | APPROVED — see Decision 2 scope extension above |
| **5. Total-loss relocation + 4.2/4.3 deduplication** | `paper/3_methodology.tex`, `paper/4_experimentalResults.tex` | Mechanical — move eq. to end of Sec. 3.5; replace verbose 4.2 paragraph; replace 4.3 recipe restatement |
| **6. Sec. 3.6 → 3-sentence + numbers move to 4.3** | `paper/3_methodology.tex`, `paper/4_experimentalResults.tex` | APPROVED — per Decision 3 |
| **7. Sec. 3.5 interpretation paragraph trim** | `paper/3_methodology.tex` | Mechanical — replace two-interpretation block with one-sentence deferral to Sec. 5 |
| **8. Identity-only alignment paragraph rewrite** | `paper/3_methodology.tex` | Mechanical — see proposed prose in `restructuring_plan.md` Item 8 |
| **9. Related work cleanup** | `paper/2_relatedwork.tex` | Mechanical — delete 2 commented-out paragraphs; rewrite Finsler-MDS transition sentence |

### Step 2 — Edit-small batch (from edit_small_batch.md)

Apply the 24 items in the order specified in [edit_small_batch.md](edit_small_batch.md) §Application order:

1. DELETE rows first: R-extra, R-extra2, I-extra, M14, M22
2. Notation-only swaps: R4, R5, R6, R7, I2
3. Rewording rows: A1, A2, A4, A5, R1, I3, I4, M5, M7, M21, E1, E4, E5
4. Position moves: M6 (verify destination unchanged after steps 1–3)
5. Integrated multi-line edits: E6+E7 bundled

**Note on M4:** M4 in `edit_small_batch.md` is conditional — if Item 3a (restructuring_plan.md) has been applied, M4 is N/A (the paragraph it targets has been moved to the supplement). Apply M4 only if Item 3a was not applied.

### Step 3 — Headline number update (cross-cutting, from Decision 1)

Grep all `paper/*.tex` for "44.9", "+2.5", "+2.5\%", "protocol-matched" and update each occurrence to Option A values (44.3%, +1.9%) unless the occurrence is in a table row clearly labeled as the older single-run snapshot (in which case add a footnote, do not delete the row — it can stay as a secondary result if it already appears in a results table, but the abstract and any Δ claims must use +1.9%).

---

## Do NOT change — DISAGREE items (4 total)

These comment-driven changes were rejected in the triage assessment. The implementing agent must not apply them.

| Triage ID | Comment | Reason for rejection |
|-----------|---------|---------------------|
| **I1** | Explain non-monotonic Δ profile (k=1→k=4) in the intro caption | Explanation belongs in Sec. 4.5, not the introduction; CVPR pacing |
| **M3** | Replace $c_{\max}$ with $\alpha$ | Would collide with the legacy scalar gate $\alpha$ in Supp.; the two are distinct symbols |
| **M12** | "Convex combination of $S$" is wrong | Mathematically correct — softmax outputs are non-negative and sum to 1 |
| **E7** | Replace $+0.023$ with $+0.041$ | Both numbers are correct under different definitions; $+0.023$ is the delta-of-deltas over trained Euclidean; $+0.041$ is the absolute mean Δ; the fix is to disambiguate (done in E6+E7 bundle), not to replace |

---

## Partial / Ask items — do not apply without explicit author instruction

| Triage ID | Comment | Status |
|-----------|---------|--------|
| **A6** | Toy dataset summary in abstract ("proved the intuition") | PARTIAL — abstract Option A prose (Decision 1) already handles this correctly; do not add overclaiming language |
| **R3** | "Nuisance" terminology — keep or replace globally | ASK — author has not decided; leave "nuisance" and add parenthetical "(e.g., camera, viewpoint, illumination)" at first use, which is the least-disruptive safe default |
| **M17** | $\mathbf{z}^\omega$ ↔ $\omega$ notation bridge sentence | PARTIAL — add one clarifying sentence at start of Sec. 3.3 (proposed in triage ledger M17 row); do not do a global symbol rename |
| **M8** | Randers positive-definiteness — add parenthetical | AGREE — add "(positivity of the Randers norm $F$)" at the Randers positivity constraint paragraph; trivial and safe |

---

## Verification checklist (after applying all edits)

- [ ] `pdflatex paper/main.tex` compiles without undefined references
- [ ] Abstract: 44.3% and +1.9% appear; 44.9% and +2.5% do not appear (unless in a clearly labeled historical-snapshot table row)
- [ ] Abstract: no unintroduced abbreviations (M+MS+CS, CUHK03 spelled out on first use)
- [ ] Sec. 3.4: $\mathbf{m}_k$ as "centroid of domain $k$" does not appear anywhere in main text
- [ ] Sec. 3.4–3.6: Total-loss paragraph appears after $\mathcal{L}_{\mathrm{dcc}}$ definition
- [ ] Sec. 3.6: no table-cell numbers (−7.5%, max‖ω‖ values) in methodology
- [ ] Sec. 4.2: one-line pointer to eq.~\ref{eq:total_loss} replaces the verbose loss restatement
- [ ] Sec. 4.3: no duplicate loss-recipe list; Δ claim says +1.9%, not +2.5%
- [ ] Related work: no commented-out paragraphs containing Forbidden_claims phrasing
- [ ] Arrow notation (→) does not appear in prose (only in equations and figure labels)
- [ ] Grep for "protocol-matched" → zero hits in main text
- [ ] Grep for "cautionary datapoint" → zero hits
- [ ] Grep for "continuous feature map" → zero hits
- [ ] $c_{\max}$ and $\alpha$ are distinct symbols; no conflation introduced

---

## Code-snapshot provenance reminder

The current `bau/` tree (branch `toyDataset_v4`, commit `1c0e181`) is a near-vanilla BAU revert. `FinslerDriftHead`, `finsler_drift_dist`, and `AlphaParameter` no longer exist in `bau/models/model.py` or `bau/loss/triplet.py`. All paper claims about those components reference the code as it existed at the job-log commits (1521465, 1521403, 1521019, etc.), **not** the current working tree. Do not grep the current `bau/` for these symbols and conclude they are absent from the paper's scope; the paper is correct to reference them via `publication_facts.md` job-log provenance. See `publication_facts.md` §Code_snapshot_provenance_caveat (added 2026-05-04).
