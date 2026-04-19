# Idea‑1 experiments 1d and 1e: results analysis and Euclidean \(\mathcal{L}_{\mathrm{dom}}\) sweep note

**Protocol (primary tables):** Market1501 + MSMT17 + CUHK-SYSU → CUHK03 (`resnet50_finsler`), as in `results/metric sweeps/sweep_loss_ablation_Idea1_1d1e_metrics.md` and related sweep ledgers.

**Design reference:** `changelogs/Idea1_Experiments_1d1e_Analysis.md` (1d = same-camera drift \(L_2\) attraction; 1e = Wang–Isola-style cross-camera repulsion on **raw** drift).

**Provenance:** Metrics are copy-paste / job-audited in `results/metric sweeps/` (`sweep_loss_ablation_Idea1_1d1e_metrics.md`, `sweep_auxiliary_loss_configs_metrics.md`, `sweep_unified_finsler_idea1_gap_metrics.md`, `sweep_loss_ablation_Idea1_EuclideanDom_metrics.md`, `eval_drift_true_finsler_ranking_table.md`).

---

## Slide: concise talking points

1. **Setup (one line).** Idea‑1 on M+MS+CS→CUHK03 with `resnet50_finsler`: 1d = same‑camera \(L_2\) drift attraction; 1e = Wang–Isola‑style cross‑camera repulsion on **raw** drift (`bau/trainers.py`); compare **instance** vs **domain** drift sampling × **Finsler** vs **Euclidean** domain loss in the results table.

2. **No free lunch on mAP.** Peak mAP stays at the **1a / Unified Finsler** tier (~**44.1%** under Finsler \(\mathcal{L}_{\mathrm{dom}}\) in the locked ablation tables). No 1d/1e arm clearly **beats** that on mAP; best 1c/1d‑class numbers sit around **43.8%** mAP.

3. **Where 1d helps (partial confirmation of the camera‑artifact story).** **1a+1d** with **domain** drift conditioning reaches **44.5% R1** (Finsler domain loss), i.e. a real Rank‑1 lift vs weak 1a instance cells—**but** it is **smaller** than **1a+1c** at **44.9% R1** under the same regime.

4. **1e is a “split” loss: instance OK, domain weak.** **1a+1e** is roughly **neutral** under **instance** conditioning (~**43.7 / 44.0**); under **domain** conditioning it **drops** (~**43.0 / 43.7**). That contradicts a simple “domain sampling always helps drift losses” rule.

5. **Designed complementarity (1d+1e) fails in practice.** **1a+1d+1e** is **worse than either alone** (R1 down to **~42.5–42.7**). The alignment/uniformity **template** borrowed from BAU on the identity sphere does **not** port cleanly to camera‑labeled objectives on a **bounded** drift ball with coupled gradients.

6. **Unified + 1c confirms “don’t stack lightly”.** Auxiliary sweep (`sweep_auxiliary_loss_configs_metrics.md`): **Unified Finsler + 1c (w=0.1)** hits **44.9% R1** only under **instance** drift; **domain** conditioning **collapses** R1 to **42.1%**—another sign that **multiple drift‑space pulls + wrong sampler pairing** is dangerous.

7. **Take‑home for the slide.** Under this protocol, **light, single drift auxiliary + correct conditioning** can nudge R1; **multi‑term drift stacks** and **1d+1e** in particular look **unpromising**. The leading **single** drift regularizer in the Idea‑1 series remains **1c** (cross‑camera same‑ID drift alignment), not the camera‑pure 1d/1e story.

8. **Missing Euclidean‑domain columns.** Empty **Eucl. Dom.** cells for 1d/1e in composite tables are **not** “null results”—they are **unrun** factorial corners. Interpreting them as evidence is **incorrect** until filled.

---

## Euclidean \(\mathcal{L}_{\mathrm{dom}}\) (no Finsler geometry in domain / memory loss): run or skip?

### Reasons to run (moderate priority, not a “must”)

1. **Completes a controlled factor.** The repo already documents **Finsler vs Euclidean** \(\mathcal{L}_{\mathrm{dom}}\) for 1a, 1b, refined 1b, 1c, Unified (`sweep_unified_finsler_idea1_gap_metrics.md`; Euclidean block in `sweep_loss_ablation_Idea1_1d1e_metrics.md`). Without 1d/1e under `--use-euclidean-domain-loss`, the paper cannot claim whether **1d/1e** behavior is **intrinsic** to those losses or **contingent** on full‑embedding domain repulsion in Finsler geometry.

2. **1c shows \(\mathcal{L}_{\mathrm{dom}}\) geometry changes which drift regime wins.** Under Finsler \(\mathcal{L}_{\mathrm{dom}}\), **1a+1c** prefers **domain** drift sampling for R1 (**44.9%**); under Euclidean \(\mathcal{L}_{\mathrm{dom}}\), the same loss reads **43.8 / 42.9** (instance) vs **43.7 / 42.9** (domain)—the **large domain‑conditioning advantage disappears** (`sweep_loss_ablation_Idea1_1d1e_metrics.md`, Euclidean subsection). That is direct precedent that **domain‑loss geometry × drift sampler** interactions are **non‑ignorable** for drift‑only auxiliaries.

3. **If the main reported metric is Euclidean ranking (`--eval-drift false`)**, an Euclidean‑\(\mathcal{L}_{\mathrm{dom}}\) row is **on‑manifold** for reporting: training target aligns with eval geometry for the identity slice, and drift auxiliaries are isolated as **regularizers** rather than coupled to full‑embedding domain repulsion.

4. **Scientifically**, the camera‑hypothesis tests (1d/1e) are **orthogonal** in intent to “should domain repulsion live on \(d_F\) or \(d_E\)?”. A small factorial answers whether **identity‑space** domain loss suffices for **camera‑structured drift** regularization—consistent with the README split between domain loss on full embedding vs identity.

### Reasons not to prioritize (or to scope narrowly)

1. **Primary evidence already shows failure modes without the extra sweep.** **1d+1e** and **1c+1d+1e** fail under Finsler \(\mathcal{L}_{\mathrm{dom}}\); **Unified+1c** is highly conditioning‑sensitive (`sweep_auxiliary_loss_configs_metrics.md`). It is **unlikely** that switching \(\mathcal{L}_{\mathrm{dom}}\) to Euclidean will **reverse** those into strong gains; at best expect **quantitative** shifts and **clearer Euclidean‑eval** reporting.

2. **Cost / diminishing returns.** Runs require **2 drift samplers × several arms × seeds**; the changelog documents **RNG / job variance** on the order of **~0.6 mAP** between sweeps. **Marginal** paper value is **boundary completion**, not a new headline number—unless Euclidean \(\mathcal{L}_{\mathrm{dom}}\) **systematically** rescues 1d/1e (no prior hint).

3. **If the story is asymmetric test ranking (`--eval-drift true`)**, training with **Euclidean** \(\mathcal{L}_{\mathrm{dom}}\) can **hurt** Finsler‑eval on stressed arms: see Table B in `eval_drift_true_finsler_ranking_table.md` (e.g. **1b**: Finsler eval **−6.1** mAP vs Euclidean eval when trained with Euclidean \(\mathcal{L}_{\mathrm{dom}}\)). A 1d/1e sweep under Euclidean \(\mathcal{L}_{\mathrm{dom}}\) must be reported **together** with **both** eval modes or it **misstates** deployable performance.

4. **Interpretability.** 1d/1e only touch \(\omega\); \(\mathcal{L}_{\mathrm{dom}}\) in Euclidean mode uses the **identity** memory bank (`--memory-bank-mode identity`). The interaction is **weaker** than for objectives that used the **full** Finsler embedding in the domain triplet (1b). Incremental insight may be modest.

**Recommendation:** Treat a **Euclidean‑\(\mathcal{L}_{\mathrm{dom}}\)** replication of **1a, 1a+1d, 1a+1e, 1a+1d+1e** (and optionally **1a+1c** as calibration) as a **targeted gap‑fill** if the paper commits to a **two‑way** table (Finsler vs Euclidean \(\mathcal{L}_{\mathrm{dom}}\)) for **all** Idea‑1 drift auxiliaries **or** if reviewers ask for **eval‑geometry alignment**. It is **not** justified as a broad second sweep hunting for SOTA; existing sweeps already bound **qualitative** conclusions below.

---

## Paper‑ready detailed analysis

### A. Protocol and evaluation alignment

Reported numbers mix **training** choices: Finsler \(\mathcal{L}_{\mathrm{dom}}\) (`memory-bank-mode full`) vs Euclidean \(\mathcal{L}_{\mathrm{dom}}\) (`--use-euclidean-domain-loss`), and **test** choices: identity‑only Euclidean ranking vs full‑embedding Finsler ranking. Tables in `sweep_loss_ablation_Idea1_1d1e_metrics.md` label **Fins. Dom.** / **Euc. Dom.** columns accordingly. For claims about **asymmetric test ranking**, cite `eval_drift_true_finsler_ranking_table.md` explicitly—Δ is often **~0** for well‑behaved arms under Finsler‑trained \(\mathcal{L}_{\mathrm{dom}}\), but **not** when \(\mathcal{L}_{\mathrm{dom}}\) is Euclidean and the model is stressed (e.g. 1b).

When substituting into `paper/draft_4/`, add one explicit sentence: **1d/1e under Euclidean \(\mathcal{L}_{\mathrm{dom}}\) were not run** in the primary ledger, so **cross‑column comparisons involving empty cells are not supported** until those jobs exist.

### B. Relation to the design intent (`Idea1_Experiments_1d1e_Analysis.md`)

The analysis document motivates **1d** as testing whether drift should capture **shared camera artifacts** (same‑camera clustering over all identities) and **1e** as **cross‑camera uniformity** complementary to **1d**, analogously to BAU’s alignment + uniformity—but on **camera labels** and **raw** drift inside the Randers ball.

**Empirical verdict:**

- **1d (domain drift sampling)** provides **partial** alignment with the “camera‑structured drift” hypothesis: **43.8% mAP / 44.5% R1** is a **competitive** R1 point relative to baseline **domain** R1 in the 1d/1e job. However, **same‑PID cross‑camera alignment (1c)** still achieves **higher R1 (44.9%)** and comparable mAP (**43.8%**). So the data support **camera‑level clustering** as a **usable** inductive bias **when paired with domain drift conditioning**—not as the **strongest** drift regularizer in the suite.

- **1e** does **not** behave like a stable second factor: **roughly neutral** under **instance** sampling and **harmful** under **domain** sampling. That undercuts the hoped‑for clean **“1d aligns / 1e uniformizes”** decomposition; **geometry** (bounded drift, softmax weighting, batch statistics) and **label statistics** (few cameras vs many IDs) likely dominate the Wang–Isola intuition.

### C. The 1d+1e combination and full stacks

**1a+1d+1e** performs **below** **1a+1d**, **1a+1e**, and **1a** on R1 (**~42.5–42.7** vs **~43–44**). This matches the analysis document’s **over‑constraint** failure mode for the drift slice. **1a+1c+1d+1e** is similarly poor (**42.2% R1** instance in `sweep_loss_ablation_Idea1_1d1e_metrics.md`). **Conclusion:** The **drift subspace does not tolerate naive stacking** of multiple structured losses under the tested weights—consistent with the broader Idea‑1 theme that **only mild, single‑purpose drift supervision** survives.

### D. Comparison to Unified Finsler and 1c

- **Unified Finsler** remains the **mAP‑centric** reference (**44.1%** both conditionings under Finsler \(\mathcal{L}_{\mathrm{dom}}\) in the gap table), with a documented **Euclidean‑eval** mAP dip for Unified in the Euclidean‑\(\mathcal{L}_{\mathrm{dom}}\) rows (`sweep_unified_finsler_idea1_gap_metrics.md`)—a **representation / metric mismatch** effect.

- **1c** remains the **clearest R1 winner** under Finsler \(\mathcal{L}_{\mathrm{dom}}\) + **domain** drift sampling (**44.9% R1**), and it **contradicts** the naive “drift = camera artifact only” story: it **aligns** drift across **cameras** for the **same person**. **1d** partially rehabilitates camera structure but **does not surpass 1c**.

### E. Auxiliary sweep cross‑check

`sweep_auxiliary_loss_configs_metrics.md`: **Unified + 1c (w=0.1)** achieves **43.9% / 44.9%** (instance)—tying the best R1 tier—but **domain** conditioning falls to **42.8% / 42.1%**. This **echoes** **conditioning sensitivity** for 1e and reinforces that **sampler choice** is not monotone; it **interacts** with the auxiliary’s role.

### F. What a Euclidean‑\(\mathcal{L}_{\mathrm{dom}}\) sweep would likely change (hypothesis)

Based on **1a+1c** crossing behavior:

- **Expect** weaker or **reordered** gains from **domain** drift conditioning when \(\mathcal{L}_{\mathrm{dom}}\) is Euclidean.

- **Do not expect** **1d+1e** to become strong; the **interaction failure** is likely **loss‑algebraic** as much as **purely** from \(\mathcal{L}_{\mathrm{dom}}\) geometry.

---

## Bottom line for the main text

**Camera‑labeled drift objectives 1d and 1e do not improve the Idea‑1 frontier on mAP.** **1d with domain drift sampling** is the **only** new arm that remains **competitive on R1** but **still below 1c**. **1d+1e** is **counterproductive**. **Empty Euclidean‑\(\mathcal{L}_{\mathrm{dom}}\)** table cells for 1d/1e are **unmeasured**, not evidence of zero effect.

A **narrow** Euclidean‑\(\mathcal{L}_{\mathrm{dom}}\) replication is **warranted for completeness and reviewer‑proofing**, not because current Finsler‑\(\mathcal{L}_{\mathrm{dom}}\) evidence suggests a **missed** large gain.

---

## Publication phase note

When `paper/draft_4/` follows `.cursor/paper/publication_facts.md`, respect **Drafting_phase**: in **results_implementation_lock**, abstract and claims may cite provenanced scores and protocol details; align wording with `paper-code-truth` and the sweep ledgers above.
