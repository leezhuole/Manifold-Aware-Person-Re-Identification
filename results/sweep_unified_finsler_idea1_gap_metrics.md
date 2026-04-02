# Sweep: `sweep_unified_finsler_idea1_gap` (job 1516604)

Metrics from **`Loaded best model for final evaluation`** → the following **`Mean AP`** and **`top-1`** lines in each `log.txt` under `logs/sweep_unified_finsler_idea1_gap/`.

Configuration recap:

- **1b (original):** `--no-triplet --use-domain-triplet` (camera mining, weight 1.0), no PID triplet.
- **Unified Finsler:** single Finsler batch-hard triplet on PIDs (full embedding), no domain triplet.
- **Fins. Dom.:** `--memory-bank-mode full` + `--use-omega-reg`.
- **Euc. Dom.:** `--use-euclidean-domain-loss` (identity memory bank).
- **Instance drift:** `RandomIdentitySampler`; **domain drift:** `RandomMultipleGallery` (per current sbatch).

---

## Copy-paste: row **1b** (all cells)

| Short Name | Drift Cond. | Fins. Dom. mAP | Fins. Dom. R1 | Euc. Dom. mAP | Euc. Dom. R1 |
|------------|-------------|----------------|---------------|---------------|--------------|
| **1b** | instance | 35.1 | 33.9 | 30.6 | 30.6 |
| | domain | 33.2 | 34.1 | 25.6 | 27.0 |

---

## Copy-paste: row **Unified Finsler** — **Euc. Dom.** only (Fins. columns from this job)

| Short Name | Drift Cond. | Euc. Dom. mAP | Euc. Dom. R1 |
|------------|-------------|---------------|--------------|
| **Unified Finsler** | instance | 43.4 | 43.9 |
| | domain | 43.4 | 43.0 |

**Fins. Dom. (same job, for cross-check vs your slide):**

| Drift Cond. | Fins. Dom. mAP | Fins. Dom. R1 |
|-------------|----------------|---------------|
| instance | 44.1 | 44.0 |
| domain | 44.1 | 44.3 |

---

## Log paths (job 1516604)

| Table row | L_dom | driftInst | driftDom |
|-----------|-------|-----------|----------|
| **1b** | Fins | `job_1516604_idea1bOnly_domTriW1.0__driftInst__LdomFins/log.txt` | `job_1516604_idea1bOnly_domTriW1.0__driftDom__LdomFins/log.txt` |
| **1b** | Eucl | `job_1516604_idea1bOnly_domTriW1.0__driftInst__LdomEucl/log.txt` | `job_1516604_idea1bOnly_domTriW1.0__driftDom__LdomEucl/log.txt` |
| **Unified** | Fins | `job_1516604_unifiedFin_onlyTri__driftInst__LdomFins/log.txt` | `job_1516604_unifiedFin_onlyTri__driftDom__LdomFins/log.txt` |
| **Unified** | Eucl | `job_1516604_unifiedFin_onlyTri__driftInst__LdomEucl/log.txt` | `job_1516604_unifiedFin_onlyTri__driftDom__LdomEucl/log.txt` |
