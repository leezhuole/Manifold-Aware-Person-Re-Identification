# Updated Sweep Metrics (Idea-1 Ablations Re-Run)

Metrics extracted from the final evaluation block (`Loaded best model for final evaluation` -> `Mean AP` / `top-1`) in the `log.txt` files located under `logs/sweep_loss_ablation_Idea1_2/` and `logs/sweep_loss_ablation_Idea1_EuclideanDom_2/`.

Note: The run `1a + 1b` with `instance` conditioning and `Euc. Dom.` (`04_domTri_cam_w1.0_withLdom_driftInst_euclLdom`) is currently still running, hence the missing values for that cell.

---

### Copy-paste Table

| Short Name | Description | Drift Cond. | Fins. Dom. mAP | Fins. Dom. R1 | Euc. Dom. mAP | Euc. Dom. R1 |
|------------|-------------|-------------|----------------|---------------|---------------|--------------|
| 1a | Baseline: Euclidean identity triplet only. | instance | 43.6 | 44.1 | 43.9 | 43.9 |
| | | domain | 43.7 | 43.2 | 43.8 | 43.9 |
| 1b | Domain Triplet Only: Finsler triplet mining camera labels | instance | 35.1 | 33.9 | 30.6 | 30.6 |
| | | domain | 33.2 | 34.1 | 25.6 | 27.0 |
| 1a + 1b | Conflicting Objectives: Both are active simultaneously | instance | 36.2 | 36.4 | 36.9 | 36.2 |
| | | domain | 36.1 | 34.9 | 37.8 | 38.9 |
| 1a + 1b<br>(no `L_dom`) | Same as above + disabling BAU domain repulsion to isolate domain-attraction. | instance | 36.4 | 34.6 | 33.6 | 35.0 |
| | | domain | 28.5 | 24.9 | 26.4 | 25.5 |
| 1a + Refined 1b | Cross-Camera Contrastive: Shrinks Finsler distance $d_F$ for the *same* identity across *different* cameras | instance | 43.7 | 43.1 | 43.4 | 42.8 |
| | | domain | 43.2 | 42.1 | 43.1 | 42.4 |
| 1a + 1c | Drift $L_2$ Alignment: Explicitly penalizes the $L_2$ variance of drift vectors for the same identity across cameras. | instance | 43.4 | 43.0 | 42.9 | 43.8 |
| | | domain | 44.9 | 43.8 | 42.9 | 43.7 |
| Unified Finsler | Single Finsler Triplet: Replaces the Euclidean identity metric with the full asymmetric Finsler distance $d_F$. | instance | 44.1 | 44.0 | 43.4 | 43.9 |
| | | domain | 44.1 | 44.3 | 43.4 | 43.0 |

*(Note: "1b" and "Unified Finsler" rows carry over the values parsed from `sweep_unified_finsler_idea1_gap` as requested previously, as they were not part of the `sweep_loss_ablation_Idea1_2` script array.)*

---

### File Origins (for completed tasks)

**Fins. Dom.** (`logs/sweep_loss_ablation_Idea1_2`)
- **1a** (Inst): `job_1516935_00_baseline_1a_only_driftInst` (mAP: 44.1, R1: 43.6)
- **1a** (Dom): `job_1516935_00_baseline_1a_only_driftDom` (mAP: 43.2, R1: 43.7)
- **1a+1b** (Inst): `job_1516935_04_domTri_cam_w1.0_withLdom_driftInst` (mAP: 36.4, R1: 36.2)
- **1a+1b** (Dom): `job_1516935_04_domTri_cam_w1.0_withLdom_driftDom` (mAP: 34.9, R1: 36.1)
- **1a+1b, no L_dom** (Inst): `job_1516935_02_domTri_cam_w1.0_noLdom_driftInst` (mAP: 34.6, R1: 36.4)
- **1a+1b, no L_dom** (Dom): `job_1516935_02_domTri_cam_w1.0_noLdom_driftDom` (mAP: 24.9, R1: 28.5)
- **1a+Refined 1b** (Inst): `job_1516935_01_cam_xdom_w0.1_driftInst` (mAP: 43.1, R1: 43.7)
- **1a+Refined 1b** (Dom): `job_1516935_01_cam_xdom_w0.1_driftDom` (mAP: 42.1, R1: 43.2)
- **1a+1c** (Inst): `job_1516935_03_driftOnly_xcam_w0.1_driftInst` (mAP: 43.0, R1: 43.4)
- **1a+1c** (Dom): `job_1516935_03_driftOnly_xcam_w0.1_driftDom` (mAP: 43.8, R1: 44.9)

**Euc. Dom.** (`logs/sweep_loss_ablation_Idea1_EuclideanDom_2`)
- **1a** (Inst): `job_1517230_00_baseline_1a_only_driftInst_euclLdom` (mAP: 43.9, R1: 43.9)
- **1a** (Dom): `job_1517230_00_baseline_1a_only_driftDom_euclLdom` (mAP: 43.8, R1: 43.9)
- **1a+1b** (Inst): `job_1517230_04_domTri_cam_w1.0_withLdom_driftInst_euclLdom` *(Still running! PID 1517230_8)*
- **1a+1b** (Dom): `job_1517230_04_domTri_cam_w1.0_withLdom_driftDom_euclLdom` (mAP: 37.8, R1: 38.9)
- **1a+1b, no L_dom** (Inst): `job_1517230_02_domTri_cam_w1.0_noLdom_driftInst_euclLdom` (mAP: 33.6, R1: 35.0)
- **1a+1b, no L_dom** (Dom): `job_1517230_02_domTri_cam_w1.0_noLdom_driftDom_euclLdom` (mAP: 25.5, R1: 26.4)
- **1a+Refined 1b** (Inst): `job_1517230_01_cam_xdom_w0.1_driftInst_euclLdom` (mAP: 42.8, R1: 43.4)
- **1a+Refined 1b** (Dom): `job_1517230_01_cam_xdom_w0.1_driftDom_euclLdom` (mAP: 42.4, R1: 43.1)
- **1a+1c** (Inst): `job_1517230_03_driftOnly_xcam_w0.1_driftInst_euclLdom` (mAP: 43.8, R1: 42.9)
- **1a+1c** (Dom): `job_1517230_03_driftOnly_xcam_w0.1_driftDom_euclLdom` (mAP: 43.7, R1: 42.9)