# Sweep: `sweep_loss_ablation_Idea1_EuclideanDom` (job 1514816)

Metrics taken from **final evaluation** (`Loaded best model for final evaluation` → second `Mean AP` / `top-1` block) in each `log.txt` under `logs/sweep_loss_ablation_Idea1_EuclideanDom/`.

## Copy-paste table (Euclidean domain loss)

| Short Name | Drift Conditioning | Euc. Dom. mAP (%) | Euc. Dom. Rank-1 (%) |
|------------|---------------------|-------------------|----------------------|
| **baseline_1a_only** | instance | 43.0 | 42.9 |
| | domain | 43.2 | 43.2 |
| **cam_crossview_contrast** | instance | 41.6 | 41.9 |
| | domain | 40.7 | 40.7 |
| **domain_triplet_no_memory_domain** | instance | 23.8 | 27.7 |
| | domain | 23.3 | 27.1 |
| **drift_only_cross_camera_l2** | instance | 43.1 | 42.6 |
| | domain | 42.7 | 42.3 |
| **domain_triplet_with_memory_domain** | instance | 27.5 | 28.0 |
| | domain | 25.7 | 27.1 |

## Log paths (job 1514816)

| Arm | driftInst | driftDom |
|-----|-----------|----------|
| 00 baseline | `job_1514816_00_baseline_1a_only_driftInst_euclLdom/log.txt` | `job_1514816_00_baseline_1a_only_driftDom_euclLdom/log.txt` |
| 01 cam xdom | `job_1514816_01_cam_xdom_w0.1_driftInst_euclLdom/log.txt` | `job_1514816_01_cam_xdom_w0.1_driftDom_euclLdom/log.txt` |
| 02 dom tri no Ldom | `job_1514816_02_domTri_cam_w1.0_noLdom_driftInst_euclLdom/log.txt` | `job_1514816_02_domTri_cam_w1.0_noLdom_driftDom_euclLdom/log.txt` |
| 03 drift only | `job_1514816_03_driftOnly_xcam_w0.1_driftInst_euclLdom/log.txt` | `job_1514816_03_driftOnly_xcam_w0.1_driftDom_euclLdom/log.txt` |
| 04 dom tri + Ldom | `job_1514816_04_domTri_cam_w1.0_withLdom_driftInst_euclLdom/log.txt` | `job_1514816_04_domTri_cam_w1.0_withLdom_driftDom_euclLdom/log.txt` |
